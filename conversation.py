# conversation.py
"""
Модуль для управления состояниями диалога и памятью разговоров.
Отвечает за понимание контекста беседы, переходы между состояниями
и сохранение истории общения с каждым пользователем.
"""

import redis
import threading
import logging
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from config import config


class ConversationManager:
    """
    Класс для управления состояниями диалога и памятью разговоров.
    
    Принципы работы:
    1. Каждый пользователь имеет свое состояние диалога (greeting, fact_finding, problem_solving, closing)
    2. История сообщений сохраняется в Redis (или fallback памяти)
    3. Состояния переключаются на основе анализа сообщений пользователя
    4. Все операции thread-safe для работы в многопоточной среде
    """
    
    # Определяем возможные состояния диалога
    DIALOGUE_STATES = {
        'greeting': 'Приветствие и первое знакомство',
        'problem_solving': 'Решение проблем ребенка, консультирование', 
        'fact_finding': 'Поиск информации о курсах, ценах, расписании',
        'closing': 'Готовность к записи на пробный урок'
    }
    
    # Ключевые слова для автоматического определения состояний
    STATE_KEYWORDS = {
        'problem_solving': ['проблем', 'сложно', 'трудно', 'застенчив', 'боится', 
                           'не слушается', 'агрессивн', 'замкн', 'помогите'],
        'fact_finding': ['цена', 'стоимость', 'расписание', 'время', 'когда', 
                        'сколько', 'преподаватель', 'группа', 'возраст'],
        'closing': ['записат', 'попробова', 'хочу', 'готов', 'решил', 
                   'интересно', 'согласен', 'давайте']
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Инициализируем Redis соединение с обработкой ошибок
        self.redis_client = None
        self.redis_available = False
        self._init_redis()
        
        # Fallback память для случая, когда Redis недоступен
        self.fallback_memory = {}
        self.fallback_memory_lock = threading.Lock()
        
        # Thread safety: блокировки для каждого пользователя
        self.user_locks = {}
        self.user_locks_lock = threading.Lock()
        
        self.logger.info("🧠 Менеджер диалогов инициализирован")
    
    def _init_redis(self):
        """
        Инициализирует Redis соединение с graceful degradation.
        Если Redis недоступен, система будет работать с fallback памятью.
        """
        if not config.REDIS_URL:
            self.logger.info("🔶 Redis URL не настроен, используем fallback память")
            return
        
        try:
            self.redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
            self.redis_client.ping()  # Проверяем соединение
            self.redis_available = True
            self.logger.info("✅ Redis подключен успешно")
        except Exception as e:
            self.redis_available = False
            self.logger.warning(f"⚠️ Redis недоступен: {e}")
            self.logger.info("🔄 Система будет работать с fallback памятью")
    
    def _get_user_lock(self, chat_id: str) -> threading.Lock:
        """
        Получает блокировку для конкретного пользователя.
        Это критически важно для предотвращения race conditions
        в многопоточной среде.
        """
        chat_id = str(chat_id)
        with self.user_locks_lock:
            if chat_id not in self.user_locks:
                self.user_locks[chat_id] = threading.Lock()
            return self.user_locks[chat_id]
    
    def _normalize_chat_id(self, chat_id) -> str:
        """Нормализует chat_id для консистентного использования"""
        if chat_id is None:
            return ""
        return str(chat_id)
    
    def get_dialogue_state(self, chat_id: str) -> str:
        """
        Получает текущее состояние диалога для пользователя.
        
        Алгоритм:
        1. Пытаемся получить из Redis
        2. Если Redis недоступен или состояние не найдено, анализируем историю
        3. Если история пустая, возвращаем 'greeting'
        """
        chat_id = self._normalize_chat_id(chat_id)
        if not chat_id:
            return 'greeting'
        
        user_lock = self._get_user_lock(chat_id)
        with user_lock:
            # Пытаемся получить из Redis
            if self.redis_available:
                try:
                    state_key = f"state:{chat_id}"
                    state = self.redis_client.get(state_key)
                    if state and state in self.DIALOGUE_STATES:
                        return state
                except Exception as e:
                    self.logger.warning(f"Ошибка чтения состояния из Redis: {e}")
            
            # Если не нашли в Redis, анализируем историю сообщений
            history = self._get_conversation_history_internal(chat_id)
            return self._infer_state_from_history(history)
    
    def _infer_state_from_history(self, history: List[str]) -> str:
        """
        Выводит состояние диалога на основе истории сообщений.
        Использует эвристический анализ ключевых слов.
        """
        if not history:
            return 'greeting'
        
        # Анализируем последние 4 сообщения пользователя
        user_messages = [msg for msg in history if msg.startswith("Пользователь:")][-4:]
        recent_text = ' '.join(user_messages).lower()
        
        # Ищем ключевые слова для определения состояния
        for state, keywords in self.STATE_KEYWORDS.items():
            if any(keyword in recent_text for keyword in keywords):
                self.logger.info(f"Определено состояние '{state}' по ключевым словам")
                return state
        
        # Fallback логика на основе длины истории
        if len(history) < 4:
            return 'greeting'
        elif len(history) < 8:
            return 'fact_finding'
        else:
            return 'problem_solving'
    
    def update_dialogue_state(self, chat_id: str, new_state: str):
        """
        Обновляет состояние диалога для пользователя.
        Thread-safe операция с записью в Redis и fallback.
        """
        chat_id = self._normalize_chat_id(chat_id)
        if not chat_id or new_state not in self.DIALOGUE_STATES:
            self.logger.warning(f"Невалидные параметры состояния: {chat_id}, {new_state}")
            return
        
        user_lock = self._get_user_lock(chat_id)
        with user_lock:
            if self.redis_available:
                try:
                    state_key = f"state:{chat_id}"
                    self.redis_client.set(state_key, new_state, ex=config.CONVERSATION_EXPIRATION_SECONDS)
                    self.logger.info(f"Состояние диалога обновлено: {chat_id} -> {new_state}")
                except Exception as e:
                    self.logger.error(f"Ошибка записи состояния в Redis: {e}")
    
    def get_conversation_history(self, chat_id: str) -> List[str]:
        """
        Получает историю диалога для пользователя.
        Возвращает список строк в формате ["Пользователь: ...", "Ассистент: ..."]
        """
        chat_id = self._normalize_chat_id(chat_id)
        if not chat_id:
            return []
        
        user_lock = self._get_user_lock(chat_id)
        with user_lock:
            return self._get_conversation_history_internal(chat_id)
    
    def _get_conversation_history_internal(self, chat_id: str) -> List[str]:
        """
        Внутренний метод получения истории (без дополнительной блокировки).
        Используется когда блокировка уже установлена вызывающим кодом.
        """
        if self.redis_available:
            try:
                history_key = f"history:{chat_id}"
                # Redis возвращает в обратном порядке, исправляем
                return self.redis_client.lrange(history_key, 0, -1)[::-1]
            except Exception as e:
                self.logger.warning(f"Ошибка чтения истории из Redis: {e}")
        
        # Fallback на локальную память
        with self.fallback_memory_lock:
            return self.fallback_memory.get(chat_id, [])
    
    def update_conversation_history(self, chat_id: str, user_message: str, ai_response: str):
        """
        Обновляет историю диалога, добавляя новое сообщение пользователя и ответ AI.
        
        Алгоритм:
        1. Очищаем токены действий из ответа AI (для чистой истории)
        2. Добавляем сообщения в Redis/fallback
        3. Обрезаем историю до максимального размера
        4. Устанавливаем время жизни записи
        """
        chat_id = self._normalize_chat_id(chat_id)
        if not chat_id or not user_message:
            return
        
        # Очищаем токены действий перед сохранением
        clean_response = ai_response.replace("[ACTION:SEND_LESSON_LINK]", "[ССЫЛКА_НА_УРОК]")
        
        user_lock = self._get_user_lock(chat_id)
        with user_lock:
            timestamp = datetime.now().isoformat()
            
            if self.redis_available:
                try:
                    history_key = f"history:{chat_id}"
                    metadata_key = f"metadata:{chat_id}"
                    
                    # Используем pipeline для атомарности операций
                    pipe = self.redis_client.pipeline()
                    pipe.lpush(history_key, f"Ассистент: {clean_response}")
                    pipe.lpush(history_key, f"Пользователь: {user_message}")
                    pipe.ltrim(history_key, 0, (config.CONVERSATION_MEMORY_SIZE * 2) - 1)
                    pipe.expire(history_key, config.CONVERSATION_EXPIRATION_SECONDS)
                    
                    # Сохраняем метаданные
                    metadata = {
                        "last_activity": timestamp,
                        "message_count": len(self._get_conversation_history_internal(chat_id)) // 2 + 1
                    }
                    pipe.hset(metadata_key, mapping=metadata)
                    pipe.expire(metadata_key, config.CONVERSATION_EXPIRATION_SECONDS)
                    
                    pipe.execute()
                    
                except Exception as e:
                    self.logger.error(f"Ошибка записи истории в Redis: {e}")
                    self._update_fallback_memory(chat_id, user_message, clean_response)
            else:
                self._update_fallback_memory(chat_id, user_message, clean_response)
    
    def _update_fallback_memory(self, chat_id: str, user_message: str, ai_response: str):
        """
        Обновляет fallback память когда Redis недоступен.
        Включает автоматическую очистку для предотвращения переполнения памяти.
        """
        with self.fallback_memory_lock:
            if chat_id not in self.fallback_memory:
                self.fallback_memory[chat_id] = []
            
            self.fallback_memory[chat_id].append(f"Пользователь: {user_message}")
            self.fallback_memory[chat_id].append(f"Ассистент: {ai_response}")
            
            # Обрезаем до максимального размера
            max_lines = config.CONVERSATION_MEMORY_SIZE * 2
            if len(self.fallback_memory[chat_id]) > max_lines:
                self.fallback_memory[chat_id] = self.fallback_memory[chat_id][-max_lines:]
            
            # Периодическая очистка памяти
            self._cleanup_fallback_memory()
    
    def _cleanup_fallback_memory(self):
        """
        Очищает fallback память от лишних записей для предотвращения переполнения.
        Вызывается автоматически при обновлении истории.
        """
        if len(self.fallback_memory) > config.MAX_FALLBACK_USERS:
            # Удаляем половину самых старых записей
            old_keys = list(self.fallback_memory.keys())[:len(self.fallback_memory)//2]
            for key in old_keys:
                del self.fallback_memory[key]
            self.logger.info(f"Очищена fallback память: удалено {len(old_keys)} старых записей")
    
    def analyze_message_for_state_transition(self, user_message: str, current_state: str) -> str:
        """
        Анализирует сообщение пользователя и определяет новое состояние диалога.
        
        Алгоритм:
        1. Проверяем прямые запросы урока (-> closing)
        2. Анализируем ключевые слова
        3. Для коротких сообщений используем простую логику
        4. Для длинных сообщений используем более сложный анализ
        """
        if not user_message:
            return current_state
        
        message_lower = user_message.lower()
        
        # Прямые запросы урока имеют высший приоритет
        direct_lesson_keywords = ["пробн", "бесплатн", "попробова", "записат", "хочу урок", "дайте ссылку"]
        if any(word in message_lower for word in direct_lesson_keywords):
            self.logger.info("Детектирован прямой запрос урока -> closing")
            return 'closing'
        
        # Анализ по ключевым словам
        for state, keywords in self.STATE_KEYWORDS.items():
            if any(keyword in message_lower for keyword in keywords):
                self.logger.info(f"Детектировано состояние '{state}' по ключевым словам")
                return state
        
        # Для коротких сообщений (менее 5 слов) состояние обычно не меняется
        if len(user_message.split()) < 5:
            return current_state
        
        # Логика переходов по умолчанию для длинных сообщений
        if current_state == 'greeting':
            return 'fact_finding'
        elif current_state == 'fact_finding' and len(user_message.split()) > 10:
            return 'problem_solving'
        
        return current_state
    
    def get_conversation_stats(self, chat_id: str) -> Dict[str, Any]:
        """
        Возвращает статистику диалога для пользователя.
        Полезно для отладки и мониторинга.
        """
        chat_id = self._normalize_chat_id(chat_id)
        history = self.get_conversation_history(chat_id)
        current_state = self.get_dialogue_state(chat_id)
        
        return {
            "chat_id": chat_id,
            "current_state": current_state,
            "message_count": len(history),
            "redis_available": self.redis_available,
            "last_messages": history[-4:] if history else []
        }


# Создаем глобальный экземпляр менеджера диалогов
conversation_manager = ConversationManager()
