# conversation.py
"""
✅ ФИНАЛЬНАЯ ВЕРСИЯ v3: Потокобезопасный менеджер диалогов с Машиной Состояний.
- Добавлен метод analyze_message_for_state_transition для управления логикой диалога.
- Сохранена отказоустойчивость и потокобезопасность.
"""

import redis
import threading
import logging
import time
from typing import List, Dict, Any, Optional
from config import config


class ReadWriteLock:
    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
    def acquire_read(self, timeout: float = 5.0): return ReadContext(self, timeout)
    def acquire_write(self, timeout: float = 5.0): return WriteContext(self, timeout)
    def _acquire_read_internal(self, timeout: float) -> bool:
        if not self._read_ready.acquire(timeout=timeout): return False
        try:
            self._readers += 1
            return True
        finally: self._read_ready.release()
    def _release_read_internal(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0: self._read_ready.notifyAll()
    def _acquire_write_internal(self, timeout: float) -> bool:
        if not self._read_ready.acquire(timeout=timeout): return False
        try:
            start_time = time.time()
            while self._readers > 0:
                remaining_timeout = timeout - (time.time() - start_time)
                if remaining_timeout <= 0: return False
                self._read_ready.wait(remaining_timeout)
            return True
        finally: pass
    def _release_write_internal(self):
        try: self._read_ready.release()
        except Exception: pass

class ReadContext:
    def __init__(self, rw_lock: ReadWriteLock, timeout: float):
        self.rw_lock = rw_lock
        self.timeout = timeout
        self.acquired = False
    def __enter__(self):
        self.acquired = self.rw_lock._acquire_read_internal(self.timeout)
        if not self.acquired: raise TimeoutError("Failed to acquire read lock")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired: self.rw_lock._release_read_internal()

class WriteContext:
    def __init__(self, rw_lock: ReadWriteLock, timeout: float):
        self.rw_lock = rw_lock
        self.timeout = timeout
        self.acquired = False
    def __enter__(self):
        self.acquired = self.rw_lock._acquire_write_internal(self.timeout)
        if not self.acquired: raise TimeoutError("Failed to acquire write lock")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired: self.rw_lock._release_write_internal()


class ConversationManager:
    """
    THREAD-SAFE версия менеджера диалогов с Машиной Состояний.
    """
    # ✅ ДОБАВЛЕНО: Определения состояний и ключевых слов перенесены сюда
    DIALOGUE_STATES = {
        'greeting': 'Приветствие и первое знакомство',
        'problem_solving': 'Решение проблем ребенка, консультирование', 
        'fact_finding': 'Поиск информации о курсах, ценах, расписании',
        'closing': 'Готовность к записи на пробный урок'
    }
    STATE_KEYWORDS = {
        'problem_solving': ['проблем', 'сложно', 'трудно', 'застенчив', 'боится', 
                           'не слушается', 'агрессивн', 'замкн', 'помогите'],
        'fact_finding': ['цена', 'стоимость', 'расписание', 'время', 'когда', 
                        'сколько', 'преподаватель', 'группа', 'возраст', 'расскажите'],
        'closing': ['записат', 'попробова', 'хочу', 'готов', 'решил', 
                   'интересно', 'согласен', 'давайте']
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        self.redis_available = False
        self._init_redis()
        
        self.fallback_memory = {}
        self.fallback_memory_lock = threading.RLock()
        
        self.user_rw_locks = {}
        self.user_locks_lock = threading.RLock()
        
        self.performance_stats = {
            'successful_reads': 0, 'successful_writes': 0,
            'timeout_errors': 0, 'redis_errors': 0, 'fallback_usage': 0
        }
        self.stats_lock = threading.Lock()
        
        self.logger.info("🧠 Thread-safe менеджер диалогов (v3) инициализирован")
        
        if config.CLEAR_MEMORY_ON_START:
            self.clear_all_conversations()
            self.logger.info("🧹 Вся память диалогов очищена при старте (режим тестирования)")
    
    def _init_redis(self):
        try:
            if config.REDIS_URL:
                self.redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
                self.redis_client.ping()
                self.redis_available = True
                self.logger.info("✅ Redis соединение установлено")
            else:
                self.logger.info("ℹ️ Redis URL не найден, используется fallback память")
        except Exception as e:
            self.logger.warning(f"⚠️ Redis недоступен: {e}. Используется fallback память")
            self.redis_available = False
    
    def _get_user_rw_lock(self, chat_id: str) -> ReadWriteLock:
        with self.user_locks_lock:
            if chat_id not in self.user_rw_locks:
                self.user_rw_locks[chat_id] = ReadWriteLock()
            return self.user_rw_locks[chat_id]
    
    def _normalize_chat_id(self, chat_id: str) -> str:
        return str(chat_id).strip() if chat_id else ""
    
    def get_dialogue_state(self, chat_id: str) -> str:
        chat_id = self._normalize_chat_id(chat_id)
        if not chat_id: return 'greeting'
        user_rw_lock = self._get_user_rw_lock(chat_id)
        try:
            with user_rw_lock.acquire_read(timeout=3.0):
                if self.redis_available:
                    state = self.redis_client.get(f"state:{chat_id}")
                    return state if state in self.DIALOGUE_STATES else 'greeting'
                with self.fallback_memory_lock:
                    return self.fallback_memory.get(chat_id, {}).get('state', 'greeting')
        except (TimeoutError, redis.exceptions.RedisError) as e:
            self.logger.warning(f"Ошибка получения состояния для {chat_id}: {e}, используется fallback")
            with self.fallback_memory_lock:
                return self.fallback_memory.get(chat_id, {}).get('state', 'greeting')

    def set_dialogue_state(self, chat_id: str, state: str):
        chat_id = self._normalize_chat_id(chat_id)
        if not chat_id or state not in self.DIALOGUE_STATES: return
        user_rw_lock = self._get_user_rw_lock(chat_id)
        try:
            with user_rw_lock.acquire_write(timeout=3.0):
                if self.redis_available:
                    self.redis_client.setex(f"state:{chat_id}", config.CONVERSATION_EXPIRATION_SECONDS, state)
                    return
                with self.fallback_memory_lock:
                    if chat_id not in self.fallback_memory: self.fallback_memory[chat_id] = {}
                    self.fallback_memory[chat_id]['state'] = state
        except (TimeoutError, redis.exceptions.RedisError) as e:
            self.logger.warning(f"Ошибка установки состояния для {chat_id}: {e}, используется fallback")
            with self.fallback_memory_lock:
                 if chat_id not in self.fallback_memory: self.fallback_memory[chat_id] = {}
                 self.fallback_memory[chat_id]['state'] = state

    def get_conversation_history(self, chat_id: str) -> List[str]:
        chat_id = self._normalize_chat_id(chat_id)
        if not chat_id: return []
        user_rw_lock = self._get_user_rw_lock(chat_id)
        try:
            with user_rw_lock.acquire_read(timeout=3.0):
                if self.redis_available:
                    return self.redis_client.lrange(f"history:{chat_id}", 0, -1)
                with self.fallback_memory_lock:
                    return self.fallback_memory.get(chat_id, {}).get('history', [])
        except (TimeoutError, redis.exceptions.RedisError) as e:
            self.logger.warning(f"Ошибка получения истории для {chat_id}: {e}, используется fallback")
            with self.fallback_memory_lock:
                return self.fallback_memory.get(chat_id, {}).get('history', [])

    def update_conversation_history(self, chat_id: str, user_message: str, ai_response: str):
        chat_id = self._normalize_chat_id(chat_id)
        if not chat_id or not user_message: return
        user_rw_lock = self._get_user_rw_lock(chat_id)
        user_entry = f"Пользователь: {user_message}"
        ai_entry = f"Ассистент: {ai_response}"
        try:
            with user_rw_lock.acquire_write(timeout=3.0):
                if self.redis_available:
                    pipe = self.redis_client.pipeline()
                    pipe.lpush(f"history:{chat_id}", ai_entry, user_entry)
                    pipe.ltrim(f"history:{chat_id}", 0, (config.CONVERSATION_MEMORY_SIZE * 2) - 1)
                    pipe.expire(f"history:{chat_id}", config.CONVERSATION_EXPIRATION_SECONDS)
                    pipe.execute()
                    return
                with self.fallback_memory_lock:
                    if chat_id not in self.fallback_memory: self.fallback_memory[chat_id] = {'history': [], 'state': 'greeting'}
                    history = self.fallback_memory[chat_id]['history']
                    history.extend([user_entry, ai_entry])
                    max_lines = config.CONVERSATION_MEMORY_SIZE * 2
                    if len(history) > max_lines:
                        self.fallback_memory[chat_id]['history'] = history[-max_lines:]
                    self.fallback_memory[chat_id]['last_update'] = time.time()
                    self._cleanup_fallback_memory()
        except (TimeoutError, redis.exceptions.RedisError) as e:
             self.logger.warning(f"Ошибка обновления истории для {chat_id}: {e}, используется fallback")
             with self.fallback_memory_lock:
                if chat_id not in self.fallback_memory: self.fallback_memory[chat_id] = {'history': [], 'state': 'greeting'}
                history = self.fallback_memory[chat_id]['history']
                history.extend([user_entry, ai_entry])


    def _cleanup_fallback_memory(self):
        if len(self.fallback_memory) > config.MAX_FALLBACK_USERS:
            sorted_entries = sorted(self.fallback_memory.items(), key=lambda x: x[1].get('last_update', 0))
            entries_to_remove = sorted_entries[:len(self.fallback_memory)//2]
            for chat_id, _ in entries_to_remove: del self.fallback_memory[chat_id]
            if entries_to_remove: self.logger.info(f"🧹 Очищена fallback память: удалено {len(entries_to_remove)} старых записей")

    # ✅ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: МЕТОД ПЕРЕНЕСЕН СЮДА
    def analyze_message_for_state_transition(self, user_message: str, current_state: str) -> str:
        """
        Анализирует сообщение пользователя и определяет новое состояние диалога.
        """
        if not user_message:
            return current_state
        
        message_lower = user_message.lower()
        
        # Анализ по ключевым словам для смены состояния
        for state, keywords in self.STATE_KEYWORDS.items():
            if any(keyword in message_lower for keyword in keywords):
                # Если нашли ключевое слово, возвращаем новое состояние
                if state != current_state:
                    self.logger.info(f"Смена состояния по ключевому слову '{keywords[0]}...': {current_state} -> {state}")
                return state
        
        # Если ключевых слов не найдено, состояние не меняется
        return current_state

    def clear_all_conversations(self):
        self.logger.info("🧹 Запущена процедура очистки всей памяти...")
        if self.redis_available and self.redis_client:
            try:
                keys_to_delete = list(self.redis_client.scan_iter("*:*"))
                if keys_to_delete: self.redis_client.delete(*keys_to_delete)
            except Exception as e:
                self.logger.error(f"Ошибка при попытке очистки Redis: {e}", exc_info=True)
        with self.fallback_memory_lock: self.fallback_memory.clear()
        with self.user_locks_lock: self.user_rw_locks.clear()
        self.logger.info("✅ Процедура очистки памяти завершена.")

conversation_manager = ConversationManager()