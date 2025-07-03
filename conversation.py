# conversation.py (CRITICAL THREADING FIX)
"""
КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Устранение deadlocks при parallel execution

ПРОБЛЕМА: app.py использует ThreadPoolExecutor для параллельных операций:
- get_dialogue_state(chat_id)  
- get_conversation_history(chat_id)
- update_conversation_history(chat_id, ...)

При одновременном доступе к одному chat_id возможны deadlocks из-за user-level locks.

РЕШЕНИЕ:
1. Timeout-based locks для предотвращения deadlocks
2. Read-Write lock pattern для concurrent reads
3. Atomic operations для thread-safe updates
4. Graceful degradation при lock timeouts
"""

import redis
import threading
import logging
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from config import config


class TimeoutLock:
    """
    Lock с timeout для предотвращения deadlocks
    """
    
    def __init__(self, timeout: float = 5.0):
        self._lock = threading.Lock()
        self.timeout = timeout
        self.logger = logging.getLogger(f"{__name__}.TimeoutLock")
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Попытка получить блокировку с timeout"""
        timeout = timeout or self.timeout
        return self._lock.acquire(timeout=timeout)
    
    def release(self):
        """Освобождение блокировки"""
        try:
            self._lock.release()
        except Exception as e:
            self.logger.warning(f"Lock release warning: {e}")
    
    def __enter__(self):
        acquired = self.acquire()
        if not acquired:
            raise TimeoutError("Failed to acquire lock within timeout")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class ReadWriteLock:
    """
    Read-Write lock для concurrent reads без блокировки
    """
    
    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
        
    def acquire_read(self, timeout: float = 5.0):
        """Получить read lock"""
        return ReadContext(self, timeout)
    
    def acquire_write(self, timeout: float = 5.0):
        """Получить write lock"""
        return WriteContext(self, timeout)
    
    def _acquire_read_internal(self, timeout: float) -> bool:
        """Внутренний метод для read lock"""
        if not self._read_ready.acquire(timeout=timeout):
            return False
        try:
            self._readers += 1
            return True
        finally:
            self._read_ready.release()
    
    def _release_read_internal(self):
        """Внутренний метод для release read lock"""
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notifyAll()
    
    def _acquire_write_internal(self, timeout: float) -> bool:
        """Внутренний метод для write lock"""
        if not self._read_ready.acquire(timeout=timeout):
            return False
        try:
            start_time = time.time()
            while self._readers > 0:
                remaining_timeout = timeout - (time.time() - start_time)
                if remaining_timeout <= 0:
                    return False
                self._read_ready.wait(remaining_timeout)
            return True
        finally:
            pass  # Не освобождаем lock здесь - это сделает WriteContext
    
    def _release_write_internal(self):
        """Внутренний метод для release write lock"""
        try:
            self._read_ready.release()
        except Exception:
            pass


class ReadContext:
    """Context manager для read operations"""
    
    def __init__(self, rw_lock: ReadWriteLock, timeout: float):
        self.rw_lock = rw_lock
        self.timeout = timeout
        self.acquired = False
    
    def __enter__(self):
        self.acquired = self.rw_lock._acquire_read_internal(self.timeout)
        if not self.acquired:
            raise TimeoutError("Failed to acquire read lock within timeout")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            self.rw_lock._release_read_internal()


class WriteContext:
    """Context manager для write operations"""
    
    def __init__(self, rw_lock: ReadWriteLock, timeout: float):
        self.rw_lock = rw_lock
        self.timeout = timeout
        self.acquired = False
    
    def __enter__(self):
        self.acquired = self.rw_lock._acquire_write_internal(self.timeout)
        if not self.acquired:
            raise TimeoutError("Failed to acquire write lock within timeout")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            self.rw_lock._release_write_internal()


class ConversationManager:
    """
    THREAD-SAFE версия менеджера диалогов с поддержкой parallel execution
    
    КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:
    1. Read-Write locks для concurrent reads
    2. Timeout-based locks для предотвращения deadlocks  
    3. Atomic operations для thread-safe updates
    4. Graceful degradation при timeouts
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
        
        # ИСПРАВЛЕНО: Fallback память с thread-safe доступом
        self.fallback_memory = {}
        self.fallback_memory_lock = threading.RLock()
        
        # ИСПРАВЛЕНО: Read-Write locks для каждого пользователя
        self.user_rw_locks = {}
        self.user_locks_lock = threading.RLock()
        
        # Performance metrics для мониторинга
        self.performance_stats = {
            'successful_reads': 0,
            'successful_writes': 0,
            'timeout_errors': 0,
            'redis_errors': 0,
            'fallback_usage': 0
        }
        self.stats_lock = threading.Lock()
        
        self.logger.info("🧠 Thread-safe менеджер диалогов инициализирован")
        
        # Очистка памяти при старте (для тестирования)
        if config.CLEAR_MEMORY_ON_START:
            self._clear_all_memory()
            self.logger.info("🧹 Вся память диалогов очищена (режим тестирования)")
    
    def _init_redis(self):
        """
        Инициализирует Redis соединение с graceful degradation.
        """
        try:
            if config.REDIS_URL:
                self.redis_client = redis.from_url(config.REDIS_URL, decode_responses=True)
                # Проверяем соединение
                self.redis_client.ping()
                self.redis_available = True
                self.logger.info("✅ Redis соединение установлено")
            else:
                self.logger.info("ℹ️ Redis URL не найден, используется fallback память")
        except Exception as e:
            self.logger.warning(f"⚠️ Redis недоступен: {e}. Используется fallback память")
            self.redis_available = False
    
    def _get_user_rw_lock(self, chat_id: str) -> ReadWriteLock:
        """
        ИСПРАВЛЕНО: Получает Read-Write lock для пользователя с timeout protection
        """
        with self.user_locks_lock:
            if chat_id not in self.user_rw_locks:
                self.user_rw_locks[chat_id] = ReadWriteLock()
                
                # Cleanup старых locks для предотвращения memory leak
                if len(self.user_rw_locks) > config.MAX_FALLBACK_USERS:
                    old_chat_ids = list(self.user_rw_locks.keys())[:len(self.user_rw_locks)//4]
                    for old_id in old_chat_ids:
                        del self.user_rw_locks[old_id]
                    self.logger.info(f"🧹 Очищены старые user locks: {len(old_chat_ids)}")
            
            return self.user_rw_locks[chat_id]
    
    def _normalize_chat_id(self, chat_id: str) -> str:
        """Нормализует chat_id для consistent доступа"""
        return str(chat_id).strip() if chat_id else ""
    
    def get_dialogue_state(self, chat_id: str) -> str:
        """
        THREAD-SAFE получение состояния диалога с concurrent read support
        """
        chat_id = self._normalize_chat_id(chat_id)
        if not chat_id:
            return 'greeting'
        
        user_rw_lock = self._get_user_rw_lock(chat_id)
        
        try:
            with user_rw_lock.acquire_read(timeout=3.0):
                state = self._get_dialogue_state_internal(chat_id)
                with self.stats_lock:
                    self.performance_stats['successful_reads'] += 1
                return state
                
        except TimeoutError:
            # Graceful degradation при timeout
            with self.stats_lock:
                self.performance_stats['timeout_errors'] += 1
            self.logger.warning(f"Read timeout for chat_id: {chat_id}, using default state")
            return 'greeting'
        except Exception as e:
            self.logger.error(f"Error getting dialogue state: {e}")
            return 'greeting'
    
    def _get_dialogue_state_internal(self, chat_id: str) -> str:
        """Внутренний метод получения состояния (без дополнительной блокировки)"""
        if self.redis_available:
            try:
                state_key = f"state:{chat_id}"
                state = self.redis_client.get(state_key)
                return state if state in self.DIALOGUE_STATES else 'greeting'
            except Exception as e:
                self.logger.warning(f"Redis state read error: {e}")
                with self.stats_lock:
                    self.performance_stats['redis_errors'] += 1
        
        # Fallback на локальную память
        with self.fallback_memory_lock:
            user_data = self.fallback_memory.get(chat_id, {})
            return user_data.get('state', 'greeting')
    
    def set_dialogue_state(self, chat_id: str, state: str):
        """
        THREAD-SAFE установка состояния диалога
        """
        chat_id = self._normalize_chat_id(chat_id)
        if not chat_id or state not in self.DIALOGUE_STATES:
            return
        
        user_rw_lock = self._get_user_rw_lock(chat_id)
        
        try:
            with user_rw_lock.acquire_write(timeout=3.0):
                self._set_dialogue_state_internal(chat_id, state)
                with self.stats_lock:
                    self.performance_stats['successful_writes'] += 1
                    
        except TimeoutError:
            with self.stats_lock:
                self.performance_stats['timeout_errors'] += 1
            self.logger.warning(f"Write timeout for chat_id: {chat_id}, state: {state}")
        except Exception as e:
            self.logger.error(f"Error setting dialogue state: {e}")
    
    def _set_dialogue_state_internal(self, chat_id: str, state: str):
        """Внутренний метод установки состояния"""
        if self.redis_available:
            try:
                state_key = f"state:{chat_id}"
                self.redis_client.setex(state_key, config.CONVERSATION_EXPIRATION_SECONDS, state)
                return
            except Exception as e:
                self.logger.warning(f"Redis state write error: {e}")
                with self.stats_lock:
                    self.performance_stats['redis_errors'] += 1
        
        # Fallback на локальную память
        with self.fallback_memory_lock:
            if chat_id not in self.fallback_memory:
                self.fallback_memory[chat_id] = {}
            self.fallback_memory[chat_id]['state'] = state
            self.fallback_memory[chat_id]['last_update'] = time.time()
            
            with self.stats_lock:
                self.performance_stats['fallback_usage'] += 1
    
    def get_conversation_history(self, chat_id: str) -> List[str]:
        """
        THREAD-SAFE получение истории диалога с concurrent read support
        """
        chat_id = self._normalize_chat_id(chat_id)
        if not chat_id:
            return []
        
        user_rw_lock = self._get_user_rw_lock(chat_id)
        
        try:
            with user_rw_lock.acquire_read(timeout=3.0):
                history = self._get_conversation_history_internal(chat_id)
                with self.stats_lock:
                    self.performance_stats['successful_reads'] += 1
                return history
                
        except TimeoutError:
            with self.stats_lock:
                self.performance_stats['timeout_errors'] += 1
            self.logger.warning(f"History read timeout for chat_id: {chat_id}")
            return []
        except Exception as e:
            self.logger.error(f"Error getting conversation history: {e}")
            return []
    
    def _get_conversation_history_internal(self, chat_id: str) -> List[str]:
        """Внутренний метод получения истории"""
        if self.redis_available:
            try:
                history_key = f"history:{chat_id}"
                # Redis возвращает в обратном порядке, исправляем
                return self.redis_client.lrange(history_key, 0, -1)[::-1]
            except Exception as e:
                self.logger.warning(f"Redis history read error: {e}")
                with self.stats_lock:
                    self.performance_stats['redis_errors'] += 1
        
        # Fallback на локальную память
        with self.fallback_memory_lock:
            user_data = self.fallback_memory.get(chat_id, {})
            return user_data.get('history', [])
    
    def update_conversation_history(self, chat_id: str, user_message: str, ai_response: str):
        """
        THREAD-SAFE обновление истории диалога
        """
        chat_id = self._normalize_chat_id(chat_id)
        if not chat_id or not user_message:
            return
        
        # Очищаем токены действий перед сохранением
        clean_response = ai_response.replace("[ACTION:SEND_LESSON_LINK]", "[ССЫЛКА_НА_УРОК]")
        
        user_rw_lock = self._get_user_rw_lock(chat_id)
        
        try:
            with user_rw_lock.acquire_write(timeout=3.0):
                self._update_conversation_history_internal(chat_id, user_message, clean_response)
                with self.stats_lock:
                    self.performance_stats['successful_writes'] += 1
                    
        except TimeoutError:
            with self.stats_lock:
                self.performance_stats['timeout_errors'] += 1
            self.logger.warning(f"History update timeout for chat_id: {chat_id}")
        except Exception as e:
            self.logger.error(f"Error updating conversation history: {e}")
    
    def _update_conversation_history_internal(self, chat_id: str, user_message: str, ai_response: str):
        """Внутренний метод обновления истории"""
        timestamp = datetime.now().isoformat()
        
        if self.redis_available:
            try:
                history_key = f"history:{chat_id}"
                metadata_key = f"metadata:{chat_id}"
                
                # Используем pipeline для атомарности операций
                pipe = self.redis_client.pipeline()
                pipe.lpush(history_key, f"Ассистент: {ai_response}")
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
                return
                
            except Exception as e:
                self.logger.error(f"Redis history update error: {e}")
                with self.stats_lock:
                    self.performance_stats['redis_errors'] += 1
        
        # Fallback на локальную память
        self._update_fallback_memory(chat_id, user_message, ai_response)
    
    def _update_fallback_memory(self, chat_id: str, user_message: str, ai_response: str):
        """Thread-safe обновление fallback памяти"""
        with self.fallback_memory_lock:
            if chat_id not in self.fallback_memory:
                self.fallback_memory[chat_id] = {'history': [], 'state': 'greeting'}
            
            history = self.fallback_memory[chat_id]['history']
            history.append(f"Пользователь: {user_message}")
            history.append(f"Ассистент: {ai_response}")
            
            # Обрезаем до максимального размера
            max_lines = config.CONVERSATION_MEMORY_SIZE * 2
            if len(history) > max_lines:
                history[:] = history[-max_lines:]
            
            self.fallback_memory[chat_id]['last_update'] = time.time()
            
            with self.stats_lock:
                self.performance_stats['fallback_usage'] += 1
            
            # Периодическая очистка памяти
            self._cleanup_fallback_memory()
    
    def _cleanup_fallback_memory(self):
        """Thread-safe очистка fallback памяти"""
        if len(self.fallback_memory) > config.MAX_FALLBACK_USERS:
            # Удаляем половину самых старых записей
            current_time = time.time()
            old_entries = []
            
            for chat_id, data in self.fallback_memory.items():
                last_update = data.get('last_update', 0)
                if current_time - last_update > 3600:  # Старше часа
                    old_entries.append(chat_id)
            
            # Если недостаточно старых записей, удаляем самые старые
            if len(old_entries) < len(self.fallback_memory) // 4:
                sorted_entries = sorted(
                    self.fallback_memory.items(),
                    key=lambda x: x[1].get('last_update', 0)
                )
                old_entries.extend([chat_id for chat_id, _ in sorted_entries[:len(self.fallback_memory)//4]])
            
            for chat_id in old_entries[:len(self.fallback_memory)//2]:
                del self.fallback_memory[chat_id]
            
            if old_entries:
                self.logger.info(f"🧹 Очищена fallback память: удалено {len(old_entries)} записей")
    
    def analyze_message_for_state_transition(self, user_message: str, current_state: str) -> str:
        """
        Анализирует сообщение пользователя и определяет новое состояние диалога.
        Этот метод не требует блокировок, так как он stateless.
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
        Thread-safe возвращение статистики диалога
        """
        chat_id = self._normalize_chat_id(chat_id)
        
        try:
            history = self.get_conversation_history(chat_id)
            current_state = self.get_dialogue_state(chat_id)
            
            with self.stats_lock:
                performance_stats = self.performance_stats.copy()
            
            return {
                "chat_id": chat_id,
                "current_state": current_state,
                "message_count": len(history),
                "redis_available": self.redis_available,
                "last_messages": history[-4:] if history else [],
                "performance_stats": performance_stats
            }
        except Exception as e:
            self.logger.error(f"Error getting conversation stats: {e}")
            return {
                "chat_id": chat_id,
                "error": str(e),
                "redis_available": self.redis_available
            }
    
    def _clear_all_memory(self):
        """
        Thread-safe очистка всей памяти диалогов
        """
        cleared_redis = 0
        cleared_fallback = 0
        
        # Очищаем Redis
        if self.redis_available:
            try:
                keys_to_delete = []
                for pattern in ['history:*', 'state:*', 'metadata:*']:
                    keys = self.redis_client.keys(pattern)
                    keys_to_delete.extend(keys)
                
                if keys_to_delete:
                    cleared_redis = self.redis_client.delete(*keys_to_delete)
                    self.logger.info(f"Очищено {cleared_redis} ключей из Redis")
            except Exception as e:
                self.logger.warning(f"Ошибка очистки Redis: {e}")
        
        # Очищаем fallback память
        with self.fallback_memory_lock:
            cleared_fallback = len(self.fallback_memory)
            self.fallback_memory.clear()
        
        # Очищаем user locks
        with self.user_locks_lock:
            self.user_rw_locks.clear()
        
        if cleared_fallback > 0:
            self.logger.info(f"Очищено {cleared_fallback} записей из fallback памяти")


# Создаем глобальный экземпляр thread-safe менеджера диалогов
conversation_manager = ConversationManager()