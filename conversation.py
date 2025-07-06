# conversation.py
"""
✅ ФИНАЛЬНАЯ ВЕРСИЯ: Потокобезопасный менеджер диалогов.
- Сохранена архитектура с Read-Write locks для предотвращения deadlocks.
- Исправлена функция очистки памяти: теперь она не падает, если Redis недоступен.
"""

import redis
import threading
import logging
import time
from typing import List, Dict, Any, Optional
from config import config


class ReadWriteLock:
    """
    Read-Write lock для concurrent reads без блокировки.
    """
    def __init__(self):
        self._read_ready = threading.Condition(threading.RLock())
        self._readers = 0
    def acquire_read(self, timeout: float = 5.0):
        return ReadContext(self, timeout)
    def acquire_write(self, timeout: float = 5.0):
        return WriteContext(self, timeout)
    def _acquire_read_internal(self, timeout: float) -> bool:
        if not self._read_ready.acquire(timeout=timeout):
            return False
        try:
            self._readers += 1
            return True
        finally:
            self._read_ready.release()
    def _release_read_internal(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notifyAll()
    def _acquire_write_internal(self, timeout: float) -> bool:
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
            pass
    def _release_write_internal(self):
        try:
            self._read_ready.release()
        except Exception:
            pass


class ReadContext:
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
    THREAD-SAFE версия менеджера диалогов с отказоустойчивой очисткой памяти.
    """
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
        
        self.logger.info("🧠 Thread-safe менеджер диалогов инициализирован")
        
        if config.CLEAR_MEMORY_ON_START:
            self.clear_all_conversations() # Имя изменено для консистентности
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
                if len(self.user_rw_locks) > config.MAX_FALLBACK_USERS:
                    old_chat_ids = list(self.user_rw_locks.keys())[:len(self.user_rw_locks)//4]
                    for old_id in old_chat_ids: del self.user_rw_locks[old_id]
                    self.logger.info(f"🧹 Очищены старые user locks: {len(old_chat_ids)}")
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
                    try:
                        state = self.redis_client.get(f"state:{chat_id}")
                        return state if state else 'greeting'
                    except Exception as e:
                        self.logger.warning(f"Redis state read error: {e}")
                with self.fallback_memory_lock:
                    return self.fallback_memory.get(chat_id, {}).get('state', 'greeting')
        except TimeoutError:
            self.logger.warning(f"Read timeout for chat_id: {chat_id}, using default state")
            return 'greeting'

    def set_dialogue_state(self, chat_id: str, state: str):
        chat_id = self._normalize_chat_id(chat_id)
        if not chat_id: return
        user_rw_lock = self._get_user_rw_lock(chat_id)
        try:
            with user_rw_lock.acquire_write(timeout=3.0):
                if self.redis_available:
                    try:
                        self.redis_client.setex(f"state:{chat_id}", config.CONVERSATION_EXPIRATION_SECONDS, state)
                        return
                    except Exception as e:
                        self.logger.warning(f"Redis state write error: {e}")
                with self.fallback_memory_lock:
                    if chat_id not in self.fallback_memory: self.fallback_memory[chat_id] = {}
                    self.fallback_memory[chat_id]['state'] = state
                    self.fallback_memory[chat_id]['last_update'] = time.time()
        except TimeoutError:
            self.logger.warning(f"Write timeout for chat_id: {chat_id}, state: {state}")

    def get_conversation_history(self, chat_id: str) -> List[str]:
        chat_id = self._normalize_chat_id(chat_id)
        if not chat_id: return []
        user_rw_lock = self._get_user_rw_lock(chat_id)
        try:
            with user_rw_lock.acquire_read(timeout=3.0):
                if self.redis_available:
                    try:
                        return self.redis_client.lrange(f"history:{chat_id}", 0, -1)
                    except Exception as e:
                        self.logger.warning(f"Redis history read error: {e}")
                with self.fallback_memory_lock:
                    return self.fallback_memory.get(chat_id, {}).get('history', [])
        except TimeoutError:
            self.logger.warning(f"History read timeout for chat_id: {chat_id}")
            return []

    def update_conversation_history(self, chat_id: str, user_message: str, ai_response: str):
        chat_id = self._normalize_chat_id(chat_id)
        if not chat_id or not user_message: return
        user_rw_lock = self._get_user_rw_lock(chat_id)
        
        # Форматируем сообщения для консистентности
        user_entry = f"Пользователь: {user_message}"
        ai_entry = f"Ассистент: {ai_response}"

        try:
            with user_rw_lock.acquire_write(timeout=3.0):
                if self.redis_available:
                    try:
                        pipe = self.redis_client.pipeline()
                        pipe.lpush(f"history:{chat_id}", ai_entry, user_entry)
                        pipe.ltrim(f"history:{chat_id}", 0, (config.CONVERSATION_MEMORY_SIZE * 2) - 1)
                        pipe.expire(f"history:{chat_id}", config.CONVERSATION_EXPIRATION_SECONDS)
                        pipe.execute()
                        return
                    except Exception as e:
                        self.logger.error(f"Redis history update error: {e}")
                
                # Fallback на локальную память
                with self.fallback_memory_lock:
                    if chat_id not in self.fallback_memory:
                        self.fallback_memory[chat_id] = {'history': [], 'state': 'greeting'}
                    history = self.fallback_memory[chat_id]['history']
                    history.extend([user_entry, ai_entry])
                    max_lines = config.CONVERSATION_MEMORY_SIZE * 2
                    if len(history) > max_lines:
                        self.fallback_memory[chat_id]['history'] = history[-max_lines:]
                    self.fallback_memory[chat_id]['last_update'] = time.time()
                    self._cleanup_fallback_memory()
        except TimeoutError:
            self.logger.warning(f"History update timeout for chat_id: {chat_id}")
    
    def _cleanup_fallback_memory(self):
        if len(self.fallback_memory) > config.MAX_FALLBACK_USERS:
            sorted_entries = sorted(self.fallback_memory.items(), key=lambda x: x[1].get('last_update', 0))
            entries_to_remove = sorted_entries[:len(self.fallback_memory)//2]
            for chat_id, _ in entries_to_remove:
                del self.fallback_memory[chat_id]
            if entries_to_remove:
                self.logger.info(f"🧹 Очищена fallback память: удалено {len(entries_to_remove)} старых записей")

    def clear_all_conversations(self):
        """
        ✅ ИСПРАВЛЕНО: Thread-safe и ОТКАЗОУСТОЙЧИВАЯ очистка всей памяти диалогов.
        """
        self.logger.info("🧹 Запущена процедура очистки всей памяти...")
        cleared_redis_keys = 0
        cleared_fallback_items = 0

        # ✅ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Очищаем Redis, только если он доступен
        if self.redis_available and self.redis_client:
            try:
                # Используем scan_iter для безопасного перебора ключей
                keys_to_delete = list(self.redis_client.scan_iter("*:*"))
                if keys_to_delete:
                    cleared_redis_keys = self.redis_client.delete(*keys_to_delete)
                    self.logger.info(f"Очищено {cleared_redis_keys} ключей из Redis.")
                else:
                    self.logger.info("В Redis не найдено ключей для удаления.")
            except Exception as e:
                self.logger.error(f"Ошибка при попытке очистки Redis: {e}", exc_info=True)
        else:
            self.logger.info("Пропуск очистки Redis, т.к. он недоступен.")
        
        # Очищаем fallback память
        with self.fallback_memory_lock:
            cleared_fallback_items = len(self.fallback_memory)
            self.fallback_memory.clear()
        
        # Очищаем user locks, чтобы не было утечек
        with self.user_locks_lock:
            self.user_rw_locks.clear()
        
        self.logger.info(f"Очищено {cleared_fallback_items} записей из fallback памяти.")
        self.logger.info(f"Очищены все пользовательские блокировки (locks).")
        self.logger.info("✅ Процедура очистки памяти завершена.")


# Создаем глобальный экземпляр thread-safe менеджера диалогов
conversation_manager = ConversationManager()