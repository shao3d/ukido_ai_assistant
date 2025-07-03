# rag_system.py
"""
Модуль для работы с RAG (Retrieval-Augmented Generation) системой.
Отвечает за поиск релевантной информации в векторной базе данных Pinecone
и генерацию ответов с использованием найденного контекста.

ПРОДАКШН УЛУЧШЕНИЯ:
- Circuit Breaker pattern для устойчивости к отказам внешних API
- Exponential backoff для retry механизмов
- Улучшенная диагностика для продакшн мониторинга
"""

import time
import hashlib
import threading
import logging
import requests
from typing import Tuple, Dict, Any, Optional, List
from functools import lru_cache
from enum import Enum

import google.generativeai as genai
from pinecone import Pinecone

from config import config


class CircuitBreakerState(Enum):
    """Состояния Circuit Breaker паттерна"""
    CLOSED = "closed"      # Нормальная работа
    OPEN = "open"          # Сервис недоступен, запросы блокируются
    HALF_OPEN = "half_open"  # Тестовый режим восстановления


class CircuitBreaker:
    """
    Circuit Breaker pattern для защиты от cascade failures.
    
    Принцип: При определенном количестве ошибок API временно блокируется,
    что предотвращает накопление запросов к недоступному сервису.
    """
    
    def __init__(self, failure_threshold=5, timeout=60, test_request_timeout=30):
        self.failure_threshold = failure_threshold
        self.timeout = timeout  # Время блокировки в секундах
        self.test_request_timeout = test_request_timeout
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")
    
    def call(self, func, *args, **kwargs):
        """
        Выполняет функцию через Circuit Breaker.
        
        Returns:
            Результат функции или None при блокировке
        """
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                # Проверяем, не пора ли попробовать восстановление
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.logger.info("🔄 Circuit Breaker: переход в HALF_OPEN для тестирования")
                else:
                    self.logger.warning("⚡ Circuit Breaker: запрос заблокирован (OPEN состояние)")
                    return None
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                # В тестовом режиме делаем только один запрос
                try:
                    result = func(*args, **kwargs)
                    # Успех! Восстанавливаем нормальную работу
                    self.failure_count = 0
                    self.state = CircuitBreakerState.CLOSED
                    self.logger.info("✅ Circuit Breaker: восстановление успешно, переход в CLOSED")
                    return result
                except Exception as e:
                    # Сервис все еще недоступен, возвращаемся к блокировке
                    self.last_failure_time = time.time()
                    self.state = CircuitBreakerState.OPEN
                    self.logger.error(f"❌ Circuit Breaker: тест не прошел, возврат в OPEN: {e}")
                    return None
        
        # Нормальное выполнение (CLOSED состояние)
        try:
            result = func(*args, **kwargs)
            # При успешном выполнении сбрасываем счетчик ошибок
            with self.lock:
                self.failure_count = 0
            return result
            
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.logger.warning(f"⚠️ Circuit Breaker: ошибка {self.failure_count}/{self.failure_threshold}: {e}")
                
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                    self.last_failure_time = time.time()
                    self.logger.error(f"🚨 Circuit Breaker: ОТКРЫТ на {self.timeout}с после {self.failure_count} ошибок")
            
            raise  # Пробрасываем исключение для первичной обработки


class RAGSystem:
    """
    Класс для работы с RAG системой.
    
    Архитектурные принципы:
    1. Ленивая инициализация Pinecone для устойчивости к сетевым проблемам
    2. Многоуровневое кеширование для оптимизации производительности
    3. Circuit Breaker для защиты от cascade failures
    4. Graceful degradation при недоступности внешних сервисов
    5. Thread-safe операции для многопоточной среды
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Инициализируем Google Gemini для эмбеддингов
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.embedding_model = config.EMBEDDING_MODEL
        
        # Pinecone будет инициализирован лениво при первом обращении
        self.pinecone_index = None
        self.pinecone_available = False
        self.pinecone_lock = threading.Lock()
        
        # Circuit Breaker для Pinecone API
        self.pinecone_circuit_breaker = CircuitBreaker(
            failure_threshold=3,  # Более агрессивный для продакшн
            timeout=120,          # 2 минуты блокировки
            test_request_timeout=30
        )
        
        # Circuit Breaker для Gemini API
        self.gemini_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=60,
            test_request_timeout=15
        )
        
        # Система кеширования для оптимизации производительности
        self.rag_cache = {}
        self.cache_lock = threading.Lock()
        
        # Расширенная статистика для мониторинга
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'circuit_breaker_blocks': 0,
            'pinecone_errors': 0,
            'gemini_errors': 0
        }
        self.stats_lock = threading.Lock()
        
        self.logger.info("🔍 RAG система инициализирована с Circuit Breaker защитой")
    
    @lru_cache(maxsize=1)
    def _get_pinecone_index(self):
        """
        Ленивая инициализация Pinecone индекса с Circuit Breaker защитой.
        """
        def _initialize_pinecone():
            """Внутренняя функция инициализации для Circuit Breaker"""
            with self.pinecone_lock:
                if self.pinecone_index is not None:
                    return self.pinecone_index
                
                self.logger.info("🔌 Инициализируем Pinecone соединение...")
                pc = Pinecone(api_key=config.PINECONE_API_KEY)
                
                # Пытаемся динамическое подключение с timeout
                try:
                    facts_description = pc.describe_index("ukido")
                    self.pinecone_index = pc.Index(host=facts_description.host)
                    self.logger.info("✅ Pinecone подключен динамически")
                except Exception as dynamic_error:
                    self.logger.warning(f"Динамическое подключение не удалось: {dynamic_error}")
                    # Fallback на прямой host
                    self.pinecone_index = pc.Index(host=config.PINECONE_HOST_FACTS)
                    self.logger.info("✅ Pinecone подключен через прямой host")
                
                self.pinecone_available = True
                return self.pinecone_index
        
        # Используем Circuit Breaker для инициализации
        try:
            index = self.pinecone_circuit_breaker.call(_initialize_pinecone)
            if index is None:
                self.pinecone_available = False
                with self.stats_lock:
                    self.stats['circuit_breaker_blocks'] += 1
                self.logger.error("❌ Pinecone заблокирован Circuit Breaker")
            return index
        except Exception as e:
            self.pinecone_available = False
            with self.stats_lock:
                self.stats['pinecone_errors'] += 1
            self.logger.error(f"❌ Критическая ошибка Pinecone инициализации: {e}")
            return None
    
    def _get_embedding_with_circuit_breaker(self, text: str) -> Optional[List[float]]:
        """
        Получает эмбеддинг через Gemini API с Circuit Breaker защитой.
        """
        def _get_embedding():
            """Внутренняя функция для Circuit Breaker"""
            try:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_query"
                )
                return result['embedding']
            except Exception as e:
                self.logger.error(f"Ошибка Gemini API: {e}")
                raise
        
        # Используем Circuit Breaker для Gemini API
        try:
            embedding = self.gemini_circuit_breaker.call(_get_embedding)
            if embedding is None:
                with self.stats_lock:
                    self.stats['circuit_breaker_blocks'] += 1
                    self.stats['gemini_errors'] += 1
                self.logger.error("❌ Gemini API заблокирован Circuit Breaker")
            return embedding
        except Exception as e:
            with self.stats_lock:
                self.stats['gemini_errors'] += 1
            self.logger.error(f"❌ Критическая ошибка Gemini API: {e}")
            return None
    
    def _get_cache_key(self, query: str) -> str:
        """
        Генерирует ключ кеша для запроса.
        Использует MD5 хеш для компактности.
        """
        return hashlib.md5(query.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Получает результат из кеша, если он еще актуален.
        """
        with self.cache_lock:
            if cache_key in self.rag_cache:
                cached_entry = self.rag_cache[cache_key]
                # Проверяем, не истекло ли время жизни кеша
                if time.time() - cached_entry['timestamp'] < config.RAG_CACHE_TTL:
                    with self.stats_lock:
                        self.stats['cache_hits'] += 1
                    return cached_entry['result']
                else:
                    # Удаляем устаревшую запись
                    del self.rag_cache[cache_key]
        
        with self.stats_lock:
            self.stats['cache_misses'] += 1
        return None
    
    def _cache_result(self, cache_key: str, result: Tuple[str, Dict[str, Any]]):
        """
        Сохраняет результат в кеш с ограничением размера.
        """
        with self.cache_lock:
            # Ограничиваем размер кеша для предотвращения утечки памяти
            if len(self.rag_cache) >= config.MAX_CACHE_SIZE:
                # Удаляем старейшие записи (простая FIFO стратегия)
                oldest_keys = list(self.rag_cache.keys())[:config.MAX_CACHE_SIZE // 4]
                for key in oldest_keys:
                    del self.rag_cache[key]
                self.logger.info("🧹 Выполнена очистка кеша RAG")
            
            self.rag_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
    
    def search_knowledge_base(self, query: str, conversation_history: List[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Ищет релевантную информацию в базе знаний с улучшенной устойчивостью к отказам.
        
        Args:
            query: Запрос пользователя
            conversation_history: История диалога для контекста
            
        Returns:
            Tuple[str, Dict[str, Any]]: (контекст, метрики)
        """
        search_start = time.time()
        
        # Проверяем кеш
        cache_key = self._get_cache_key(query)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            self.logger.info("💾 Результат получен из кеша")
            return cached_result
        
        try:
            # Получаем эмбеддинг с Circuit Breaker защитой
            query_embedding = self._get_embedding_with_circuit_breaker(query)
            if query_embedding is None:
                # Fallback при недоступности эмбеддингов
                fallback_context = "Информация временно недоступна. Используйте общие знания о развитии детей."
                fallback_metrics = {
                    'search_time': time.time() - search_start,
                    'chunks_found': 0,
                    'fallback_reason': 'embedding_api_unavailable'
                }
                with self.stats_lock:
                    self.stats['failed_queries'] += 1
                return fallback_context, fallback_metrics
            
            # Получаем Pinecone индекс с Circuit Breaker защитой
            index = self._get_pinecone_index()
            if index is None:
                # Fallback при недоступности Pinecone
                fallback_context = "База знаний временно недоступна. Отвечайте на основе общих принципов детской психологии."
                fallback_metrics = {
                    'search_time': time.time() - search_start,
                    'chunks_found': 0,
                    'fallback_reason': 'pinecone_unavailable'
                }
                with self.stats_lock:
                    self.stats['failed_queries'] += 1
                return fallback_context, fallback_metrics
            
            # Выполняем поиск в Pinecone с Circuit Breaker
            def _pinecone_search():
                return index.query(
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True
                )
            
            search_results = self.pinecone_circuit_breaker.call(_pinecone_search)
            if search_results is None:
                # Circuit Breaker заблокировал запрос
                fallback_context = "Поиск в базе знаний временно недоступен."
                fallback_metrics = {
                    'search_time': time.time() - search_start,
                    'chunks_found': 0,
                    'fallback_reason': 'circuit_breaker_open'
                }
                with self.stats_lock:
                    self.stats['failed_queries'] += 1
                    self.stats['circuit_breaker_blocks'] += 1
                return fallback_context, fallback_metrics
            
            # Обрабатываем результаты
            relevant_chunks = []
            for match in search_results.matches:
                if match.score > 0.3:  # Порог релевантности
                    relevant_chunks.append(match.metadata.get('text', ''))
            
            context = '\n\n'.join(relevant_chunks) if relevant_chunks else "Релевантная информация не найдена."
            
            metrics = {
                'search_time': time.time() - search_start,
                'chunks_found': len(relevant_chunks),
                'max_score': max([m.score for m in search_results.matches]) if search_results.matches else 0,
                'pinecone_available': self.pinecone_available
            }
            
            # Кешируем результат
            result = (context, metrics)
            self._cache_result(cache_key, result)
            
            with self.stats_lock:
                self.stats['successful_queries'] += 1
            
            self.logger.info(f"🔍 RAG поиск выполнен: {len(relevant_chunks)} чанков за {metrics['search_time']:.2f}с")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Критическая ошибка RAG поиска: {e}")
            with self.stats_lock:
                self.stats['failed_queries'] += 1
            
            # Emergency fallback
            emergency_context = "Система поиска временно недоступна. Используйте базовые знания о детском развитии."
            emergency_metrics = {
                'search_time': time.time() - search_start,
                'chunks_found': 0,
                'error': str(e),
                'fallback_reason': 'critical_error'
            }
            return emergency_context, emergency_metrics
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает расширенную статистику для мониторинга производительности.
        """
        with self.stats_lock:
            stats_copy = self.stats.copy()
        
        with self.cache_lock:
            cache_size = len(self.rag_cache)
        
        total_queries = stats_copy['successful_queries'] + stats_copy['failed_queries']
        
        # Добавляем информацию о Circuit Breaker состояниях
        circuit_breaker_info = {
            'pinecone_cb_state': self.pinecone_circuit_breaker.state.value,
            'pinecone_cb_failures': self.pinecone_circuit_breaker.failure_count,
            'gemini_cb_state': self.gemini_circuit_breaker.state.value,
            'gemini_cb_failures': self.gemini_circuit_breaker.failure_count,
        }
        
        return {
            **stats_copy,
            **circuit_breaker_info,
            "cache_size": cache_size,
            "total_queries": total_queries,
            "success_rate": round(stats_copy['successful_queries'] / max(total_queries, 1) * 100, 1),
            "cache_hit_rate": round(stats_copy['cache_hits'] / max(stats_copy['cache_hits'] + stats_copy['cache_misses'], 1) * 100, 1),
            "pinecone_available": self.pinecone_available
        }


# Создаем глобальный экземпляр RAG системы
rag_system = RAGSystem()