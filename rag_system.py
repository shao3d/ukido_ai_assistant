# rag_system.py
"""
Модуль для работы с RAG (Retrieval-Augmented Generation) системой.
Отвечает за поиск релевантной информации в векторной базе данных Pinecone
и генерацию ответов с использованием найденного контекста.
"""

import time
import hashlib
import threading
import logging
import requests
from typing import Tuple, Dict, Any, Optional, List
from functools import lru_cache

import google.generativeai as genai
from pinecone import Pinecone

from config import config


class RAGSystem:
    """
    Класс для работы с RAG системой.
    
    Архитектурные принципы:
    1. Ленивая инициализация Pinecone для устойчивости к сетевым проблемам
    2. Многоуровневое кеширование для оптимизации производительности
    3. Graceful degradation при недоступности внешних сервисов
    4. Thread-safe операции для многопоточной среды
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
        
        # Система кеширования для оптимизации производительности
        self.rag_cache = {}
        self.cache_lock = threading.Lock()
        
        # Статистика для мониторинга
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'successful_queries': 0,
            'failed_queries': 0
        }
        self.stats_lock = threading.Lock()
        
        self.logger.info("🔍 RAG система инициализирована")
    
    @lru_cache(maxsize=1)
    def _get_pinecone_index(self):
        """
        Ленивая инициализация Pinecone индекса.
        
        Принцип: Не создаем соединение при старте приложения,
        а только когда оно действительно нужно. Это повышает
        устойчивость к сетевым проблемам при деплое.
        """
        with self.pinecone_lock:
            if self.pinecone_index is not None:
                return self.pinecone_index
            
            try:
                self.logger.info("🔌 Инициализируем Pinecone соединение...")
                pc = Pinecone(api_key=config.PINECONE_API_KEY)
                
                # Сначала пытаемся динамическое подключение
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
                
            except Exception as e:
                self.pinecone_available = False
                self.logger.error(f"❌ Pinecone полностью недоступен: {e}")
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
            # Если кеш переполнен, удаляем самую старую запись
            if len(self.rag_cache) >= config.MAX_CACHE_SIZE:
                oldest_key = min(self.rag_cache.keys(), 
                               key=lambda k: self.rag_cache[k]['timestamp'])
                del self.rag_cache[oldest_key]
            
            self.rag_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
    
    def _rewrite_query_for_better_search(self, query: str, conversation_history: List[str]) -> str:
        """
        Переписывает короткие или контекстно-зависимые запросы в более полные.
        
        Пример: "а сколько стоит?" -> "сколько стоит курс Юный Оратор в Ukido"
        
        Это критически важно для качества поиска в векторной БД.
        """
        if not conversation_history or len(query.split()) > 3:
            return query  # Запрос уже достаточно подробный
        
        # Берем последние 3 сообщения пользователя для контекста
        user_messages = [msg for msg in conversation_history 
                        if msg.startswith("Пользователь:")][-3:]
        
        if not user_messages:
            return query
        
        # Простая эвристика для переписывания
        context = ' '.join(user_messages).lower()
        
        # Если обсуждались курсы, добавляем контекст
        if any(word in context for word in ['курс', 'занятие', 'урок']):
            if any(word in query.lower() for word in ['цена', 'стоимость', 'сколько']):
                return f"стоимость курсов Ukido {query}"
            elif any(word in query.lower() for word in ['время', 'когда', 'расписание']):
                return f"расписание курсов Ukido {query}"
        
        return query
    
    def _create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Создает векторное представление текста с помощью Google Gemini.
        Включает обработку ошибок и retry логику.
        """
        try:
            response = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="RETRIEVAL_QUERY"
            )
            return response['embedding']
        except Exception as e:
            self.logger.error(f"Ошибка создания эмбеддинга: {e}")
            return None
    
    def _search_in_pinecone(self, query_embedding: List[float], top_k: int = 3) -> Dict[str, Any]:
        """
        Выполняет поиск в Pinecone векторной базе данных.
        """
        index = self._get_pinecone_index()
        if not index:
            return {'matches': []}
        
        try:
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            return results
        except Exception as e:
            self.logger.error(f"Ошибка поиска в Pinecone: {e}")
            return {'matches': []}
    
    def _format_search_results(self, results: Dict[str, Any]) -> Tuple[str, List[Dict], float]:
        """
        Форматирует результаты поиска для использования в промпте.
        
        Returns:
            Tuple[str, List[Dict], float]: (контекст, отладочная_инфо, лучший_скор)
        """
        context_chunks = []
        debug_info = []
        best_score = 0
        
        for match in results.get('matches', []):
            score = match.get('score', 0)
            
            # Используем только достаточно релевантные результаты
            if score > 0.5:
                text = match.get('metadata', {}).get('text', '')
                context_chunks.append(text)
                best_score = max(best_score, score)
                
                debug_info.append({
                    "score": round(score, 3),
                    "source": match.get('metadata', {}).get('source', 'unknown'),
                    "text_preview": text[:150] + "..." if len(text) > 150 else text
                })
        
        context = "\n".join(context_chunks)
        return context, debug_info, best_score
    
    def _get_fallback_context(self) -> str:
        """
        Возвращает базовую информацию о Ukido, если RAG система недоступна.
        Это пример graceful degradation - система продолжает работать
        даже при полном отказе внешних сервисов.
        """
        return """Ukido - онлайн-школа развития soft skills для детей. 
        
Основные курсы:
- 'Юный Оратор' (7-10 лет, 6000 грн/мес) - развитие навыков публичных выступлений
- 'Эмоциональный Компас' (9-12 лет, 7500 грн/мес) - эмоциональный интеллект
- 'Капитан Проектов' (11-14 лет, 8000 грн/мес) - управление проектами
- 'Диалог' (6-8 лет, 5500 грн/мес) - основы коммуникации

Все курсы проводятся онлайн в мини-группах до 6 человек."""
    
    def search_knowledge_base(self, query: str, conversation_history: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Основной метод для поиска информации в базе знаний.
        
        Алгоритм:
        1. Проверяем кеш
        2. Переписываем запрос для лучшего поиска
        3. Создаем эмбеддинг
        4. Ищем в Pinecone
        5. Форматируем результаты
        6. Кешируем результат
        
        Returns:
            Tuple[str, Dict[str, Any]]: (найденный_контекст, метрики)
        """
        search_start = time.time()
        
        # Нормализуем входные данные
        if not query or not query.strip():
            return "", {"error": "Пустой запрос", "search_time": 0}
        
        conversation_history = conversation_history or []
        
        # Улучшаем запрос на основе контекста
        enhanced_query = self._rewrite_query_for_better_search(query, conversation_history)
        cache_key = self._get_cache_key(enhanced_query)
        
        # Проверяем кеш
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            self.logger.info(f"Кеш попадание для запроса: {enhanced_query}")
            return cached_result
        
        try:
            # Создаем эмбеддинг запроса
            embedding_start = time.time()
            query_embedding = self._create_embedding(enhanced_query)
            embedding_time = time.time() - embedding_start
            
            if not query_embedding:
                raise Exception("Не удалось создать эмбеддинг")
            
            # Ищем в Pinecone
            search_start_time = time.time()
            search_results = self._search_in_pinecone(query_embedding)
            search_time = time.time() - search_start_time
            
            # Форматируем результаты
            context, debug_info, best_score = self._format_search_results(search_results)
            
            total_time = time.time() - search_start
            
            # Подготавливаем метрики
            metrics = {
                "search_time": round(total_time, 2),
                "embedding_time": round(embedding_time, 2),
                "pinecone_time": round(search_time, 2),
                "chunks_found": len(debug_info),
                "best_score": round(best_score, 3),
                "relevance_desc": self._get_relevance_description(best_score),
                "speed_desc": self._get_speed_description(total_time),
                "success": True,
                "original_query": query,
                "enhanced_query": enhanced_query,
                "found_chunks_debug": debug_info,
                "cache_hit": False
            }
            
            result = (context, metrics)
            
            # Кешируем результат
            self._cache_result(cache_key, result)
            
            with self.stats_lock:
                self.stats['successful_queries'] += 1
            
            return result
            
        except Exception as e:
            total_time = time.time() - search_start
            self.logger.error(f"Ошибка в RAG системе: {e}")
            
            # Graceful degradation - возвращаем базовую информацию
            fallback_context = self._get_fallback_context()
            metrics = {
                "search_time": round(total_time, 2),
                "error": str(e),
                "fallback_used": True,
                "chunks_found": 1,
                "success": False,
                "cache_hit": False
            }
            
            with self.stats_lock:
                self.stats['failed_queries'] += 1
            
            return fallback_context, metrics
    
    def _get_relevance_description(self, score: float) -> str:
        """Возвращает человекочитаемое описание релевантности результата"""
        if score >= 0.9:
            return "Отличное совпадение"
        elif score >= 0.7:
            return "Хорошее совпадение"
        elif score >= 0.5:
            return "Среднее совпадение"
        else:
            return "Слабое совпадение"
    
    def _get_speed_description(self, seconds: float) -> str:
        """Возвращает человекочитаемое описание скорости поиска"""
        if seconds < 2:
            return "Быстро"
        elif seconds <= 5:
            return "Нормально"
        else:
            return "Медленно"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику работы RAG системы.
        Полезно для мониторинга и оптимизации производительности.
        """
        with self.stats_lock:
            stats_copy = self.stats.copy()
        
        with self.cache_lock:
            cache_size = len(self.rag_cache)
        
        total_queries = stats_copy['successful_queries'] + stats_copy['failed_queries']
        
        return {
            **stats_copy,
            "cache_size": cache_size,
            "total_queries": total_queries,
            "success_rate": round(stats_copy['successful_queries'] / max(total_queries, 1) * 100, 1),
            "cache_hit_rate": round(stats_copy['cache_hits'] / max(stats_copy['cache_hits'] + stats_copy['cache_misses'], 1) * 100, 1),
            "pinecone_available": self.pinecone_available
        }


# Создаем глобальный экземпляр RAG системы
rag_system = RAGSystem()
