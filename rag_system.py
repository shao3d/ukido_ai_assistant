# rag_system.py
"""
Модуль для работы с RAG (Retrieval-Augmented Generation) системой.
Простая, надежная реализация без излишних оптимизаций.
"""

import time
import hashlib
import threading
import logging
from typing import Tuple, Dict, Any, Optional, List

import google.generativeai as genai
from pinecone import Pinecone

from config import config


class RAGSystem:
    """
    Простой и надежный класс для работы с RAG системой.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Инициализируем Google Gemini
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.embedding_model = config.EMBEDDING_MODEL
        
        # Pinecone инициализация
        self.pinecone_index = None
        self.pinecone_available = False
        
        # Простой кеш
        self.rag_cache = {}
        
        self.logger.info("🔍 RAG система инициализирована")

    def _rerank_chunks_by_keywords(self, query: str, matches: list) -> list:
        """
        Переранжирование чанков по наличию ключевых слов из запроса.
        """
        query_words = query.lower().split()
        scored_matches = []
        for match in matches:
            chunk_text = match.metadata.get('text', '').lower()
            keyword_score = 0
            for word in query_words:
                if len(word) > 3:
                    keyword_score += chunk_text.count(word)
            if "дмитрий" in query_words and "дмитрий" in chunk_text:
                keyword_score += 5
            if "петров" in query_words and "петров" in chunk_text:
                keyword_score += 5
            scored_matches.append((match, keyword_score))
        scored_matches.sort(key=lambda x: (x[1], x[0].score), reverse=True)
        return [match for match, _ in scored_matches]

    def _extract_relevant_sentences(self, chunk_text: str, query: str) -> str:
        """
        Извлекает релевантные предложения из чанка.
        """
        sentences = chunk_text.split('.')
        query_words = set(word.lower() for word in query.split() if len(word) > 3)
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in query_words):
                relevant_sentences.append(sentence.strip())
        if not relevant_sentences and sentences:
            relevant_sentences = sentences[:2]
        return '. '.join(relevant_sentences) + '.'

    def _get_pinecone_index(self):
        """
        Ленивая инициализация Pinecone.
        """
        if self.pinecone_index is not None:
            return self.pinecone_index
        
        try:
            self.logger.info("🔌 Подключаемся к Pinecone...")
            pc = Pinecone(api_key=config.PINECONE_API_KEY)
            
            try:
                facts_description = pc.describe_index("ukido")
                self.pinecone_index = pc.Index(host=facts_description.host)
                self.logger.info("✅ Pinecone подключен")
            except:
                self.pinecone_index = pc.Index(host=config.PINECONE_HOST_FACTS)
                self.logger.info("✅ Pinecone подключен (fallback)")
            
            self.pinecone_available = True
            return self.pinecone_index
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка Pinecone: {e}")
            self.pinecone_available = False
            return None

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Получает эмбеддинг через Gemini API.
        """
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            self.logger.error(f"❌ Ошибка Gemini API: {e}")
            return None

    def search_knowledge_base(self, query: str, conversation_history: List[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Поиск в базе знаний - простая и надежная реализация.
        """
        search_start = time.time()
        
        # Простая проверка кеша
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.rag_cache:
            cached_entry = self.rag_cache[cache_key]
            if time.time() - cached_entry['timestamp'] < 300:  # 5 минут TTL
                return cached_entry['result']
        
        try:
            # Получаем эмбеддинг
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                return self._fallback_response("embedding_error", search_start)
            
            # Получаем Pinecone индекс
            index = self._get_pinecone_index()
            if index is None:
                return self._fallback_response("pinecone_error", search_start)
            
            # Простое увеличение количества чанков БЕЗ сжатия
            search_results = index.query(
                vector=query_embedding,
                top_k=10,  # Увеличиваем с 8 до 10
                include_metadata=True
            )
            
            if not search_results.matches:
                return self._fallback_response("no_results", search_start)
            
            # Собираем ВСЕ чанки без сжатия
            relevant_chunks = []
            for match in search_results.matches:
                chunk_text = match.metadata.get('text', '')
                if chunk_text:
                    relevant_chunks.append(chunk_text)
            
            context = '\n\n'.join(relevant_chunks) if relevant_chunks else "Релевантная информация не найдена."
            metrics = {
                'search_time': time.time() - search_start,
                'chunks_found': len(relevant_chunks),
                'max_score': max([m.score for m in top_matches]) if top_matches else 0
            }
            
            # Кешируем результат
            result = (context, metrics)
            self.rag_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
            
            # Ограничиваем размер кеша
            if len(self.rag_cache) > 100:
                oldest_keys = list(self.rag_cache.keys())[:20]
                for key in oldest_keys:
                    del self.rag_cache[key]
            
            self.logger.info(f"🔍 RAG поиск: {len(relevant_chunks)} чанков за {metrics['search_time']:.2f}с")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Критическая ошибка RAG: {e}")
            return self._fallback_response("critical_error", search_start)

    def _fallback_response(self, reason: str, search_start: float) -> Tuple[str, Dict[str, Any]]:
        """
        Простой fallback ответ.
        """
        context = "К сожалению, в моей базе знаний нет информации по этому вопросу."
        metrics = {
            'search_time': time.time() - search_start,
            'chunks_found': 0,
            'fallback_reason': reason
        }
        return context, metrics

    def get_stats(self) -> Dict[str, Any]:
        """
        Простая статистика.
        """
        return {
            "cache_size": len(self.rag_cache),
            "pinecone_available": self.pinecone_available
        }


# Создаем глобальный экземпляр RAG системы
rag_system = RAGSystem()