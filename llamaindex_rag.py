# llamaindex_rag.py
"""
✅ ИСПРАВЛЕНО: LlamaIndex RAG система с поддержкой истории диалога
"""
import logging
import time
from typing import Tuple, Dict, Any, List

import pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

# Импорт debug логгера
try:
    from rag_debug_logger import rag_debug
except ImportError:
    # Fallback если debug логгер недоступен
    class DummyDebug:
        def log_enricher_input(self, *args): pass
        def log_enricher_prompt(self, *args): pass
        def log_enricher_output(self, *args): pass
        def log_retrieval_results(self, *args): pass
    rag_debug = DummyDebug()

try:
    from config import config
except ImportError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import config

class LlamaIndexRAG:
    """
    ✅ ИСПРАВЛЕНО: RAG система с ChatEngine и поддержкой истории диалога.
    Использует GPT-4o mini для понимания неполных вопросов.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pinecone_index_name = "ukido"
        self.chat_engine = None

        try:
            # Настройка моделей
            Settings.embed_model = GeminiEmbedding(
                model_name=config.EMBEDDING_MODEL, 
                api_key=config.GEMINI_API_KEY
            )
            Settings.llm = OpenRouter(
                api_key=config.OPENROUTER_API_KEY, 
                model="openai/gpt-4o-mini"
            )

            # Подключение к Pinecone
            pc = pinecone.Pinecone(api_key=config.PINECONE_API_KEY)
            pinecone_index = pc.Index(self.pinecone_index_name)
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

            # Настройка реранкера
            reranker = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
                top_n=4
            )

            # Создание ChatEngine
            self.chat_engine = ContextChatEngine.from_defaults(
                retriever=index.as_retriever(
                    similarity_top_k=15,
                    node_postprocessors=[reranker]
                ),
                llm=Settings.llm,
                system_prompt="""Ты помощник по поиску в базе знаний школы Ukido. 

Если спрашивают "Сколько стоит?" - ищи информацию про стоимость курсов.
Если спрашивают "Что нужно?" - добавь контекст о чем речь.
Если спрашивают "Когда?" - уточни про расписание.

Возвращай только найденную информацию, не добавляй от себя."""
            )

            self.logger.info("✅ LlamaIndex ChatEngine инициализирован")

        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации: {e}")
            raise

    def _prepare_smart_history(self, conversation_history: List[str] = None) -> List[str]:
        """
        Умная подготовка истории диалога - максимум 4 сообщения
        """
        if not conversation_history or len(conversation_history) == 0:
            return []  # Первое сообщение
        
        # Адаптивная история: максимум 4 сообщения (2 пары вопрос-ответ)
        return conversation_history[-4:]

    def search_knowledge_base(self, query: str, conversation_history: List[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        ✅ ИСПРАВЛЕНО: Поиск в базе знаний с поддержкой истории диалога
        """
        search_start = time.time()
        
        # Подготавливаем умную историю
        smart_history = self._prepare_smart_history(conversation_history)
        
        # Debug логирование с РЕАЛЬНОЙ историей
        rag_debug.log_enricher_input(query, smart_history)
        
        if not self.chat_engine:
            self.logger.error("ChatEngine не готов")
            return "Ошибка: система не готова", {}

        try:
            self.logger.info(f"🔍 Поиск: '{query}' | История: {len(smart_history)} сообщений")
            
            # Системный промпт
            system_prompt = "Ты помощник по поиску в базе знаний школы Ukido..."
            
            # Обогащаем запрос историей если она есть
            enriched_query = query
            if smart_history:
                history_context = "\n".join(smart_history[-2:])  # Последние 2 сообщения для контекста
                enriched_query = f"Контекст диалога: {history_context}\n\nВопрос: {query}"
            
            # Логируем обогащенный промпт
            rag_debug.log_enricher_prompt(f"SYSTEM: {system_prompt}\nENRICHED QUERY: {enriched_query}")
            
            # ChatEngine обрабатывает обогащенный запрос
            enrichment_start = time.time()
            response = self.chat_engine.chat(enriched_query)
            enrichment_time = time.time() - enrichment_start
            
            # Debug логирование результата
            rag_debug.log_enricher_output(f"ChatEngine обработал запрос с историей ({len(smart_history)} msg)", enrichment_time)
            
            # Извлекаем чанки
            context_chunks = []
            scores = []
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                context_chunks = [node.get_content() for node in response.source_nodes]
                scores = [getattr(node, 'score', 0.5) for node in response.source_nodes]
            
            # Fallback если чанки не найдены
            if not context_chunks and hasattr(response, 'response'):
                context_chunks = [str(response.response)]
                scores = [0.7]
            
            context = "\n\n".join(context_chunks) if context_chunks else "Информация не найдена"
            search_time = time.time() - search_start
            
            # Дополняем scores до нужного количества
            while len(scores) < len(context_chunks):
                scores.append(0.5)
            
            # Debug логирование результатов поиска
            rag_debug.log_retrieval_results(
                chunks=context_chunks,
                scores=scores[:len(context_chunks)],
                time_taken=search_time,
                total_before_rerank=15
            )

            # Метрики
            average_score = sum(scores) / len(scores) if scores else 0.5
            metrics = {
                'search_time': search_time,
                'chunks_found': len(context_chunks),
                'average_score': average_score,
                'max_score': max(scores) if scores else 0.5,
                'history_used': len(smart_history)  # Добавляем метрику использования истории
            }

            self.logger.info(f"✅ Найдено {len(context_chunks)} чанков за {search_time:.2f}s (история: {len(smart_history)})")
            return context, metrics

        except Exception as e:
            self.logger.error(f"❌ Ошибка поиска: {e}")
            return "Ошибка при поиске информации", {}

# Глобальная инициализация
try:
    llama_index_rag = LlamaIndexRAG()
except Exception as e:
    llama_index_rag = None
    logging.getLogger(__name__).error(f"Не удалось создать LlamaIndexRAG: {e}")