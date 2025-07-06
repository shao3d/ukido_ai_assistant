# llamaindex_rag.py
"""
Простая LlamaIndex RAG система с ChatEngine для умного обогащения контекста
"""
import logging
import time
from typing import Tuple, Dict, Any

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
    Простая RAG система с ChatEngine для обогащения контекста.
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

    def search_knowledge_base(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Поиск в базе знаний с обогащением контекста
        """
        search_start = time.time()
        
        # Debug логирование
        rag_debug.log_enricher_input(query, [])
        
        if not self.chat_engine:
            self.logger.error("ChatEngine не готов")
            return "Ошибка: система не готова", {}

        try:
            self.logger.info(f"🔍 Поиск: '{query}'")
            
            # Логируем системный промпт
            system_prompt = "Ты помощник по поиску в базе знаний школы Ukido..."
            rag_debug.log_enricher_prompt(f"SYSTEM: {system_prompt}\nUSER: {query}")
            
            # ChatEngine обрабатывает запрос
            enrichment_start = time.time()
            response = self.chat_engine.chat(query)
            enrichment_time = time.time() - enrichment_start
            
            # Debug логирование результата
            rag_debug.log_enricher_output("ChatEngine обработал запрос", enrichment_time)
            
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
            }

            self.logger.info(f"✅ Найдено {len(context_chunks)} чанков за {search_time:.2f}s")
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