# llamaindex_rag.py (ФИНАЛЬНАЯ ВЕРСИЯ с обработкой лимитов API)
import logging
import time
from typing import Tuple, Dict, Any

import pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

# --- НОВЫЙ БЛОК: Импорты для обработки лимитов ---
from tenacity import retry, stop_after_attempt, wait_exponential
from google.api_core.exceptions import ResourceExhausted
# -------------------------------------------------

try:
    from config import config
except ImportError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import config

class LlamaIndexRAG:
    """
    Отказоустойчивый класс для работы с RAG системой, который умеет
    обрабатывать лимиты API Gemini.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pinecone_index_name = "ukido"
        self.query_engine = None

        try:
            # Настройка моделей
            Settings.embed_model = GeminiEmbedding(model_name=config.EMBEDDING_MODEL, api_key=config.GEMINI_API_KEY)
            Settings.llm = Gemini(model_name="models/gemini-1.5-pro-latest", api_key=config.GEMINI_API_KEY)

            # Подключение к Pinecone
            pc = pinecone.Pinecone(api_key=config.PINECONE_API_KEY)
            pinecone_index = pc.Index(self.pinecone_index_name)
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

            # Настройка реранкера
            reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=4)

            # Создание Query Engine
            self.query_engine = index.as_query_engine(
                similarity_top_k=15,
                node_postprocessors=[reranker]
            )

            self.logger.info("✅ LlamaIndex RAG система с реранкером успешно инициализирована (модель: Gemini 1.5 Pro)")

        except Exception as e:
            self.logger.error(f"❌ Ошибка при инициализации LlamaIndex RAG: {e}", exc_info=True)
            raise

    # --- 🔥 КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Декоратор для обработки лимитов 🔥 ---
    @retry(
        wait=wait_exponential(multiplier=2, min=5, max=30), # Ждем 5с, потом 10с, потом 20с, макс 30с
        stop=stop_after_attempt(4), # Делаем 4 попытки
        retry_error_callback=lambda state: logging.warning(f"Достигнут лимит API Gemini. Попытка #{state.attempt_number}, ждем..."),
        retry=retry_if_exception_type(ResourceExhausted) # Повторяем только при ошибке лимита
    )
    def search_knowledge_base(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Поиск в базе знаний с автоматической обработкой лимитов API.
        """
        search_start = time.time()
        if not self.query_engine:
            self.logger.error("LlamaIndex query engine не инициализирован.")
            return "Ошибка: LlamaIndex RAG не готов к работе.", {}

        try:
            self.logger.info(f"🔍 LlamaIndex RAG: Поиск по запросу: '{query}'")
            # Этот вызов теперь защищен декоратором @retry
            response = self.query_engine.query(query)

            context_chunks = [node.get_content() for node in response.source_nodes]
            context = "\n\n".join(context_chunks)

            search_time = time.time() - search_start
            scores = [node.get_score() for node in response.source_nodes]
            average_score = sum(scores) / len(scores) if scores else 0.0

            metrics = {
                'search_time': search_time,
                'chunks_found': len(context_chunks),
                'average_score': average_score,
                'max_score': max(scores) if scores else 0.0,
            }

            self.logger.info(f"✅ LlamaIndex RAG: Найдено {metrics['chunks_found']} чанков за {search_time:.2f}с")
            return context, metrics

        except ResourceExhausted as e:
            # Этот блок сработает, если все 4 попытки не увенчались успехом
            self.logger.error(f"❌ Не удалось выполнить запрос к Gemini после нескольких попыток: {e}")
            return "К сожалению, сервер AI сейчас перегружен. Пожалуйста, повторите ваш запрос через минуту.", {}
        except Exception as e:
            self.logger.error(f"❌ Ошибка во время поиска LlamaIndex RAG: {e}", exc_info=True)
            return f"К сожалению, произошла внутренняя ошибка при поиске информации.", {}

# Вспомогательная функция для декоратора
def retry_if_exception_type(exception_type):
    return lambda e: isinstance(e, exception_type)

try:
    llama_index_rag = LlamaIndexRAG()
except Exception as e:
    llama_index_rag = None
    logging.getLogger(__name__).critical(f"Не удалось создать LlamaIndexRAG: {e}")
