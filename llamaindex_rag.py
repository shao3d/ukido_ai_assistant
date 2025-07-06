# llamaindex_rag.py (Версия для тестирования gpt-4o-mini)
"""
Модуль для работы с RAG системой на базе LlamaIndex.
Используется в основном приложении (app.py) для поиска по базе знаний.
Включает реранкер для максимальной точности.

Конфигурация LLM для ответов:
- Модель: openai/gpt-4o-mini (для проведения тестов)
- Эмбеддинги: Gemini
"""
import logging
import time
from typing import Tuple, Dict, Any

import pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

# Попытка импортировать 'config' из родительской директории, если скрипт запускается из подпапки.
# Если не удается, предполагаем, что config.py находится в той же директории.
try:
    from config import config
except ImportError:
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import config


class LlamaIndexRAG:
    """
    Класс для работы с RAG системой на базе LlamaIndex с реранкером.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pinecone_index_name = "ukido"
        self.query_engine = None

        try:
            # 1. Настройка моделей
            # Модель для создания векторов (остается Gemini)
            Settings.embed_model = GeminiEmbedding(
                model_name=config.EMBEDDING_MODEL, api_key=config.GEMINI_API_KEY
            )
            # Модель для генерации финального ответа (тестируем gpt-4o-mini)
            Settings.llm = OpenRouter(
                api_key=config.OPENROUTER_API_KEY,
                model="openai/gpt-4o-mini", # 🎯 МОДЕЛЬ ДЛЯ ТЕСТА!
                max_tokens=2048, # Увеличим лимит для более полных ответов
                temperature=0.2, # Чуть больше креативности
            )

            # 2. Подключение к Pinecone
            pc = pinecone.Pinecone(api_key=config.PINECONE_API_KEY)
            pinecone_index = pc.Index(self.pinecone_index_name)

            # 3. Создание VectorStore из существующего индекса
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

            # 4. Настройка реранкера для повышения точности
            reranker = SentenceTransformerRerank(
                model="rerank-english-v1.0", # Модель для реранкинга
                top_n=4  # Оставляем 4 лучших чанка после пересортировки
            )

            # 5. Создание Query Engine с реранкером
            self.query_engine = index.as_query_engine(
                similarity_top_k=15,          # Сначала находим 15 кандидатов
                node_postprocessors=[reranker] # Затем применяем реранкер
            )

            self.logger.info("✅ LlamaIndex RAG система с реранкером успешно инициализирована (модель: gpt-4o-mini)")

        except Exception as e:
            self.logger.error(f"❌ Ошибка при инициализации LlamaIndex RAG: {e}", exc_info=True)
            raise

    def search_knowledge_base(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Поиск в базе знаний с использованием LlamaIndex.
        Возвращает контекст и метрики, совместимые со старой системой.
        """
        search_start = time.time()
        if not self.query_engine:
            self.logger.error("LlamaIndex query engine не инициализирован.")
            return "Ошибка: LlamaIndex RAG не готов к работе.", {}

        try:
            self.logger.info(f"🔍 LlamaIndex RAG: Поиск по запросу: '{query}'")
            response = self.query_engine.query(query)

            # Собираем контекст из найденных нод
            context_chunks = [node.get_content() for node in response.source_nodes]
            context = "\n\n".join(context_chunks)

            # Рассчитываем метрики для совместимости и анализа
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

        except Exception as e:
            self.logger.error(f"❌ Ошибка во время поиска LlamaIndex RAG: {e}", exc_info=True)
            return f"К сожалению, произошла ошибка при поиске информации: {e}", {
                'search_time': time.time() - search_start,
                'chunks_found': 0,
                'average_score': 0.0,
                'fallback_reason': 'llama_index_error'
            }

# Создаем глобальный экземпляр LlamaIndex RAG системы
try:
    llama_index_rag = LlamaIndexRAG()
except Exception as e:
    # Если инициализация не удалась, создаем "пустышку", чтобы приложение не упало
    llama_index_rag = None
    logging.getLogger(__name__).critical(f"Не удалось создать экземпляр LlamaIndexRAG: {e}")
