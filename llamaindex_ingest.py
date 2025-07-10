# llamaindex_ingest.py (ИСПРАВЛЕННАЯ ВЕРСИЯ: решена ошибка с prompt_template)
"""
Финальная, промышленная версия скрипта для создания и загрузки индекса в Pinecone.
Использует двухступенчатый парсинг (Markdown -> Semantic) и генерацию
вопросов на основе примеров для максимальной точности RAG.
"""
import logging
import os
import pinecone
from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    PromptTemplate,
)
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.openrouter import OpenRouter
from custom_metadata_extractor import CustomMetadataExtractor

try:
    from config import config
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import config

# --- НАСТРОЙКА ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PINECONE_INDEX_NAME = "ukido"
DATA_DIRECTORY = "data_facts"
QUESTIONS_FILE = "questions.txt"

def load_questions(filepath: str) -> str:
    """Загружает и форматирует вопросы из файла."""
    logger = logging.getLogger(__name__)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        logger.info(f"✅ Загружено {len(questions)} вопросов-примеров из '{filepath}'.")
        return "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    except FileNotFoundError:
        logger.error(f"❌ Файл с вопросами '{filepath}' не найден! Убедитесь, что он лежит рядом со скриптом.")
        raise

def main():
    """Основная функция для запуска процесса индексации."""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Запуск ФИНАЛЬНОЙ версии индексации (Структура -> Семантика)...")

    example_questions = load_questions(QUESTIONS_FILE)

    logger.info("⚙️  Шаг 1: Настройка моделей...")
    llm = OpenRouter(api_key=config.OPENROUTER_API_KEY, model="mistralai/mistral-7b-instruct-v0.2")
    embed_model = GeminiEmbedding(model_name=config.EMBEDDING_MODEL, api_key=config.GEMINI_API_KEY)
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # --- 🔥 КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ 🔥 ---
    # Создаем финальную СТРОКУ для промпта, вставляя в нее примеры вопросов
    qa_generate_prompt_str = (
        "Ниже предоставлен контекст:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Используя этот контекст и не используя предварительные знания, "
        "сгенерируй {num_questions} вопросов, на которые отвечает данный контекст.\n"
        "Вопросы должны быть в стиле и по тематике, как в этих примерах:\n"
        "--- ПРИМЕРЫ ВОПРОСОВ ---\n"
        f"{example_questions}\n"
        "------------------------\n"
        "Сгенерированные вопросы:\n"
    )

    logger.info(f"📂 Шаг 2: Загрузка документов из '{DATA_DIRECTORY}'...")
    documents = SimpleDirectoryReader(DATA_DIRECTORY, required_exts=[".md"]).load_data()

    logger.info("🔪 Шаг 3: Настройка ДВУХСТУПЕНЧАТОГО конвейера обработки...")
    
    # Передаем в экстрактор готовую СТРОКУ, как он и просит
    question_extractor = QuestionsAnsweredExtractor(
        questions=5,
        prompt_template=qa_generate_prompt_str,
        llm=llm,
    )

    pipeline = IngestionPipeline(
        transformations=[
            MarkdownNodeParser(include_metadata=True),
            SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=embed_model
            ),
            question_extractor,
            CustomMetadataExtractor(),  # НОВАЯ СТРОКА
            embed_model,
        ],
    )
    
    logger.info("🧠 Обработка документов... Это может занять несколько минут.")
    nodes = pipeline.run(documents=documents, show_progress=True)
    logger.info(f"✅ Создано {len(nodes)} финальных, умных чанков.")

    logger.info(f"🌲 Шаг 4: Подключение к Pinecone и очистка индекса '{PINECONE_INDEX_NAME}'...")
    pc = pinecone.Pinecone(api_key=config.PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    pinecone_index.delete(delete_all=True)
    logger.info("✅ Индекс очищен.")

    logger.info("📤 Шаг 5: Загрузка нового индекса в Pinecone...")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    vector_store.add(nodes)
    
    stats = pinecone_index.describe_index_stats()
    logger.info(f"🎉🎉🎉 УСПЕХ! Индексация завершена. В базе {stats.get('total_vector_count', 'N/A')} векторов. 🎉🎉🎉")

if __name__ == "__main__":
    if not os.path.isdir(DATA_DIRECTORY):
        print(f"❌ Папка '{DATA_DIRECTORY}' не найдена. Пожалуйста, создайте ее и положите туда ваши .md файлы.")
    elif not os.path.exists(QUESTIONS_FILE):
        print(f"❌ Файл '{QUESTIONS_FILE}' не найден. Пожалуйста, создайте его и наполните вопросами.")
    else:
        main()