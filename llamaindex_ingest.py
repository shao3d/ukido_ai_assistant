# llamaindex_ingest.py (ФИНАЛЬНАЯ ВЕРСИЯ: Структурный + Семантический парсинг)
"""
Финальная, промышленная версия скрипта для создания и загрузки индекса в Pinecone.
Использует двухступенчатый парсинг (Markdown -> Semantic) и генерацию
вопросов на основе примеров для максимальной точности RAG.

Для запуска:
1. Убедитесь, что установленны все зависимости из requirements.txt.
2. Создайте файл 'questions.txt' с вопросами-примерами.
3. Создайте папку 'data_facts' с вашими .md файлами.
4. Запустите скрипт: python llamaindex_ingest.py
"""
import logging
import os
import pinecone
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
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

# Попытка импортировать 'config' из родительской директории, если скрипт запускается из подпапки.
# Если не удается, предполагаем, что config.py находится в той же директории.
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

    # --- 1. ЗАГРУЗКА ВОПРОСОВ-ПРИМЕРОВ ---
    example_questions = load_questions(QUESTIONS_FILE)

    # --- 2. НАСТРОЙКА МОДЕЛЕЙ ---
    logger.info("⚙️  Шаг 1: Настройка моделей...")
    # LLM для генерации мета-вопросов (быстрая и дешевая)
    llm = OpenRouter(api_key=config.OPENROUTER_API_KEY, model="mistralai/mistral-7b-instruct-v0.2")
    # Модель для векторизации (эмбеддингов)
    embed_model = GeminiEmbedding(model_name=config.EMBEDDING_MODEL, api_key=config.GEMINI_API_KEY)
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # --- 3. НАСТРОЙКА ПРОМПТА ДЛЯ ГЕНЕРАЦИИ ВОПРОСОВ ---
    qa_generate_prompt_tmpl = PromptTemplate(
        "Ниже предоставлен контекст:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Используя этот контекст и не используя предварительные знания, "
        "сгенерируй {num_questions} вопросов, на которые отвечает данный контекст.\n"
        "Вопросы должны быть в стиле и по тематике, как в этих примерах:\n"
        "--- ПРИМЕРЫ ВОПРОСОВ ---\n"
        "{example_questions}\n"
        "------------------------\n"
        "Сгенерированные вопросы:\n"
    )
    qa_generate_prompt = qa_generate_prompt_tmpl.partial_format(example_questions=example_questions)

    # --- 4. ЗАГРУЗКА ДОКУМЕНТОВ ---
    logger.info(f"📂 Шаг 2: Загрузка документов из '{DATA_DIRECTORY}'...")
    documents = SimpleDirectoryReader(DATA_DIRECTORY, required_exts=[".md"]).load_data()

    # --- 5. НАСТРОЙКА ДВУХСТУПЕНЧАТОГО КОНВЕЙЕРА ОБРАБОТКИ ---
    logger.info("🔪 Шаг 3: Настройка ДВУХСТУПЕНЧАТОГО конвейера обработки...")
    
    question_extractor = QuestionsAnsweredExtractor(
        questions=5,
        prompt_template=qa_generate_prompt,
        llm=llm,
    )

    pipeline = IngestionPipeline(
        transformations=[
            # ЭТАП 1: Грубая структурная разбивка по заголовкам Markdown
            MarkdownNodeParser(include_metadata=True),
            
            # ЭТАП 2: Тонкая семантическая нарезка внутри каждого структурного блока
            SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=embed_model
            ),
            
            # ЭТАП 3: Обогащение финальных чанков мета-вопросами
            question_extractor,
            
            # ЭТАП 4: Векторизация
            embed_model,
        ],
    )
    
    logger.info("🧠 Обработка документов... Это может занять несколько минут.")
    nodes = pipeline.run(documents=documents, show_progress=True)
    logger.info(f"✅ Создано {len(nodes)} финальных, умных чанков.")

    # --- 6. ПОДКЛЮЧЕНИЕ К PINECONE И ЗАГРУЗКА ---
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