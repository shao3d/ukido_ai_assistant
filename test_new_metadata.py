"""
Тестирование улучшений метаданных с CustomMetadataExtractor
Обрабатывает указанный файл и показывает результаты анализа.
"""

import logging
import os
import argparse
from collections import Counter
from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.openrouter import OpenRouter
from custom_metadata_extractor import CustomMetadataExtractor

try:
    from config import config
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import config

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Константы по умолчанию
DEFAULT_TEST_FILE = "data_facts/faq.md"
QUESTIONS_FILE = "questions.txt"

def load_questions(filepath: str) -> str:
    """Загружает и форматирует вопросы из файла."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        logger.info(f"✅ Загружено {len(questions)} вопросов-примеров из '{filepath}'.")
        return "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    except FileNotFoundError:
        logger.error(f"❌ Файл с вопросами '{filepath}' не найден!")
        return "1. Что это за курс?\n2. Сколько это стоит?\n3. Как записаться?"

def setup_models():
    """Настраивает модели LLM и эмбеддинга."""
    logger.info("⚙️ Настройка моделей...")
    
    llm = OpenRouter(
        api_key=config.OPENROUTER_API_KEY, 
        model="mistralai/mistral-7b-instruct-v0.2"
    )
    embed_model = GeminiEmbedding(
        model_name=config.EMBEDDING_MODEL, 
        api_key=config.GEMINI_API_KEY
    )
    
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    return llm, embed_model

def create_pipeline(llm, embed_model):
    """Создает pipeline для обработки документов."""
    logger.info("🔪 Создание pipeline...")
    
    # Загружаем примеры вопросов
    example_questions = load_questions(QUESTIONS_FILE)
    
    # Создаем промпт для генерации вопросов
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
    
    # Создаем экстрактор вопросов
    question_extractor = QuestionsAnsweredExtractor(
        questions=5,
        prompt_template=qa_generate_prompt_str,
        llm=llm,
    )
    
    # Создаем pipeline
    pipeline = IngestionPipeline(
        transformations=[
            MarkdownNodeParser(include_metadata=True),
            SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=embed_model
            ),
            question_extractor,
            CustomMetadataExtractor(),  # НАША НОВАЯ ФУНКЦИЯ
            embed_model,
        ],
    )
    
    return pipeline

def display_chunk_details(chunk, chunk_num):
    """Отображает детали чанка в красивом формате."""
    print(f"\n{'='*50}")
    print(f"ЧАНК {chunk_num}")
    print(f"{'='*50}")
    
    # Показываем первые 200 символов текста
    text = getattr(chunk, 'text', '')
    print(f"Текст (первые 200 символов): {text[:200]}...")
    
    # Показываем метаданные
    metadata = getattr(chunk, 'metadata', {})
    print(f"\nМетаданные:")
    
    # Список важных метаданных для показа
    important_metadata = [
        'questions_this_excerpt_can_answer',
        'content_type',
        'has_pricing',
        'course_mentioned',
        'is_teacher_info',
        'is_faq',
        'text_length',
        'has_courses'
    ]
    
    for key in important_metadata:
        if key in metadata:
            value = metadata[key]
            if isinstance(value, list) and len(value) > 3:
                # Ограничиваем длинные списки
                print(f"  • {key}: {value[:3]}... ({len(value)} всего)")
            else:
                print(f"  • {key}: {value}")
    
    # Показываем дополнительные метаданные
    other_metadata = {k: v for k, v in metadata.items() if k not in important_metadata}
    if other_metadata:
        print(f"\nДополнительные метаданные:")
        for key, value in other_metadata.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  • {key}: {value[:100]}...")
            else:
                print(f"  • {key}: {value}")

def display_statistics(chunks):
    """Отображает статистику по обработанным чанкам."""
    print(f"\n{'='*60}")
    print("СТАТИСТИКА")
    print(f"{'='*60}")
    
    total_chunks = len(chunks)
    print(f"📊 Всего чанков обработано: {total_chunks}")
    
    # Статистика по has_pricing
    pricing_count = sum(1 for chunk in chunks 
                       if getattr(chunk, 'metadata', {}).get('has_pricing', False))
    print(f"💰 Чанков с ценовой информацией (has_pricing=True): {pricing_count}")
    
    # Статистика по is_faq
    faq_count = sum(1 for chunk in chunks 
                   if getattr(chunk, 'metadata', {}).get('is_faq', False))
    print(f"❓ Чанков с FAQ (is_faq=True): {faq_count}")
    
    # Статистика по is_teacher_info
    teacher_count = sum(1 for chunk in chunks 
                       if getattr(chunk, 'metadata', {}).get('is_teacher_info', False))
    print(f"👨‍🏫 Чанков с информацией о преподавателях (is_teacher_info=True): {teacher_count}")
    
    # Статистика по content_type
    content_types = Counter()
    for chunk in chunks:
        content_type = getattr(chunk, 'metadata', {}).get('content_type', 'unknown')
        content_types[content_type] += 1
    
    print(f"\n📋 Распределение по типам контента:")
    for content_type, count in content_types.most_common():
        print(f"  • {content_type}: {count} чанков")
    
    # Статистика по упоминаниям курсов
    all_courses = []
    for chunk in chunks:
        courses = getattr(chunk, 'metadata', {}).get('course_mentioned', [])
        all_courses.extend(courses)
    
    if all_courses:
        course_counts = Counter(all_courses)
        print(f"\n🎓 Упоминания курсов:")
        for course, count in course_counts.most_common():
            print(f"  • {course}: {count} раз")
    else:
        print(f"\n🎓 Курсы не упоминаются в данном файле")

def parse_arguments():
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Тестирование улучшений метаданных с CustomMetadataExtractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python test_new_metadata.py                           # Использует файл по умолчанию: data_facts/faq.md
  python test_new_metadata.py --file data_facts/pricing.md
  python test_new_metadata.py --file data_facts/courses_detailed.md
  python test_new_metadata.py -f data_facts/methodology.md
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        default=DEFAULT_TEST_FILE,
        help=f'Путь к файлу для тестирования (по умолчанию: {DEFAULT_TEST_FILE})'
    )
    
    return parser.parse_args()

def main():
    """Основная функция для тестирования."""
    # Парсим аргументы командной строки
    args = parse_arguments()
    test_file = args.file
    
    print("🧪 ТЕСТИРОВАНИЕ УЛУЧШЕНИЙ МЕТАДАННЫХ")
    print("=" * 60)
    print(f"📁 Тестируемый файл: {test_file}")
    print("=" * 60)
    
    # Проверяем наличие файла
    if not os.path.exists(test_file):
        print(f"❌ Файл '{test_file}' не найден!")
        print(f"\n💡 Убедитесь, что файл существует по указанному пути.")
        print(f"📋 Доступные файлы в data_facts/:")
        
        # Показываем доступные файлы в data_facts, если папка существует
        data_facts_dir = "data_facts"
        if os.path.exists(data_facts_dir):
            files = [f for f in os.listdir(data_facts_dir) if f.endswith('.md')]
            for file in sorted(files):
                print(f"  • {os.path.join(data_facts_dir, file)}")
        
        return
    
    try:
        # Настраиваем модели
        llm, embed_model = setup_models()
        
        # Создаем pipeline
        pipeline = create_pipeline(llm, embed_model)
        
        # Загружаем документ
        logger.info(f"📂 Загрузка документа '{test_file}'...")
        documents = SimpleDirectoryReader(
            input_files=[test_file]
        ).load_data()
        
        # Обрабатываем документ
        logger.info("🧠 Обработка документа... Это может занять несколько минут.")
        chunks = pipeline.run(documents=documents, show_progress=True)
        
        logger.info(f"✅ Создано {len(chunks)} чанков.")
        
        # Показываем первые 6 чанков
        print(f"\n🔍 АНАЛИЗ ПЕРВЫХ 6 ЧАНКОВ:")
        for i, chunk in enumerate(chunks[:6], 1):
            display_chunk_details(chunk, i)
        
        # Показываем общую статистику
        display_statistics(chunks)
        
        print(f"\n🎉 Тестирование завершено успешно!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка во время тестирования: {e}")
        print(f"\n❌ Произошла ошибка: {e}")

if __name__ == "__main__":
    main()