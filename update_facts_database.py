import os
import time
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import re

# --- НАСТРОЙКИ И ЗАГРУЗКА КЛЮЧЕЙ ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST_FACTS = os.getenv("PINECONE_HOST_FACTS")

if not all([GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_HOST_FACTS]):
    raise ValueError("Необходимо задать все ключи в файле .env")

# --- КОНФИГУРАЦИЯ КЛИЕНТОВ ---
genai.configure(api_key=GEMINI_API_KEY)
embedding_model = 'models/text-embedding-004'
pc = Pinecone(api_key=PINECONE_API_KEY)

def clear_pinecone_index(index):
    """
    Полностью очищает индекс Pinecone от всех векторов.
    Это нужно для полной замены данных новыми.
    """
    print("🗑️ Очищаем индекс от старых данных...")
    try:
        # Получаем статистику индекса
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        
        if total_vectors == 0:
            print("   ✅ Индекс уже пуст")
            return
        
        print(f"   📊 Найдено {total_vectors} векторов для удаления")
        
        # Удаляем все векторы (delete_all - это специальная команда Pinecone)
        index.delete(delete_all=True)
        
        # Ждем завершения операции удаления
        print("   ⏳ Ожидаем завершения удаления...")
        time.sleep(5)
        
        # Проверяем результат
        stats = index.describe_index_stats()
        print(f"   ✅ Индекс очищен. Осталось векторов: {stats.total_vector_count}")
        
    except Exception as e:
        print(f"   ❌ Ошибка при очистке индекса: {e}")
        raise e

def analyze_chunk_completeness(chunk_text, context_info=""):
    """
    Использует Gemini для анализа семантической целостности чанка.
    Возвращает рекомендации по улучшению границ чанка.
    """
    analysis_prompt = f"""
Ты - эксперт по анализу текстов. Проанализируй следующий фрагмент текста на предмет его семантической завершенности.

КОНТЕКСТ: Это часть справочной информации о детской школе развития soft-skills "Ukido". 
{context_info}

АНАЛИЗИРУЕМЫЙ ФРАГМЕНТ:
"{chunk_text}"

ЗАДАЧА: Определи, является ли этот фрагмент семантически завершенным и подходящим для самостоятельного использования в системе поиска.

Ответь ТОЛЬКО в следующем формате:
СТАТУС: [ЗАВЕРШЕН/НЕЗАВЕРШЕН/ЧАСТИЧНО_ЗАВЕРШЕН]
ПРИЧИНА: [краткое объяснение]
РЕКОМЕНДАЦИЯ: [что можно улучшить или как скорректировать границы]
КЛЮЧЕВЫЕ_ТЕМЫ: [2-3 основные темы фрагмента]
"""
    
    try:
        response = genai.GenerativeModel('gemini-1.5-flash').generate_content(analysis_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"      ⚠️ Ошибка анализа чанка: {e}")
        return "СТАТУС: ЧАСТИЧНО_ЗАВЕРШЕН\nПРИЧИНА: Ошибка анализа\nРЕКОМЕНДАЦИЯ: Использовать как есть"

def create_intelligent_chunks(content, filename):
    """
    Создает семантически осмысленные чанки с помощью AI-анализа.
    Это продвинутая версия чанкинга, которая учитывает смысловую целостность.
    """
    print(f"   🧠 Начинаю интеллектуальное разбиение файла {filename}")
    
    # Этап 1: Первичное структурное разбиение
    # Сначала разделяем по явным разделителям, которые авторы поставили специально
    primary_sections = re.split(r'\n---\n', content)
    print(f"   📋 Найдено {len(primary_sections)} основных разделов")
    
    intelligent_chunks = []
    
    for section_idx, section in enumerate(primary_sections):
        section = section.strip()
        if not section or len(section) < 100:
            continue
            
        print(f"   🔍 Анализирую раздел {section_idx + 1} (длина: {len(section)} символов)")
        
        # Этап 2: Если раздел умеренного размера, анализируем его целиком
        if len(section) <= 1200:
            analysis = analyze_chunk_completeness(
                section, 
                f"Раздел {section_idx + 1} из файла {filename}"
            )
            print(f"      🤖 AI анализ: {analysis.split('СТАТУС:')[1].split('ПРИЧИНА:')[0].strip()}")
            intelligent_chunks.append(section)
            
        # Этап 3: Если раздел большой, разбиваем умно
        else:
            print(f"      ✂️ Раздел большой, выполняю подразбиение")
            # Разбиваем по абзацам (двойной перенос строки)
            paragraphs = re.split(r'\n\n', section)
            
            current_chunk = ""
            for para_idx, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                # Проверяем, не станет ли чанк слишком большим
                potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
                
                if len(potential_chunk) > 1000 and current_chunk:
                    # Текущий чанк готов, анализируем его
                    analysis = analyze_chunk_completeness(
                        current_chunk,
                        f"Часть раздела {section_idx + 1}, абзацы до {para_idx}"
                    )
                    
                    # Извлекаем статус из анализа
                    status = "ЧАСТИЧНО_ЗАВЕРШЕН"  # значение по умолчанию
                    if "СТАТУС:" in analysis:
                        status = analysis.split("СТАТУС:")[1].split("ПРИЧИНА:")[0].strip()
                    
                    print(f"         🎯 Подчанк готов, AI статус: {status}")
                    intelligent_chunks.append(current_chunk)
                    current_chunk = paragraph
                else:
                    current_chunk = potential_chunk
            
            # Добавляем последний чанк раздела
            if current_chunk:
                analysis = analyze_chunk_completeness(
                    current_chunk,
                    f"Завершающая часть раздела {section_idx + 1}"
                )
                status = "ЧАСТИЧНО_ЗАВЕРШЕН"
                if "СТАТУС:" in analysis:
                    status = analysis.split("СТАТУС:")[1].split("ПРИЧИНА:")[0].strip()
                print(f"         🎯 Финальный подчанк, AI статус: {status}")
                intelligent_chunks.append(current_chunk)
    
    print(f"   ✅ Интеллектуальное разбиение завершено: {len(intelligent_chunks)} чанков")
    return intelligent_chunks

def process_and_upload_updated_data(directory_path, pinecone_index, index_name):
    """
    Читает все .txt файлы из директории, создает семантически осмысленные чанки
    с помощью AI-анализа, создает эмбеддинги и загружает в Pinecone.
    
    Эта версия использует продвинутый семантический чанкинг.
    """
    print(f"\n📚 Начинаю обработку обновленной базы знаний: {directory_path}")
    print("🧠 Используется интеллектуальный семантический чанкинг с AI-анализом")
    
    vector_id_counter = 0
    total_chunks = 0
    
    # Получаем список всех txt файлов
    txt_files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
    print(f"📄 Найдено файлов для обработки: {len(txt_files)}")
    
    for file_idx, filename in enumerate(txt_files):
        file_path = os.path.join(directory_path, filename)
        print(f"\n📖 Обрабатываю файл {file_idx + 1}/{len(txt_files)}: {filename}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print(f"   ⚠️ Файл {filename} пуст, пропускаю")
            continue
        
        # Используем новый интеллектуальный чанкинг
        intelligent_chunks = create_intelligent_chunks(content, filename)
        
        # Обрабатываем каждый семантически осмысленный чанк
        for chunk_idx, chunk in enumerate(intelligent_chunks):
            if not chunk or len(chunk.strip()) < 50:
                continue
                
            print(f"   🔄 Создаю вектор для чанка {chunk_idx + 1}/{len(intelligent_chunks)} (длина: {len(chunk)} символов)")
            
            try:
                # Создаем векторное представление
                embedding = genai.embed_content(
                    model=embedding_model,
                    content=chunk,
                    task_type="RETRIEVAL_DOCUMENT",
                    title=f"Intelligent chunk from {filename}"
                )
                
                # Подготавливаем данные для загрузки в Pinecone
                vector_to_upsert = {
                    "id": f"{index_name}-{vector_id_counter}",
                    "values": embedding['embedding'],
                    "metadata": {
                        "text": chunk,
                        "source": filename,
                        "chunk_index": chunk_idx,
                        "chunk_length": len(chunk),
                        "chunking_method": "intelligent_semantic"
                    }
                }
                
                # Загружаем в Pinecone
                pinecone_index.upsert(vectors=[vector_to_upsert])
                print(f"      ✅ Вектор {vector_to_upsert['id']} загружен успешно")
                
                vector_id_counter += 1
                total_chunks += 1
                
                # Пауза между запросами для стабильности
                time.sleep(1)

            except Exception as e:
                print(f"      ❌ Ошибка при обработке чанка №{chunk_idx + 1}: {e}")
                print(f"      📄 Проблемный текст: {chunk[:100]}...")
                continue

    print(f"\n🎉 Обработка завершена!")
    print(f"📊 Статистика:")
    print(f"   📁 Обработано файлов: {len(txt_files)}")
    print(f"   📝 Создано семантических чанков: {total_chunks}")
    print(f"   💾 Загружено векторов в индекс: {vector_id_counter}")
    print("🧠 Все чанки прошли AI-анализ на семантическую целостность")

def main():
    """
    Основная функция для обновления базы фактов
    """
    print("🚀 ОБНОВЛЕНИЕ БАЗЫ ЗНАНИЙ UKIDO")
    print("=" * 50)
    
    try:
        # Подключаемся к индексу фактов
        print("🔌 Подключаюсь к Pinecone...")
        index_facts = pc.Index(host=PINECONE_HOST_FACTS)
        print("   ✅ Подключение установлено")
        
        # Показываем текущую статистику
        stats = index_facts.describe_index_stats()
        print(f"📊 Текущее состояние индекса: {stats.total_vector_count} векторов")
        
        # Спрашиваем подтверждение на очистку
        print("\n⚠️  ВНИМАНИЕ: Сейчас будет выполнена полная замена данных!")
        print("   Все старые векторы будут удалены и заменены новыми.")
        
        # В продакшн среде можно добавить input() для подтверждения
        # response = input("Продолжить? (yes/no): ")
        # if response.lower() != 'yes':
        #     print("Операция отменена")
        #     return
        
        # Очищаем индекс
        clear_pinecone_index(index_facts)
        
        # Загружаем новые данные
        process_and_upload_updated_data("data_facts", index_facts, "ukido")
        
        # Финальная проверка
        print("\n🔍 Проверяю результат...")
        time.sleep(3)  # Ждем индексации
        final_stats = index_facts.describe_index_stats()
        print(f"📊 Новое состояние индекса: {final_stats.total_vector_count} векторов")
        
        print("\n🎊 ОБНОВЛЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("   RAG система теперь использует расширенную базу знаний")
        
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        print("   Обратитесь к техническому специалисту")

if __name__ == "__main__":
    main()
