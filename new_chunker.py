"""
================================================================
UKIDO MARKDOWN CHUNKER - ПРОСТОЙ РАБОЧИЙ "МОЛОТОК"
================================================================
Простой, надежный чанкер для обработки Markdown файлов школы Ukido.
Без излишеств - только то что нужно для работы.
"""

import os
import time
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import List, Dict
import re
import logging

# === ПРОСТАЯ НАСТРОЙКА ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

# Проверка переменных окружения
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
PINECONE_HOST = os.getenv("PINECONE_HOST_FACTS")

if not all([GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_HOST]):
    raise ValueError("❌ Отсутствуют переменные окружения")

logger.info("✅ Конфигурация проверена")

class SimpleMarkdownChunker:

    """
    Простой чанкер для файлов Ukido - без излишеств!
    """

    def __init__(self):
        # Настройка Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("🚀 Простой чанкер готов к работе")

    def chunk_courses(self, content: str) -> List[Dict]:
        """
        Простое чанкование курсов по разделителям ---
        """
        chunks = []
        sections = content.split('\n---\n')  # Простой split
        for section in sections:
            section = section.strip()
            if len(section) < 50:  # Слишком короткая секция
                continue
            # Ищем название курса
            course_match = re.search(r'#\s*КУРС\s+"([^"]+)"', section)
            if course_match:
                course_name = course_match.group(1).lower().replace(' ', '_')
                logger.info(f"📚 Найден курс: {course_match.group(1)}")
            else:
                course_name = "unknown_course"
                logger.warning("⚠️ Курс без названия")
            chunks.append({
                "text": section,
                "type": "course_detail",
                "course": course_name
            })
        return chunks

    def chunk_teachers(self, content: str) -> List[Dict]:
        """
        Простое чанкование преподавателей по разделителям ---
        """
        chunks = []
        sections = content.split('\n---\n')  # Простой split
        
        for section in sections:
            section = section.strip()
            if len(section) < 100:
                continue
                
            # Ищем имя преподавателя
            # Ищем заголовок, состоящий ТОЛЬКО из 2-3 заглавных слов (Имя Фамилия)
            name_match = re.search(r'#\s*([А-ЯЁ]+\s+[А-ЯЁ]+(\s+[А-ЯЁ]+)?)\s*$', section)
            if name_match:
                teacher_name = name_match.group(1)
                logger.info(f"👨‍🏫 Найден преподаватель: {teacher_name}")
                
                # Основной чанк
                chunks.append({
                    "text": section,
                    "type": "teacher_overview",
                    "teacher": teacher_name.lower().replace(' ', '_')
                })
                
                # Микро-чанки для фактов
                if "опыт" in section.lower() and re.search(r'(\d+)', section):
                    experience_match = re.search(r'(\d+)[^\d]*лет', section, re.I)
                    if experience_match:
                        years = experience_match.group(1)
                        chunks.append({
                            "text": f"Опыт работы {teacher_name} составляет {years} лет.",
                            "type": "teacher_experience",
                            "teacher": teacher_name.lower().replace(' ', '_')
                        })
        
        return chunks

    def chunk_standard_file(self, content: str, filename: str) -> List[Dict]:
        """
        Стандартное чанкование по разделителям ---
        """
        chunks = []
        sections = content.split('\n---\n')
        for section in sections:
            section = section.strip()
            chunks.append({
                "text": section,
                "type": filename.replace('.md', '').replace('.txt', '')
            })
        return chunks

    def process_files(self, directory: str) -> List[Dict]:
        """
        Простая обработка всех файлов
        """
        all_chunks = []
        chunk_id = 0
        
        # Ищем файлы .md и .txt
        files = [f for f in os.listdir(directory) if f.endswith(('.md', '.txt'))]
        logger.info(f"📁 Найдено {len(files)} файлов")
        
        for filename in files:
            logger.info(f"📄 Обрабатываю: {filename}")
            
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                content = f.read()

            # Выбираем метод чанкования
            if 'courses' in filename:
                chunks = self.chunk_courses(content)
            elif 'teachers' in filename:
                chunks = self.chunk_teachers(content)
            else:
                # ВСЕ ОСТАЛЬНЫЕ ФАЙЛЫ (pricing, conditions, faq и т.д.)
                # обрабатываются стандартным методом по разделителю '---'
                chunks = self.chunk_standard_file(content, filename)
            
            # Добавляем ID и метаданные
            for chunk in chunks:
                chunk_id += 1
                all_chunks.append({
                    "id": f"ukido-{chunk_id}",
                    "text": chunk["text"],
                    "metadata": {
                        "source": filename,
                        "type": chunk["type"],
                        **{k: v for k, v in chunk.items() if k not in ["text", "type"]}
                    }
                })
        
        logger.info(f"✅ Создано {len(all_chunks)} чанков")
        return all_chunks

    def vectorize_and_upload(self, chunks: List[Dict]) -> bool:
        """
        Простая векторизация и загрузка в Pinecone
        """
        if not chunks:
            logger.error("❌ Нет чанков для загрузки")
            return False
        
        logger.info(f"🔄 Векторизую {len(chunks)} чанков...")
        
        vectors = []
        for i, chunk in enumerate(chunks):
            try:
                # Создаем эмбеддинг
                response = genai.embed_content(
                    model='models/text-embedding-004',
                    content=chunk['text'],
                    task_type="RETRIEVAL_DOCUMENT"
                )
                
                vectors.append({
                    "id": chunk['id'],
                    "values": response['embedding'],
                    "metadata": {
                        "text": chunk['text'][:500],  # Короткий отрывок
                        **chunk['metadata']
                    }
                })
                
                if (i + 1) % 5 == 0:
                    logger.info(f"  📊 {i + 1}/{len(chunks)} готово")
                    time.sleep(0.5)  # Пауза для API
                    
            except Exception as e:
                logger.error(f"❌ Ошибка векторизации {chunk['id']}: {e}")
                continue
        
        if not vectors:
            logger.error("❌ Не удалось создать ни одного вектора")
            return False
        
        # Загружаем в Pinecone
        logger.info("🔌 Подключение к Pinecone...")
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(host=PINECONE_HOST)
            
            # Очищаем старые данные
            logger.info("🧹 Очистка базы...")
            index.delete(delete_all=True)
            time.sleep(5)
            
            # Загружаем новые данные
            logger.info(f"📤 Загружаю {len(vectors)} векторов...")
            batch_size = 50
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch)
                logger.info(f"  📦 Батч {i//batch_size + 1} загружен")
                time.sleep(1)
            
            # Проверка больше не нужна, т.к. upsert выдал бы ошибку.
            # Просто сообщаем об успехе.
            stats = index.describe_index_stats()
            final_count = stats.get('total_vector_count', 'неизвестно (статистика обновляется)')
            logger.info(f"🎉 Команда на загрузку {len(vectors)} векторов успешно отправлена! Финальное количество в базе: {final_count}")

            return True # Считаем успешным, если не было исключений
        except Exception as e:
            logger.error(f"❌ Ошибка Pinecone: {e}")
            return False

def main():
    """
    Простой запуск чанкера
    """
    logger.info("🚀 UKIDO CHUNKER - ПРОСТОЙ МОЛОТОК")
    logger.info("=" * 50)
    
    try:
        chunker = SimpleMarkdownChunker()
        
        # Обрабатываем файлы
        chunks = chunker.process_files("data_facts")
        
        # Загружаем в Pinecone
        success = chunker.vectorize_and_upload(chunks)
        
        if success:
            logger.info("🎉 ГОТОВО! База знаний обновлена!")
        else:
            logger.error("❌ Что-то пошло не так")
            
    except Exception as e:
        logger.error(f"💥 Ошибка: {e}")

if __name__ == "__main__":
    main()