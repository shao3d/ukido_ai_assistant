import os
import time
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import re
from typing import List, Dict, Optional
from datetime import datetime
import pdfplumber

# --- НАСТРОЙКИ ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST_STYLE = os.getenv("PINECONE_HOST_STYLE")

# Проверяем наличие всех необходимых ключей
if not all([GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_HOST_STYLE]):
    missing = [name for name, value in [
        ('GEMINI_API_KEY', GEMINI_API_KEY),
        ('PINECONE_API_KEY', PINECONE_API_KEY), 
        ('PINECONE_HOST_STYLE', PINECONE_HOST_STYLE)
    ] if not value]
    raise ValueError(f"Отсутствуют обязательные переменные: {', '.join(missing)}")

# Инициализация Gemini API для векторизации
genai.configure(api_key=GEMINI_API_KEY)

class SimpleZhvanetskyProcessor:
    """
    Упрощенный процессор для текстов Жванецкого.
    Фокусируется только на основной функциональности:
    1. Чтение PDF файлов
    2. Умное чанкование с сохранением стиля
    3. Векторизация через text-embedding-004
    4. Загрузка в Pinecone
    """
    
    def __init__(self):
        # Настройки для чанкования
        self.min_chunk_size = 300
        self.ideal_chunk_size = 800
        self.max_chunk_size = 1500
        
        # Настройки для векторизации
        self.embedding_model = 'models/text-embedding-004'
        self.delay_between_requests = 0.1  # Простая задержка 100ms между запросами
        
        print("🎭 ПРОСТОЙ ПРОЦЕССОР ТЕКСТОВ ЖВАНЕЦКОГО")
        print(f"📝 Размеры чанков: {self.min_chunk_size}-{self.ideal_chunk_size} символов")
        print(f"⏱️ Задержка между запросами: {self.delay_between_requests}с")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Извлекает текст из PDF файла"""
        print(f"📄 Извлечение текста из PDF: {os.path.basename(pdf_path)}")
        
        try:
            extracted_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"   📊 Всего страниц: {total_pages}")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        # Простая очистка текста
                        cleaned_text = re.sub(r'\n{3,}', '\n\n', page_text)
                        cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
                        extracted_text += cleaned_text + "\n\n"
                    
                    # Показываем прогресс каждые 50 страниц
                    if page_num % 50 == 0:
                        print(f"   📖 Обработано страниц: {page_num}/{total_pages}")
                
                print(f"   ✅ Извлечено {len(extracted_text)} символов")
                return extracted_text.strip()
                
        except Exception as e:
            print(f"   ❌ Ошибка при извлечении текста: {e}")
            return ""

    def is_dialogue(self, text: str) -> bool:
        """Простая проверка на диалог в тексте Жванецкого"""
        dialogue_markers = [
            r'—\s*[А-ЯЁ]',  # Прямая речь с тире
            r'[А-ЯЁ][а-яё]*:',  # Персонаж с двоеточием
            r'Директор:',  # Типичные персонажи
            r'Костоглазов:',
        ]
        
        marker_count = sum(1 for pattern in dialogue_markers if re.search(pattern, text))
        return marker_count >= 2

    def create_chunks(self, content: str, filename: str) -> List[str]:
        """Создает чанки из текста с сохранением стиля Жванецкого"""
        print(f"✂️ Создание чанков: {filename}")
        
        # Разбиваем на абзацы
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        print(f"   📄 Найдено абзацев: {len(paragraphs)}")
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            
            # Если это диалог и он не слишком длинный, сохраняем целиком
            if self.is_dialogue(paragraph) and paragraph_length <= self.max_chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                chunks.append(paragraph)
                print(f"   💬 Диалог сохранен: {paragraph_length} символов")
                continue
            
            # Обычная логика объединения абзацев
            potential_size = current_size + paragraph_length + 2  # +2 для \n\n
            
            if potential_size > self.ideal_chunk_size and current_size >= self.min_chunk_size:
                # Сохраняем накопленный чанк и начинаем новый
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_size = paragraph_length
            else:
                # Добавляем к текущему чанку
                current_chunk.append(paragraph)
                current_size = potential_size
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # Простая постобработка: объединяем слишком короткие чанки
        processed_chunks = []
        for chunk in chunks:
            if len(chunk) < self.min_chunk_size and processed_chunks:
                # Объединяем с предыдущим, если возможно
                if len(processed_chunks[-1] + '\n\n' + chunk) <= self.max_chunk_size:
                    processed_chunks[-1] = processed_chunks[-1] + '\n\n' + chunk
                    continue
            
            processed_chunks.append(chunk)
        
        print(f"   🎯 Создано чанков: {len(processed_chunks)}")
        return processed_chunks

    def generate_safe_id(self, index_name: str, filename: str, chunk_idx: int) -> str:
        """Создает ASCII-совместимый ID для Pinecone"""
        # Простая транслитерация основных символов
        transliteration = {
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
            'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
            'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
            'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
            'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
            'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
            'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
            'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
            'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Shch',
            'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya'
        }
        
        # Убираем расширение файла
        clean_filename = os.path.splitext(filename)[0]
        
        # Применяем транслитерацию
        transliterated = ""
        for char in clean_filename:
            if char in transliteration:
                transliterated += transliteration[char]
            elif char.isalnum() or char in '-_':
                transliterated += char
            elif char in ' .()[]{}':
                transliterated += '-'
        
        # Очищаем множественные дефисы и ограничиваем длину
        normalized = re.sub(r'-+', '-', transliterated).strip('-')[:50]
        
        return f"{index_name}-{normalized}-{chunk_idx}"

    def vectorize_chunk(self, chunk: str, chunk_id: str) -> Optional[Dict]:
        """Векторизует чанк текста"""
        try:
            # Простая задержка между запросами
            time.sleep(self.delay_between_requests)
            
            # Создаем векторное представление
            response = genai.embed_content(
                model=self.embedding_model,
                content=chunk,
                task_type="RETRIEVAL_DOCUMENT",
                title="Zhvanetsky Style Sample"
            )
            
            # Определяем тип контента
            content_type = "dialogue" if self.is_dialogue(chunk) else "narrative"
            
            return {
                "id": chunk_id,
                "values": response['embedding'],
                "metadata": {
                    "text": chunk,
                    "chunk_size": len(chunk),
                    "content_type": content_type,
                    "style_source": "zhvanetsky",
                    "created_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"   ❌ Ошибка векторизации {chunk_id}: {e}")
            return None

    def process_directory(self, directory_path: str, index_name: str) -> Dict:
        """Обрабатывает директорию с файлами Жванецкого"""
        start_time = time.time()
        
        print(f"\n🎭 ОБРАБОТКА ТЕКСТОВ ЖВАНЕЦКОГО")
        print(f"📂 Директория: {directory_path}")
        print(f"🎯 Индекс: {index_name}")
        print("=" * 50)
        
        # Подключаемся к Pinecone
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(host=PINECONE_HOST_STYLE)
            print("🔌 Подключение к Pinecone успешно")
        except Exception as e:
            print(f"❌ Ошибка подключения к Pinecone: {e}")
            return {"success": False, "error": str(e)}
        
        # Очищаем индекс
        print("🗑️ Очистка существующих данных...")
        index.delete(delete_all=True)
        time.sleep(3)
        
        # Получаем список файлов
        try:
            all_files = os.listdir(directory_path)
            supported_files = [f for f in all_files if f.endswith(('.txt', '.pdf'))]
        except Exception as e:
            print(f"❌ Ошибка чтения директории: {e}")
            return {"success": False, "error": str(e)}
        
        if not supported_files:
            print("❌ Не найдено поддерживаемых файлов")
            return {"success": False, "error": "No files found"}
        
        print(f"📁 Найдено файлов: {len(supported_files)}")
        
        # Статистика
        stats = {
            "files_processed": 0,
            "total_chunks": 0,
            "vectors_uploaded": 0,
            "processing_time": 0,
            "file_details": []
        }
        
        # Обрабатываем каждый файл
        for file_idx, filename in enumerate(supported_files):
            print(f"\n📖 Файл {file_idx + 1}/{len(supported_files)}: {filename}")
            file_path = os.path.join(directory_path, filename)
            file_start = time.time()
            
            try:
                # Извлекаем содержимое
                if filename.endswith('.pdf'):
                    content = self.extract_text_from_pdf(file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                
                if len(content) < 100:
                    print(f"   ⚠️ Файл слишком мал, пропускаем")
                    continue
                
                # Создаем чанки
                chunks = self.create_chunks(content, filename)
                if not chunks:
                    print(f"   ⚠️ Не удалось создать чанки")
                    continue
                
                # Векторизуем и загружаем
                print(f"   🔄 Векторизация {len(chunks)} чанков...")
                vectors_uploaded = 0
                
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = self.generate_safe_id(index_name, filename, chunk_idx)
                    vector_data = self.vectorize_chunk(chunk, chunk_id)
                    
                    if vector_data:
                        index.upsert(vectors=[vector_data])
                        vectors_uploaded += 1
                        
                        # Показываем прогресс
                        if (chunk_idx + 1) % 10 == 0:
                            print(f"      📊 Обработано: {chunk_idx + 1}/{len(chunks)}")
                
                file_time = time.time() - file_start
                
                # Сохраняем статистику файла
                file_stat = {
                    "filename": filename,
                    "chunks_created": len(chunks),
                    "vectors_uploaded": vectors_uploaded,
                    "processing_time": file_time
                }
                
                stats["file_details"].append(file_stat)
                stats["files_processed"] += 1
                stats["total_chunks"] += len(chunks)
                stats["vectors_uploaded"] += vectors_uploaded
                
                print(f"   ✅ Файл обработан за {file_time:.1f}с")
                print(f"      📦 Чанков: {len(chunks)}")
                print(f"      💾 Векторов: {vectors_uploaded}")
                
            except Exception as e:
                print(f"   ❌ Ошибка обработки файла {filename}: {e}")
                continue
        
        # Финальная статистика
        total_time = time.time() - start_time
        stats["processing_time"] = total_time
        
        # Проверяем результат в Pinecone
        time.sleep(3)
        final_stats = index.describe_index_stats()
        
        print(f"\n🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
        print("=" * 40)
        print(f"📊 Результаты:")
        print(f"   📁 Файлов обработано: {stats['files_processed']}")
        print(f"   📝 Всего чанков: {stats['total_chunks']}")
        print(f"   💾 Векторов загружено: {stats['vectors_uploaded']}")
        print(f"   ⏱️ Время: {total_time/60:.1f} минут")
        print(f"✅ Векторов в Pinecone: {final_stats.total_vector_count}")
        
        return {"success": True, "stats": stats}

def main():
    """Основная функция"""
    processor = SimpleZhvanetskyProcessor()
    result = processor.process_directory("data_style", "ukido-style")
    
    if result["success"]:
        print("\n✨ Тексты Жванецкого успешно обработаны!")
        print("🎭 AI-ассистент готов к работе в стиле великого сатирика")
    else:
        print(f"\n❌ Произошла ошибка: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()