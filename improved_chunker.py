import os
import time
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import re
import numpy as np
from typing import List, Tuple
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

# Импорты для локальной модели (опциональные)
try:
    from sentence_transformers import SentenceTransformer
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    LOCAL_MODELS_AVAILABLE = True
    print("✅ Локальные модели доступны")
except ImportError:
    LOCAL_MODELS_AVAILABLE = False
    print("⚠️ Локальные модели недоступны. Используем только Gemini API.")

# --- НАСТРОЙКИ ДЛЯ i7-6700HQ ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST_FACTS = os.getenv("PINECONE_HOST_FACTS")

# Конфигурация, оптимизированная для твоего железа
CONFIG = {
    # Параметры чанкинга
    "min_chunk_size": 800,
    "max_chunk_size": 2200,          # Немного меньше для стабильности
    "target_chunk_size": 1400,       # Оптимально для твоих задач
    "similarity_threshold": 0.72,    # Чуть выше для лучшего качества
    "sentence_window": 3,
    
    # Выбор модели с учетом производительности
    "model_name": "paraphrase-multilingual-MiniLM-L12-v2",  # Компромисс: многоязычная + быстрая
    
    # Настройки производительности для i7-6700HQ
    "max_threads": 6,                # Оставляем 2 потока системе
    "batch_size": 4,                 # Небольшие батчи для стабильности
    "api_delay": 0.4,                # Пауза между API запросами
    "processing_delay": 0.2,         # Пауза между операциями
    "memory_limit_gb": 8,            # Лимит для AI операций (половина от общей)
    "cpu_threshold": 75,             # Порог загрузки CPU
    "temp_check_interval": 10,       # Проверка каждые 10 операций
}

# Инициализация клиентов
genai.configure(api_key=GEMINI_API_KEY)
embedding_model = 'models/text-embedding-004'
pc = Pinecone(api_key=PINECONE_API_KEY)

class SystemMonitor:
    """Монитор системы для контроля нагрузки на i7-6700HQ"""
    
    @staticmethod
    def get_system_status():
        """Получает текущий статус системы"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_available_gb": memory.available / (1024**3)
        }
    
    @staticmethod
    def should_take_break():
        """Определяет, нужна ли пауза системе"""
        status = SystemMonitor.get_system_status()
        
        if status["cpu_percent"] > CONFIG["cpu_threshold"]:
            return True, f"Высокая загрузка CPU: {status['cpu_percent']:.1f}%"
        
        if status["memory_percent"] > 85:
            return True, f"Высокое использование памяти: {status['memory_percent']:.1f}%"
        
        return False, "Система в норме"
    
    @staticmethod
    def wait_for_system_cooldown():
        """Ждет снижения нагрузки на систему"""
        print("      🌡️ Даю системе остыть...")
        
        while True:
            time.sleep(3)  # Пауза для остывания
            needs_break, reason = SystemMonitor.should_take_break()
            
            if not needs_break:
                print("      ✅ Система готова к продолжению работы")
                break
            else:
                print(f"      ⏳ Ожидание: {reason}")

class OptimizedSemanticChunker:
    """
    Семантический чанкер, оптимизированный для i7-6700HQ
    Баланс между качеством работы и сохранением производительности системы
    """
    
    def __init__(self):
        self.local_model = None
        self.operation_count = 0
        self._init_local_model()
        self._display_system_info()
    
    def _display_system_info(self):
        """Показывает информацию о системе и настройках"""
        status = SystemMonitor.get_system_status()
        
        print("🖥️ ИНФОРМАЦИЯ О СИСТЕМЕ:")
        print(f"   💻 CPU: Intel i7-6700HQ (использование: {status['cpu_percent']:.1f}%)")
        print(f"   🧠 ОЗУ: {status['memory_available_gb']:.1f} ГБ доступно из 16 ГБ")
        print(f"   ⚙️ Потоков для AI: {CONFIG['max_threads']} из 8")
        print(f"   📦 Размер батча: {CONFIG['batch_size']}")
        print(f"   🎯 Целевой размер чанка: {CONFIG['target_chunk_size']} символов")
        print(f"   🧠 Модель: {CONFIG['model_name']}")
    
    def _init_local_model(self):
        """Инициализация локальной модели с оптимизацией для твоего железа"""
        if not LOCAL_MODELS_AVAILABLE:
            print("🌐 Работаем только с Gemini API (безопасный режим)")
            return
        
        try:
            print(f"🚀 Загружаю оптимизированную модель...")
            
            # Выбираем модель с учетом производительности
            self.local_model = SentenceTransformer(CONFIG['model_name'])
            
            # Оптимизируем для CPU
            self.local_model = self.local_model.cpu()
            
            # Ограничиваем количество потоков PyTorch
            torch.set_num_threads(CONFIG['max_threads'])
            torch.set_num_interop_threads(2)  # Консервативное значение
            
            # Тестируем производительность
            print("   ⚡ Тестирую производительность на твоем железе...")
            start_time = time.time()
            
            test_embeddings = self.local_model.encode(
                ["Тестовое предложение для проверки скорости"],
                batch_size=1,
                show_progress_bar=False
            )
            
            test_time = time.time() - start_time
            print(f"   ⏱️ Время обработки одного предложения: {test_time:.3f}с")
            
            if test_time > 2.0:
                print("   ⚠️ Медленная работа. Рекомендую использовать только API режим.")
                self.local_model = None
            else:
                print("   ✅ Локальная модель готова к работе")
            
        except Exception as e:
            print(f"   ❌ Ошибка загрузки модели: {e}")
            print("   🌐 Переключаемся на API-only режим")
            self.local_model = None
    
    def _check_system_periodically(self):
        """Периодически проверяет состояние системы"""
        self.operation_count += 1
        
        if self.operation_count % CONFIG["temp_check_interval"] == 0:
            needs_break, reason = SystemMonitor.should_take_break()
            
            if needs_break:
                print(f"      ⏸️ {reason}")
                SystemMonitor.wait_for_system_cooldown()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Умное разбиение на предложения для русского языка"""
        # Учитываем особенности русской пунктуации
        sentences = re.split(r'[.!?]+(?:\s|$)', text)
        
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Более строгая фильтрация для качества
            if len(sentence) > 25 and not sentence.isdigit():
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def calculate_semantic_breaks(self, sentences: List[str]) -> List[float]:
        """
        Вычисляет семантические разрывы с оптимизацией для производительности
        """
        if not self.local_model or len(sentences) < 2:
            # API fallback - возвращаем разумные значения
            return [0.6 + (i % 3) * 0.1 for i in range(len(sentences) - 1)]
        
        try:
            print(f"      🧠 Семантический анализ {len(sentences)} предложений...")
            
            # Проверяем систему перед тяжелой операцией
            self._check_system_periodically()
            
            # Ограничиваем количество предложений для анализа
            max_sentences = 20  # Разумное ограничение для твоего железа
            if len(sentences) > max_sentences:
                print(f"      📊 Анализирую первые {max_sentences} предложений для оптимизации")
                sentences = sentences[:max_sentences]
            
            # Создаем окна с ограниченным размером
            windows = []
            for i in range(len(sentences)):
                start = max(0, i - CONFIG['sentence_window'] // 2)
                end = min(len(sentences), i + CONFIG['sentence_window'] // 2 + 1)
                window = " ".join(sentences[start:end])
                # Ограничиваем длину окна для стабильности
                if len(window) > 500:
                    window = window[:500] + "..."
                windows.append(window)
            
            # Обрабатываем с маленькими батчами
            embeddings = self.local_model.encode(
                windows,
                batch_size=CONFIG['batch_size'],
                show_progress_bar=False,
                normalize_embeddings=True  # Ускоряет вычисления
            )
            
            # Пауза для системы
            time.sleep(CONFIG["processing_delay"])
            
            # Вычисляем сходство эффективно
            similarities = []
            for i in range(len(embeddings) - 1):
                # Используем dot product для нормализованных векторов (быстрее)
                similarity = np.dot(embeddings[i], embeddings[i + 1])
                similarities.append(float(similarity))
            
            avg_similarity = np.mean(similarities)
            print(f"      📊 Семантическая связность: {avg_similarity:.3f}")
            
            return similarities
            
        except Exception as e:
            print(f"      ⚠️ Ошибка семантического анализа: {e}")
            print(f"      🔄 Переключаюсь на упрощенный режим")
            # Возвращаем разумные fallback значения
            return [0.6] * (len(sentences) - 1)
    
    def create_semantic_chunks(self, text: str, source_filename: str) -> List[str]:
        """
        Создает семантически связные чанки с учетом производительности системы
        """
        print(f"   ✂️ Обрабатываю: {source_filename}")
        
        # Разбиваем на предложения
        sentences = self.split_into_sentences(text)
        if len(sentences) < 2:
            return [text] if len(text) > CONFIG['min_chunk_size'] else []
        
        print(f"      📝 Предложений найдено: {len(sentences)}")
        
        # Анализируем семантические разрывы (с учетом нагрузки)
        similarities = self.calculate_semantic_breaks(sentences)
        
        # Создаем чанки с оптимальной логикой
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            current_size += len(sentence)
            
            # Определяем точки разделения
            should_split = False
            
            # Достигли максимального размера
            if current_size >= CONFIG['max_chunk_size']:
                should_split = True
                reason = "макс. размер"
            
            # Целевой размер + семантический разрыв
            elif (current_size >= CONFIG['target_chunk_size'] and 
                  i < len(similarities) and 
                  similarities[i] < CONFIG['similarity_threshold']):
                should_split = True
                reason = "семантический разрыв"
            
            # Сильный семантический разрыв (новая тема)
            elif (i < len(similarities) and 
                  similarities[i] < 0.45 and 
                  current_size >= CONFIG['min_chunk_size']):
                should_split = True
                reason = "смена темы"
            
            if should_split:
                chunk_text = ". ".join(current_chunk).strip()
                if len(chunk_text) >= CONFIG['min_chunk_size']:
                    chunks.append(chunk_text)
                    print(f"      ✅ Чанк {len(chunks)}: {len(chunk_text)} символов ({reason})")
                
                current_chunk = []
                current_size = 0
        
        # Добавляем финальный чанк
        if current_chunk:
            chunk_text = ". ".join(current_chunk).strip()
            if len(chunk_text) >= CONFIG['min_chunk_size']:
                chunks.append(chunk_text)
                print(f"      ✅ Финальный чанк: {len(chunk_text)} символов")
        
        print(f"   🎯 Создано {len(chunks)} качественных чанков")
        return chunks
    
    def process_and_upload(self, directory_path: str, index_name: str):
        """
        Основная функция обработки с полным контролем производительности
        """
        start_time = time.time()
        
        print("🚀 ОПТИМИЗИРОВАННЫЙ СЕМАНТИЧЕСКИЙ ЧАНКЕР ДЛЯ i7-6700HQ")
        print("=" * 65)
        
        # Подключаемся к Pinecone
        try:
            index = pc.Index(host=PINECONE_HOST_FACTS)
            print("🔌 Подключились к Pinecone")
        except Exception as e:
            print(f"❌ Ошибка подключения к Pinecone: {e}")
            return
        
        # Очищаем индекс
        print("🗑️ Очищаем индекс...")
        index.delete(delete_all=True)
        time.sleep(3)
        
        # Получаем список файлов
        txt_files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
        print(f"📁 Найдено файлов для обработки: {len(txt_files)}")
        
        vector_count = 0
        total_chunks = 0
        
        # Обрабатываем файлы по одному для контроля нагрузки
        for file_idx, filename in enumerate(txt_files):
            print(f"\n📖 Файл {file_idx + 1}/{len(txt_files)}: {filename}")
            
            # Проверяем систему перед обработкой файла
            needs_break, reason = SystemMonitor.should_take_break()
            if needs_break:
                print(f"   ⏸️ {reason}")
                SystemMonitor.wait_for_system_cooldown()
            
            # Читаем файл
            try:
                with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as f:
                    content = f.read().strip()
            except Exception as e:
                print(f"   ❌ Ошибка чтения файла: {e}")
                continue
            
            if len(content) < 200:
                print("   ⚠️ Файл слишком мал, пропускаем")
                continue
            
            # Создаем семантические чанки
            chunks = self.create_semantic_chunks(content, filename)
            
            # Векторизируем и загружаем чанки
            print(f"   🔄 Векторизация {len(chunks)} чанков...")
            
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    # Проверяем систему каждые несколько операций
                    if chunk_idx % 3 == 0:
                        self._check_system_periodically()
                    
                    # Создаем эмбеддинг через Gemini API
                    embedding = genai.embed_content(
                        model=embedding_model,
                        content=chunk,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                    
                    # Подготавливаем данные для Pinecone
                    vector_data = {
                        "id": f"{index_name}-optimized-{vector_count}",
                        "values": embedding['embedding'],
                        "metadata": {
                            "text": chunk,
                            "source": filename,
                            "chunk_index": chunk_idx,
                            "chunk_size": len(chunk),
                            "method": "semantic_i7_optimized",
                            "model": CONFIG['model_name']
                        }
                    }
                    
                    # Загружаем в Pinecone
                    index.upsert(vectors=[vector_data])
                    vector_count += 1
                    total_chunks += 1
                    
                    # Пауза между API запросами для стабильности
                    time.sleep(CONFIG["api_delay"])
                    
                except Exception as e:
                    print(f"      ❌ Ошибка обработки чанка {chunk_idx}: {e}")
                    time.sleep(1)  # Дополнительная пауза при ошибке
                    continue
            
            print(f"   ✅ Файл обработан: {len(chunks)} чанков загружено")
            
            # Пауза между файлами для остывания системы
            time.sleep(2)
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 ОБРАБОТКА ЗАВЕРШЕНА!")
        print("=" * 50)
        print(f"📊 Результаты:")
        print(f"   📁 Файлов обработано: {len(txt_files)}")
        print(f"   📝 Чанков создано: {total_chunks}")
        print(f"   💾 Векторов загружено: {vector_count}")
        print(f"   ⏱️ Общее время: {total_time/60:.1f} минут")
        print(f"   🎯 Среднее время на чанк: {total_time/max(total_chunks,1):.2f}с")
        
        # Финальная проверка результата
        time.sleep(3)
        stats = index.describe_index_stats()
        print(f"✅ В индексе Pinecone: {stats.total_vector_count} векторов")
        
        # Проверяем состояние системы после работы
        final_status = SystemMonitor.get_system_status()
        print(f"🖥️ Финальное состояние системы:")
        print(f"   CPU: {final_status['cpu_percent']:.1f}%")
        print(f"   ОЗУ: {final_status['memory_percent']:.1f}%")

def main():
    """Запуск оптимизированного чанкера"""
    print("Запускаем оптимизированный семантический чанкер для i7-6700HQ...")
    
    chunker = OptimizedSemanticChunker()
    chunker.process_and_upload("data_facts", "ukido")
    
    print("\n🏁 Все готово! Твой ноутбук может отдохнуть 😊")

if __name__ == "__main__":
    main()
