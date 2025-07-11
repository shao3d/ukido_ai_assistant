#!/usr/bin/env python3
"""
Скрипт для создания резервной копии всех векторов из Pinecone индекса.
Сохраняет все векторы с их метаданными в JSON файл с timestamp в названии.
"""

import json
import logging
import os
import pinecone
from datetime import datetime
from typing import Dict, List, Any

try:
    from config import config
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import config

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Константы
PINECONE_INDEX_NAME = "ukido"
BACKUP_DIR = "backups"
BATCH_SIZE = 100  # Размер батча для выгрузки векторов

def ensure_backup_directory():
    """Создает директорию для бэкапов если её нет."""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        logger.info(f"✅ Создана директория для бэкапов: {BACKUP_DIR}")

def generate_backup_filename() -> str:
    """Генерирует имя файла бэкапа с timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pinecone_backup_{PINECONE_INDEX_NAME}_{timestamp}.json"
    return os.path.join(BACKUP_DIR, filename)

def get_all_vector_ids(index) -> List[str]:
    """
    Получает список всех ID векторов в индексе.
    Использует list_paginated для получения всех векторов.
    """
    logger.info("🔍 Получение списка всех векторов...")
    
    all_ids = []
    pagination_token = None
    
    while True:
        try:
            if pagination_token:
                response = index.list_paginated(pagination_token=pagination_token)
            else:
                response = index.list_paginated()
            
            # Получаем список ID из ответа
            vector_ids = response.get('vectors', [])
            if vector_ids:
                # Извлекаем ID из векторов
                ids = [vector['id'] for vector in vector_ids]
                all_ids.extend(ids)
                logger.info(f"📦 Получено {len(ids)} ID векторов (всего: {len(all_ids)})")
            
            # Проверяем, есть ли еще страницы
            pagination_token = response.get('pagination', {}).get('next')
            if not pagination_token:
                break
                
        except Exception as e:
            logger.error(f"❌ Ошибка при получении списка векторов: {e}")
            # Пробуем альтернативный метод через describe_index_stats
            stats = index.describe_index_stats()
            total_count = stats.get('total_vector_count', 0)
            logger.warning(f"⚠️ Переход к альтернативному методу. Ожидается ~{total_count} векторов")
            
            # Генерируем ID на основе известного паттерна (если он есть)
            # Для этого нужно знать как генерируются ID в вашей системе
            break
    
    logger.info(f"✅ Найдено {len(all_ids)} векторов в индексе")
    return all_ids

def fetch_vectors_batch(index, vector_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Получает векторы батчами с их метаданными.
    """
    vectors_data = []
    
    for i in range(0, len(vector_ids), BATCH_SIZE):
        batch_ids = vector_ids[i:i + BATCH_SIZE]
        
        try:
            logger.info(f"📥 Загрузка батча {i//BATCH_SIZE + 1}/{(len(vector_ids) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch_ids)} векторов)")
            
            # Получаем векторы с метаданными
            response = index.fetch(ids=batch_ids)
            
            # Обрабатываем ответ
            for vector_id, vector_data in response.get('vectors', {}).items():
                vector_info = {
                    'id': vector_id,
                    'values': vector_data.get('values', []),
                    'metadata': vector_data.get('metadata', {}),
                    'sparse_values': vector_data.get('sparse_values', {})
                }
                vectors_data.append(vector_info)
            
            logger.info(f"✅ Загружен батч: {len(response.get('vectors', {}))} векторов")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке батча {i//BATCH_SIZE + 1}: {e}")
            continue
    
    return vectors_data

def save_backup_to_json(vectors_data: List[Dict[str, Any]], filename: str):
    """
    Сохраняет данные векторов в JSON файл.
    """
    logger.info(f"💾 Сохранение бэкапа в файл: {filename}")
    
    backup_data = {
        'backup_info': {
            'timestamp': datetime.now().isoformat(),
            'index_name': PINECONE_INDEX_NAME,
            'total_vectors': len(vectors_data),
            'created_by': 'backup_pinecone.py'
        },
        'vectors': vectors_data
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2)
        
        file_size = os.path.getsize(filename) / (1024 * 1024)  # Размер в MB
        logger.info(f"✅ Бэкап сохранен: {filename} ({file_size:.2f} MB)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении файла: {e}")
        return False

def verify_backup(filename: str, expected_count: int) -> bool:
    """
    Проверяет целостность созданного бэкапа.
    """
    logger.info(f"🔍 Проверка целостности бэкапа...")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        saved_count = len(backup_data.get('vectors', []))
        backup_info = backup_data.get('backup_info', {})
        
        logger.info(f"📊 Статистика бэкапа:")
        logger.info(f"  • Индекс: {backup_info.get('index_name', 'N/A')}")
        logger.info(f"  • Время создания: {backup_info.get('timestamp', 'N/A')}")
        logger.info(f"  • Ожидалось векторов: {expected_count}")
        logger.info(f"  • Сохранено векторов: {saved_count}")
        
        if saved_count == expected_count:
            logger.info("✅ Бэкап прошел проверку целостности")
            return True
        else:
            logger.warning(f"⚠️ Несоответствие количества векторов: ожидалось {expected_count}, сохранено {saved_count}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка при проверке бэкапа: {e}")
        return False

def main():
    """Основная функция для создания бэкапа."""
    logger.info("🚀 Запуск создания резервной копии Pinecone индекса...")
    
    try:
        # Создаем директорию для бэкапов
        ensure_backup_directory()
        
        # Подключаемся к Pinecone
        logger.info("🔌 Подключение к Pinecone...")
        pc = pinecone.Pinecone(api_key=config.PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Получаем статистику индекса
        stats = index.describe_index_stats()
        total_vectors = stats.get('total_vector_count', 0)
        logger.info(f"📊 Статистика индекса '{PINECONE_INDEX_NAME}':")
        logger.info(f"  • Всего векторов: {total_vectors}")
        logger.info(f"  • Размерность: {stats.get('dimension', 'N/A')}")
        
        if total_vectors == 0:
            logger.warning("⚠️ Индекс пустой, нечего бэкапить")
            return
        
        # Получаем список всех векторов
        vector_ids = get_all_vector_ids(index)
        
        if not vector_ids:
            logger.error("❌ Не удалось получить список векторов")
            return
        
        # Загружаем все векторы
        logger.info(f"📥 Начало загрузки {len(vector_ids)} векторов...")
        vectors_data = fetch_vectors_batch(index, vector_ids)
        
        if not vectors_data:
            logger.error("❌ Не удалось загрузить векторы")
            return
        
        # Генерируем имя файла и сохраняем
        backup_filename = generate_backup_filename()
        success = save_backup_to_json(vectors_data, backup_filename)
        
        if success:
            # Проверяем целостность
            verify_backup(backup_filename, len(vector_ids))
            logger.info(f"🎉 Бэкап успешно создан: {backup_filename}")
        else:
            logger.error("❌ Не удалось сохранить бэкап")
            
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        raise

if __name__ == "__main__":
    main()