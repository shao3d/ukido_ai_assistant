"""
Анализ содержимого Pinecone индекса "ukido"
Этот скрипт подключается к Pinecone и анализирует структуру данных в индексе.
"""

import os
import json
import random
import logging
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional
from pinecone import Pinecone
from dotenv import load_dotenv
from config import config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PineconeAnalyzer:
    """
    Класс для анализа содержимого Pinecone индекса
    """
    
    def __init__(self):
        """Инициализация подключения к Pinecone"""
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = self.pc.Index(host=config.PINECONE_HOST_FACTS)
        self.index_name = "ukido"
        self.analysis_results = {}
        
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Получение общей статистики индекса
        """
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"📊 Получена статистика индекса: {stats}")
            return stats
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {}
    
    def get_random_vectors(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Получение случайных векторов для анализа метаданных
        """
        try:
            # Используем query с случайным вектором для получения случайных результатов
            dummy_vector = [random.random() for _ in range(768)]  # Размерность 768 для Gemini
            
            results = self.index.query(
                vector=dummy_vector,
                top_k=count,
                include_metadata=True
            )
            
            logger.info(f"🎲 Получено {len(results['matches'])} случайных векторов")
            return results['matches']
        except Exception as e:
            logger.error(f"❌ Ошибка получения случайных векторов: {e}")
            return []
    
    def analyze_metadata_structure(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Анализ структуры метаданных векторов
        """
        metadata_fields = defaultdict(int)
        field_types = defaultdict(set)
        field_samples = defaultdict(list)
        
        for vector in vectors:
            metadata = vector.get('metadata', {})
            
            # Подсчет полей
            for field, value in metadata.items():
                metadata_fields[field] += 1
                field_types[field].add(type(value).__name__)
                
                # Сохраняем примеры значений (первые 3)
                if len(field_samples[field]) < 3:
                    field_samples[field].append(str(value)[:100])  # Ограничиваем длину
        
        # Преобразуем sets в lists для JSON сериализации
        field_types_json = {field: list(types) for field, types in field_types.items()}
        
        analysis = {
            'total_vectors_analyzed': len(vectors),
            'metadata_fields': dict(metadata_fields),
            'field_types': field_types_json,
            'field_samples': dict(field_samples)
        }
        
        logger.info(f"🔍 Найдено {len(metadata_fields)} различных полей метаданных")
        return analysis
    
    def analyze_content_patterns(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Анализ паттернов в содержимом
        """
        content_lengths = []
        sources = Counter()
        chunk_types = Counter()
        
        for vector in vectors:
            metadata = vector.get('metadata', {})
            
            # Анализ длины контента
            content = metadata.get('content', '')
            if content:
                content_lengths.append(len(content))
            
            # Анализ источников
            source = metadata.get('source', 'Unknown')
            sources[source] += 1
            
            # Анализ типов чанков
            chunk_type = metadata.get('chunk_type', 'Unknown')
            chunk_types[chunk_type] += 1
        
        analysis = {
            'content_length_stats': {
                'min': min(content_lengths) if content_lengths else 0,
                'max': max(content_lengths) if content_lengths else 0,
                'avg': sum(content_lengths) / len(content_lengths) if content_lengths else 0
            },
            'top_sources': dict(sources.most_common(10)),
            'chunk_types': dict(chunk_types)
        }
        
        return analysis
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Запуск полного анализа индекса
        """
        logger.info("🚀 Начинаю полный анализ Pinecone индекса")
        
        # 1. Получение статистики индекса
        logger.info("1️⃣ Получение общей статистики...")
        index_stats = self.get_index_stats()
        
        # 2. Получение случайных векторов
        logger.info("2️⃣ Получение случайных векторов...")
        random_vectors = self.get_random_vectors(10)
        
        # 3. Анализ метаданных
        logger.info("3️⃣ Анализ структуры метаданных...")
        metadata_analysis = self.analyze_metadata_structure(random_vectors)
        
        # 4. Анализ содержимого
        logger.info("4️⃣ Анализ паттернов содержимого...")
        content_analysis = self.analyze_content_patterns(random_vectors)
        
        # Составление итогового отчета
        self.analysis_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'index_name': self.index_name,
            'index_statistics': index_stats,
            'metadata_analysis': metadata_analysis,
            'content_analysis': content_analysis,
            'sample_vectors': [
                {
                    'id': vector['id'],
                    'score': vector.get('score', 0),
                    'metadata_keys': list(vector.get('metadata', {}).keys())
                }
                for vector in random_vectors[:5]  # Показываем только первые 5
            ]
        }
        
        logger.info("✅ Анализ завершен успешно")
        return self.analysis_results
    
    def save_report(self, filename: str = 'index_analysis.txt') -> None:
        """
        Сохранение красивого отчета в файл
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("🔍 АНАЛИЗ PINECONE ИНДЕКСА 'UKIDO'\n")
                f.write("=" * 80 + "\n\n")
                
                # Время анализа
                f.write(f"📅 Время анализа: {self.analysis_results['analysis_timestamp']}\n\n")
                
                # Общая статистика
                f.write("📊 ОБЩАЯ СТАТИСТИКА ИНДЕКСА\n")
                f.write("-" * 40 + "\n")
                stats = self.analysis_results['index_statistics']
                if stats:
                    f.write(f"• Всего векторов: {stats.get('total_vector_count', 'N/A')}\n")
                    f.write(f"• Размерность: {stats.get('dimension', 'N/A')}\n")
                    f.write(f"• Заполненность: {stats.get('index_fullness', 'N/A')}\n")
                    
                    # Информация о неймспейсах
                    namespaces = stats.get('namespaces', {})
                    if namespaces:
                        f.write(f"• Количество неймспейсов: {len(namespaces)}\n")
                        for ns_name, ns_data in namespaces.items():
                            f.write(f"  - {ns_name}: {ns_data.get('vector_count', 0)} векторов\n")
                f.write("\n")
                
                # Анализ метаданных
                f.write("🏷️ АНАЛИЗ МЕТАДАННЫХ\n")
                f.write("-" * 40 + "\n")
                metadata = self.analysis_results['metadata_analysis']
                f.write(f"• Проанализировано векторов: {metadata['total_vectors_analyzed']}\n")
                f.write(f"• Найдено полей метаданных: {len(metadata['metadata_fields'])}\n\n")
                
                f.write("📋 Поля метаданных:\n")
                for field, count in metadata['metadata_fields'].items():
                    types = ", ".join(metadata['field_types'][field])
                    f.write(f"  • {field}: {count} векторов (тип: {types})\n")
                    
                    # Примеры значений
                    samples = metadata['field_samples'].get(field, [])
                    if samples:
                        f.write(f"    Примеры: {', '.join(samples[:2])}\n")
                f.write("\n")
                
                # Анализ содержимого
                f.write("📝 АНАЛИЗ СОДЕРЖИМОГО\n")
                f.write("-" * 40 + "\n")
                content = self.analysis_results['content_analysis']
                
                # Статистика длины контента
                length_stats = content['content_length_stats']
                f.write(f"• Длина контента:\n")
                f.write(f"  - Минимум: {length_stats['min']} символов\n")
                f.write(f"  - Максимум: {length_stats['max']} символов\n")
                f.write(f"  - Среднее: {length_stats['avg']:.1f} символов\n\n")
                
                # Топ источников
                f.write("📚 Топ источников данных:\n")
                for source, count in content['top_sources'].items():
                    f.write(f"  • {source}: {count} векторов\n")
                f.write("\n")
                
                # Типы чанков
                f.write("🧩 Типы чанков:\n")
                for chunk_type, count in content['chunk_types'].items():
                    f.write(f"  • {chunk_type}: {count} векторов\n")
                f.write("\n")
                
                # Примеры векторов
                f.write("🎯 ПРИМЕРЫ ВЕКТОРОВ\n")
                f.write("-" * 40 + "\n")
                for i, vector in enumerate(self.analysis_results['sample_vectors'], 1):
                    f.write(f"{i}. ID: {vector['id']}\n")
                    f.write(f"   Score: {vector['score']:.4f}\n")
                    f.write(f"   Поля метаданных: {', '.join(vector['metadata_keys'])}\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("✅ Анализ завершен. Отчет создан автоматически.\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"📄 Отчет сохранен в файл: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения отчета: {e}")

def main():
    """
    Основная функция для запуска анализа
    """
    print("🚀 Запуск анализа Pinecone индекса 'ukido'...")
    
    try:
        # Создаем анализатор
        analyzer = PineconeAnalyzer()
        
        # Запускаем анализ
        results = analyzer.run_full_analysis()
        
        # Сохраняем отчет
        analyzer.save_report()
        
        print("\n✅ Анализ завершен успешно!")
        print("📄 Отчет сохранен в файл 'index_analysis.txt'")
        
        # Краткая сводка
        stats = results['index_statistics']
        if stats:
            print(f"\n📊 Краткая сводка:")
            print(f"   • Всего векторов: {stats.get('total_vector_count', 'N/A')}")
            print(f"   • Размерность: {stats.get('dimension', 'N/A')}")
            print(f"   • Полей метаданных: {len(results['metadata_analysis']['metadata_fields'])}")
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        print(f"\n❌ Произошла ошибка: {e}")

if __name__ == "__main__":
    main()