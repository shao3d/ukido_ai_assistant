"""
Кастомный экстрактор метаданных для LlamaIndex
Анализирует содержимое node.text и добавляет дополнительные метаданные
для улучшения качества поиска в RAG системе.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Sequence
from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import BaseNode

# Настройка логирования
logger = logging.getLogger(__name__)

class CustomMetadataExtractor(BaseExtractor):
    """
    Кастомный экстрактор метаданных для анализа содержимого узлов
    и добавления структурированных метаданных.
    """
    
    # Определяем поля класса для сериализации
    courses: Dict[str, List[str]] = {}
    content_patterns: Dict[str, List[str]] = {}
    pricing_patterns: List[str] = []
    teacher_patterns: List[str] = []
    faq_patterns: List[str] = []
    teacher_names: List[str] = []
    
    def __init__(self):
        """Инициализация экстрактора"""
        super().__init__()
        
        # Определяем курсы для поиска (полные названия и ключевые слова)
        self.courses = {
            "Капитан Проектов": ["Капитан Проектов", "Капитан", "капитан проектов", "капитан"],
            "Юный Оратор": ["Юный Оратор", "Оратор", "юный оратор", "оратор"],
            "Эмоциональный Компас": ["Эмоциональный Компас", "Компас", "эмоциональный компас", "компас"]
        }
        
        # Паттерны для определения типов контента
        self.content_patterns = {
            "pricing": [
                r"грн",
                r"оплат",
                r"стоимост",
                r"цен",
                r"скидк",
                r"тариф",
                r"руб",
                r"₴",
                r"₽"
            ],
            "teachers": [
                r"КВАЛИФИКАЦИЯ",
                r"ОПЫТ РАБОТЫ",
                r"ОБРАЗОВАНИЕ:",
                r"преподаватель",
                r"тренер",
                r"ментор"
            ],
            "faq": [
                r"Q:",
                r"A:",
                r"Вопрос:",
                r"Ответ:",
                r"Часто задаваемые",
                r"FAQ"
            ],
            "schedule": [
                r"расписание",
                r"время",
                r"дата",
                r"занятие",
                r"урок",
                r"час"
            ],
            "course_description": [
                r"программа",
                r"курс",
                r"обучение",
                r"модуль",
                r"урок"
            ]
        }
        
        # Паттерны для поиска ценовой информации
        self.pricing_patterns = [
            r"грн",
            r"оплат",
            r"стоимост",
            r"цен",
            r"скидк"
        ]
        
        # Паттерны для определения информации о преподавателях
        self.teacher_patterns = [
            r"КВАЛИФИКАЦИЯ",
            r"ОПЫТ РАБОТЫ",
            r"ОБРАЗОВАНИЕ:",
            r"преподаватель",
            r"тренер",
            r"ментор"
        ]
        
        # Имена преподавателей для поиска
        self.teacher_names = [
            "Анна Коваленко",
            "Дмитрий Петров", 
            "Елена Сидорова"
        ]
        
        # Паттерны для определения FAQ
        self.faq_patterns = [
            r"Q:",
            r"A:",
            r"Вопрос:",
            r"Ответ:"
        ]
        
        logger.info("✅ CustomMetadataExtractor инициализирован")
    
    def _determine_content_type(self, text: str) -> str:
        """
        Определяет тип контента на основе анализа текста
        """
        text_lower = text.lower()
        
        # Проверяем каждый тип контента
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return content_type
        
        # Если не найдено специфичных паттернов, возвращаем общий тип
        return "general"
    
    def _has_pricing_info(self, text: str) -> bool:
        """
        Проверяет наличие ценовой информации в тексте
        """
        text_lower = text.lower()
        
        for pattern in self.pricing_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _find_mentioned_courses(self, text: str) -> List[str]:
        """
        Находит упоминания курсов в тексте с учетом кавычек и частичных совпадений
        """
        mentioned_courses = []
        
        for course_name, search_variants in self.courses.items():
            found = False
            
            # Проверяем каждый вариант названия курса
            for variant in search_variants:
                # Создаем паттерны для поиска с кавычками и без
                patterns = [
                    rf'["\']?{re.escape(variant)}["\']?',  # С кавычками или без
                    rf'{re.escape(variant)}',              # Точное совпадение
                ]
                
                # Проверяем каждый паттерн
                for pattern in patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        found = True
                        break
                
                if found:
                    break
            
            if found and course_name not in mentioned_courses:
                mentioned_courses.append(course_name)
        
        return mentioned_courses
    
    def _is_teacher_info(self, text: str) -> bool:
        """
        Проверяет, содержит ли текст информацию о преподавателях
        """
        # Проверяем паттерны
        for pattern in self.teacher_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Проверяем имена преподавателей
        for teacher_name in self.teacher_names:
            if re.search(re.escape(teacher_name), text, re.IGNORECASE):
                return True
        
        return False
    
    def _is_faq(self, text: str) -> bool:
        """
        Проверяет, является ли текст FAQ
        """
        for pattern in self.faq_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _extract_metadata_from_text(self, text: str) -> Dict[str, Any]:
        """
        Извлекает все метаданные из текста узла
        """
        metadata = {}
        
        # Определяем тип контента
        metadata["content_type"] = self._determine_content_type(text)
        
        # Проверяем наличие ценовой информации
        metadata["has_pricing"] = self._has_pricing_info(text)
        
        # Находим упоминания курсов
        metadata["course_mentioned"] = self._find_mentioned_courses(text)
        
        # Проверяем, является ли это информацией о преподавателях
        metadata["is_teacher_info"] = self._is_teacher_info(text)
        
        # Проверяем, является ли это FAQ
        metadata["is_faq"] = self._is_faq(text)
        
        # Дополнительные метаданные
        metadata["text_length"] = len(text)
        metadata["has_courses"] = len(metadata["course_mentioned"]) > 0
        
        return metadata
    
    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict[str, Any]]:
        """
        Основной метод для извлечения метаданных из списка узлов
        
        Args:
            nodes: Список узлов для обработки
            
        Returns:
            Список словарей с метаданными для каждого узла
        """
        metadata_list = []
        
        logger.info(f"🔍 Начинаю извлечение кастомных метаданных из {len(nodes)} узлов")
        
        for i, node in enumerate(nodes):
            try:
                # Получаем текст узла
                text = getattr(node, 'text', '')
                
                if not text:
                    logger.warning(f"⚠️ Узел {i} не содержит текста")
                    metadata_list.append({})
                    continue
                
                # Извлекаем метаданные
                extracted_metadata = self._extract_metadata_from_text(text)
                
                # Добавляем метаданные к существующим метаданным узла
                if hasattr(node, 'metadata') and node.metadata:
                    # Объединяем существующие метаданные с новыми
                    node.metadata.update(extracted_metadata)
                else:
                    # Создаем новые метаданные
                    node.metadata = extracted_metadata
                
                metadata_list.append(extracted_metadata)
                
                # Логируем результат для первых нескольких узлов
                if i < 3:
                    logger.info(f"✅ Узел {i}: {extracted_metadata}")
                
            except Exception as e:
                logger.error(f"❌ Ошибка при обработке узла {i}: {e}")
                metadata_list.append({})
        
        logger.info(f"🎉 Завершено извлечение метаданных для {len(nodes)} узлов")
        
        # Статистика по типам контента
        content_types = {}
        for metadata in metadata_list:
            content_type = metadata.get("content_type", "unknown")
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        logger.info(f"📊 Статистика по типам контента: {content_types}")
        
        return metadata_list
    
    async def aextract(self, nodes: Sequence[BaseNode]) -> List[Dict[str, Any]]:
        """Асинхронная версия extract - просто вызывает синхронную версию"""
        return self.extract(nodes)

# Пример использования
def create_custom_metadata_extractor():
    """
    Фабричная функция для создания экстрактора
    """
    return CustomMetadataExtractor()

# Для тестирования
if __name__ == "__main__":
    # Создаем тестовый экстрактор
    extractor = CustomMetadataExtractor()
    
    # Тестовые тексты
    test_texts = [
        "Курс Капитан Проектов стоит 2500 грн. Скидка 20% для ранних регистраций.",
        "КВАЛИФИКАЦИЯ: Преподаватель имеет 10 лет опыта. ОБРАЗОВАНИЕ: Магистр психологии.",
        "Q: Сколько длится курс Юный Оратор? A: Курс длится 8 недель.",
        "Программа курса Эмоциональный Компас включает 12 модулей обучения."
    ]
    
    # Создаем псевдо-узлы для тестирования
    class TestNode:
        def __init__(self, text):
            self.text = text
            self.metadata = {}
    
    test_nodes = [TestNode(text) for text in test_texts]
    
    # Извлекаем метаданные
    results = extractor.extract(test_nodes)
    
    # Выводим результаты
    print("\n🧪 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    for i, (text, metadata) in enumerate(zip(test_texts, results)):
        print(f"\n--- Тест {i+1} ---")
        print(f"Текст: {text[:50]}...")
        print(f"Метаданные: {metadata}")