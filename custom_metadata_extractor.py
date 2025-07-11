"""
Кастомный экстрактор метаданных для LlamaIndex
Анализирует содержимое node.text и добавляет дополнительные метаданные
для улучшения качества поиска в RAG системе.
Использует оптимизированную функцию extract_metadata() с топ-10 полями.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Sequence
from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import BaseNode
from extract_metadata import extract_metadata

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
        
        logger.info("✅ CustomMetadataExtractor инициализирован с новой функцией extract_metadata()")
    
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
                # Простая проверка - есть ли вариант в тексте (без учета регистра)
                if variant.lower() in text.lower():
                    found = True
                    break
                    
                # Дополнительная проверка с кавычками
                if f'"{variant}"' in text or f"'{variant}'" in text:
                    found = True
                    break
                    
                # Проверка с экранированными кавычками
                if f'«{variant}»' in text:
                    found = True
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
        Извлекает все метаданные из текста узла используя новую функцию extract_metadata()
        с топ-10 полями для решения проблемных запросов
        """
        # Используем новую оптимизированную функцию
        extracted_metadata = extract_metadata(text)
        
        # Добавляем совместимость со старыми полями для обратной совместимости
        legacy_metadata = {}
        
        # Маппинг новых полей на старые для совместимости
        legacy_metadata["content_type"] = extracted_metadata.get("content_category", "general")
        legacy_metadata["has_pricing"] = extracted_metadata.get("has_pricing", False)
        legacy_metadata["course_mentioned"] = extracted_metadata.get("courses_offered", [])
        legacy_metadata["is_teacher_info"] = False  # Убираем ссылку на несуществующее поле
        legacy_metadata["is_faq"] = extracted_metadata.get("content_category") == "FAQ"
        legacy_metadata["text_length"] = len(text)
        legacy_metadata["has_courses"] = len(legacy_metadata["course_mentioned"]) > 0
        
        # Объединяем новые и старые метаданные
        combined_metadata = {**extracted_metadata, **legacy_metadata}
        
        return combined_metadata
    
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
                
                # Проверяем курсы в существующих метаданных (например, в вопросах)
                existing_metadata = getattr(node, 'metadata', {})
                if existing_metadata:
                    # Ищем курсы в других полях метаданных
                    for key, value in existing_metadata.items():
                        if isinstance(value, (str, list)):
                            # Если это список, объединяем в строку
                            search_text = ' '.join(value) if isinstance(value, list) else value
                            # Находим курсы в этом тексте
                            additional_courses = self._find_mentioned_courses(search_text)
                            # Добавляем найденные курсы к основным
                            for course in additional_courses:
                                if course not in extracted_metadata["course_mentioned"]:
                                    extracted_metadata["course_mentioned"].append(course)
                    
                    # Обновляем has_courses после добавления курсов
                    extracted_metadata["has_courses"] = len(extracted_metadata["course_mentioned"]) > 0
                
                # Добавляем метаданные к существующим метаданным узла
                if hasattr(node, 'metadata') and node.metadata:
                    # Объединяем существующие метаданные с новыми
                    node.metadata.update(extracted_metadata)
                else:
                    # Создаем новые метаданные
                    node.metadata = extracted_metadata
                
                metadata_list.append(extracted_metadata)
                
                # Логируем результат для первых нескольких узлов (только ключевые поля)
                if i < 3:
                    key_fields = {
                        "content_category": extracted_metadata.get("content_category"),
                        "pricing_info": extracted_metadata.get("has_pricing", False),
                        "special_needs": extracted_metadata.get("has_special_needs_info", False),
                        "courses": extracted_metadata.get("courses_offered", []),
                        "age_groups": extracted_metadata.get("age_groups_mentioned", [])
                    }
                    logger.info(f"✅ Узел {i}: {key_fields}")
                
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
    
    # Тестовые тексты для проверки новой функциональности
    test_texts = [
        "Курс Капитан Проектов стоит 6000 грн. Семейная скидка 15% для двух детей.",
        "Для детей с СДВГ мы используем короткие блоки 5-7 минут и визуальные подсказки.",
        "Сын увлекается программированием - рекомендуем курс Капитан Проектов для возраста 11-14 лет.",
        "Анна Коваленко ведет курс Юный Оратор для детей 7-10 лет. Опыт 8 лет."
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