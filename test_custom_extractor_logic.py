#!/usr/bin/env python3
"""
Тест логики custom_metadata_extractor без зависимостей LlamaIndex
"""

from extract_metadata import extract_metadata

def test_metadata_mapping():
    """Тестируем маппинг полей из extract_metadata на legacy поля"""
    
    test_text = "Курс Капитан Проектов стоит 6000 грн в месяц. Есть семейная скидка 15%."
    
    # Получаем метаданные из основной функции
    extracted_metadata = extract_metadata(test_text)
    
    print("🔍 Извлеченные метаданные:")
    print(f"  has_pricing: {extracted_metadata.get('has_pricing')}")
    print(f"  courses_offered: {extracted_metadata.get('courses_offered')}")
    print(f"  content_category: {extracted_metadata.get('content_category')}")
    
    # Создаем legacy маппинг как в custom_metadata_extractor.py
    legacy_metadata = {}
    legacy_metadata["content_type"] = extracted_metadata.get("content_category", "general")
    legacy_metadata["has_pricing"] = extracted_metadata.get("has_pricing", False)
    legacy_metadata["course_mentioned"] = extracted_metadata.get("courses_offered", [])
    legacy_metadata["is_teacher_info"] = False
    legacy_metadata["is_faq"] = extracted_metadata.get("content_category") == "FAQ"
    legacy_metadata["text_length"] = len(test_text)
    legacy_metadata["has_courses"] = len(legacy_metadata["course_mentioned"]) > 0
    
    print("\n🔄 Legacy маппинг:")
    print(f"  content_type: {legacy_metadata['content_type']}")
    print(f"  has_pricing: {legacy_metadata['has_pricing']}")
    print(f"  course_mentioned: {legacy_metadata['course_mentioned']}")
    print(f"  has_courses: {legacy_metadata['has_courses']}")
    
    # Объединяем метаданные
    combined_metadata = {**extracted_metadata, **legacy_metadata}
    
    print(f"\n✅ Комбинированные метаданные содержат {len(combined_metadata)} полей")
    print(f"  Поле 'courses_offered' существует: {'courses_offered' in combined_metadata}")
    print(f"  Поле 'course_mentioned' существует: {'course_mentioned' in combined_metadata}")
    print(f"  Значения совпадают: {combined_metadata.get('courses_offered') == combined_metadata.get('course_mentioned')}")

if __name__ == "__main__":
    test_metadata_mapping()