# diagnose_metadata.py
"""
Диагностика: почему метаданные не извлекаются из реальных файлов
"""
import os
from extract_metadata import extract_metadata

# Читаем реальные файлы из data_facts
data_dir = "data_facts"

print("🔍 ДИАГНОСТИКА ИЗВЛЕЧЕНИЯ МЕТАДАННЫХ ИЗ РЕАЛЬНЫХ ФАЙЛОВ\n")

# Файлы которые точно должны содержать нужную информацию
test_files = [
    "pricing.md",           # Должна быть информация о ценах
    "faq.md",               # Могут быть и цены, и курсы  
    "teachers_team.md",     # Информация о преподавателях
    "conditions.md",        # Может быть инфо об особых потребностях
    "mission_values_history.md"  # Общая информация о школе
]

for filename in test_files:
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"❌ Файл не найден: {filename}")
        continue
        
    print(f"\n📄 ФАЙЛ: {filename}")
    print("-" * 60)
    
    # Читаем первые 1000 символов файла
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        preview = content[:500] + "..." if len(content) > 500 else content
        
    print(f"Превью: {preview}\n")
    
    # Тестируем на разных частях файла
    # Начало файла
    chunk1 = content[:1000]
    metadata1 = extract_metadata(chunk1)
    
    # Середина файла  
    mid = len(content) // 2
    chunk2 = content[mid-500:mid+500]
    metadata2 = extract_metadata(chunk2)
    
    print("Метаданные из НАЧАЛА файла:")
    print(f"  has_pricing: {metadata1.get('has_pricing')}")
    print(f"  prices_mentioned: {metadata1.get('prices_mentioned')}")
    print(f"  courses_offered: {metadata1.get('courses_offered')}")
    print(f"  has_special_needs_info: {metadata1.get('has_special_needs_info')}")
    
    print("\nМетаданные из СЕРЕДИНЫ файла:")
    print(f"  has_pricing: {metadata2.get('has_pricing')}")
    print(f"  prices_mentioned: {metadata2.get('prices_mentioned')}")  
    print(f"  courses_offered: {metadata2.get('courses_offered')}")
    print(f"  has_special_needs_info: {metadata2.get('has_special_needs_info')}")
    
    # Ищем ключевые слова вручную для проверки
    print(f"\n🔎 Ручная проверка:")
    if "грн" in content or "₴" in content or "цен" in content.lower():
        print("  ✓ Найдены слова о ценах")
    if "Капитан Проектов" in content or "Юный Оратор" in content or "Эмоциональный Компас" in content:
        print("  ✓ Найдены названия курсов")
    if "СДВГ" in content or "аутизм" in content or "диабет" in content:
        print("  ✓ Найдена информация об особых потребностях")
