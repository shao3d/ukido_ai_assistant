from extract_metadata import extract_metadata

# Тестовые тексты
test_texts = [
    "Курс Капитан Проектов стоит 6000 грн в месяц. Есть семейная скидка 15%.",
    "Мы работаем с детьми с СДВГ, аутизмом, диабетом. Адаптируем программу.",
    "Капитан Проектов - курс для детей 11-14 лет, которые любят технологии."
]

print("🧪 ТЕСТИРУЕМ EXTRACT_METADATA:\n")

for i, text in enumerate(test_texts, 1):
    print(f"Тест {i}: {text[:50]}...")
    metadata = extract_metadata(text)
    print(f"  has_pricing: {metadata.get('has_pricing')}")
    print(f"  has_special_needs_info: {metadata.get('has_special_needs_info')}")
    print(f"  courses_offered: {metadata.get('courses_offered')}")
    print(f"  prices_mentioned: {metadata.get('prices_mentioned')}")
    print("-" * 50)