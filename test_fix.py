from extract_metadata import extract_metadata

test_texts = [
    'Курс "Юный Оратор" для детей 7-10 лет',
    'Записаться на Юный Оратор можно онлайн',
    'курс Эмоциональный Компас поможет ребенку',
    'КАПИТАН ПРОЕКТОВ - лучший выбор',
    'Курс «Эмоциональный Компас» с кавычками елочками'
]

print("ТЕСТ ИСПРАВЛЕНИЯ:")
for text in test_texts:
    metadata = extract_metadata(text)
    courses = metadata.get('courses_offered', [])
    print(f"Текст: {text}")
    print(f"Найдены курсы: {courses}")
    print("-" * 50)