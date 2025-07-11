from extract_metadata import extract_metadata

# Читаем начало faq.md
with open('data_facts/faq.md', 'r', encoding='utf-8') as f:
    faq_content = f.read()[:2000]  # Первые 2000 символов

print("ОТЛАДКА FAQ.MD:")
print("=" * 60)
print("Текст для анализа:")
print(faq_content[:500] + "...")
print("=" * 60)

metadata = extract_metadata(faq_content)
print(f"\nРезультат:")
print(f"courses_offered: {metadata.get('courses_offered', [])}")