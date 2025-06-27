import os
import google.generativeai as genai
from pinecone import Pinecone
import time
from dotenv import load_dotenv

# --- НАСТРОЙКИ И КЛЮЧИ ---
# Загружаем переменные окружения из .env
load_dotenv()

# Ключ для Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Данные для Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST_FACTS")  # или другой переменной, если нужно
INDEX_NAME = "ukido" # Имя нашего индекса
# --- КОНЕЦ НАСТРОЕК ---


# 1. Создаем вектор с помощью Gemini
print("1. Создаю тестовый вектор с помощью Gemini...")
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Та же модель, что создает векторы размерностью 768
    embedding_model = 'models/text-embedding-004' 
    text_to_embed = "Это тестовое предложение для проверки Pinecone"
    
    # Создаем вектор (эмбеддинг)
    embedding = genai.embed_content(model=embedding_model, content=text_to_embed)
    test_vector = embedding['embedding']
    print("   ... вектор успешно создан!")

except Exception as e:
    print(f"   !!! Ошибка при создании вектора: {e}")
    exit() # Выходим из скрипта, если вектор не создался

# --- НАША НОВАЯ ПАУЗА ---
print("\nДаю Pinecone 10 секунд на индексацию...")
time.sleep(10) # <--- ДОБАВЛЕНА ПАУЗА В 10 СЕКУНД
# ------------------------

# 2. Подключаемся к Pinecone и сохраняем вектор
print("\n2. Подключаюсь к Pinecone и сохраняю вектор...")
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    # Подключаемся к нашему конкретному индексу по его адресу
    host = PINECONE_HOST 
    index = pc.Index(host=host)

    # Сохраняем наш вектор под ID 'vec1'
    index.upsert(vectors=[{'id': 'vec1', 'values': test_vector}])
    print("   ... вектор успешно сохранен в индексе 'ukido'!")

except Exception as e:
    print(f"   !!! Ошибка при работе с Pinecone: {e}")
    exit()


# 3. Ищем наш вектор в Pinecone
print("\n3. Ищу только что сохраненный вектор...")
try:
    # Ищем вектор, наиболее похожий на наш тестовый (это должен быть он сам)
    query_result = index.query(vector=test_vector, top_k=1)
    found_id = query_result['matches'][0]['id']
    
    print(f"   ... поиск завершен! Найден вектор с ID: {found_id}")

    if found_id == 'vec1':
        print("\nУРА! ВСЕ РАБОТАЕТ! Мы успешно сохранили и нашли вектор.")
    else:
        print("\nЧто-то пошло не так. Найденный ID не совпадает.")

except Exception as e:
    print(f"   !!! Ошибка при поиске в Pinecone: {e}")