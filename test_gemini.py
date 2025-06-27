import os
import google.generativeai as genai
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

# Получаем ключ из переменных окружения
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Передаем ключ в библиотеку
genai.configure(api_key=GEMINI_API_KEY)

# Выбираем модель, которую будем использовать. 
# 'gemini-1.5-flash' — одна из самых быстрых и современных.
model = genai.GenerativeModel('gemini-1.5-flash')

print("Отправляю первый запрос к Gemini...")

try:
    # Задаем наш вопрос
    prompt = "Объясни просто и кратко, как для новичка, что такое API?"
    
    # Отправляем запрос и ждем ответа
    response = model.generate_content(prompt)

    # Печатаем ответ, который сгенерировал AI
    print("\nОтвет от Gemini:")
    print("----------------")
    print(response.text)
    print("----------------")

except Exception as e:
    print(f"Возникла ошибка: {e}")