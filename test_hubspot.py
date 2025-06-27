import os
import hubspot
from hubspot.crm.contacts import SimplePublicObjectInput
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

# Получаем ключ из переменных окружения
HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")

# Создаем клиент для работы с API
client = hubspot.Client.create(access_token=HUBSPOT_API_KEY)

# Готовим данные для нового контакта
# Мы создадим легендарного охотника за головами :)
contact_properties = {
    "email": "boba.fett@tatooine.com",
    "firstname": "Боба",
    "lastname": "Фетт",
    "phone": "+1234567890",
    "jobtitle": "Охотник за головами"
}

# Создаем объект контакта
contact_to_create = SimplePublicObjectInput(properties=contact_properties)

print("Пытаюсь создать контакт в HubSpot...")

try:
    # Отправляем запрос на создание контакта
    api_response = client.crm.contacts.basic_api.create(
        simple_public_object_input_for_create=contact_to_create
    )
    print("Успех! Контакт создан.")
    print(f"ID нового контакта: {api_response.id}")

except Exception as e:
    print(f"Возникла ошибка: {e}")