import httpx
import asyncio
import json
import time
import random
from datetime import datetime

# --- КОНФИГУРАЦИЯ ---
# Убедитесь, что здесь указан URL вашего локально запущенного сервера
APP_URL = "http://127.0.0.1:5000" 
SCENARIOS_FILE = "test_scenarios.json"
TEST_USER_ID_BASE = random.randint(10000000, 99999999)

# --- ЦВЕТА ДЛЯ ВЫВОДА ---
class colors:
    USER = '\033[94m'
    BOT = '\033[92m'
    SYSTEM = '\033[93m'
    ERROR = '\033[91m'
    SUCCESS = '\033[96m'
    ENDC = '\033[0m'

# ... (все функции print_* остаются без изменений) ...
def print_user_message(message):
    print(f"{colors.USER}👤 [USER]: {message}{colors.ENDC}")

def print_bot_message(response_text, response_time):
    print(f"{colors.BOT}🤖 [BOT]: {response_text}{colors.ENDC}")
    print(f"{colors.SYSTEM}⚡ Время ответа: {response_time:.2f}s{colors.ENDC}")

def print_system_message(message):
    print(f"{colors.SYSTEM}--- {message} ---{colors.ENDC}")

def print_success_message(message):
    print(f"{colors.SUCCESS}✅ {message}{colors.ENDC}")

def print_error_message(message):
    print(f"{colors.ERROR}❌ {message}{colors.ENDC}")


# --- 🔥 НОВАЯ ФУНКЦИЯ ДЛЯ СБРОСА ПАМЯТИ 🔥 ---
async def clear_server_memory(client):
    """Отправляет запрос на /clear-memory для очистки состояния сервера."""
    clear_url = f"{APP_URL}/clear-memory"
    print_system_message(f"🧹 Попытка сброса памяти сервера по адресу: {clear_url}")
    try:
        response = await client.post(clear_url, timeout=10.0)
        if response.status_code == 200:
            print_success_message("Память сервера успешно сброшена (команда отправлена).")
        else:
            print_error_message(f"Не удалось сбросить память. Сервер ответил: {response.status_code}")
    except httpx.RequestError as exc:
        print_error_message(f"Ошибка подключения при сбросе памяти: {exc}")
        print_system_message("Продолжаем тестирование без сброса. Убедитесь, что сервер app.py запущен.")


async def send_test_message(client, message_text, user_id):
    """Отправляет сообщение на специальный test endpoint."""
    test_endpoint_url = f"{APP_URL}/test-message"
    test_payload = {"message": message_text, "user_id": user_id}

    try:
        # Увеличим таймаут, т.к. Gemini 1.5 Pro может отвечать дольше
        response = await client.post(test_endpoint_url, json=test_payload, timeout=60.0)
        
        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'bot_response': result.get('bot_response', 'Нет ответа'),
                'response_time': result.get('response_time', 0),
            }
        else:
            return {'success': False, 'bot_response': f"HTTP ошибка: {response.status_code} {response.text}"}
            
    except httpx.TimeoutException:
        return {'success': False, 'bot_response': "❌ Таймаут: Gemini 1.5 Pro может отвечать долго. Попробуйте еще раз."}
    except httpx.RequestError as exc:
        return {'success': False, 'bot_response': f"❌ Ошибка подключения: {exc}"}


async def main():
    """Главная функция для запуска тестирования."""
    try:
        with open(SCENARIOS_FILE, 'r', encoding='utf-8') as f:
            scenarios = json.load(f)
    except FileNotFoundError:
        print_error_message(f"Файл сценариев '{SCENARIOS_FILE}' не найден.")
        return

    print_system_message("🚀 ЗАПУСК ТЕСТИРОВАНИЯ UKIDO AI ASSISTANT (ЛОКАЛЬНЫЙ РЕЖИМ)")
    print_system_message(f"🎯 Target URL: {APP_URL}")
    print("=" * 80)

    async with httpx.AsyncClient() as client:
        # --- 🔥 ВЫЗЫВАЕМ СБРОС ПАМЯТИ В НАЧАЛЕ 🔥 ---
        await clear_server_memory(client)
        print("=" * 80)
        
        total_scenarios = len(scenarios)
        for idx, scenario in enumerate(scenarios, 1):
            scenario_user_id = f"{TEST_USER_ID_BASE}_{idx:02d}"
            print_system_message(f"СЦЕНАРИЙ {idx}/{total_scenarios}: {scenario['scenario_name']}")
            print_system_message(f"👤 User ID для этого сценария: {scenario_user_id}")
            
            for step_idx, step in enumerate(scenario['steps'], 1):
                print(f"\n[{step_idx}/{len(scenario['steps'])}]")
                print_user_message(step)
                
                result = await send_test_message(client, step, scenario_user_id)
                
                if result['success']:
                    print_bot_message(result['bot_response'], result['response_time'])
                else:
                    print_error_message(f"Ошибка: {result['bot_response']}")
                
                # Задержка, чтобы не превысить лимиты Gemini 1.5 Pro (2 запроса в минуту)
                await asyncio.sleep(31) 
            
            print_system_message(f"✅ ЗАВЕРШЕН: {scenario['scenario_name']}")
            print("=" * 80)
            if idx < total_scenarios:
                await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
    print_system_message("🏁 Тестирование завершено")

