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

async def clear_server_memory(client, scenario_name=""):
    """✅ ИСПРАВЛЕНО: Очистка памяти сервера с указанием сценария"""
    clear_url = f"{APP_URL}/clear-memory"
    context = f"для сценария '{scenario_name}'" if scenario_name else "глобальная очистка"
    print_system_message(f"🧹 Сброс памяти сервера {context}")
    
    try:
        response = await client.post(clear_url, timeout=10.0)
        if response.status_code == 200:
            print_success_message(f"Память сервера сброшена {context}")
        else:
            print_error_message(f"Не удалось сбросить память. Код: {response.status_code}")
    except httpx.RequestError as exc:
        print_error_message(f"Ошибка подключения при сбросе памяти: {exc}")
        print_system_message("Продолжаем тестирование без сброса.")

async def send_test_message(client, message_text, user_id):
    """Отправляет сообщение на специальный test endpoint"""
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
    """✅ ИСПРАВЛЕНО: Главная функция с очисткой памяти перед каждым сценарием"""
    
    # Загружаем сценарии тестирования
    try:
        with open(SCENARIOS_FILE, 'r', encoding='utf-8') as f:
            scenarios = json.load(f)
    except FileNotFoundError:
        print_error_message(f"Файл сценариев '{SCENARIOS_FILE}' не найден.")
        return

    print_system_message("🚀 ЗАПУСК ТЕСТИРОВАНИЯ UKIDO AI ASSISTANT (ЛОКАЛЬНЫЙ РЕЖИМ)")
    print_system_message(f"🎯 Target URL: {APP_URL}")
    print_system_message(f"📁 Сценарии: {SCENARIOS_FILE}")
    print("=" * 80)

    async with httpx.AsyncClient() as client:
        # ✅ ИСПРАВЛЕНО: Глобальная очистка в начале тестирования
        await clear_server_memory(client, "НАЧАЛО ТЕСТИРОВАНИЯ")
        print("=" * 80)
        
        total_scenarios = len(scenarios)
        
        for idx, scenario in enumerate(scenarios, 1):
            scenario_user_id = f"{TEST_USER_ID_BASE}_{idx:02d}"
            
            print_system_message(f"🎭 НОВЫЙ ДИАЛОГ: СЦЕНАРИЙ {idx}/{total_scenarios}")
            print_system_message(f"📝 Название: {scenario['scenario_name']}")
            print_system_message(f"👤 User ID: {scenario_user_id}")
            
            # ✅ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Очистка памяти перед каждым сценарием
            if idx > 1:  # Для 2-го и последующих сценариев (1-й уже очищен глобально)
                print()  # Пустая строка для читаемости
                await clear_server_memory(client, scenario['scenario_name'])
                print()
            
            # Выполняем шаги текущего сценария
            for step_idx, step in enumerate(scenario['steps'], 1):
                print(f"\n[{step_idx}/{len(scenario['steps'])}]")
                print_user_message(step)
                
                result = await send_test_message(client, step, scenario_user_id)
                
                if result['success']:
                    print_bot_message(result['bot_response'], result['response_time'])
                else:
                    print_error_message(f"Ошибка: {result['bot_response']}")
                
                # Задержка между запросами для соблюдения лимитов API
                if step_idx < len(scenario['steps']):  # Не ждем после последнего шага в сценарии
                    await asyncio.sleep(2)  # Короткая пауза между шагами внутри диалога
            
            print_success_message(f"✅ ЗАВЕРШЕН ДИАЛОГ: {scenario['scenario_name']}")
            print("=" * 80)
            
            # Пауза между диалогами (сценариями)
            if idx < total_scenarios:
                print_system_message("💤 Пауза перед следующим диалогом...")
                await asyncio.sleep(3)  # Пауза между диалогами

if __name__ == "__main__":
    asyncio.run(main())
    print_system_message("🏁 ВСЕ ДИАЛОГИ ЗАВЕРШЕНЫ - Тестирование окончено")
    print_system_message("📊 Проверьте логи в терминале с app.py и в папке rag_debug_logs/")