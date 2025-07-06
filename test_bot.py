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
        # ✅ ИСПРАВЛЕНО: Таймаут для GPT-4o mini (быстрее чем Gemini)
        response = await client.post(test_endpoint_url, json=test_payload, timeout=30.0)
        
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
        # ✅ ИСПРАВЛЕНО: Обновленное сообщение про GPT-4o mini
        return {'success': False, 'bot_response': "❌ Таймаут: GPT-4o mini отвечает дольше обычного. Попробуйте еще раз."}
    except httpx.RequestError as exc:
        return {'success': False, 'bot_response': f"❌ Ошибка подключения: {exc}"}

async def main():
    """✅ УНИВЕРСАЛЬНАЯ ФУНКЦИЯ: Автоматически адаптируется к любому количеству сценариев"""
    
    # Загружаем сценарии тестирования
    try:
        with open(SCENARIOS_FILE, 'r', encoding='utf-8') as f:
            scenarios = json.load(f)
    except FileNotFoundError:
        print_error_message(f"Файл сценариев '{SCENARIOS_FILE}' не найден.")
        return
    
    # ✅ АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ КОЛИЧЕСТВА СЦЕНАРИЕВ
    total_scenarios = len(scenarios)
    
    print_system_message("🚀 ЗАПУСК ТЕСТИРОВАНИЯ UKIDO AI ASSISTANT (ЛОКАЛЬНЫЙ РЕЖИМ)")
    print_system_message(f"🎯 Target URL: {APP_URL}")
    print_system_message(f"📁 Сценарии: {SCENARIOS_FILE}")
    print_system_message(f"📊 Обнаружено сценариев: {total_scenarios}")
    print("=" * 80)

    async with httpx.AsyncClient() as client:
        # ✅ ГЛОБАЛЬНАЯ ОЧИСТКА в начале тестирования
        await clear_server_memory(client, "НАЧАЛО ТЕСТИРОВАНИЯ")
        print("=" * 80)
        
        # ✅ УНИВЕРСАЛЬНЫЙ ЦИКЛ для любого количества сценариев
        for idx, scenario in enumerate(scenarios, 1):
            scenario_user_id = f"{TEST_USER_ID_BASE}_{idx:02d}"
            
            print_system_message(f"🎭 НОВЫЙ ДИАЛОГ: СЦЕНАРИЙ {idx}/{total_scenarios}")
            print_system_message(f"📝 Название: {scenario['scenario_name']}")
            print_system_message(f"👤 User ID: {scenario_user_id}")
            
            # ✅ ОЧИСТКА ПАМЯТИ между сценариями (кроме первого)
            if idx > 1:  
                print()  # Пустая строка для читаемости
                await clear_server_memory(client, scenario['scenario_name'])
                print()
            
            # ✅ ВЫПОЛНЕНИЕ ШАГОВ сценария
            steps = scenario.get('steps', [])
            total_steps = len(steps)
            
            for step_idx, step in enumerate(steps, 1):
                print(f"\n[{step_idx}/{total_steps}]")
                print_user_message(step)
                
                result = await send_test_message(client, step, scenario_user_id)
                
                if result['success']:
                    print_bot_message(result['bot_response'], result['response_time'])
                else:
                    print_error_message(f"Ошибка: {result['bot_response']}")
                
                # ✅ ПАУЗА между шагами внутри диалога
                if step_idx < total_steps:  
                    await asyncio.sleep(1.5)  # Уменьшена с 2s до 1.5s для GPT-4o mini
            
            print_success_message(f"✅ ЗАВЕРШЕН ДИАЛОГ: {scenario['scenario_name']}")
            print("=" * 80)
            
            # ✅ ПАУЗА между диалогами (сценариями)
            if idx < total_scenarios:
                print_system_message("💤 Пауза перед следующим диалогом...")
                await asyncio.sleep(2)  # Уменьшена с 3s до 2s

if __name__ == "__main__":
    asyncio.run(main())
    print_system_message("🏁 ВСЕ ДИАЛОГИ ЗАВЕРШЕНЫ - Тестирование окончено")
    print_system_message(f"📊 Проверьте логи в терминале с app.py и в папке rag_debug_logs/")
    print_system_message(f"🎯 Архитектура: 2x GPT-4o mini (ChatEngine + Финальный ответ)")
