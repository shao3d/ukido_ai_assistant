
import httpx
import asyncio
import json
import time
import random
from datetime import datetime

# --- КОНФИГУРАЦИЯ ---
APP_URL = "https://ukidoaiassistant-production.up.railway.app/test-message"  # НОВЫЙ endpoint!
SCENARIOS_FILE = "test_scenarios.json"
TEST_USER_ID = random.randint(10000000, 99999999) 

# --- ЦВЕТА ДЛЯ КРАСИВОГО ВЫВОДА В КОНСОЛЬ ---
class colors:
    USER = '\033[94m'      # Синий
    BOT = '\033[92m'       # Зеленый
    SYSTEM = '\033[93m'    # Желтый
    ERROR = '\033[91m'     # Красный
    SUCCESS = '\033[96m'   # Голубой
    ENDC = '\033[0m'       # Сброс цвета

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

async def send_test_message(client, message_text):
    """
    Отправляет сообщение на специальный test endpoint и получает реальный ответ бота
    """
    test_payload = {
        "message": message_text,
        "user_id": TEST_USER_ID
    }

    try:
        response = await client.post(APP_URL, json=test_payload, timeout=45.0)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return {
                    'success': True,
                    'bot_response': result.get('bot_response', 'Нет ответа'),
                    'response_time': result.get('response_time', 0),
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'bot_response': f"Ошибка API: {result.get('error', 'Неизвестная ошибка')}",
                    'response_time': 0,
                    'error': result.get('error')
                }
        else:
            error_text = response.text if response.text else f"HTTP {response.status_code}"
            return {
                'success': False,
                'bot_response': f"HTTP ошибка: {error_text}",
                'response_time': 0,
                'error': error_text
            }
            
    except httpx.TimeoutException:
        return {
            'success': False,
            'bot_response': "❌ Таймаут: приложение может быть в режиме сна",
            'response_time': 0,
            'error': 'timeout'
        }
    except httpx.RequestError as exc:
        return {
            'success': False,
            'bot_response': f"❌ Ошибка подключения: {exc}",
            'response_time': 0,
            'error': str(exc)
        }
    except Exception as e:
        return {
            'success': False,
            'bot_response': f"❌ Неожиданная ошибка: {e}",
            'response_time': 0,
            'error': str(e)
        }

async def analyze_bot_response(user_message, bot_response):
    """
    Анализирует качество ответа бота
    """
    user_lower = user_message.lower()
    response_lower = bot_response.lower()
    
    issues = []
    
    # Проверяем на fallback ответы
    fallback_phrases = [
        "к сожалению, у меня нет",
        "в моей базе данных нет",
        "не удалось найти",
        "информация временно недоступна",
        "недостаточно информации"
    ]
    
    if any(phrase in response_lower for phrase in fallback_phrases):
        issues.append("🚨 FALLBACK ответ")
    
    # Проверяем быстрые ответы
    if any(word in user_lower for word in ['цена', 'стоимость', 'сколько стоит', 'сколько стоят']):
        if '6000' in bot_response or '7000' in bot_response or '8000' in bot_response:
            issues.append("✅ Цены указаны корректно")
        else:
            issues.append("❌ Цены НЕ указаны")
    
    # Проверяем ответы на эмоциональные сообщения
    if user_message in [':)))', ':)', 'лол', 'ха-ха']:
        if len(bot_response) < 50:
            issues.append("❌ Слишком краткий ответ на эмоцию")
        else:
            issues.append("✅ Развернутый ответ на эмоцию")
    
    # Проверяем информативность для вопросов о школе
    if any(word in user_lower for word in ['школа', 'ukido', 'расскажи о', 'что такое']):
        if 'ukido' in response_lower or 'курс' in response_lower:
            issues.append("✅ Информативный ответ о школе")
        else:
            issues.append("❌ Недостаточно информации о школе")
    
    return issues

async def main():
    """
    Главная функция для запуска улучшенного тестирования
    """
    # Читаем тестовые сценарии из файла test_scenarios.json
    try:
        with open(SCENARIOS_FILE, 'r', encoding='utf-8') as f:
            scenarios = json.load(f)
    except FileNotFoundError:
        print_error_message(f"Файл сценариев '{SCENARIOS_FILE}' не найден.")
        print_system_message("Создайте файл test_scenarios.json с тестовыми сценариями")
        return
    except json.JSONDecodeError:
        print_error_message(f"Неверный формат JSON в файле '{SCENARIOS_FILE}'.")
        return

    print_system_message("🚀 ЗАПУСК УЛУЧШЕННОГО ТЕСТИРОВАНИЯ UKIDO AI ASSISTANT")
    print_system_message(f"📱 Test User ID: {TEST_USER_ID}")
    print_system_message(f"🎯 Target URL: {APP_URL}")
    print_system_message("🔍 Теперь видим РЕАЛЬНЫЕ ответы бота!")
    print("=" * 80)

    # Статистика тестирования
    stats = {
        'total_tests': 0,
        'successful_tests': 0,
        'fallback_responses': 0,
        'fast_responses': 0,
        'avg_response_time': 0,
        'total_time': 0
    }

    async with httpx.AsyncClient() as client:
        total_scenarios = len(scenarios)
        
        for idx, scenario in enumerate(scenarios, 1):
            print_system_message(f"СЦЕНАРИЙ {idx}/{total_scenarios}: {scenario['scenario_name']}")
            
            for step_idx, step in enumerate(scenario['steps'], 1):
                print(f"\n[{step_idx}/{len(scenario['steps'])}]")
                print_user_message(step)
                
                start_time = time.time()
                result = await send_test_message(client, step)
                local_response_time = time.time() - start_time
                
                # Используем время ответа от API если есть, иначе локальное
                response_time = result.get('response_time', local_response_time)
                
                stats['total_tests'] += 1
                stats['total_time'] += response_time
                
                if result['success']:
                    stats['successful_tests'] += 1
                    print_bot_message(result['bot_response'], response_time)
                    
                    # Анализируем качество ответа
                    issues = await analyze_bot_response(step, result['bot_response'])
                    for issue in issues:
                        if '✅' in issue:
                            print_success_message(issue)
                        elif '❌' in issue or '🚨' in issue:
                            print_error_message(issue)
                            if 'FALLBACK' in issue:
                                stats['fallback_responses'] += 1
                        else:
                            print_system_message(issue)
                    
                    # Определяем тип ответа
                    if response_time < 1.0:
                        stats['fast_responses'] += 1
                        print_success_message("⚡ БЫСТРЫЙ ответ (возможно fast response)")
                    
                else:
                    print_error_message(f"Ошибка: {result['bot_response']}")
                
                await asyncio.sleep(1.5)
            
            print_system_message(f"✅ ЗАВЕРШЕН: {scenario['scenario_name']}")
            print("=" * 80)
            
            if idx < total_scenarios:
                await asyncio.sleep(2)
    
    # Финальная статистика
    if stats['total_tests'] > 0:
        stats['avg_response_time'] = stats['total_time'] / stats['total_tests']
        success_rate = (stats['successful_tests'] / stats['total_tests']) * 100
        fallback_rate = (stats['fallback_responses'] / stats['total_tests']) * 100
        fast_rate = (stats['fast_responses'] / stats['total_tests']) * 100
        
        print_system_message("📊 ФИНАЛЬНАЯ СТАТИСТИКА")
        print_success_message(f"Успешных тестов: {stats['successful_tests']}/{stats['total_tests']} ({success_rate:.1f}%)")
        print_success_message(f"Среднее время ответа: {stats['avg_response_time']:.2f}s")
        print_success_message(f"Быстрых ответов: {stats['fast_responses']} ({fast_rate:.1f}%)")
        
        if stats['fallback_responses'] > 0:
            print_error_message(f"Fallback ответов: {stats['fallback_responses']} ({fallback_rate:.1f}%)")
        else:
            print_success_message("Fallback ответов: 0 🎉")

if __name__ == "__main__":
    print_system_message("🔧 Запуск улучшенного скрипта с реальными ответами бота")
    print_system_message("⏳ Начинаем через 2 секунды...")
    time.sleep(2)
    asyncio.run(main())
    print_system_message("🏁 Тестирование завершено")