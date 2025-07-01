import os
import requests
import google.generativeai as genai
from flask import Flask, request, render_template
from dotenv import load_dotenv
from pinecone import Pinecone
import redis
import time
import json
from datetime import datetime

# --- НАСТРОЙКИ И ЗАГРУЗКА КЛЮЧЕЙ ---
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST_FACTS = os.getenv("PINECONE_HOST_FACTS")
HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

# Проверяем наличие основных обязательных переменных (REDIS_URL опционален)
if not all([TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_HOST_FACTS, HUBSPOT_API_KEY]):
    required_vars = {
        'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN, 'GEMINI_API_KEY': GEMINI_API_KEY, 
        'PINECONE_API_KEY': PINECONE_API_KEY, 'PINECONE_HOST_FACTS': PINECONE_HOST_FACTS, 
        'HUBSPOT_API_KEY': HUBSPOT_API_KEY
    }
    missing_vars = [name for name, value in required_vars.items() if not value]
    raise ValueError(f"Отсутствуют обязательные переменные: {', '.join(missing_vars)}")

# --- КОНФИГУРАЦИЯ КЛИЕНТОВ ---
genai.configure(api_key=GEMINI_API_KEY)
generation_model = genai.GenerativeModel('gemini-1.5-flash')
embedding_model = 'models/text-embedding-004'

# --- ИНИЦИАЛИЗАЦИЯ REDIS С ОБРАБОТКОЙ ОШИБОК ---
redis_client = None
redis_available = False

def init_redis():
    """Инициализация Redis с обработкой ошибок"""
    global redis_client, redis_available
    try:
        print("🔍 Инициализируем Redis client...")
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()  # Проверяем подключение
        redis_available = True
        print("✅ Redis подключен успешно")
    except Exception as e:
        redis_available = False
        print(f"❌ Redis недоступен: {e}")
        print("⚠️ Система будет работать без постоянной памяти диалогов")

# Запускаем инициализацию Redis при старте
init_redis()

# Fallback память для случаев когда Redis недоступен
fallback_memory = {}

# --- ЛЕНИВАЯ ИНИЦИАЛИЗАЦИЯ PINECONE С FALLBACK ---
pinecone_index = None
pinecone_available = False

def get_pinecone_index():
    """Ленивая инициализация Pinecone с fallback стратегией"""
    global pinecone_index, pinecone_available
    
    if pinecone_index is not None:
        return pinecone_index
    
    try:
        print("🔍 Инициализируем Pinecone client...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Стратегия fallback: сначала динамический, потом прямой
        try:
            # Подход А - Динамический
            facts_description = pc.describe_index("ukido")
            pinecone_index = pc.Index(host=facts_description.host)
            print("✅ Pinecone подключен динамически")
        except Exception as dynamic_error:
            print(f"⚠️ Динамическое подключение не удалось: {dynamic_error}")
            # Подход Б - Прямой через переменную окружения
            pinecone_index = pc.Index(host=PINECONE_HOST_FACTS)
            print("✅ Pinecone подключен через прямой host")
        
        pinecone_available = True
        return pinecone_index
        
    except Exception as e:
        pinecone_available = False
        print(f"❌ Pinecone полностью недоступен: {e}")
        return None

# --- НАСТРОЙКИ ПАМЯТИ ДИАЛОГОВ ---
CONVERSATION_MEMORY_SIZE = 15  # 15 обменов = 30 строк
CONVERSATION_EXPIRATION_SECONDS = 3600  # 1 час

# --- УПРОЩЕННЫЕ ПРОМПТЫ БЕЗ СТИЛЕВЫХ МОДУЛЕЙ ---
BASE_PROMPT = """Ты AI-ассистент школы soft skills для детей "Ukido". 
Отвечай дружелюбно, используй только факты из предоставленной информации о школе.
Если вопрос касается новой темы, забудь предыдущую тему и сосредоточься на новом вопросе.
Если информации нет в контексте - честно скажи об этом."""

# --- ФУНКЦИИ УПРАВЛЕНИЯ ПАМЯТЬЮ ДИАЛОГОВ ---

def get_conversation_history(chat_id):
    """Получает историю диалога с fallback на локальную память"""
    if redis_available:
        try:
            history_key = f"history:{chat_id}"
            history_list = redis_client.lrange(history_key, 0, -1)
            # Redis возвращает в обратном порядке, разворачиваем
            history_list.reverse()
            return history_list
        except Exception as e:
            print(f"⚠️ Ошибка чтения из Redis: {e}, использую fallback память")
            return fallback_memory.get(chat_id, [])
    else:
        return fallback_memory.get(chat_id, [])

def update_conversation_history(chat_id, user_message, ai_response):
    """Обновляет историю диалога с расширенной структурой данных"""
    timestamp = datetime.now().isoformat()
    
    if redis_available:
        try:
            history_key = f"history:{chat_id}"
            metadata_key = f"metadata:{chat_id}"
            
            # Батчевые операции Redis для производительности
            pipe = redis_client.pipeline()
            
            # Добавляем новые сообщения в начало списка
            pipe.lpush(history_key, f"Ассистент: {ai_response}")
            pipe.lpush(history_key, f"Пользователь: {user_message}")
            
            # Обрезаем до нужного размера (15 обменов = 30 строк)
            pipe.ltrim(history_key, 0, (CONVERSATION_MEMORY_SIZE * 2) - 1)
            
            # Устанавливаем время жизни ключа
            pipe.expire(history_key, CONVERSATION_EXPIRATION_SECONDS)
            
            # Обновляем метаданные
            metadata = {
                "last_activity": timestamp,
                "question_count": len(get_conversation_history(chat_id)) // 2 + 1,
                "session_start": timestamp
            }
            pipe.hset(metadata_key, mapping=metadata)
            pipe.expire(metadata_key, CONVERSATION_EXPIRATION_SECONDS)
            
            # Выполняем все операции одним запросом
            pipe.execute()
            
        except Exception as e:
            print(f"⚠️ Ошибка записи в Redis: {e}, использую fallback память")
            # Fallback на локальную память
            if chat_id not in fallback_memory:
                fallback_memory[chat_id] = []
            
            fallback_memory[chat_id].append(f"Пользователь: {user_message}")
            fallback_memory[chat_id].append(f"Ассистент: {ai_response}")
            
            # Ограничиваем размер fallback памяти
            max_lines = CONVERSATION_MEMORY_SIZE * 2
            if len(fallback_memory[chat_id]) > max_lines:
                fallback_memory[chat_id] = fallback_memory[chat_id][-max_lines:]
    else:
        # Работаем с fallback памятью
        if chat_id not in fallback_memory:
            fallback_memory[chat_id] = []
        
        fallback_memory[chat_id].append(f"Пользователь: {user_message}")
        fallback_memory[chat_id].append(f"Ассистент: {ai_response}")
        
        max_lines = CONVERSATION_MEMORY_SIZE * 2
        if len(fallback_memory[chat_id]) > max_lines:
            fallback_memory[chat_id] = fallback_memory[chat_id][-max_lines:]

# --- ФУНКЦИИ RAG СИСТЕМЫ ---

def get_relevance_description(score):
    """Преобразует score релевантности в понятное описание"""
    if score >= 0.9:
        return "Отличное совпадение"
    elif score >= 0.7:
        return "Хорошее совпадение"
    elif score >= 0.5:
        return "Среднее совпадение"
    else:
        return "Слабое совпадение"

def get_speed_description(seconds):
    """Преобразует время ответа в понятное описание"""
    if seconds < 2:
        return "Быстро"
    elif seconds <= 5:
        return "Нормально"
    else:
        return "Медленно"

def get_facts_from_rag(user_message):
    """Получает релевантный контекст из Pinecone с детальными метриками"""
    search_start = time.time()
    
    try:
        index = get_pinecone_index()
        if not index:
            return "", {"error": "Pinecone недоступен", "fallback_used": True}
        
        # Создаем эмбеддинг для поискового запроса
        embedding_start = time.time()
        query_embedding = genai.embed_content(
            model=embedding_model, 
            content=user_message, 
            task_type="RETRIEVAL_QUERY"
        )['embedding']
        embedding_time = time.time() - embedding_start
        
        # Ищем релевантные факты
        query_start = time.time()
        results = index.query(
            vector=query_embedding, 
            top_k=3, 
            include_metadata=True
        )
        query_time = time.time() - query_start
        
        # Собираем контекст и метрики
        context_chunks = []
        best_score = 0
        
        for match in results['matches']:
            if match['score'] > 0.5:  # Фильтруем слабые совпадения
                context_chunks.append(match['metadata']['text'])
                best_score = max(best_score, match['score'])
        
        context = "\n".join(context_chunks)
        total_time = time.time() - search_start
        
        metrics = {
            "search_time": round(total_time, 2),
            "embedding_time": round(embedding_time, 2),
            "query_time": round(query_time, 2),
            "chunks_found": len(context_chunks),
            "best_score": round(best_score, 3),
            "relevance_desc": get_relevance_description(best_score),
            "speed_desc": get_speed_description(total_time),
            "success": True
        }
        
        return context, metrics
        
    except Exception as e:
        total_time = time.time() - search_start
        print(f"⚠️ Ошибка RAG системы: {e}")
        
        # Возвращаем базовую информацию как fallback
        fallback_context = """Ukido - онлайн-школа soft skills для детей. 
Курсы: "Юный Оратор" (7-10 лет, 6000 грн/мес), "Эмоциональный Компас" (9-12 лет, 7500 грн/мес), "Капитан Проектов" (11-14 лет, 8000 грн/мес).
Занятия 2 раза в неделю по 90 минут. Доступны бесплатные пробные уроки."""
        
        metrics = {
            "search_time": round(total_time, 2),
            "error": str(e),
            "fallback_used": True,
            "chunks_found": 1,
            "success": False
        }
        
        return fallback_context, metrics

# --- ОСНОВНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ ОТВЕТОВ ---

def generate_response(chat_id, user_message, is_test_mode=False):
    """Генерирует ответ с использованием RAG и памяти диалогов"""
    start_time = time.time()
    
    # Получаем контекст из RAG системы
    facts_context, rag_metrics = get_facts_from_rag(user_message)
    
    # Получаем историю диалога
    history = get_conversation_history(chat_id)
    history_context = "\n".join(history) if history else "Это начало диалога."
    
    # Создаем промпт
    full_prompt = f"""{BASE_PROMPT}

История диалога:
{history_context}

Информация о школе Ukido:
{facts_context}

Пользователь: {user_message}
Ассистент:"""
    
    try:
        # Генерируем ответ через Gemini
        gemini_start = time.time()
        response = generation_model.generate_content(full_prompt)
        ai_response = response.text.strip()
        gemini_time = time.time() - gemini_start
        
        # Если не тестовый режим, добавляем ссылку на урок при необходимости
        if not is_test_mode and len(history) >= 10:  # После 5 обменов
            if "пробный" not in ai_response.lower():
                base_url = os.environ.get('BASE_URL', 'http://localhost:5000')
                lesson_url = f"{base_url}/lesson?user_id={chat_id}"
                ai_response += f"\n\n🎯 Хотите увидеть нашу методику в действии? Попробуйте пробный урок: {lesson_url}"
        
        # Обновляем историю диалога (только если не тестовый режим)
        if not is_test_mode:
            update_conversation_history(chat_id, user_message, ai_response)
        
        total_time = time.time() - start_time
        
        # Возвращаем ответ и метрики
        response_metrics = {
            "total_time": round(total_time, 2),
            "gemini_time": round(gemini_time, 2),
            "rag_metrics": rag_metrics,
            "history_length": len(history),
            "redis_available": redis_available,
            "pinecone_available": pinecone_available
        }
        
        return ai_response, response_metrics
        
    except Exception as e:
        print(f"❌ Ошибка генерации ответа: {e}")
        error_response = """Извините, возникла техническая проблема. 
Пожалуйста, попробуйте перефразировать вопрос или обратитесь позже."""
        
        if not redis_available:
            error_response += "\n\nℹ️ Технические работы с памятью системы. Ваши сообщения обрабатываются, но я могу не помнить предыдущие вопросы."
        
        return error_response, {"error": str(e), "total_time": time.time() - start_time}

# --- ТЕСТОВЫЕ ВОПРОСЫ ДЛЯ НАКОПИТЕЛЬНОГО ДИАЛОГА ---

TEST_QUESTIONS = [
    "Расскажи о школе Ukido",
    "Какие курсы вы предлагаете?",
    "Подробнее о курсе Юный Оратор",
    "Кто ведет этот курс?",
    "А сколько он стоит?",
    "Есть ли скидки?",
    "Какое оборудование нужно для занятий?",
    "А что с курсом Эмоциональный Компас?",
    "Какие результаты показывают ваши выпускники?",
    "На чем основана ваша методика?",
    "Какое расписание занятий?",
    "Можно ли прийти на пробный урок?",
    "Вернемся к Юному Оратору - сколько детей в группе?",
    "Какая миссия вашей школы?",
    "Подведем итог - что вы бы порекомендовали для ребенка 8 лет?"
]

# --- HUBSPOT ИНТЕГРАЦИЯ ---
def send_to_hubspot(user_data):
    """Отправляет данные пользователя в HubSpot CRM"""
    hubspot_url = "https://api.hubapi.com/crm/v3/objects/contacts"
    
    headers = {
        "Authorization": f"Bearer {HUBSPOT_API_KEY}",
        "Content-Type": "application/json"
    }
    
    contact_data = {
        "properties": {
            "firstname": user_data["firstName"],
            "lastname": user_data["lastName"],
            "email": user_data["email"],
            "telegram_user_id": str(user_data.get("userId", ""))
        }
    }
    
    try:
        response = requests.post(hubspot_url, headers=headers, json=contact_data)
        
        if response.status_code == 201:
            print("✅ Контакт успешно создан в HubSpot!")
            return True
        else:
            print(f"❌ Ошибка HubSpot API: {response.status_code}")
            print(f"Ответ сервера: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при отправке в HubSpot: {str(e)}")
        return False

def generate_first_follow_up_message(first_name):
    """Генерирует первое автоматическое сообщение"""
    return f"""👋 Привет, {first_name}!

Как впечатления от нашего урока об открытых вопросах? Удалось попробовать технику в реальной жизни?

🎯 Если понравилось, предлагаю записаться на полноценное пробное занятие с тренером Ukido. Это бесплатно и поможет лучше понять нашу методику.

Интересно?"""

def generate_second_follow_up_message(first_name):
    """Генерирует второе автоматическое сообщение"""
    return f"""🌟 {first_name}, не хочу быть навязчивым, но очень хочется узнать ваше мнение!

Искусство правильных вопросов действительно может изменить отношения в семье. Многие родители замечают улучшения уже после первых попыток применить технику.

💡 Если готовы погрузиться глубже, наши тренеры покажут еще больше эффективных методов развития soft skills у детей.

Запишем на бесплатную консультацию?"""

# --- ОСНОВНОЕ ПРИЛОЖЕНИЕ FLASK ---
app = Flask(__name__)

def send_telegram_message(chat_id, text):
    """Отправляет сообщение пользователю в Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Сообщение успешно отправлено пользователю {chat_id}")
                return True
            else:
                print(f"⚠️ Telegram API вернул статус {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"⏱️ Таймаут при отправке сообщения (попытка {attempt + 1}/{max_retries})")
        except Exception as e:
            print(f"❌ Ошибка при отправке сообщения в Telegram (попытка {attempt + 1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(1)
    
    print(f"❌ Не удалось отправить сообщение пользователю {chat_id} после {max_retries} попыток")
    return False

# --- МАРШРУТЫ FLASK ---

@app.route('/lesson')
def show_lesson_page():
    """Отображает страницу урока с возможностью персонализации"""
    user_id = request.args.get('user_id')
    return render_template('lesson.html', user_id=user_id)

@app.route('/', methods=['POST'])
def webhook():
    """Главный маршрут для обработки сообщений от Telegram"""
    update = request.get_json()
    
    if "message" in update and "text" in update["message"]:
        chat_id = update["message"]["chat"]["id"]
        received_text = update["message"]["text"]
        
        # Генерируем ответ
        ai_response, metrics = generate_response(chat_id, received_text)
        send_telegram_message(chat_id, ai_response)
        
        # Логируем метрики для мониторинга
        print(f"📊 Обработан запрос от {chat_id}: {metrics['total_time']}с, Redis: {metrics['redis_available']}, Pinecone: {metrics['pinecone_available']}")

    return "ok", 200

@app.route('/test-rag')
def test_rag_system():
    """НАКОПИТЕЛЬНОЕ ТЕСТИРОВАНИЕ RAG СИСТЕМЫ"""
    print("\n" + "="*60)
    print("🧪 НАЧАЛО НАКОПИТЕЛЬНОГО ТЕСТИРОВАНИЯ RAG СИСТЕМЫ")
    print("="*60)
    
    test_chat_id = "test_user_session"
    
    # Очищаем предыдущую тестовую сессию
    if redis_available:
        try:
            redis_client.delete(f"history:{test_chat_id}")
            redis_client.delete(f"metadata:{test_chat_id}")
        except:
            pass
    
    if test_chat_id in fallback_memory:
        del fallback_memory[test_chat_id]
    
    total_test_start = time.time()
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n🧪 === RAG ТЕСТ №{i}/15 ===")
        print(f"❓ ВОПРОС: {question}")
        
        # Генерируем ответ
        response, metrics = generate_response(test_chat_id, question, is_test_mode=False)
        
        # Детальное логирование для анализа
        rag_metrics = metrics.get('rag_metrics', {})
        
        if rag_metrics.get('success', False):
            print(f"\n🔍 ПОИСК В PINECONE:")
            print(f"   ⏱️  Время поиска: {rag_metrics['search_time']} сек ({rag_metrics['speed_desc']})")
            print(f"   📊 Найдено чанков: {rag_metrics['chunks_found']}")
            print(f"   🎯 Лучший score: {rag_metrics['best_score']} ({rag_metrics['relevance_desc']})")
        else:
            print(f"\n⚠️ ПРОБЛЕМА С PINECONE:")
            print(f"   ❌ Ошибка: {rag_metrics.get('error', 'Неизвестная ошибка')}")
            print(f"   🔄 Использован fallback: {rag_metrics.get('fallback_used', False)}")
        
        print(f"\n🤖 ОТВЕТ GEMINI:")
        print(f"{response}")
        
        print(f"\n✅ МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:")
        print(f"   ⏱️  Общее время ответа: {metrics['total_time']} сек")
        print(f"   🧠 Время Gemini: {metrics['gemini_time']} сек")
        print(f"   💾 История диалога: {metrics['history_length']} строк")
        print(f"   🔗 Redis статус: {'✅ Работает' if metrics['redis_available'] else '❌ Недоступен'}")
        print(f"   🔍 Pinecone статус: {'✅ Работает' if metrics['pinecone_available'] else '❌ Недоступен'}")
        
        print("="*50)
        
        # Небольшая пауза между вопросами для реалистичности
        time.sleep(0.5)
    
    total_test_time = time.time() - total_test_start
    
    print(f"\n🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print(f"⏱️  Общее время тестирования: {total_test_time:.1f} секунд")
    print(f"📊 Среднее время на вопрос: {total_test_time/15:.1f} секунд")
    print(f"💾 Система памяти: {'Redis' if redis_available else 'Fallback'}")
    print(f"🔍 RAG система: {'Pinecone' if pinecone_available else 'Fallback'}")
    print("\n📋 Скопируйте весь лог выше для анализа качества RAG grounding!")
    print("="*60)
    
    return {
        "message": "Накопительное тестирование RAG завершено",
        "questions_tested": len(TEST_QUESTIONS),
        "total_time": round(total_test_time, 2),
        "redis_status": "available" if redis_available else "unavailable",
        "pinecone_status": "available" if pinecone_available else "unavailable"
    }, 200

@app.route('/submit-lesson-form', methods=['POST'])
def submit_lesson_form():
    """Обрабатывает данные формы урока и сохраняет их в HubSpot CRM"""
    form_data = request.get_json()
    
    print("=== Получены данные формы ===")
    print(f"Имя: {form_data.get('firstName')}")
    print(f"Фамилия: {form_data.get('lastName')}")
    print(f"Email: {form_data.get('email')}")
    print("==========================")
    
    hubspot_success = send_to_hubspot(form_data)
    
    if hubspot_success:
        print("🎉 Данные успешно сохранены в CRM!")
        return {"success": True, "message": "Данные сохранены в CRM"}, 200
    else:
        print("⚠️ Не удалось сохранить в CRM, но форма обработана")
        return {"success": True, "message": "Данные получены"}, 200

@app.route('/hubspot-webhook', methods=['POST'])
def hubspot_webhook():
    """Обрабатывает webhook'и от HubSpot для автоматических сообщений"""
    try:
        webhook_data = request.get_json()
        contact_id = webhook_data.get('vid')
        
        if contact_id:
            properties = webhook_data.get('properties', {})
            first_name = properties.get('firstname', {}).get('value', 'друг')
            telegram_id = properties.get('telegram_user_id', {}).get('value')
            
            print(f"🆔 Contact ID: {contact_id}")
            print(f"👋 Имя: {first_name}")
            print(f"📱 Telegram ID: {telegram_id}")
            
            message_type = request.args.get('message_type', 'first_follow_up')
            print(f"📝 Тип сообщения: {message_type}")
            
            if telegram_id:
                if message_type == 'first_follow_up':
                    follow_up_message = generate_first_follow_up_message(first_name)
                    print(f"📤 Отправляю ПЕРВОЕ follow-up сообщение пользователю {first_name}")
                elif message_type == 'second_follow_up':
                    follow_up_message = generate_second_follow_up_message(first_name)
                    print(f"📤 Отправляю ВТОРОЕ follow-up сообщение пользователю {first_name}")
                else:
                    print(f"⚠️ Неизвестный тип сообщения: {message_type}")
                    return "Unknown message type", 400
                
                send_telegram_message(telegram_id, follow_up_message)
                print(f"✅ Follow-up сообщение ({message_type}) отправлено пользователю {telegram_id}")
            else:
                print("❌ Не найден telegram_user_id для контакта")
            
            return "OK", 200
        else:
            print("❌ Не удалось извлечь contact_id из webhook данных")
            return "No contact ID found", 400
        
    except Exception as e:
        print(f"❌ Ошибка обработки webhook: {e}")
        return "Error", 500

# --- ТОЧКА ВХОДА В ПРОГРАММУ ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print("="*60)
    print(f"🚀 ЗАПУСК ОБНОВЛЕННОЙ СИСТЕМЫ UKIDO AI ASSISTANT")
    print(f"🌐 Порт: {port}")
    print(f"🔧 Debug режим: {'включен' if debug_mode else 'отключен'}")
    print(f"💾 Redis: {'✅ Подключен' if redis_available else '❌ Недоступен (fallback режим)'}")
    print(f"🔍 Pinecone: готов к ленивой инициализации")
    print(f"🧠 Память диалогов: {CONVERSATION_MEMORY_SIZE} обменов ({CONVERSATION_MEMORY_SIZE * 2} строк)")
    print(f"⏱️  TTL диалогов: {CONVERSATION_EXPIRATION_SECONDS} секунд")
    print(f"🧪 Тестирование: доступно на /test-rag")
    print("="*60)
    print("📊 Для тестирования RAG откройте: https://ваш-url.railway.app/test-rag")
    print("🔍 Логи тестирования появятся в Railway Deploy Logs")
    print("="*60)
    
    app.run(debug=debug_mode, port=port, host='0.0.0.0')