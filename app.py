import os
import requests
import google.generativeai as genai
from flask import Flask, request, render_template
from dotenv import load_dotenv
from pinecone import Pinecone

# --- НАСТРОЙКИ И ЗАГРУЗКА КЛЮЧЕЙ ---
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST_FACTS = os.getenv("PINECONE_HOST_FACTS")
HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")

# Проверяем только необходимые переменные (убрали PINECONE_HOST_STYLE)
if not all([TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_HOST_FACTS, HUBSPOT_API_KEY]):
    required_vars = {
        'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
        'GEMINI_API_KEY': GEMINI_API_KEY, 
        'PINECONE_API_KEY': PINECONE_API_KEY,
        'PINECONE_HOST_FACTS': PINECONE_HOST_FACTS,
        'HUBSPOT_API_KEY': HUBSPOT_API_KEY
    }
    missing_vars = [name for name, value in required_vars.items() if not value]
    raise ValueError(f"Отсутствуют обязательные переменные: {', '.join(missing_vars)}")

# --- КОНФИГУРАЦИЯ КЛИЕНТОВ ---
genai.configure(api_key=GEMINI_API_KEY)
generation_model = genai.GenerativeModel('gemini-1.5-flash')
embedding_model = 'models/text-embedding-004'

def get_pinecone_index():
    """
    Ленивая инициализация соединения с Pinecone Facts (убрали Style RAG)
    """
    if not hasattr(get_pinecone_index, 'initialized'):
        try:
            print("🔍 Инициализируем Pinecone client...")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Получаем актуальную информацию только о facts индексе
            print("🔍 Получаем актуальную информацию об индексе фактов...")
            facts_description = pc.describe_index("ukido")
            
            print(f"🔍 Facts индекс host: {facts_description.host}")
            
            # Создаем подключение только к facts индексу
            get_pinecone_index.pc = pc
            get_pinecone_index.index_facts = pc.Index(host=facts_description.host)
            get_pinecone_index.initialized = True
            
            print("✅ Соединение с Pinecone Facts инициализировано успешно")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации Pinecone: {e}")
            raise e
    
    return get_pinecone_index.index_facts

# --- ПАМЯТЬ ДИАЛОГОВ ---
conversation_history = {}
CONVERSATION_MEMORY_SIZE = 9  # Бот помнит и использует 9 обменов (память = контекст)

# --- ОБОГАЩЕННАЯ СИСТЕМА ПРОМПТОВ СО СТИЛЕМ ЖВАНЕЦКОГО ---

# Базовый промпт
BASE_PROMPT = """Ты AI-ассистент украинской школы soft skills для детей "Ukido". Отвечай на "вы" с уважением. Обслуживаешь родителей украинских детей среднего класса."""

# Обогащенные стилевые модули с ярким стилем Жванецкого
STYLE_MODULES = {
    "informational": """
МАКСИМУМ 1-2 предложения. Дай ТОЛЬКО факт из контекста четко и коротко.
Добавь ОДНУ легкую одесскую деталь:
- "Ну что ж, так у нас" 
- "А что тут такого"
- "Вот и все дела"
- "Нормально же"
СТРОГО ЗАПРЕЩЕНО философствовать! Только факты + прагматичная концовка.""",

    "trial_lesson": """
До 3 предложений + ОБЯЗАТЕЛЬНО ссылка [LESSON_LINK]. 
Одесская прагматичность с конкретными оборотами:
- "Конечно можно! А что тут такого. [LESSON_LINK]"
- "Да без проблем: [LESSON_LINK]. Попробуете - поймете"
- "Ну а как же! [LESSON_LINK] - вот ссылочка"
- "Само собой: [LESSON_LINK]. Детям понравится"
Тон: практичный, без лишних слов.""",

    "consultational": """
3-5 предложений МАКСИМУМ. Структура Жванецкого:
1. Конкретный практичный совет (1-2 предложения)
2. ОДНО житейское наблюдение с деталями в стиле Жванецкого
3. Можно предложить курс из контекста

ОБЯЗАТЕЛЬНЫЕ речевые обороты:
- "А что тут поделаешь"
- "Слушайте"
- "Понимаете, что получается"
- "Чем больше [действие], тем меньше [результат]"
- "Ну что ж, дети они и есть дети"

КОНКРЕТНЫЕ детали (как у Жванецкого):
- НЕ "проблемы" → "не слушается, огрызается, в телефоне торчит"
- НЕ "много развиваем" → "английский записали, теннис, шахматы, рисование"
- Используй звукоподражания: "тык-тык в телефоне", "хлоп дверью"""",

    "philosophical": """
4-6 предложений МАКСИМУМ. Полная структура Жванецкого:
1. Начни с конкретной житейской ситуации с деталями
2. Одно наблюдение-парадокс через накопление деталей  
3. Практический вывод

ОБЯЗАТЕЛЬНЫЕ зачины:
- "Слушайте..."
- "Понимаете, что получается..."
- "Вот смотрите..." 
- "А что в итоге..."

ТЕХНИКИ Жванецкого:
- Накопление через "И": "и английский записали, и теннис, и шахматы"
- Переключения: "а что получается", "а в итоге что"
- Звукоподражания: "тык-тык", "хлоп", "бух на диван"
- Конкретные детали вместо абстракций

ОБЯЗАТЕЛЬНЫЕ концовки:
- "Вот и вся наука"
- "А что тут скажешь" 
- "Ну что ж"

Пример структуры: "Слушайте, покупаем планшет, записываем на английский, на теннис - и то хотим, и это нужно. А что получается - сидит ребенок, тык-тык в телефоне, говорит 'Скучно мне'. Понимаете, мы думаем - счастье в количестве кружков, а ему просто внимания не хватает. Вот и вся наука воспитания."""
}

def analyze_request_type(user_message):
    """
    Анализатор типа запроса для выбора стилевого модуля
    """
    message_lower = user_message.lower()
    
    # Запросы на пробный урок (высший приоритет)
    trial_keywords = ['пробный', 'попробовать', 'посмотреть', 'попробуем', 'можно ли попробовать', 'хочу попробовать']
    if any(word in message_lower for word in trial_keywords):
        return "trial_lesson"
    
    # Информационные запросы
    info_keywords = ['цена', 'стоимость', 'расписание', 'запись', 'когда', 'сколько', 'время', 'адрес', 'телефон', 'как записаться', 'возраст', 'длительность']
    if any(word in message_lower for word in info_keywords):
        return "informational"
    
    # Философские вопросы
    philosophical_keywords = ['как правильно', 'смысл', 'почему', 'зачем', 'современные дети', 'поколение', 'в наше время', 'раньше было', 'принципы воспитания']
    if any(word in message_lower for word in philosophical_keywords):
        return "philosophical"
    
    # По умолчанию - консультационные (проблемы с детьми)
    return "consultational"

def should_add_lesson_link(user_message, request_type, conversation_length):
    """
    Логика добавления ссылки на урок
    """
    # Всегда добавляем для trial_lesson запросов
    if request_type == "trial_lesson":
        return True
    
    # Не добавляем для информационных запросов
    if request_type == "informational":
        return False
    
    # Не добавляем для деликатных тем
    sensitive_keywords = ['развод', 'смерть', 'болезнь', 'депрессия', 'травма']
    if any(keyword in user_message.lower() for keyword in sensitive_keywords):
        return False
    
    # Добавляем для консультаций/философии после 9+ обменов (18 строк)
    return conversation_length >= 18  # 9 обменов = 18 строк

def create_enriched_prompt(request_type, facts_context, history_context):
    """
    Создает обогащенный промпт с ярким стилем Жванецкого
    """
    # Базовый промпт + обогащенный стилевой модуль + правила
    system_prompt = f"""{BASE_PROMPT}

{STYLE_MODULES[request_type]}

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
- Опирайся ТОЛЬКО на факты из контекста о школе Ukido
- Если не знаешь факта - честно признавайся
- Максимум один смайлик за ответ
- НИКОГДА не упоминай реальные имена из литературы
- Обращение только на "вы"""
    
    return f"""{system_prompt}

[ИСТОРИЯ ДИАЛОГА]:
{history_context}

[КОНТЕКСТ из базы фактов]:
{facts_context}

Пользователь: """

def get_facts_context(prompt):
    """
    Получает релевантный контекст только из базы фактов
    """
    try:
        index_facts = get_pinecone_index()
        
        # Создаем эмбеддинг для поискового запроса
        query_embedding = genai.embed_content(
            model=embedding_model, 
            content=prompt, 
            task_type="RETRIEVAL_QUERY"
        )['embedding']
        
        # Ищем релевантные факты о школе (топ-3 результата)
        facts_results = index_facts.query(vector=query_embedding, top_k=3, include_metadata=True)
        facts_context = "\n".join([match['metadata']['text'] for match in facts_results['matches']])
        
        print("✅ RAG система работает корректно")
        return facts_context, True
        
    except Exception as e:
        print(f"⚠️ RAG система временно недоступна: {e}")
        
        # Fallback: базовая информация о школе
        fallback_facts = """Ukido - онлайн-школа soft skills для детей. 
Курсы: "Юный Оратор" (7-10 лет), "Эмоциональный Компас" (9-12 лет), "Капитан Проектов" (11-14 лет).
Стоимость: от 6000 до 8000 грн в месяц. Занятия 2 раза в неделю по 90 минут.
Доступны бесплатные пробные уроки."""
        
        return fallback_facts, False

def update_conversation_history(chat_id, user_message, ai_response):
    """Обновляет историю диалога для конкретного пользователя"""
    if chat_id not in conversation_history:
        conversation_history[chat_id] = []
    
    conversation_history[chat_id].append(f"Пользователь: {user_message}")
    conversation_history[chat_id].append(f"Ассистент: {ai_response}")
    
    # Ограничиваем размер истории (9 обменов = 18 строк)
    max_lines = CONVERSATION_MEMORY_SIZE * 2
    if len(conversation_history[chat_id]) > max_lines:
        conversation_history[chat_id] = conversation_history[chat_id][-max_lines:]

def get_optimized_gemini_response(chat_id, prompt):
    """
    ФИНАЛЬНАЯ главная функция с честной памятью: что помним - то используем
    """
    # Определяем тип запроса
    request_type = analyze_request_type(prompt)
    
    # Получаем контекст только из фактов
    facts_context, rag_available = get_facts_context(prompt)
    
    # Получаем ВСЮ историю диалога - используем всю память (9 обменов = 18 строк)
    history = conversation_history.get(chat_id, [])
    history_context = "\n".join(history)  # ВСЯ память в контекст
    conversation_length = len(history)
    
    # Создаем обогащенный промпт
    enriched_prompt = create_enriched_prompt(request_type, facts_context, history_context)
    full_prompt = enriched_prompt + prompt + "\nАссистент:"
    
    try:
        # Генерируем ответ
        response = generation_model.generate_content(full_prompt)
        ai_response = response.text.strip()
        
        # Добавляем ссылку на урок если нужно
        if should_add_lesson_link(prompt, request_type, conversation_length):
            if "[LESSON_LINK]" not in ai_response and request_type != "trial_lesson":
                ai_response += "\n\n🎯 Хотите увидеть нашу методику в действии? Попробуйте пробный урок: [LESSON_LINK]"
        
        # Обрабатываем ссылку на урок
        if "[LESSON_LINK]" in ai_response:
            base_url = os.environ.get('BASE_URL')
            if base_url:
                lesson_url = f"{base_url}/lesson?user_id={chat_id}"
            else:
                lesson_url = f"http://localhost:5000/lesson?user_id={chat_id}"
                print("⚠️ BASE_URL не настроен, используется localhost fallback")
            
            ai_response = ai_response.replace("[LESSON_LINK]", lesson_url)
        
        # Сохраняем в истории диалога
        update_conversation_history(chat_id, prompt, ai_response)
        
        print(f"✅ Ответ сгенерирован: тип={request_type}, память={conversation_length//2} обменов, RAG={'да' if rag_available else 'нет'}")
        
        return ai_response
        
    except Exception as e:
        print(f"❌ Критическая ошибка при работе с Gemini AI: {e}")
        return """Извините, в данный момент система испытывает технические трудности. 

Пожалуйста, попробуйте обратиться позже или свяжитесь с нашими консультантами напрямую."""

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
            import time
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
        
        # Используем ФИНАЛЬНУЮ функцию генерации ответов
        ai_response = get_optimized_gemini_response(chat_id, received_text)
        send_telegram_message(chat_id, ai_response)

    return "ok", 200

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
    
    print(f"🚀 Запуск ФИНАЛЬНОГО Flask приложения на порту {port}")
    print(f"🔧 Debug режим: {'включен' if debug_mode else 'отключен'}")
    print("💰 Токен-экономика: СБАЛАНСИРОВАННАЯ (качество + эффективность)")
    print("🧠 Память диалогов: 9 обменов (память = контекст)")
    print("🎭 Стиль Жванецкого: ОБОГАЩЕННЫЙ")
    print("📊 Facts RAG: АКТИВЕН")
    
    app.run(debug=debug_mode, port=port, host='0.0.0.0')