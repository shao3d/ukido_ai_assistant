import os
import requests
import google.generativeai as genai
from flask import Flask, request, render_template
from dotenv import load_dotenv
from pinecone import Pinecone
import redis

# --- НАСТРОЙКИ И ЗАГРУЗКА КЛЮЧЕЙ ---
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST_FACTS = os.getenv("PINECONE_HOST_FACTS")
HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

# Проверяем переменные, включая Redis
if not all([TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_HOST_FACTS, HUBSPOT_API_KEY, REDIS_URL]):
    required_vars = {
        'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
        'GEMINI_API_KEY': GEMINI_API_KEY, 
        'PINECONE_API_KEY': PINECONE_API_KEY,
        'PINECONE_HOST_FACTS': PINECONE_HOST_FACTS,
        'HUBSPOT_API_KEY': HUBSPOT_API_KEY,
        'REDIS_URL': REDIS_URL
    }
    missing_vars = [name for name, value in required_vars.items() if not value]
    raise ValueError(f"Отсутствуют обязательные переменные: {', '.join(missing_vars)}")

# --- КОНФИГУРАЦИЯ КЛИЕНТОВ ---
genai.configure(api_key=GEMINI_API_KEY)
generation_model = genai.GenerativeModel('gemini-1.5-flash')
embedding_model = 'models/text-embedding-004'

try:
    print("🔍 Инициализируем Redis client...")
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    print("✅ Соединение с Redis инициализировано успешно")
except Exception as e:
    print(f"❌ Критическая ошибка инициализации Redis: {e}")
    raise e

def get_pinecone_index():
    if not hasattr(get_pinecone_index, 'initialized'):
        try:
            print("🔍 Инициализируем Pinecone client...")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            facts_description = pc.describe_index("ukido")
            print(f"🔍 Facts индекс host: {facts_description.host}")
            get_pinecone_index.pc = pc
            get_pinecone_index.index_facts = pc.Index(host=facts_description.host)
            get_pinecone_index.initialized = True
            print("✅ Соединение с Pinecone Facts инициализировано успешно")
        except Exception as e:
            print(f"❌ Ошибка инициализации Pinecone: {e}")
            raise e
    return get_pinecone_index.index_facts

CONVERSATION_MEMORY_SIZE = 9
CONVERSATION_EXPIRATION_SECONDS = 3600

def get_conversation_history_from_redis(chat_id):
    history_key = f"history:{chat_id}"
    history_list = redis_client.lrange(history_key, 0, -1)
    history_list.reverse()
    return history_list

def update_conversation_history_in_redis(chat_id, user_message, ai_response):
    history_key = f"history:{chat_id}"
    pipe = redis_client.pipeline()
    pipe.lpush(history_key, f"Ассистент: {ai_response}")
    pipe.lpush(history_key, f"Пользователь: {user_message}")
    max_lines = CONVERSATION_MEMORY_SIZE * 2
    pipe.ltrim(history_key, 0, max_lines - 1)
    pipe.expire(history_key, CONVERSATION_EXPIRATION_SECONDS)
    pipe.execute()

# --- СИСТЕМА ПРОМПТОВ ---
BASE_PROMPT = """Ты AI-ассистент украинской школы soft skills для детей "Ukido". Отвечай на "вы" с уважением. Обслуживаешь родителей украинских детей среднего класса."""
STYLE_MODULES = {
    # Стили остаются без изменений, они работают отлично
    "informational": """МАКСИМУМ 1-2 предложения. Дай ТОЛЬКО факт из контекста четко и коротко.
Добавь ОДНУ легкую одесскую деталь из вариантов:
- "Ну что ж, так у нас" 
- "А что тут такого"
- "Вот и все дела"
- "Нормально же"
- "И все тут"
- "Что поделаешь"
- "Так уж заведено"
- "Как есть, так есть"
СТРОГО ЗАПРЕЩЕНО философствовать! Только факты + прагматичная концовка.""",
    "trial_lesson": """До 3 предложений + ОБЯЗАТЕЛЬНО ссылка [LESSON_LINK]. 
Одесская прагматичность с разнообразными оборотами:
- "Конечно можно! А что тут такого. [LESSON_LINK]"
- "Да без проблем: [LESSON_LINK]. Попробуете - поймете"
- "Ну а как же! [LESSON_LINK] - вот ссылочка"
- "Само собой: [LESSON_LINK]. Детям понравится"
- "Естественно: [LESSON_LINK]. А зачем без пробы покупать"
- "Разумеется: [LESSON_LINK]. Сначала смотрим, потом решаем"
- "Обязательно попробуйте: [LESSON_LINK]. Что терять-то"
Тон: практичный, дружелюбный, без лишних слов.""",
    "consultational": """3-5 предложений МАКСИМУМ. Структура Жванецкого:
1. Конкретный практичный совет (1-2 предложения)
2. ОДНО житейское наблюдение с деталями в стиле Жванецкого
3. Можно предложить курс из контекста

ОБЯЗАТЕЛЬНЫЕ речевые обороты (варьируй):
- "А что тут поделаешь"
- "Слушайте"
- "Понимаете, что получается"
- "Чем больше [действие], тем меньше [результат]"
- "Ну что ж, дети они и есть дети"
- "Вот вам и вся проблема"
- "И что в итоге"
- "Такая уж жизнь"

КОНКРЕТНЫЕ детали (как у Жванецкого):
- НЕ "проблемы" → "не слушается, огрызается, в телефоне торчит до ночи"
- НЕ "много развиваем" → "английский записали, теннис, шахматы, рисование, еще и китайский"
- Звукоподражания: "тык-тык в телефоне", "хлоп дверью", "топ-топ по коридору", "бух на диван"
- Конкретные ситуации: "домой приходит - сразу к холодильнику, потом в комнату, дверь закрыл"

ВАЖНО: начинай с практического совета, а НЕ с наблюдений!""",
    "philosophical": """4-6 предложений МАКСИМУМ. Полная структура Жванецкого:
1. Начни с конкретной житейской ситуации с деталями
2. Одно наблюдение-парадокс через накопление деталей  
3. Практический вывод

ОБЯЗАТЕЛЬНЫЕ зачины (варьируй):
- "Слушайте..."
- "Понимаете, что получается..."
- "Вот смотрите..." 
- "А что в итоге..."
- "Скажите на милость..."
- "Вот вам картинка..."

ТЕХНИКИ Жванецкого:
- Накопление через "И": "и английский записали, и теннис, и шахматы, и робототехнику еще"
- Переключения: "а что получается", "а в итоге что", "а результат какой"
- Звукоподражания: "тык-тык", "хлоп", "бух на диван", "топ-топ"
- Конкретные детали: "планшет купили за 15 тысяч, репетитора наняли за 500 в час"

ОБЯЗАТЕЛЬНЫЕ концовки (варьируй):
- "Вот и вся наука"
- "А что тут скажешь" 
- "Ну что ж"
- "Такие дела"
- "И все тут"
- "Вот вам и весь секрет"

Пример: "Слушайте, покупаем планшет за 15 тысяч, записываем на английский за 800 в месяц, на теннис еще за 1200 - и то хотим лучшее, и это самое современное. А что получается - сидит ребенок, тык-тык в телефоне, говорит 'Скучно мне, все надоело'. Понимаете, мы думаем - счастье в количестве возможностей, а ему просто родительского внимания не хватает. Вот и вся наука воспитания.""",
    "sensitive": """ДЕЛИКАТНЫЙ РЕЖИМ для болезненных тем (развод, смерть, болезни).
МАКСИМУМ 3-4 предложения. БЕЗ иронии и юмора.
Тон: сочувствующий, поддерживающий, мудрый.
Можешь использовать ТОЛЬКО мягкие обороты:
- "В таких ситуациях..."
- "Это действительно сложно..."
- "Понимаю, как это тяжело..."
- "Дети особенно чувствительны к таким изменениям..."
НИКОГДА не используй: "А что тут поделаешь", "Ну что ж", звукоподражания.
Предлагай конкретную помощь от школы, но деликатно."""
}

def analyze_request_type(user_message):
    message_lower = user_message.lower()
    sensitive_keywords = ['развод', 'развелись', 'смерть', 'умер', 'умерла', 'болезнь', 'болеет', 'депрессия', 'травма', 'горе', 'потеря']
    if any(word in message_lower for word in sensitive_keywords):
        return "sensitive"
    trial_keywords = ['пробный', 'попробовать', 'посмотреть', 'попробуем', 'можно ли попробовать', 'хочу попробовать', 'дайте попробовать', 'хочу попробовать']
    if any(word in message_lower for word in trial_keywords):
        return "trial_lesson"
    info_keywords = ['цена', 'стоимость', 'расписание', 'запись', 'когда', 'сколько', 'время', 'адрес', 'телефон', 'как записаться', 'возраст', 'длительность', 'продолжительность', 'сколько детей', 'размер группы', 'миссия', 'учителя', 'преподаватели', 'методики']
    if any(word in message_lower for word in info_keywords):
        return "informational"
    philosophical_keywords = ['как правильно', 'смысл', 'почему', 'зачем', 'современные дети', 'поколение', 'в наше время', 'раньше было', 'принципы воспитания', 'правильно воспитывать', 'невоспитанные']
    if any(word in message_lower for word in philosophical_keywords):
        return "philosophical"
    return "consultational"

def should_add_lesson_link(user_message, request_type, conversation_length):
    if request_type == "trial_lesson":
        return True
    if request_type in ["informational", "sensitive"]:
        return False
    return conversation_length >= 18

# <<< ИЗМЕНЕНИЕ: Полностью переработанная функция для борьбы с "залипанием" >>>
def create_enriched_prompt(request_type, facts_context, history_context):
    """
    Создает обогащенный промпт с четкой иерархией, чтобы предотвратить "залипание" на истории.
    """
    # Шаг 1: Определяем базовые инструкции и стиль для текущего запроса.
    system_prompt = f"""{BASE_PROMPT}

{STYLE_MODULES[request_type]}

ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
- Опирайся ТОЛЬКО на факты из [КОНТЕКСТА из базы фактов].
- Если в контексте нет ответа, честно признайся, сказав: "К сожалению, у меня нет точной информации по этому вопросу". Не выдумывай.
- Максимум один смайлик за ответ.
- НИКОГДА не упоминай реальные имена из литературы.
- Обращение только на "вы"."""

    # Шаг 2: Собираем финальный промпт с четким порядком приоритетов.
    final_prompt = f"""{system_prompt}

### КОНТЕКСТ из базы фактов (Источник правды для ответа):
{facts_context if facts_context else "Нет релевантной информации."}

### ИСТОРИЯ ПРЕДЫДУЩЕГО ДИАЛОГА (Используй только для справки, чтобы понять общий настрой разговора):
{history_context if history_context else "Это начало диалога."}

---
### ГЛАВНАЯ ЗАДАЧА:
Твоя основная цель — дать точный и релевантный ответ на **самый последний вопрос пользователя**, который находится ниже. Не позволяй истории диалога отвлечь тебя. Если пользователь спрашивает про цену, отвечай про цену. Если про расписание — отвечай про расписание. Сосредоточься на последнем запросе.

Пользователь:"""
    
    return final_prompt
# <<< КОНЕЦ ИЗМЕНЕНИЯ >>>

def get_facts_context(prompt):
    try:
        index_facts = get_pinecone_index()
        query_embedding = genai.embed_content(
            model=embedding_model, 
            content=prompt, 
            task_type="RETRIEVAL_QUERY"
        )['embedding']
        facts_results = index_facts.query(vector=query_embedding, top_k=3, include_metadata=True)
        # <<< ИЗМЕНЕНИЕ: Добавил проверку на релевантность, чтобы отсечь мусор >>>
        relevant_matches = [match['metadata']['text'] for match in facts_results['matches'] if match['score'] > 0.7]
        if not relevant_matches:
            print("⚠️ RAG не нашел достаточно релевантного контекста.")
            return "", True # Возвращаем пустую строку, чтобы модель знала, что фактов нет
        
        facts_context = "\n".join(relevant_matches)
        print("✅ RAG система работает корректно")
        return facts_context, True
    except Exception as e:
        print(f"⚠️ RAG система временно недоступна: {e}")
        fallback_facts = """Ukido - онлайн-школа soft skills для детей. 
Курсы: "Юный Оратор" (7-10 лет), "Эмоциональный Компас" (9-12 лет), "Капитан Проектов" (11-14 лет).
Стоимость: от 6000 до 8000 грн в месяц. Занятия 2 раза в неделю по 90 минут.
Доступны бесплатные пробные уроки."""
        return fallback_facts, False

def get_optimized_gemini_response(chat_id, prompt):
    request_type = analyze_request_type(prompt)
    facts_context, rag_available = get_facts_context(prompt)
    history = get_conversation_history_from_redis(chat_id)
    history_context = "\n".join(history)
    conversation_length = len(history)
    
    enriched_prompt = create_enriched_prompt(request_type, facts_context, history_context)
    full_prompt = enriched_prompt + " " + prompt + "\nАссистент:" # Добавил пробел для чистоты
    
    try:
        response = generation_model.generate_content(full_prompt)
        ai_response = response.text.strip()
        
        if should_add_lesson_link(prompt, request_type, conversation_length):
            if "[LESSON_LINK]" not in ai_response and request_type != "trial_lesson":
                ai_response += "\n\n🎯 Хотите увидеть нашу методику в действии? Попробуйте пробный урок: [LESSON_LINK]"
        
        if "[LESSON_LINK]" in ai_response:
            base_url = os.environ.get('BASE_URL')
            lesson_url = f"{base_url}/lesson?user_id={chat_id}" if base_url else f"http://localhost:5000/lesson?user_id={chat_id}"
            if not base_url:
                print("⚠️ BASE_URL не настроен, используется localhost fallback")
            ai_response = ai_response.replace("[LESSON_LINK]", lesson_url)
        
        update_conversation_history_in_redis(chat_id, prompt, ai_response)
        
        print(f"✅ Ответ сгенерирован: тип={request_type}, память={conversation_length//2} обменов, RAG={'да' if rag_available else 'нет'}")
        
        return ai_response
    except Exception as e:
        print(f"❌ Критическая ошибка при работе с Gemini AI: {e}")
        return """Извините, в данный момент система испытывает технические трудности. 

Пожалуйста, попробуйте обратиться позже или свяжитесь с нашими консультантами напрямую."""

# --- HUBSPOT И FLASK (без изменений) ---
def send_to_hubspot(user_data):
    hubspot_url = "https://api.hubapi.com/crm/v3/objects/contacts"
    headers = {"Authorization": f"Bearer {HUBSPOT_API_KEY}", "Content-Type": "application/json"}
    contact_data = {"properties": {"firstname": user_data["firstName"], "lastname": user_data["lastName"], "email": user_data["email"], "telegram_user_id": str(user_data.get("userId", ""))}}
    try:
        response = requests.post(hubspot_url, headers=headers, json=contact_data)
        if response.status_code == 201:
            print("✅ Контакт успешно создан в HubSpot!")
            return True
        else:
            print(f"❌ Ошибка HubSpot API: {response.status_code}\nОтвет сервера: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ошибка при отправке в HubSpot: {str(e)}")
        return False

def generate_first_follow_up_message(first_name):
    return f"""👋 Привет, {first_name}!\n\nКак впечатления от нашего урока об открытых вопросах? Удалось попробовать технику в реальной жизни?\n\n🎯 Если понравилось, предлагаю записаться на полноценное пробное занятие с тренером Ukido. Это бесплатно и поможет лучше понять нашу методику.\n\nИнтересно?"""

def generate_second_follow_up_message(first_name):
    return f"""🌟 {first_name}, не хочу быть навязчивым, но очень хочется узнать ваше мнение!\n\nИскусство правильных вопросов действительно может изменить отношения в семье. Многие родители замечают улучшения уже после первых попыток применить технику.\n\n💡 Если готовы погрузиться глубже, наши тренеры покажут еще больше эффективных методов развития soft skills у детей.\n\nЗапишем на бесплатную консультацию?"""

app = Flask(__name__)

def send_telegram_message(chat_id, text):
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

@app.route('/lesson')
def show_lesson_page():
    user_id = request.args.get('user_id')
    return render_template('lesson.html', user_id=user_id)

@app.route('/', methods=['POST'])
def webhook():
    update = request.get_json()
    if "message" in update and "text" in update["message"]:
        chat_id = update["message"]["chat"]["id"]
        received_text = update["message"]["text"]
        ai_response = get_optimized_gemini_response(chat_id, received_text)
        send_telegram_message(chat_id, ai_response)
    return "ok", 200

@app.route('/submit-lesson-form', methods=['POST'])
def submit_lesson_form():
    form_data = request.get_json()
    print(f"=== Получены данные формы ===\nИмя: {form_data.get('firstName')}\nФамилия: {form_data.get('lastName')}\nEmail: {form_data.get('email')}\n==========================")
    if send_to_hubspot(form_data):
        print("🎉 Данные успешно сохранены в CRM!")
        return {"success": True, "message": "Данные сохранены в CRM"}, 200
    else:
        print("⚠️ Не удалось сохранить в CRM, но форма обработана")
        return {"success": True, "message": "Данные получены"}, 200

@app.route('/hubspot-webhook', methods=['POST'])
def hubspot_webhook():
    try:
        webhook_data = request.get_json()
        contact_id = webhook_data.get('vid')
        if contact_id:
            properties = webhook_data.get('properties', {})
            first_name = properties.get('firstname', {}).get('value', 'друг')
            telegram_id = properties.get('telegram_user_id', {}).get('value')
            message_type = request.args.get('message_type', 'first_follow_up')
            print(f"🆔 Contact ID: {contact_id}, 👋 Имя: {first_name}, 📱 Telegram ID: {telegram_id}, 📝 Тип сообщения: {message_type}")
            if telegram_id:
                if message_type == 'first_follow_up':
                    follow_up_message = generate_first_follow_up_message(first_name)
                elif message_type == 'second_follow_up':
                    follow_up_message = generate_second_follow_up_message(first_name)
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'false').lower() == 'true'
    print(f"🚀 Запуск ФИНАЛЬНОГО Flask приложения на порту {port}")
    print(f"🔧 Debug режим: {'включен' if debug_mode else 'отключен'}")
    print("🧠 Память диалогов: 9 обменов (внешняя, в Redis)")
    print("🎭 Стиль Жванецкого: ОБОГАЩЕННЫЙ, фокус на последнем вопросе")
    print("📊 Facts RAG: АКТИВЕН, с порогом релевантности")
    app.run(debug=debug_mode, port=port, host='0.0.0.0')