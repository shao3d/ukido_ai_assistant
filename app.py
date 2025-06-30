import os
import requests
import google.generativeai as genai
from flask import Flask, request, render_template
from dotenv import load_dotenv
from pinecone import Pinecone
import redis
import time

# --- НАСТРОЙКИ И ЗАГРУЗКА КЛЮЧЕЙ ---
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST_FACTS = os.getenv("PINECONE_HOST_FACTS")
HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

# Проверяем наличие всех обязательных переменных
if not all([TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_HOST_FACTS, HUBSPOT_API_KEY, REDIS_URL]):
    required_vars = {
        'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN, 'GEMINI_API_KEY': GEMINI_API_KEY, 
        'PINECONE_API_KEY': PINECONE_API_KEY, 'PINECONE_HOST_FACTS': PINECONE_HOST_FACTS, 
        'HUBSPOT_API_KEY': HUBSPOT_API_KEY, 'REDIS_URL': REDIS_URL
    }
    missing_vars = [name for name, value in required_vars.items() if not value]
    raise ValueError(f"Отсутствуют обязательные переменные: {', '.join(missing_vars)}")

# --- КОНФИГУРАЦИЯ КЛИЕНТОВ ---
genai.configure(api_key=GEMINI_API_KEY)
# Модель для генерации основных ответов
generation_model = genai.GenerativeModel('gemini-1.5-flash')
# Отдельная модель для быстрой классификации запросов
classification_model = genai.GenerativeModel('gemini-1.5-flash')
embedding_model = 'models/text-embedding-004'

# Инициализация клиента Redis
try:
    print("🔍 Инициализируем Redis client...")
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    print("✅ Соединение с Redis инициализировано успешно")
except Exception as e:
    raise ValueError(f"❌ Критическая ошибка инициализации Redis: {e}")

# Ленивая инициализация Pinecone
def get_pinecone_index():
    if not hasattr(get_pinecone_index, 'initialized'):
        try:
            print("🔍 Инициализируем Pinecone client...")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            facts_description = pc.describe_index("ukido")
            get_pinecone_index.index_facts = pc.Index(host=facts_description.host)
            get_pinecone_index.initialized = True
            print("✅ Соединение с Pinecone Facts инициализировано успешно")
        except Exception as e:
            raise RuntimeError(f"❌ Ошибка инициализации Pinecone: {e}")
    return get_pinecone_index.index_facts

# --- УПРАВЛЕНИЕ ПАМЯТЬЮ ДИАЛОГОВ В REDIS ---
CONVERSATION_MEMORY_SIZE = 9
CONVERSATION_EXPIRATION_SECONDS = 3600

def get_conversation_history_from_redis(chat_id):
    """Получает историю диалога из Redis."""
    history_key = f"history:{chat_id}"
    history_list = redis_client.lrange(history_key, 0, -1)
    history_list.reverse()
    return history_list

def update_conversation_history_in_redis(chat_id, user_message, ai_response):
    """Обновляет историю диалога в Redis."""
    history_key = f"history:{chat_id}"
    pipe = redis_client.pipeline()
    pipe.lpush(history_key, f"Ассистент: {ai_response}")
    pipe.lpush(history_key, f"Пользователь: {user_message}")
    pipe.ltrim(history_key, 0, (CONVERSATION_MEMORY_SIZE * 2) - 1)
    pipe.expire(history_key, CONVERSATION_EXPIRATION_SECONDS)
    pipe.execute()


# --- СИСТЕМА ПРОМПТОВ И СТИЛЕЙ ---
BASE_PROMPT = """Ты AI-ассистент украинской школы soft skills для детей "Ukido". Отвечай на "вы" с уважением. Обслуживаешь родителей украинских детей среднего класса."""
STYLE_MODULES = {
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

# --- ЯДРО ЛОГИКИ БОТА ---

def get_request_type_with_gemini(user_message, history_context):
    """Определяет намерение пользователя с помощью Gemini для более точной логики."""
    prompt = f"""
    Проанализируй последний вопрос пользователя и классифицируй его. Учитывай историю диалога для контекста.

    Категории:
    - informational: Запрос конкретной информации о школе Ukido (цена, расписание, курсы, учителя, методики, адрес, длительность, партнерства, требования).
    - trial_lesson: Явный запрос на пробный урок или желание попробовать ("хочу попробовать", "можно ли посмотреть").
    - sensitive: Запрос на очень деликатную тему (развод, смерть, болезнь, тяжелая депрессия, горе, травма).
    - consultational: Просьба дать совет по конкретной проблеме с поведением ребенка (не слушается, сидит в телефоне, капризничает, истерики, не общается).
    - philosophical: Общий, философский вопрос о воспитании, современных детях, смысле чего-либо, сравнение поколений.

    [ИСТОРИЯ ДИАЛОГА]:
    {history_context if history_context else "Нет истории."}

    [ПОСЛЕДНИЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ]:
    {user_message}

    Ответь ОДНИМ словом - названием категории из списка выше.
    """
    try:
        response = classification_model.generate_content(prompt)
        request_type = response.text.strip().lower()
        if request_type in STYLE_MODULES:
            print(f"🧠 Классификатор Gemini определил тип: {request_type}")
            return request_type
        else:
            print(f"⚠️ Классификатор Gemini вернул невалидное значение: '{request_type}'. Используем 'consultational'.")
            return "consultational"
    except Exception as e:
        print(f"❌ Ошибка классификатора Gemini: {e}. Используем 'consultational'.")
        return "consultational"

def get_facts_from_rag(prompt):
    """Ищет факты в Pinecone ТОЛЬКО для информационных запросов."""
    try:
        index_facts = get_pinecone_index()
        query_embedding = genai.embed_content(model=embedding_model, content=prompt, task_type="RETRIEVAL_QUERY")['embedding']
        facts_results = index_facts.query(vector=query_embedding, top_k=3, include_metadata=True)
        relevant_matches = [match['metadata']['text'] for match in facts_results['matches'] if match['score'] > 0.75]
        
        if not relevant_matches:
            print("⚠️ RAG не нашел релевантного контекста.")
            return ""
        
        facts_context = "\n".join(relevant_matches)
        print("✅ RAG нашел релевантный контекст.")
        return facts_context
    except Exception as e:
        print(f"⚠️ Ошибка RAG системы: {e}")
        return ""

def create_enriched_prompt(request_type, facts_context, history_context, user_message):
    """Создает динамический промпт в зависимости от типа запроса."""
    base_rules = """ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
- Обращайся только на "вы".
- Максимум один смайлик за ответ.
- НИКОГДА не упоминай реальные имена из литературы."""

    if request_type == "informational":
        specific_instructions = f"""
        Твоя задача - ответить на вопрос пользователя, используя ТОЛЬКО [КОНТЕКСТ из базы фактов].
        - Если в контексте есть нужный факт, дай его четко и коротко в стиле informational.
        - Если в контексте НЕТ ответа, скажи: "К сожалению, у меня нет точной информации по этому вопросу. Что поделаешь?". Не выдумывай и не используй историю.
        """
        context_section = f"### КОНТЕКСТ из базы фактов (Единственный источник правды):\n{facts_context if facts_context else 'Нет информации.'}"
    else:
        specific_instructions = f"""
        Твоя задача - дать совет или поразмышлять на тему пользователя в стиле {request_type}.
        НЕ ищи факты. Опирайся на свой "жизненный опыт" из инструкций по стилю и на историю диалога для поддержания беседы.
        """
        context_section = ""

    system_prompt = f"""{BASE_PROMPT}
{STYLE_MODULES[request_type]}
{base_rules}
{specific_instructions}"""
    
    return f"""{system_prompt}
### ИСТОРИЯ ПРЕДЫДУЩЕГО ДИАЛОГА:
{history_context if history_context else "Это начало диалога."}
{context_section}
---
### ПОСЛЕДНИЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ (Отвечай именно на него):
{user_message}

Ассистент:"""

def get_optimized_gemini_response(chat_id, user_message):
    """Главная функция с разделенной логикой для разных типов запросов."""
    history = get_conversation_history_from_redis(chat_id)
    history_context = "\n".join(history)
    
    request_type = get_request_type_with_gemini(user_message, history_context)
    
    facts_context = ""
    if request_type == "informational":
        facts_context = get_facts_from_rag(user_message)

    full_prompt = create_enriched_prompt(request_type, facts_context, history_context, user_message)
    
    try:
        response = generation_model.generate_content(full_prompt)
        ai_response = response.text.strip()
        
        # Добавляем ссылку на урок, если это уместно
        if request_type != "informational" and request_type != "sensitive" and (request_type == "trial_lesson" or len(history) >= 10):
             if "[LESSON_LINK]" not in ai_response:
                ai_response += "\n\n🎯 Хотите увидеть нашу методику в действии? Попробуйте пробный урок: [LESSON_LINK]"
        
        if "[LESSON_LINK]" in ai_response:
            base_url = os.environ.get('BASE_URL')
            lesson_url = f"{base_url}/lesson?user_id={chat_id}" if base_url else f"http://localhost:5000/lesson?user_id={chat_id}"
            ai_response = ai_response.replace("[LESSON_LINK]", lesson_url)

        update_conversation_history_in_redis(chat_id, user_message, ai_response)
        return ai_response
    except Exception as e:
        print(f"❌ Критическая ошибка при генерации ответа Gemini: {e}")
        return "Извините, у меня возникла внутренняя ошибка. Пожалуйста, попробуйте перефразировать вопрос."

# --- ИНТЕГРАЦИЯ С HUBSPOT И FLASK ---
def send_to_hubspot(user_data):
    """Отправляет данные пользователя в HubSpot CRM."""
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
    """Генерирует первое автоматическое сообщение."""
    return f"""👋 Привет, {first_name}!\n\nКак впечатления от нашего урока об открытых вопросах? Удалось попробовать технику в реальной жизни?\n\n🎯 Если понравилось, предлагаю записаться на полноценное пробное занятие с тренером Ukido. Это бесплатно и поможет лучше понять нашу методику.\n\nИнтересно?"""

def generate_second_follow_up_message(first_name):
    """Генерирует второе автоматическое сообщение."""
    return f"""🌟 {first_name}, не хочу быть навязчивым, но очень хочется узнать ваше мнение!\n\nИскусство правильных вопросов действительно может изменить отношения в семье. Многие родители замечают улучшения уже после первых попыток применить технику.\n\n💡 Если готовы погрузиться глубже, наши тренеры покажут еще больше эффективных методов развития soft skills у детей.\n\nЗапишем на бесплатную консультацию?"""

app = Flask(__name__)

def send_telegram_message(chat_id, text):
    """Отправляет сообщение пользователю в Telegram с ретраями."""
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
                    return "Unknown message type", 400
                send_telegram_message(telegram_id, follow_up_message)
                print(f"✅ Follow-up сообщение ({message_type}) отправлено пользователю {telegram_id}")
            else:
                print("❌ Не найден telegram_user_id для контакта")
            return "OK", 200
        else:
            return "No contact ID found", 400
    except Exception as e:
        print(f"❌ Ошибка обработки webhook: {e}")
        return "Error", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'false').lower() == 'true'
    print(f"🚀 Запуск Flask приложения на порту {port}")
    print(f"🔧 Debug режим: {'включен' if debug_mode else 'отключен'}")
    print("🧠 Память диалогов: 9 обменов (внешняя, в Redis)")
    print("🤖 Ядро логики: ПЕРЕРАБОТАНО. Умный классификатор + разделение потоков RAG/Non-RAG.")
    app.run(debug=debug_mode, port=port, host='0.0.0.0')