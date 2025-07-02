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

# --- НАСТРОЙКИ И ЗАГРУЗКА КЛЮЧЕЙ! ---
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST_FACTS = os.getenv("PINECONE_HOST_FACTS")
HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

# Проверяем наличие основных обязательных переменных
required_vars = {
    'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN, 'GEMINI_API_KEY': GEMINI_API_KEY, 
    'PINECONE_API_KEY': PINECONE_API_KEY, 'PINECONE_HOST_FACTS': PINECONE_HOST_FACTS, 
    'HUBSPOT_API_KEY': HUBSPOT_API_KEY, 'OPENROUTER_API_KEY': OPENROUTER_API_KEY
}
missing_vars = [name for name, value in required_vars.items() if not value]
if missing_vars:
    raise ValueError(f"Отсутствуют обязательные переменные: {', '.join(missing_vars)}")

# --- КОНФИГУРАЦИЯ КЛИЕНТОВ ---
# Gemini остается для векторизации
genai.configure(api_key=GEMINI_API_KEY)
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
        redis_client.ping()
        redis_available = True
        print("✅ Redis подключен успешно")
    except Exception as e:
        redis_available = False
        print(f"❌ Redis недоступен: {e}")
        print("⚠️ Система будет работать без постоянной памяти диалогов")

init_redis()
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
        try:
            facts_description = pc.describe_index("ukido")
            pinecone_index = pc.Index(host=facts_description.host)
            print("✅ Pinecone подключен динамически")
        except Exception as dynamic_error:
            print(f"⚠️ Динамическое подключение не удалось: {dynamic_error}")
            pinecone_index = pc.Index(host=PINECONE_HOST_FACTS)
            print("✅ Pinecone подключен через прямой host")
        pinecone_available = True
        return pinecone_index
    except Exception as e:
        pinecone_available = False
        print(f"❌ Pinecone полностью недоступен: {e}")
        return None

# --- НАСТРОЙКИ ПАМЯТИ ДИАЛОГОВ ---
CONVERSATION_MEMORY_SIZE = 15
CONVERSATION_EXPIRATION_SECONDS = 3600

# --- УЛУЧШЕННЫЕ ПРОМПТЫ С МЯГКИМ ВОЗВРАЩЕНИЕМ ---
BASE_PROMPT = """Ты — AI-ассистент школы soft skills "Ukido". Твоя роль — мудрый наставник с тонким, ироничным юмором в духе Жванецкого. Твоя речь — это парадоксальные, но жизненные наблюдения. Твоя главная задача — помочь родителю, а не продать любой ценой.

### ОСНОВНЫЕ ПРИНЦИПЫ
1.  **Наставник, а не продавец:** Твоя цель — дать пользу. Продажа — естественное следствие хорошей консультации.
2.  **Юмор — это инструмент, а не самоцель:** Используй иронию и парадоксы для иллюстрации мысли, а не для плоских шуток. Стиль проявляется в наблюдениях, а не в анекдотах.
3.  **Конкретика важнее философии:** Вместо "проблемы с поведением" используй "кричит, хлопает дверью, уроки не делает". Опирайся на факты из RAG.

### СТИЛЬ ОБЩЕНИЯ И УМЕСТНОСТЬ ЮМОРА
- **Начало диалога (0-3 сообщения):** Максимально по делу, коротко, профессионально. Ответь на вопрос из RAG. Пример: "Курс стоит 6000 гривен в месяц. Вот и все дела."
- **Установление контакта (4-8 сообщений):** Здесь можно добавить легкую иронию, аналогии, истории успеха (Максим, София). Пример: "И на английский записали, и на теннис, и на шахматы... а он сидит, тык-тык в телефоне. Знакомая картина, не правда ли?"
- **Развитие диалога (9+ сообщений):** Уместны философские, но короткие замечания, подводящие к действию. Пример: "Знаете, дети не ждут, пока мы решимся. Они просто растут."

### ПРАВИЛА ТАКТИЧНОСТИ И ЭМПАТИИ (КРИТИЧЕСКИ ВАЖНО!)
- **Если родитель говорит о проблемах ребенка (застенчивость, страхи, конфликты):**
  - **ЗАПРЕЩЕНЫ ШУТКИ, САРКАЗМ И ПРЯМАЯ ИРОНИЯ.**
  - **Тон:** Максимально тактичный, поддерживающий.
  - **Что можно:** Допускается одно мудрое, эмпатичное наблюдение в стиле "Чем больше мы кричим 'Делай уроки!', тем громче он включает музыку в наушниках. А что тут поделаешь...".
  - **Действия:** Признай чувства родителя ("Я понимаю, как это может беспокоить..."). Дай конкретный, полезный совет, основанный на информации из RAG.

### РАБОТА С ИНФОРМАЦИЕЙ ИЗ RAG
- **Score > 0.7 (Высокая релевантность):** Четко и уверенно отвечай на основе найденных фактов. Цитируй детали: имена, цифры, названия.
- **Score 0.5-0.7 (Средняя релевантность):** Начинай с общей мудрости, а затем плавно переходи к фактам. Пример: "Говорят, все знают, как воспитывать чужих детей... А если серьезно, то в Ukido мы подходим к этому так: [факт из RAG]".
- **Score < 0.5 (Низкая релевантность / Off-topic):** Используй иронию для возврата в тему. Пример: "Я, конечно, не метеоролог, но точно знаю, что сейчас — отличная погода, чтобы инвестировать в будущее ребенка. Кстати, о будущем, у нас есть..."

### ПРАВИЛА ПРЕДЛОЖЕНИЯ ПРОБНОГО УРОКА (СТРОГО!)
- **НЕ ПРЕДЛАГАЙ УРОК в каждом сообщении.**
- **ПРЕДЛАГАЙ УРОК ТОЛЬКО В 3 СЛУЧАЯХ:**
  1.  **Прямой запрос:** Если пользователь сам спросил "как попробовать", "есть ли бесплатный урок", "как записаться". Отвечай прямо: "Да, конечно. А что тут такого. Вот ссылка...".
  2.  **Логическое завершение:** После обсуждения конкретной проблемы ребенка, которую решает курс. Пример: "...именно для таких случаев и создан наш курс 'Юный Оратор'. Хотите посмотреть, как это работает на практике на пробном занятии?"
  3.  **Длинный диалог:** Если диалог длится более 8 сообщений и он содержательный, можно ненавязчиво предложить урок как следующий шаг.
- **Никогда не предлагай урок в ответ на off-topic вопрос.**

### ТАБУ (ЧТО НЕЛЬЗЯ ГОВОРИТЬ)
- **Канцеляризмы:** "С радостью расскажу", "Буду рад помочь", "Данный курс".
- **Пустые фразы:** "Спасибо за ваш вопрос". Сразу переходи к делу.
- **Повторяться:** Не используй одни и те же продающие переходы.
- **Высокопарные метафоры:** Никаких "морских плаваний", "режиссуры самооценки" и "дегустаций изысканных блюд".
"""

# --- ФУНКЦИИ УПРАВЛЕНИЯ ПАМЯТЬЮ ДИАЛОГОВ ---

def get_conversation_history(chat_id):
    """Получает историю диалога с fallback на локальную память"""
    if redis_available:
        try:
            history_key = f"history:{chat_id}"
            return redis_client.lrange(history_key, 0, -1)[::-1]
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
            pipe = redis_client.pipeline()
            pipe.lpush(history_key, f"Ассистент: {ai_response}")
            pipe.lpush(history_key, f"Пользователь: {user_message}")
            pipe.ltrim(history_key, 0, (CONVERSATION_MEMORY_SIZE * 2) - 1)
            pipe.expire(history_key, CONVERSATION_EXPIRATION_SECONDS)
            metadata = {
                "last_activity": timestamp,
                "question_count": len(get_conversation_history(chat_id)) // 2 + 1
            }
            pipe.hset(metadata_key, mapping=metadata)
            pipe.expire(metadata_key, CONVERSATION_EXPIRATION_SECONDS)
            pipe.execute()
        except Exception as e:
            print(f"⚠️ Ошибка записи в Redis: {e}, использую fallback память")
            update_fallback_memory(chat_id, user_message, ai_response)
    else:
        update_fallback_memory(chat_id, user_message, ai_response)

def update_fallback_memory(chat_id, user_message, ai_response):
    """Обновляет локальную fallback-память"""
    if chat_id not in fallback_memory:
        fallback_memory[chat_id] = []
    fallback_memory[chat_id].append(f"Пользователь: {user_message}")
    fallback_memory[chat_id].append(f"Ассистент: {ai_response}")
    max_lines = CONVERSATION_MEMORY_SIZE * 2
    if len(fallback_memory[chat_id]) > max_lines:
        fallback_memory[chat_id] = fallback_memory[chat_id][-max_lines:]

# --- ФУНКЦИИ RAG СИСТЕМЫ ---

def get_relevance_description(score):
    if score >= 0.9: return "Отличное совпадение"
    if score >= 0.7: return "Хорошее совпадение"
    if score >= 0.5: return "Среднее совпадение"
    return "Слабое совпадение"

def get_speed_description(seconds):
    if seconds < 2: return "Быстро"
    if seconds <= 5: return "Нормально"
    return "Медленно"

def get_facts_from_rag(user_message):
    search_start = time.time()
    try:
        index = get_pinecone_index()
        if not index:
            return "", {"error": "Pinecone недоступен", "fallback_used": True}
        
        embedding_start = time.time()
        query_embedding = genai.embed_content(model=embedding_model, content=user_message, task_type="RETRIEVAL_QUERY")['embedding']
        embedding_time = time.time() - embedding_start
        
        query_start = time.time()
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        query_time = time.time() - query_start
        
        context_chunks, found_chunks_debug, best_score = [], [], 0
        for match in results['matches']:
            if match['score'] > 0.5:
                context_chunks.append(match['metadata']['text'])
                best_score = max(best_score, match['score'])
                found_chunks_debug.append({
                    "score": round(match['score'], 3),
                    "source": match['metadata'].get('source', 'unknown'),
                    "text_preview": match['metadata']['text'][:150] + "..."
                })
        
        context = "\n".join(context_chunks)
        total_time = time.time() - search_start
        
        metrics = {
            "search_time": round(total_time, 2), "embedding_time": round(embedding_time, 2),
            "query_time": round(query_time, 2), "chunks_found": len(context_chunks),
            "found_chunks_debug": found_chunks_debug, "best_score": round(best_score, 3),
            "relevance_desc": get_relevance_description(best_score),
            "speed_desc": get_speed_description(total_time), "success": True
        }
        return context, metrics
        
    except Exception as e:
        total_time = time.time() - search_start
        print(f"⚠️ Ошибка RAG системы: {e}")
        fallback_context = "Ukido - онлайн-школа soft skills для детей. Курсы: 'Юный Оратор' (7-10 лет, 6000 грн/мес), 'Эмоциональный Компас' (9-12 лет, 7500 грн/мес), 'Капитан Проектов' (11-14 лет, 8000 грн/мес)."
        metrics = {"search_time": round(total_time, 2), "error": str(e), "fallback_used": True, "chunks_found": 1, "success": False}
        return fallback_context, metrics

# --- ФУНКЦИЯ ДЛЯ ВЫЗОВА GPT-4o MINI ЧЕРЕЗ OPENROUTER ---
def call_gpt4o_mini(prompt):
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Ошибка GPT-4o Mini: {e}")
        return "Извините, временная проблема с генерацией ответа."

# --- ОСНОВНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ ОТВЕТОВ (ФИНАЛЬНАЯ ВЕРСИЯ) ---
def generate_response(chat_id, user_message, is_test_mode=False):
    """Генерирует ответ с использованием RAG и памяти диалогов, создавая правильный массив messages."""
    start_time = time.time()
    facts_context, rag_metrics = get_facts_from_rag(user_message)
    history_list = get_conversation_history(chat_id)

    # --- Формируем полный промпт для GPT-4o Mini ---
    history_context = "\n".join(history_list) if history_list else "Это начало диалога."
    full_prompt = f"{BASE_PROMPT}\n\nИстория диалога:\n{history_context}\n\nИнформация о школе Ukido:\n{facts_context}\n\nПользователь: {user_message}\nАссистент:"


    try:
        llm_start = time.time()
        ai_response = call_gpt4o_mini(full_prompt)
        llm_time = time.time() - llm_start

        # Генерация ссылки на пробный урок при прямом запросе
        if any(word in user_message.lower() for word in ["пробн", "бесплатн", "попробова", "записат"]):
            base_url = os.environ.get('BASE_URL', 'https://ukidoaiassistant-production.up.railway.app')
            lesson_url = f"{base_url}/lesson?user_id={chat_id}"
            if lesson_url not in ai_response:
                ai_response += f"\n\nЗаписывайтесь: {lesson_url}"
        elif not is_test_mode and len(history_list) >= 10 and "пробный" not in ai_response.lower():
            base_url = os.environ.get('BASE_URL', 'https://ukidoaiassistant-production.up.railway.app')
            lesson_url = f"{base_url}/lesson?user_id={chat_id}"
            ai_response += f"\n\n🎯 Хотите увидеть нашу методику в действии? Попробуйте пробный урок: {lesson_url}"

        if not is_test_mode:
            update_conversation_history(chat_id, user_message, ai_response)

        total_time = time.time() - start_time

        response_metrics = {
            "total_time": round(total_time, 2), "llm_time": round(llm_time, 2),
            "rag_metrics": rag_metrics, "history_length": len(history_list),
            "redis_available": redis_available, "pinecone_available": pinecone_available
        }
        return ai_response, response_metrics

    except Exception as e:
        print(f"❌ Критическая ошибка в generate_response: {e}")
        error_response = "Извините, возникла техническая проблема. Пожалуйста, попробуйте перефразировать вопрос."
        return error_response, {"error": str(e), "total_time": time.time() - start_time}

# --- ТЕСТОВЫЕ ВОПРОСЫ ДЛЯ ПРОВЕРКИ ПАМЯТИ И РАЗГОВОРНЫХ ТЕМ ---
CONVERSATION_TEST_QUESTIONS = [
    # Блок 1: Базовая информация
    "Расскажи о курсах для детей в вашей школе",
    "А кто ведет курс для самых маленьких?",
    "Сколько стоит обучение?",

    # Блок 2: Проблемы детей (проверка тактичности)
    "Мой ребенок очень стеснительный, не может говорить при людях",
    "Сын постоянно конфликтует с одноклассниками",
    "Дочь боится выступать даже перед классом",

    # Блок 3: Философские вопросы
    "Зачем вообще развивать soft skills?",
    "Не рано ли в 7 лет учить ораторскому мастерству?",

    # Блок 4: Off-topic (проверка юмора)
    "Что думаете про искусственный интеллект?",
    "Какой у вас любимый цвет?",

    # Блок 5: Конверсия (проверка ненавязчивости)
    "Звучит интересно, что дальше?",
    "Хочу попробовать, как записаться?",
    "А если не понравится?",
    "Можно сначала посмотреть как проходят занятия?",

    # Блок 6: Проверка памяти
    "Напомни, кто ведет курс для подростков?",
    "Так сколько длится курс Эмоциональный Компас?",
    "Какие результаты у того застенчивого мальчика, про которого рассказывали?",
    "Подведи итог - что вы предлагаете для 10-летнего ребенка?",
    "Спасибо за информацию!",
]

# --- ФУНКЦИИ ТЕСТОВОЙ ПАМЯТИ ---
def update_test_conversation_history(chat_id, user_message, ai_response):
    """Обновляет память только для тестов, не затрагивая продакшн"""
    if chat_id not in fallback_memory:
        fallback_memory[chat_id] = []
    fallback_memory[chat_id].append(f"Пользователь: {user_message}")
    fallback_memory[chat_id].append(f"Ассистент: {ai_response}")
    max_lines = CONVERSATION_MEMORY_SIZE * 2
    if len(fallback_memory[chat_id]) > max_lines:
        fallback_memory[chat_id] = fallback_memory[chat_id][-max_lines:]

def get_test_conversation_history(chat_id):
    """Получает тестовую историю"""
    return fallback_memory.get(chat_id, [])

# --- РАСШИРЕННЫЕ ТЕСТОВЫЕ ВОПРОСЫ ДЛЯ НАКОПИТЕЛЬНОГО ДИАЛОГА (25 ВОПРОСОВ) ---
TEST_QUESTIONS = [
    # Категория 1: Знакомство и общие вопросы
    "Привет! Расскажи в двух словах о вашей школе.",
    "Какие курсы у вас есть и для какого возраста?",
    "В чем ваше главное отличие от других школ развития?",
    "Какая миссия у вашей школы?",
    # Категория 2: Глубокие вопросы о курсах и методике
    "Расскажи максимально подробно о программе курса 'Эмоциональный Компас'.",
    "Кто такой Дмитрий Петров и почему я должен ему доверять?",
    "На чем конкретно основана ваша методика 'Практика + Игра + Рефлексия'?",
    "Какие реальные проекты делали выпускники курса 'Капитан Проектов'?",
    "Мой ребенок очень застенчивый. Вы уверены, что ему у вас понравится?",
    # Категория 3: Финансовые и организационные вопросы
    "Сколько стоит обучение? Расскажи про все варианты цен и скидок.",
    "Какой график занятий? Можно ли его подстроить под себя?",
    "Что делать, если мы пропустим занятие?",
    "Какое оборудование нужно для уроков?",
    "Можно ли вернуть деньги, если нам не подойдет?",
    # Категория 4: Проверка памяти и контекста
    "Мне интересен курс для самых маленьких, 'Юный Оратор'. Расскажи о нем.",
    "А сколько детей занимается в группе на этом курсе?",
    "Кто его ведет?",
    "Какие результаты у выпускников именно этого курса?",
    # Категория 5: Сложные и конверсионные вопросы
    "Я сравниваю вас со школой 'СуперМозг'. Чем вы лучше?",
    "Хорошо, звучит интересно. Как нам попробовать?",
    "А что, если ребенок категорически не захочет заниматься после пробного урока?",
    "Какие гарантии вы даете?",
    "Окей, я готов записаться на пробный урок. Что мне нужно сделать?",
    "Подведите итог, почему я должен выбрать именно Ukido для своего 8-летнего сына?",
    "Какие у вас есть партнерства с известными компаниями?"
]

latest_test_results = {"timestamp": None, "tests": [], "summary": {}}
latest_conversation_results = {"timestamp": None, "tests": [], "summary": {}}

# --- HUBSPOT ИНТЕГРАЦИЯ ---
def send_to_hubspot(user_data):
    """Отправляет данные пользователя в HubSpot CRM"""
    hubspot_url = "https://api.hubapi.com/crm/v3/objects/contacts"
    headers = {"Authorization": f"Bearer {HUBSPOT_API_KEY}", "Content-Type": "application/json"}
    contact_data = {"properties": {
        "firstname": user_data["firstName"], "lastname": user_data["lastName"],
        "email": user_data["email"], "telegram_user_id": str(user_data.get("userId", ""))
    }}
    try:
        response = requests.post(hubspot_url, headers=headers, json=contact_data)
        if response.status_code == 201:
            print("✅ Контакт успешно создан в HubSpot!")
            return True
        else:
            print(f"❌ Ошибка HubSpot API: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ошибка при отправке в HubSpot: {str(e)}")
        return False

# --- ОСНОВНОЕ ПРИЛОЖЕНИЕ FLASK ---
app = Flask(__name__)

def send_telegram_message(chat_id, text):
    """Отправляет сообщение пользователю в Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                print(f"✅ Сообщение успешно отправлено пользователю {chat_id}")
                return True
            print(f"⚠️ Telegram API вернул статус {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Ошибка при отправке сообщения в Telegram (попытка {attempt + 1}/3): {e}")
        if attempt < 2: time.sleep(1)
    return False

# --- МАРШРУТЫ FLASK ---
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
        ai_response, metrics = generate_response(chat_id, received_text)
        send_telegram_message(chat_id, ai_response)
        print(f"📊 Обработан запрос от {chat_id}: {metrics.get('total_time', 'N/A')}с")
    return "ok", 200

@app.route('/test-conversation')
def test_conversation_system():
    """Тестирование памяти диалогов, разговорных тем и возвращения к школе"""
    global latest_conversation_results
    print("\n" + "="*60 + "\n🧪 ТЕСТИРОВАНИЕ ПАМЯТИ И РАЗГОВОРНЫХ ТЕМ\n" + "="*60)
    
    test_chat_id = "conversation_test_session"
    
    # Очищаем тестовую память
    if test_chat_id in fallback_memory: 
        del fallback_memory[test_chat_id]
    
    total_test_start = time.time()
    latest_conversation_results = {"timestamp": datetime.now().isoformat(), "tests": [], "summary": {}}
    
    for i, question in enumerate(CONVERSATION_TEST_QUESTIONS, 1):
        print(f"\n🧪 === ТЕСТ ПАМЯТИ №{i}/{len(CONVERSATION_TEST_QUESTIONS)} ===")
        print(f"❓ ВОПРОС: {question}")
        
        # Используем RAG, но управляем памятью вручную для тестов
        facts_context, rag_metrics = get_facts_from_rag(question)
        test_history = get_test_conversation_history(test_chat_id)
        
        # Формируем промпт с тестовой историей
        history_context = "\n".join(test_history) if test_history else "Это начало диалога."
        full_prompt = f"{BASE_PROMPT}\n\nИстория диалога:\n{history_context}\n\nИнформация о школе Ukido:\n{facts_context}\n\nПользователь: {question}\nАссистент:"
        
        start_time = time.time()
        llm_start = time.time()
        response = call_gpt4o_mini(full_prompt)
        llm_time = time.time() - llm_start
        total_time = time.time() - start_time
        
        # Обновляем тестовую память
        update_test_conversation_history(test_chat_id, question, response)
        
        metrics = {
            "total_time": round(total_time, 2), 
            "llm_time": round(llm_time, 2),
            "rag_metrics": rag_metrics, 
            "history_length": len(test_history),
            "redis_available": redis_available, 
            "pinecone_available": pinecone_available
        }
        
        test_result = {
            "question_number": i, "question": question, "response": response,
            "metrics": metrics, "rag_success": rag_metrics.get('success', False),
            "search_time": rag_metrics.get('search_time', 0),
            "chunks_found": rag_metrics.get('chunks_found', 0),
            "best_score": rag_metrics.get('best_score', 0),
            "relevance_desc": rag_metrics.get('relevance_desc', 'Неизвестно'),
            "memory_length": len(test_history)
        }
        latest_conversation_results["tests"].append(test_result)
        
        if rag_metrics.get('success', False):
            print(f"🔍 RAG: {rag_metrics['search_time']}с, Чанков: {rag_metrics['chunks_found']}, Score: {rag_metrics['best_score']}")
        
        print(f"💾 ПАМЯТЬ: {len(test_history)} строк истории")
        print(f"🤖 ОТВЕТ: {response}")
        print("="*50)
        
        time.sleep(0.5)  # Небольшая пауза между вопросами
    
    # Очищаем тестовую память после теста
    if test_chat_id in fallback_memory:
        del fallback_memory[test_chat_id]
    
    total_test_time = time.time() - total_test_start
    latest_conversation_results["summary"] = {
        "total_time": round(total_test_time, 2), 
        "avg_time_per_question": round(total_test_time/len(CONVERSATION_TEST_QUESTIONS), 2),
        "redis_status": "available" if redis_available else "unavailable",
        "pinecone_status": "available" if pinecone_available else "unavailable",
        "questions_tested": len(CONVERSATION_TEST_QUESTIONS),
        "final_memory_length": latest_conversation_results["tests"][-1]["memory_length"] if latest_conversation_results["tests"] else 0
    }
    
    print(f"\n🎉 ТЕСТИРОВАНИЕ ПАМЯТИ ЗАВЕРШЕНО! Общее время: {total_test_time:.1f}с")
    print(f"💾 Финальная длина памяти: {latest_conversation_results['summary']['final_memory_length']} строк")
    print("📊 Результаты: /conversation-results")
    
    return latest_conversation_results, 200

@app.route('/conversation-results')
def show_conversation_results():
    """Отображение результатов тестирования памяти"""
    if not latest_conversation_results["tests"]:
        return "<h1>Тестирование памяти не проводилось. Запустите <a href='/test-conversation'>/test-conversation</a></h1>"
    
    summary = latest_conversation_results['summary']
    tests_html = ""
    
    for test in latest_conversation_results["tests"]:
        memory_class = "good" if test["memory_length"] > 0 else "warning"
        rag_class = "good" if test["rag_success"] else "error"
        
        tests_html += f"""
        <div class="test">
            <div class="question">❓ Вопрос №{test['question_number']}: {test['question']}</div>
            <div class="metrics">
                <strong>💾 Память:</strong> <span class="{memory_class}">{test['memory_length']} строк</span> | 
                <strong>🔍 RAG:</strong> <span class="{rag_class}">{'✅' if test["rag_success"] else '❌'}</span> | 
                Score: {test['best_score']} ({test['relevance_desc']})
            </div>
            <div class="response"><strong>🤖 Ответ:</strong><br>{test['response'].replace('\n', '<br>')}</div>
            <div class="metrics"><strong>⏱️ Время:</strong> {test['metrics']['total_time']}с</div>
        </div>"""
    
    redis_class = "good" if summary['redis_status'] == 'available' else 'error'
    pinecone_class = "good" if summary['pinecone_status'] == 'available' else 'error'
    
    html = f"""
    <!DOCTYPE html>
    <html><head><title>Результаты тестирования памяти</title>
    <style>
        body {{ font-family: Arial; margin: 20px; }}
        .summary {{ background: #f0f8ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .test {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 8px; }}
        .question {{ font-weight: bold; color: #2c3e50; margin-bottom: 8px; }}
        .response {{ background: white; padding: 10px; border-left: 4px solid #3498db; margin: 10px 0; }}
        .metrics {{ color: #7f8c8d; font-size: 0.9em; margin: 5px 0; }}
        .good {{ color: #27ae60; font-weight: bold; }}
        .warning {{ color: #f39c12; font-weight: bold; }}
        .error {{ color: #e74c3c; font-weight: bold; }}
    </style></head>
    <body>
    <h1>🧪 Результаты тестирования памяти и разговорных тем</h1>
    <div class="summary">
        <h3>📊 Суммарная статистика</h3>
        <strong>Время тестирования:</strong> {summary['total_time']}с<br>
        <strong>Среднее время на вопрос:</strong> {summary['avg_time_per_question']}с<br>
        <strong>Вопросов протестировано:</strong> {summary['questions_tested']}<br>
        <strong>Финальная длина памяти:</strong> {summary['final_memory_length']} строк<br>
        <strong>Redis:</strong> <span class="{redis_class}">{summary['redis_status']}</span><br>
        <strong>Pinecone:</strong> <span class="{pinecone_class}">{summary['pinecone_status']}</span>
    </div>
    {tests_html}
    </body></html>
    """
    return html

@app.route('/conversation-results-json')
def get_conversation_results_json():
    """JSON результаты тестирования памяти"""
    if not latest_conversation_results["tests"]:
        return {"error": "Тестирование памяти не проводилось", "hint": "Запустите /test-conversation сначала"}, 404
    return latest_conversation_results, 200

@app.route('/test-rag')
def test_rag_system():
    global latest_test_results
    print("\n" + "="*60 + "\n🧪 НАЧАЛО ТЕСТИРОВАНИЯ С GPT-4o MINI\n" + "="*60)
    test_chat_id = "test_user_session"
    if redis_available:
        try:
            redis_client.delete(f"history:{test_chat_id}", f"metadata:{test_chat_id}")
        except: pass
    if test_chat_id in fallback_memory: del fallback_memory[test_chat_id]
    
    total_test_start = time.time()
    latest_test_results = {"timestamp": datetime.now().isoformat(), "tests": [], "summary": {}}
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n🧪 === ТЕСТ №{i}/25 С GPT-4o MINI ===")
        print(f"❓ ВОПРОС: {question}")
        response, metrics = generate_response(test_chat_id, question, is_test_mode=True)
        rag_metrics = metrics.get('rag_metrics', {})
        test_result = {
            "question_number": i, "question": question, "response": response,
            "metrics": metrics, "rag_success": rag_metrics.get('success', False),
            "search_time": rag_metrics.get('search_time', 0),
            "chunks_found": rag_metrics.get('chunks_found', 0),
            "best_score": rag_metrics.get('best_score', 0),
            "relevance_desc": rag_metrics.get('relevance_desc', 'Неизвестно')
        }
        latest_test_results["tests"].append(test_result)
        
        if rag_metrics.get('success', False):
            print(f"🔍 ПОИСК: {rag_metrics['search_time']}с, Найдено: {rag_metrics['chunks_found']}, Score: {rag_metrics['best_score']} ({rag_metrics['relevance_desc']})")
        else:
            print(f"⚠️ ПРОБЛЕМА С PINECONE: {rag_metrics.get('error', 'Неизвестная ошибка')}")
        
        print(f"🤖 ОТВЕТ GPT-4o MINI: {response}")
        print(f"✅ МЕТРИКИ: Общее время: {metrics['total_time']}с, Время LLM: {metrics['llm_time']}с, История: {metrics['history_length']} строк")
        print("="*50)
        time.sleep(1) # Пауза чтобы не превышать лимиты OpenRouter
    
    total_test_time = time.time() - total_test_start
    latest_test_results["summary"] = {
        "total_time": round(total_test_time, 2), "avg_time_per_question": round(total_test_time/25, 2),
        "redis_status": "available" if redis_available else "unavailable",
        "pinecone_status": "available" if pinecone_available else "unavailable",
        "questions_tested": 25
    }
    print(f"\n🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО! Общее время: {total_test_time:.1f}с. Результаты доступны на /test-results и /test-results-json")
    return latest_test_results, 200

@app.route('/test-results')
def show_test_results():
    if not latest_test_results["tests"]:
        return "<h1>Тестирование еще не проводилось. Запустите <a href='/test-rag'>/test-rag</a></h1>"
    
    summary = latest_test_results['summary']
    redis_class = "good" if summary['redis_status'] == 'available' else 'error'
    pinecone_class = "good" if summary['pinecone_status'] == 'available' else 'error'
    
    tests_html = ""
    for test in latest_test_results["tests"]:
        rag_class = "good" if test["rag_success"] else "error"
        tests_html += f"""
        <div class="test">
            <div class="question">❓ Вопрос №{test['question_number']}: {test['question']}</div>
            <div class="metrics"><strong>🔍 RAG:</strong> <span class="{rag_class}">{'Успешно' if test["rag_success"] else 'Ошибка'}</span> | Время: {test['search_time']}с | Чанков: {test['chunks_found']} | Score: {test['best_score']} ({test['relevance_desc']})</div>
            <div class="response"><strong>🤖 Ответ GPT-4o MINI:</strong><br>{test['response'].replace('\n', '<br>')}</div>
            <div class="metrics"><strong>⏱️ Общее время:</strong> {test['metrics']['total_time']}с | <strong>🧠 Время LLM:</strong> {test['metrics']['llm_time']}с | <strong>💾 История:</strong> {test['metrics']['history_length']} строк</div>
        </div>"""
    
    return render_template('results.html', summary=summary, tests_html=tests_html, redis_class=redis_class, pinecone_class=pinecone_class)

@app.route('/test-results-json')
def get_test_results_json():
    if not latest_test_results["tests"]:
        return {"error": "Тестирование еще не проводилось", "hint": "Запустите /test-rag сначала"}, 404
    return latest_test_results, 200

## ---
@app.route('/submit-lesson-form', methods=['POST'])
def submit_lesson_form():
    form_data = request.get_json()
    print(f"=== Получены данные формы: {form_data.get('firstName')} {form_data.get('lastName')} ===")
    hubspot_success = send_to_hubspot(form_data)
    return {"success": hubspot_success}, 200

@app.route('/hubspot-webhook', methods=['POST'])
def hubspot_webhook():
    try:
        webhook_data = request.get_json()
        properties = webhook_data.get('properties', {})
        first_name = properties.get('firstname', {}).get('value', 'друг')
        telegram_id = properties.get('telegram_user_id', {}).get('value')
        message_type = request.args.get('message_type', 'first_follow_up')
        
        if telegram_id:
            message_generators = {
                'first_follow_up': f"👋 Привет, {first_name}! Как впечатления от урока? Если понравилось, предлагаю записаться на полноценное пробное занятие. Интересно?",
                'second_follow_up': f"🌟 {first_name}, не хочу быть навязчивым, но очень хочется узнать ваше мнение! Готовы погрузиться глубже? Запишем на бесплатную консультацию?"
            }
            message_to_send = message_generators.get(message_type)
            if message_to_send:
                send_telegram_message(telegram_id, message_to_send)
                print(f"✅ Follow-up '{message_type}' отправлено {telegram_id}")
            else:
                print(f"⚠️ Неизвестный тип сообщения: {message_type}")
        else:
            print("❌ Не найден telegram_user_id для контакта")
        return "OK", 200
    except Exception as e:
        print(f"❌ Ошибка обработки HubSpot webhook: {e}")
        return "Error", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'false').lower() == 'true'
    print("="*60 + f"\n🚀 ЗАПУСК UKIDO AI ASSISTANT С GPT-4o MINI\n" + "="*60)
    app.run(debug=debug_mode, port=port, host='0.0.0.0')