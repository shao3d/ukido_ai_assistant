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
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") # НОВЫЙ КЛЮЧ
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

# --- УПРОЩЕННЫЕ ПРОМПТЫ ---
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

# --- НОВАЯ ФУНКЦИЯ ДЛЯ ВЫЗОВА DEEPSEEK (ФИНАЛЬНАЯ ВЕРСИЯ) ---
def call_deepseek_model(messages_list):
    """Вызывает модель DeepSeek через OpenRouter API с правильной структурой messages."""
    base_url = os.environ.get('BASE_URL', 'https://ukidoaiassistant-production.up.railway.app')

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": base_url,
                "X-Title": "Ukido AI Assistant"
            },
            json={
                "model": "deepseek/deepseek-chat",
                "messages": messages_list  # <--- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"❌ Ошибка вызова DeepSeek API: {e}")
        # Добавляем детализацию ошибки для логов
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Ответ сервера: {e.response.text}")
        return "Извините, у нас временные трудности с AI. Пожалуйста, попробуйте позже."

# --- ОСНОВНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ ОТВЕТОВ (ФИНАЛЬНАЯ ВЕРСИЯ) ---
def generate_response(chat_id, user_message, is_test_mode=False):
    """Генерирует ответ с использованием RAG и памяти диалогов, создавая правильный массив messages."""
    start_time = time.time()
    facts_context, rag_metrics = get_facts_from_rag(user_message)
    history_list = get_conversation_history(chat_id)

    # --- СБОРКА МАССИВА MESSAGES ---
    messages = []
    
    # 1. Системный промпт
    messages.append({"role": "system", "content": BASE_PROMPT})
    
    # 2. История диалога (если есть)
    for line in history_list:
        if line.startswith("Пользователь:"):
            messages.append({"role": "user", "content": line.replace("Пользователь: ", "", 1)})
        elif line.startswith("Ассистент:"):
            messages.append({"role": "assistant", "content": line.replace("Ассистент: ", "", 1)})
            
    # 3. Финальный промпт от пользователя с контекстом
    user_final_prompt = f"Контекст из базы знаний:\n---\n{facts_context}\n---\n\nВопрос пользователя: {user_message}"
    messages.append({"role": "user", "content": user_final_prompt})

    try:
        llm_start = time.time()
        # Вызываем модель с ПРАВИЛЬНОЙ структурой
        ai_response = call_deepseek_model(messages)
        llm_time = time.time() - llm_start
        
        # Call-to-action (логика остается)
        if not is_test_mode and len(history_list) >= 10 and "пробный" not in ai_response.lower():
            base_url = os.environ.get('BASE_URL', 'http://localhost:5000')
            lesson_url = f"{base_url}/lesson?user_id={chat_id}"
            ai_response += f"\n\n🎯 Хотите увидеть нашу методику в действии? Попробуйте пробный урок: {lesson_url}"
        
        # Обновление истории (логика остается)
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

# --- ТЕСТОВЫЕ ВОПРОСЫ ---
TEST_QUESTIONS = [
    "Расскажи о школе Ukido", "Какие курсы вы предлагаете?", "Подробнее о курсе Юный Оратор",
    "Кто ведет этот курс?", "А сколько он стоит?", "Есть ли скидки?",
    "Какое оборудование нужно для занятий?", "А что с курсом Эмоциональный Компас?",
    "Какие результаты показывают ваши выпускники?", "На чем основана ваша методика?",
    "Какое расписание занятий?", "Можно ли прийти на пробный урок?",
    "Вернемся к Юному Оратору - сколько детей в группе?", "Какая миссия вашей школы?",
    "Подведем итог - что вы бы порекомендовали для ребенка 8 лет?"
]
latest_test_results = {"timestamp": None, "tests": [], "summary": {}}

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

@app.route('/test-rag')
def test_rag_system():
    global latest_test_results
    print("\n" + "="*60 + "\n🧪 НАЧАЛО ТЕСТИРОВАНИЯ С DEEPSEEK\n" + "="*60)
    test_chat_id = "test_user_session"
    if redis_available:
        try:
            redis_client.delete(f"history:{test_chat_id}", f"metadata:{test_chat_id}")
        except: pass
    if test_chat_id in fallback_memory: del fallback_memory[test_chat_id]
    
    total_test_start = time.time()
    latest_test_results = {"timestamp": datetime.now().isoformat(), "tests": [], "summary": {}}
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n🧪 === ТЕСТ №{i}/15 С DEEPSEEK ===")
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
        
        print(f"🤖 ОТВЕТ DEEPSEEK: {response}")
        print(f"✅ МЕТРИКИ: Общее время: {metrics['total_time']}с, Время LLM: {metrics['llm_time']}с, История: {metrics['history_length']} строк")
        print("="*50)
        time.sleep(1) # Пауза чтобы не превышать лимиты OpenRouter
    
    total_test_time = time.time() - total_test_start
    latest_test_results["summary"] = {
        "total_time": round(total_test_time, 2), "avg_time_per_question": round(total_test_time/15, 2),
        "redis_status": "available" if redis_available else "unavailable",
        "pinecone_status": "available" if pinecone_available else "unavailable",
        "questions_tested": len(TEST_QUESTIONS)
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
            <div class="response"><strong>🤖 Ответ DeepSeek:</strong><br>{test['response'].replace('\n', '<br>')}</div>
            <div class="metrics"><strong>⏱️ Общее время:</strong> {test['metrics']['total_time']}с | <strong>🧠 Время LLM:</strong> {test['metrics']['llm_time']}с | <strong>💾 История:</strong> {test['metrics']['history_length']} строк</div>
        </div>"""
    
    return render_template('results.html', summary=summary, tests_html=tests_html, redis_class=redis_class, pinecone_class=pinecone_class)

@app.route('/test-results-json')
def get_test_results_json():
    if not latest_test_results["tests"]:
        return {"error": "Тестирование еще не проводилось", "hint": "Запустите /test-rag сначала"}, 404
    return latest_test_results, 200

# Код для HubSpot webhook и запуска приложения остается без изменений...
# Я его скрыл для краткости, но он должен быть в вашем файле
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
    print("="*60 + f"\n🚀 ЗАПУСК UKIDO AI ASSISTANT С DEEPSEEK\n" + "="*60)
    app.run(debug=debug_mode, port=port, host='0.0.0.0')