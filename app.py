import os
import requests
import google.generativeai as genai
# --- НОВЫЙ ИМПОРТ ---
from flask import Flask, request, render_template
# --------------------
from dotenv import load_dotenv
from pinecone import Pinecone

# --- НАСТРОЙКИ И ЗАГРУЗКА КЛЮЧЕЙ ---
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST_FACTS = os.getenv("PINECONE_HOST_FACTS")
PINECONE_HOST_STYLE = os.getenv("PINECONE_HOST_STYLE")
HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")

# ИСПРАВЛЕНИЕ 1: Добавлена проверка HUBSPOT_API_KEY в список обязательных переменных
# Раньше эта переменная загружалась, но не проверялась, что могло приводить к ошибкам
if not all([TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_HOST_FACTS, PINECONE_HOST_STYLE, HUBSPOT_API_KEY]):
    # Более информативная диагностика - показываем какие именно переменные отсутствуют
    required_vars = {
        'TELEGRAM_BOT_TOKEN': TELEGRAM_BOT_TOKEN,
        'GEMINI_API_KEY': GEMINI_API_KEY, 
        'PINECONE_API_KEY': PINECONE_API_KEY,
        'PINECONE_HOST_FACTS': PINECONE_HOST_FACTS,
        'PINECONE_HOST_STYLE': PINECONE_HOST_STYLE,
        'HUBSPOT_API_KEY': HUBSPOT_API_KEY
    }
    missing_vars = [name for name, value in required_vars.items() if not value]
    raise ValueError(f"Отсутствуют обязательные переменные: {', '.join(missing_vars)}")

# --- КОНФИГУРАЦИЯ КЛИЕНТОВ ---
genai.configure(api_key=GEMINI_API_KEY)
generation_model = genai.GenerativeModel('gemini-1.5-flash')
embedding_model = 'models/text-embedding-004'

# ИСПРАВЛЕНИЕ 2: Убрали раннюю инициализацию Pinecone соединений
# Раньше эти строки выполнялись при запуске приложения, что могло вызывать сбои в облаке
# если Pinecone серверы были недоступны в момент инициализации
# pc = Pinecone(api_key=PINECONE_API_KEY)
# index_facts = pc.Index(host=PINECONE_HOST_FACTS)
# index_style = pc.Index(host=PINECONE_HOST_STYLE)

def get_pinecone_indexes():
    """
    ИСПРАВЛЕНИЕ 2: Ленивая инициализация Pinecone соединений
    
    Вместо создания соединений при запуске приложения, мы создаем их только
    когда они действительно нужны. Это делает приложение более устойчивым
    к временным проблемам с сетью и ускоряет время запуска.
    
    Паттерн "ленивой инициализации" широко используется в облачных приложениях
    для повышения надежности и производительности.
    """
    if not hasattr(get_pinecone_indexes, 'initialized'):
        try:
            # Создаем соединения только при первом обращении
            get_pinecone_indexes.pc = Pinecone(api_key=PINECONE_API_KEY)
            get_pinecone_indexes.index_facts = get_pinecone_indexes.pc.Index(host=PINECONE_HOST_FACTS)
            get_pinecone_indexes.index_style = get_pinecone_indexes.pc.Index(host=PINECONE_HOST_STYLE)
            get_pinecone_indexes.initialized = True
            print("✅ Pinecone соединения инициализированы успешно")
        except Exception as e:
            print(f"❌ Ошибка инициализации Pinecone: {e}")
            # Не устанавливаем флаг initialized, чтобы можно было попробовать снова позже
            raise e
    
    return get_pinecone_indexes.index_facts, get_pinecone_indexes.index_style

# --- ПАМЯТЬ И СИСТЕМНАЯ РОЛЬ ---
conversation_history = {}
CONVERSATION_MEMORY_SIZE = 15
SYSTEM_PROMPT = """Ты — 'Ассистент Ukido', дружелюбный и эмпатичный помощник с "перчинкой" в общении.

# Твоя Роль и Стиль:
- Стиль общения: провокационный, образный, с яркими метафорами. Избегай банальностей.
- Обращайся к пользователям уважительно на "вы".
- Твоя главная задача — помогать родителям, основываясь на КОНТЕКСТЕ ФАКТОВ и ПРИМЕРАХ СТИЛЯ.
- Никогда не выдумывай факты. Если в контексте нет ответа, честно скажи, что не знаешь.

# ПРАВИЛА ДЛИНЫ ОТВЕТА (ОЧЕНЬ ВАЖНО):
- Твоя речь должна быть живой и динамичной по длине.
- На простые, конкретные вопросы (цена, расписание, да/нет) отвечай МАКСИМАЛЬНО КРАТКО, в 1-2 предложения.
- На более общие или творческие вопросы отвечай более развернуто, но старайся уложиться в 3-5 предложений.
- Не вываливай на пользователя всю информацию сразу. Лучше дай краткий ответ и предложи рассказать подробнее, например, спросив: "Хотите узнать детали?"

# ПРЕДЛОЖЕНИЕ ПРОБНОГО УРОКА:
- Если пользователь спрашивает о методиках, курсах, эффективности обучения или сомневается - предложи пробный урок.
- Используй фразу: "🎯 Хотите проверить на практике? Попробуйте наш 5-минутный пробный урок: [LESSON_LINK]"
- [LESSON_LINK] будет автоматически заменен на персональную ссылку.
"""

# --- ОСНОВНОЕ ПРИЛОЖЕНИЕ FLASK ---
app = Flask(__name__)

def send_telegram_message(chat_id, text):
    """
    Отправляет сообщение пользователю в Telegram
    
    УЛУЧШЕНИЕ: Добавлена проверка успешности отправки и retry механизм
    """
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
        
        # Небольшая пауза перед повторной попыткой
        if attempt < max_retries - 1:
            import time
            time.sleep(1)
    
    print(f"❌ Не удалось отправить сообщение пользователю {chat_id} после {max_retries} попыток")
    return False

def get_rag_context(prompt):
    """
    Получает релевантный контекст из векторных баз знаний Pinecone
    
    УЛУЧШЕНИЕ: Устойчивая к отказам архитектура с fallback стратегией
    Возвращает tuple (facts_context, style_context, rag_available)
    где rag_available указывает, работает ли RAG система
    """
    try:
        # Пытаемся получить соединения с Pinecone
        index_facts, index_style = get_pinecone_indexes()
        
        # Создаем эмбеддинг для поискового запроса
        query_embedding = genai.embed_content(model=embedding_model, content=prompt, task_type="RETRIEVAL_QUERY")['embedding']
        
        # Ищем релевантные факты о школе (топ-3 результата)
        facts_results = index_facts.query(vector=query_embedding, top_k=3, include_metadata=True)
        facts_context = "\n".join([match['metadata']['text'] for match in facts_results['matches']])
        
        # Ищем примеры стиля общения (топ-2 результата)
        style_results = index_style.query(vector=query_embedding, top_k=2, include_metadata=True)
        style_context = "\n".join([match['metadata']['text'] for match in style_results['matches']])
        
        print("✅ RAG система работает корректно")
        return facts_context, style_context, True
        
    except Exception as e:
        print(f"⚠️ RAG система временно недоступна: {e}")
        print("🔄 Переключаемся на базовый режим работы")
        
        # Fallback: возвращаем базовую информацию о школе
        fallback_facts = """Ukido - онлайн-школа soft skills для детей. 
        Мы предлагаем курсы по развитию коммуникативных навыков, эмоционального интеллекта и лидерства.
        Доступны пробные уроки. За подробной информацией обращайтесь к нашим консультантам."""
        
        fallback_style = """Отвечайте в дружелюбном, но профессиональном тоне. 
        Используйте простые объяснения и предлагайте практические решения."""
        
        return fallback_facts, fallback_style, False

def update_conversation_history(chat_id, user_message, ai_response):
    """Обновляет историю диалога для конкретного пользователя"""
    # Создаем историю для нового пользователя, если её еще нет
    if chat_id not in conversation_history:
        conversation_history[chat_id] = []
    
    # Добавляем новый обмен сообщениями в формате диалога
    conversation_history[chat_id].append(f"Пользователь: {user_message}")
    conversation_history[chat_id].append(f"Ассистент: {ai_response}")
    
    # Ограничиваем размер истории (сохраняем только последние N обменов)
    max_lines = CONVERSATION_MEMORY_SIZE * 2  # *2 потому что каждый обмен = 2 строки
    if len(conversation_history[chat_id]) > max_lines:
        # Удаляем самые старые сообщения, оставляя только последние
        conversation_history[chat_id] = conversation_history[chat_id][-max_lines:]

def get_gemini_response(chat_id, prompt):
    """
    Генерирует ответ AI с учетом контекста и истории диалога
    
    УЛУЧШЕНИЕ: Адаптивное поведение в зависимости от доступности RAG системы
    """
    # Получаем релевантный контекст из базы знаний с информацией о доступности
    facts_context, style_context, rag_available = get_rag_context(prompt)
    
    # Получаем историю предыдущих сообщений с этим пользователем
    history = conversation_history.get(chat_id, [])
    history_context = "\n".join(history)
    
    # Адаптируем системный промпт в зависимости от доступности RAG
    if rag_available:
        system_prompt = SYSTEM_PROMPT
        status_note = ""
    else:
        # Модифицированный промпт для работы без полной RAG системы
        system_prompt = """Ты — 'Ассистент Ukido', дружелюбный и эмпатичный помощник онлайн-школы soft skills для детей.

# Твоя роль:
- Помогай родителям с вопросами о развитии детей
- Используй дружелюбный, профессиональный тон
- Предлагай практические решения
- При необходимости подробной информации направляй к консультантам школы

# Важно:
- Отвечай кратко и по существу
- Если не знаешь точного ответа, честно скажи об этом
- Предлагай пробные уроки при вопросах о курсах"""
        
        status_note = "\n[СИСТЕМНОЕ УВЕДОМЛЕНИЕ: Работаю в базовом режиме. Для получения подробной информации рекомендую связаться с консультантами.]"
    
    # Формируем полный промпт для AI
    full_prompt = f"""{system_prompt}

[ИСТОРИЯ ДИАЛОГА]:
{history_context}

[КОНТЕКСТ] из базы фактов для [ОТВЕТА]:
{facts_context}

[ПРИМЕРЫ СТИЛЯ] для [ОТВЕТА]:
{style_context}

Пользователь: {prompt}
Ассистент:"""

    try:
        # Отправляем промпт в Gemini AI для генерации ответа
        response = generation_model.generate_content(full_prompt)
        ai_response = response.text.strip()
        
        # Добавляем системное уведомление если RAG недоступен (только для внутреннего логирования)
        if not rag_available:
            print(f"ℹ️ Ответ сгенерирован в базовом режиме для пользователя {chat_id}")
        
        # Проверяем, содержит ли ответ маркер для ссылки на урок
        if "[LESSON_LINK]" in ai_response:
            # ИСПРАВЛЕНИЕ 3: Более надежный подход к определению базового URL
            # Вместо попытки угадать переменную Railway, используем явную конфигурацию
            base_url = os.environ.get('BASE_URL')
            if base_url:
                # Продакшн URL из переменной окружения (настроим вручную в Railway)
                lesson_url = f"{base_url}/lesson?user_id={chat_id}"
            else:
                # Fallback для локальной разработки
                lesson_url = f"http://localhost:5000/lesson?user_id={chat_id}"
                print("⚠️ BASE_URL не настроен, используется localhost fallback")
            
            ai_response = ai_response.replace("[LESSON_LINK]", lesson_url)
        
        # Сохраняем этот обмен сообщениями в истории диалога
        update_conversation_history(chat_id, prompt, ai_response)
        
        return ai_response
        
    except Exception as e:
        print(f"❌ Критическая ошибка при работе с Gemini AI: {e}")
        # Последний fallback - возвращаем простое сообщение
        fallback_message = """Извините, в данный момент система испытывает технические трудности. 
        
Пожалуйста, попробуйте обратиться позже или свяжитесь с нашими консультантами напрямую для получения помощи по вопросам развития детей."""
        
        return fallback_message
    
def send_to_hubspot(user_data):
    """Отправляет данные пользователя в HubSpot CRM"""
    # URL для создания контактов в HubSpot API
    hubspot_url = "https://api.hubapi.com/crm/v3/objects/contacts"
    
    # Заголовки для HTTP-запроса с авторизацией
    headers = {
        "Authorization": f"Bearer {HUBSPOT_API_KEY}",  # Авторизация через наш API ключ
        "Content-Type": "application/json"             # Указываем, что отправляем JSON
    }
    
    # Подготавливаем данные в формате, который понимает HubSpot
    contact_data = {
        "properties": {
            "firstname": user_data["firstName"],
            "lastname": user_data["lastName"],
            "email": user_data["email"],
            "telegram_user_id": str(user_data.get("userId", ""))  # Новое поле для Telegram ID
        }
    }
    
    try:
        # Отправляем POST-запрос к HubSpot API
        response = requests.post(hubspot_url, headers=headers, json=contact_data)
        
        if response.status_code == 201:
            # HTTP код 201 означает "успешно создано"
            print("✅ Контакт успешно создан в HubSpot!")
            return True
        else:
            # Выводим информацию об ошибке для отладки
            print(f"❌ Ошибка HubSpot API: {response.status_code}")
            print(f"Ответ сервера: {response.text}")
            return False
            
    except Exception as e:
        # Обрабатываем любые технические ошибки (сеть, таймауты и т.д.)
        print(f"❌ Ошибка при отправке в HubSpot: {str(e)}")
        return False
    
def get_contact_from_hubspot(contact_id):
    """Получает информацию о контакте из HubSpot по ID"""
    url = f"https://api.hubapi.com/crm/v3/objects/contacts/{contact_id}"
    
    headers = {
        "Authorization": f"Bearer {HUBSPOT_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            contact_data = response.json()
            return contact_data['properties']
        else:
            print(f"❌ Ошибка получения контакта: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Ошибка API HubSpot: {e}")
        return None

# --- ФУНКЦИИ ДЛЯ FOLLOW-UP СООБЩЕНИЙ ---

def generate_first_follow_up_message(first_name):
    """Генерирует первое автоматическое сообщение (через 1 минуту после урока)"""
    return f"""👋 Привет, {first_name}!

Как впечатления от нашего урока об открытых вопросах? Удалось попробовать технику в реальной жизни?

🎯 Если понравилось, предлагаю записаться на полноценное пробное занятие с тренером Ukido. Это бесплатно и поможет лучше понять нашу методику.

Интересно?"""

def generate_second_follow_up_message(first_name):
    """Генерирует второе автоматическое сообщение (через 2 минуты после урока)"""
    return f"""🌟 {first_name}, не хочу быть навязчивым, но очень хочется узнать ваше мнение!

Искусство правильных вопросов действительно может изменить отношения в семье. Многие родители замечают улучшения уже после первых попыток применить технику.

💡 Если готовы погрузиться глубже, наши тренеры покажут еще больше эффективных методов развития soft skills у детей.

Запишем на бесплатную консультацию?"""

# --- МАРШРУТЫ FLASK ---

@app.route('/lesson')
def show_lesson_page():
    """Отображает страницу урока с возможностью персонализации"""
    # Получаем user_id из URL параметров (например: /lesson?user_id=123)
    user_id = request.args.get('user_id')
    # Передаем user_id в HTML шаблон для JavaScript
    return render_template('lesson.html', user_id=user_id)

@app.route('/', methods=['POST'])
def webhook():
    """Главный маршрут для обработки сообщений от Telegram"""
    update = request.get_json()
    
    # Проверяем, что это текстовое сообщение (не стикер, фото и т.д.)
    if "message" in update and "text" in update["message"]:
        chat_id = update["message"]["chat"]["id"]
        received_text = update["message"]["text"]
        
        # Генерируем ответ AI и отправляем пользователю
        ai_response = get_gemini_response(chat_id, received_text)
        send_telegram_message(chat_id, ai_response)

    return "ok", 200

@app.route('/submit-lesson-form', methods=['POST'])
def submit_lesson_form():
    """Обрабатывает данные формы урока и сохраняет их в HubSpot CRM"""
    # Получаем JSON-данные от браузера
    form_data = request.get_json()
    
    # Выводим полученные данные в консоль для контроля
    print("=== Получены данные формы ===")
    print(f"Имя: {form_data.get('firstName')}")
    print(f"Фамилия: {form_data.get('lastName')}")
    print(f"Email: {form_data.get('email')}")
    print("==========================")
    
    # Пытаемся отправить данные в HubSpot CRM
    hubspot_success = send_to_hubspot(form_data)
    
    if hubspot_success:
        print("🎉 Данные успешно сохранены в CRM!")
        return {"success": True, "message": "Данные сохранены в CRM"}, 200
    else:
        print("⚠️ Не удалось сохранить в CRM, но форма обработана")
        # Применяем принцип graceful degradation: даже если HubSpot недоступен,
        # пользователь все равно может пройти урок
        return {"success": True, "message": "Данные получены"}, 200

@app.route('/hubspot-webhook', methods=['POST'])
def hubspot_webhook():
    """Обрабатывает webhook'и от HubSpot для автоматических сообщений"""
    try:
        webhook_data = request.get_json()
        
        # --- ИСПРАВЛЕННАЯ ВЕРСИЯ: Извлекаем данные контакта напрямую из webhook ---
        contact_id = webhook_data.get('vid')  # HubSpot использует 'vid' как Contact ID
        
        if contact_id:
            # Извлекаем данные напрямую из webhook payload
            properties = webhook_data.get('properties', {})
            
            # Получаем имя и Telegram ID напрямую из данных webhook
            first_name = properties.get('firstname', {}).get('value', 'друг')
            telegram_id = properties.get('telegram_user_id', {}).get('value')
            
            print(f"🆔 Contact ID: {contact_id}")
            print(f"👋 Имя: {first_name}")
            print(f"📱 Telegram ID: {telegram_id}")
            
            # Определяем тип сообщения из URL параметров
            message_type = request.args.get('message_type', 'first_follow_up')
            print(f"📝 Тип сообщения: {message_type}")
            
            if telegram_id:
                # Генерируем сообщение в зависимости от типа
                if message_type == 'first_follow_up':
                    follow_up_message = generate_first_follow_up_message(first_name)
                    print(f"📤 Отправляю ПЕРВОЕ follow-up сообщение пользователю {first_name}")
                elif message_type == 'second_follow_up':
                    follow_up_message = generate_second_follow_up_message(first_name)
                    print(f"📤 Отправляю ВТОРОЕ follow-up сообщение пользователю {first_name}")
                else:
                    print(f"⚠️ Неизвестный тип сообщения: {message_type}")
                    return "Unknown message type", 400
                # Отправляем сообщение в Telegram
                send_telegram_message(telegram_id, follow_up_message)
                print(f"✅ Follow-up сообщение ({message_type}) отправлено пользователю {telegram_id}")
            else:
                print("❌ Не найден telegram_user_id для контакта")
                print(f"Доступные поля контакта: {list(properties.keys()) if properties else 'properties пустой'}")
            return "OK", 200
        else:
            print("❌ Не удалось извлечь contact_id из webhook данных")
            return "No contact ID found", 400
        
    except Exception as e:
        print(f"❌ Ошибка обработки webhook: {e}")
        return "Error", 500


# --- ТОЧКА ВХОДА В ПРОГРАММУ ---
if __name__ == '__main__':
    # Получаем порт от Railway через переменную окружения, 
    # если она есть, иначе используем 5000 для локальной разработки
    port = int(os.environ.get('PORT', 5000))
    
    # ИСПРАВЛЕНИЕ 4: Более надежное определение debug режима
    # Вместо полагания на FLASK_ENV, используем явную переменную DEBUG
    debug_mode = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print(f"🚀 Запуск Flask приложения на порту {port}")
    print(f"🔧 Debug режим: {'включен' if debug_mode else 'отключен'}")
    
    app.run(debug=debug_mode, port=port, host='0.0.0.0')