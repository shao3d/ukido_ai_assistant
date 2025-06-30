import os
import requests
import google.generativeai as genai
# --- НОВЫЙ ИМПОРТ ---
from flask import Flask, request, render_template
# --------------------
from dotenv import load_dotenv
from pinecone import Pinecone  # Восстановлен импорт Pinecone

# --- НАСТРОЙКИ И ЗАГРУЗКА КЛЮЧЕЙ ---
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST_FACTS = os.getenv("PINECONE_HOST_FACTS")
PINECONE_HOST_STYLE = os.getenv("PINECONE_HOST_STYLE")
HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")

if not all([TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_HOST_FACTS, PINECONE_HOST_STYLE, HUBSPOT_API_KEY]):
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

def get_pinecone_indexes():
    """
    Ленивая инициализация соединений с Pinecone (обновленная версия)
    """
    if not hasattr(get_pinecone_indexes, 'initialized'):
        try:
            print("🔍 Инициализируем Pinecone client...")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Динамически получаем актуальные host URLs
            print("🔍 Получаем актуальную информацию об индексах...")
            facts_description = pc.describe_index("ukido")
            style_description = pc.describe_index("ukido-style")
            
            print(f"🔍 Facts индекс host: {facts_description.host}")
            print(f"🔍 Style индекс host: {style_description.host}")
            
            # Создаем подключения с актуальными host URLs
            get_pinecone_indexes.pc = pc
            get_pinecone_indexes.index_facts = pc.Index(host=facts_description.host)
            get_pinecone_indexes.index_style = pc.Index(host=style_description.host)
            get_pinecone_indexes.initialized = True
            
            print("✅ Соединения с Pinecone инициализированы успешно")
            
        except Exception as e:
            print(f"❌ Ошибка инициализации Pinecone: {e}")
            raise e
    
    return get_pinecone_indexes.index_facts, get_pinecone_indexes.index_style

# --- ПАМЯТЬ И СИСТЕМНАЯ РОЛЬ ---
conversation_history = {}
CONVERSATION_MEMORY_SIZE = 15

# --- НОВЫЕ ФУНКЦИИ СИСТЕМЫ СТИЛЕЙ ЖВАНЕЦКОГО ---

def analyze_conversation_context(user_message, conversation_history):
    """
    Улучшенный анализатор контекста с более точными ключевыми словами
    """
    message_lower = user_message.lower()
    info_keywords = [
        'цена', 'стоимость', 'расписание', 'запись', 'когда', 'сколько', 'время', 
        'адрес', 'телефон', 'как записаться', 'пробный урок',
        'сколько детей', 'размер группы', 'продолжительность', 'длительность',
        'есть ли курсы', 'какие курсы', 'возраст', 'формат занятий'
    ]
    problem_keywords = [
        'не слушается', 'проблемы', 'сложно', 'трудно', 'замкнулся', 'не говорит',
        'агрессивный', 'стеснительный', 'не знаю что делать', 'помогите',
        'истерики', 'капризы', 'плачет', 'злится', 'боится', 'тревожный',
        'нет друзей', 'конфликты', 'дерется', 'не умеет общаться',
        'боится выступать', 'краснеет', 'заикается', 'молчит в классе'
    ]
    sensitive_keywords = [
        'развод', 'смерть', 'болезнь', 'депрессия', 'травма', 'горе',
        'потеря', 'расставание', 'больница', 'лечение', 'психолог'
    ]
    philosophical_keywords = [
        'как правильно', 'смысл', 'почему', 'зачем', 'что такое',
        'современные дети', 'поколение', 'в наше время', 'раньше было', 
        'невоспитанные', 'другие времена',
        'принципы воспитания', 'правильно воспитывать', 'методики', 'подходы',
        'зачем детям', 'нужны ли', 'важность', 'развиваться вместе'
    ]
    is_info = any(keyword in message_lower for keyword in info_keywords)
    is_problem = any(keyword in message_lower for keyword in problem_keywords)
    is_sensitive = any(keyword in message_lower for keyword in sensitive_keywords)
    is_philosophical = any(keyword in message_lower for keyword in philosophical_keywords)
    if is_sensitive:
        return "sensitive", "high"
    elif is_philosophical:
        return "philosophical", "low"  
    elif is_info and not is_problem:
        return "informational", "low"
    elif is_problem:
        return "consultational", "medium"
    else:
        return "general", "medium"

def should_add_school_bridge(user_message, ai_response, conversation_history):
    """
    КАРДИНАЛЬНО ПЕРЕРАБОТАННАЯ логика переходов к школе
    """
    message_lower = user_message.lower()
    info_keywords = ['цена', 'стоимость', 'расписание', 'запись', 'сколько детей', 'возраст']
    if any(keyword in message_lower for keyword in info_keywords):
        return False
    sensitive_keywords = ['развод', 'смерть', 'болезнь', 'депрессия', 'травма']
    if any(keyword in message_lower for keyword in sensitive_keywords):
        return False
    strong_action_signals = [
        'что делать', 'как помочь', 'где научиться', 'как развить', 
        'нужна помощь', 'что посоветуете', 'как быть'
    ]
    interest_signals = [
        'интересно', 'расскажите больше', 'хочу узнать', 'подробнее',
        'а если', 'а как', 'а что'
    ]
    conversation_length = len(conversation_history)
    is_established_conversation = conversation_length >= 4
    has_strong_action = any(signal in message_lower for signal in strong_action_signals)
    has_interest = any(signal in message_lower for signal in interest_signals)
    return is_established_conversation and (has_strong_action or has_interest)

def generate_school_bridge(user_message, ai_response):
    """
    Улучшенная логика выбора курсов на основе анализа тестирования
    """
    text_to_analyze = (user_message + " " + ai_response).lower()
    speech_keywords = [
        'стеснительный', 'боится говорить', 'боится выступать', 'молчит',
        'краснеет', 'заикается', 'не отвечает у доски', 'тихий голос'
    ]
    emotion_keywords = [
        'истерики', 'капризы', 'злится', 'плачет', 'эмоции', 'чувства',
        'не умеет контролировать', 'вспышки гнева', 'тревожный'
    ]
    leadership_keywords = [
        'лидер', 'лидерство', 'команда', 'ответственность', 'проект', 
        'инициатива', 'организовать', 'руководить'
    ]
    if any(word in text_to_analyze for word in speech_keywords):
        course = "Юный Оратор"
        bridge = f"\n\nИменно уверенная речь и преодоление стеснительности — основа курса \"{course}\". Дети учатся выражать мысли так, чтобы их действительно слышали."
    elif any(word in text_to_analyze for word in emotion_keywords):
        course = "Эмоциональный Компас"  
        bridge = f"\n\nТакие эмоциональные навыки мы развиваем в курсе \"{course}\". Дети учатся не подавлять чувства, а управлять ими — это принципиально разные вещи."
    elif any(word in text_to_analyze for word in leadership_keywords):
        course = "Капитан Проектов"
        bridge = f"\n\nЭти лидерские качества мы целенаправленно развиваем в курсе \"{course}\". Дети учатся не командовать, а вдохновлять — почувствуйте разницу."
    else:
        bridge = "\n\nИменно такие жизненные навыки мы развиваем в Ukido. Не абстрактную теорию, а практические умения для ежедневной жизни."
    bridge += " 🎯 Хотите увидеть нашу методику в действии? Попробуйте пробный урок: [LESSON_LINK]"
    return bridge

def create_adaptive_system_prompt(request_type, delicacy_level):
    """
    КАРДИНАЛЬНО УЛУЧШЕННЫЕ системные промпты с встроенными примерами стиля
    """
    base_prompt = """Ты — 'Ассистент Ukido', мудрый помощник с острым умом и тонким юмором.

# Твоя уникальная манера общения:
- Используй парадоксы и неожиданные повороты мысли
- Создавай запоминающиеся наблюдения о жизни и воспитании  
- Обращайся к пользователям уважительно на \"вы\"
- НИКОГДА не упоминай чужих имен — говори от себя
- Добавляй смайлики умеренно, только после удачных наблюдений

# Примеры твоего стиля (используй как вдохновение):
- "Воспитание — единственная профессия, где все считают себя экспертами, потому что когда-то были детьми"
- "Мы учим детей жизни, сами еще в ней разбираясь"  
- "Детские капризы — это не бунт против вас, это сигнал о том, что ребенок еще не научился выражать потребности словами"

# Основные правила:
- Опирайся ТОЛЬКО на факты из контекста
- Если не знаешь — честно признавайся"""
    if request_type == "informational":
        adaptive_part = """
# ИНФОРМАЦИОННЫЙ РЕЖИМ:
- Максимальная краткость: 1-2 предложения
- Фокус на точных фактах о школе Ukido
- Легкий штрих остроумия для запоминаемости
- Пример тональности: "Пробный урок бесплатный — лучшие вещи в жизни действительно бесплатны, правда, за них потом приходится платить временем на размышления 😉"
"""
    elif request_type == "sensitive":
        adaptive_part = """
# ДЕЛИКАТНЫЙ РЕЖИМ:
- Максимальная эмпатия и бережность в словах
- НИКАКОЙ иронии или сарказма
- Фокус на поддержке и понимании
- Длина ответа: 3-4 предложения
- Пример тональности: "В трудные моменты дети особенно нуждаются в нашем понимании. Не всегда нужно что-то говорить — иногда достаточно просто быть рядом"
"""
    elif request_type == "consultational":
        adaptive_part = """
# КОНСУЛЬТАЦИОННЫЙ РЕЖИМ:
- Сочетай мудрые наблюдения с практическими советами
- Используй метафоры применительно к воспитанию
- Длина ответа: 3-5 предложений
- Пример тональности: "Гиперактивность ребенка — это не его недостаток, а особенность характера, как темперамент у художника. Нужно не ломать, а направлять эту энергию в конструктивное русло"
"""
    elif request_type == "philosophical":
        adaptive_part = """
# ФИЛОСОФСКИЙ РЕЖИМ (УСИЛЕННЫЙ):
- Используй всю глубину мудрых наблюдений о воспитании и жизни
- Создавай неожиданные повороты мысли и парадоксы
- Длина ответа: 4-7 предложений, варьируй для естественности
- Можешь "разгуляться" философски, но оставайся в теме воспитания

Примеры философских оборотов:
- "Современное родительство — это марафон, где все бегут с разной скоростью, но почему-то сравнивают результаты на одном километре"
- "Мы покупаем детям игрушки, чтобы они играли сами, а потом удивляемся, что они не умеют играть с нами"
- "Каждый родитель мечтает дать ребенку то, чего не было у него самого, забывая дать то, что у него было"
"""
    else:
        adaptive_part = """
# УНИВЕРСАЛЬНЫЙ РЕЖИМ:
- Баланс между мудростью и практичностью
- Используй остроумие умеренно, в зависимости от ситуации
- Длина ответа: 2-4 предложения
- Будь полезным, но не скучным
"""
    return base_prompt + adaptive_part

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

def get_enhanced_gemini_response(chat_id, prompt):
    """
    Улучшенная версия генерации ответов с интеллектуальным применением стиля
    ГЛАВНАЯ ФУНКЦИЯ с системой анализа стилей Жванецкого
    """
    
    # Анализируем контекст запроса
    request_type, delicacy_level = analyze_conversation_context(
        prompt, 
        conversation_history.get(chat_id, [])
    )
    
    # Получаем релевантный контекст из базы знаний (используем существующую функцию)
    facts_context, style_context, rag_available = get_rag_context(prompt)
    
    # Получаем историю диалога
    history = conversation_history.get(chat_id, [])
    history_context = "\n".join(history)
    
    # Создаем адаптированный системный промпт
    adaptive_system_prompt = create_adaptive_system_prompt(request_type, delicacy_level)
    
    # Формируем полный промпт для AI
    full_prompt = f"""{adaptive_system_prompt}

[ИСТОРИЯ ДИАЛОГА]:
{history_context}

[КОНТЕКСТ] из базы фактов для [ОТВЕТА]:
{facts_context}

[ПРИМЕРЫ СТИЛЯ] для [ОТВЕТА]:
{style_context}

[ТИП ЗАПРОСА]: {request_type}
[УРОВЕНЬ ДЕЛИКАТНОСТИ]: {delicacy_level}

Пользователь: {prompt}
Ассистент:"""

    try:
        # Генерируем основной ответ
        response = generation_model.generate_content(full_prompt)
        ai_response = response.text.strip()
        
        # Определяем, нужно ли добавить переход к школе
        if should_add_school_bridge(prompt, ai_response, history):
            bridge = generate_school_bridge(prompt, ai_response)
            ai_response += bridge
        
        # Обрабатываем ссылку на урок (существующая логика)
        if "[LESSON_LINK]" in ai_response:
            base_url = os.environ.get('BASE_URL')
            if base_url:
                lesson_url = f"{base_url}/lesson?user_id={chat_id}"
            else:
                lesson_url = f"http://localhost:5000/lesson?user_id={chat_id}"
                print("⚠️ BASE_URL не настроен, используется localhost fallback")
            
            ai_response = ai_response.replace("[LESSON_LINK]", lesson_url)
        
        # Сохраняем этот обмен сообщениями в истории диалога
        update_conversation_history(chat_id, prompt, ai_response)
        
        # Выводим информацию о режиме работы для отладки
        print(f"✅ Ответ сгенерирован: тип={request_type}, деликатность={delicacy_level}")
        
        return ai_response
        
    except Exception as e:
        print(f"❌ Критическая ошибка при работе с Gemini AI: {e}")
        return """Извините, в данный момент система испытывает технические трудности. 
        
Пожалуйста, попробуйте обратиться позже или свяжитесь с нашими консультантами напрямую для получения помощи по вопросам развития детей."""
    
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
        
        # Генерируем ответ AI и отправляем пользователю - ОБНОВЛЕНО!
        ai_response = get_enhanced_gemini_response(chat_id, received_text)
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