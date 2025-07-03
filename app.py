import os
import requests
import google.generativeai as genai
from flask import Flask, request, render_template
from dotenv import load_dotenv
from pinecone import Pinecone
import redis
import time
import json
import hashlib
import threading
from datetime import datetime
from functools import lru_cache, wraps
from typing import Optional, Tuple, Dict, Any

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

# --- ПРОИЗВОДИТЕЛЬНОСТЬ: КЕШИРОВАНИЕ И ОПТИМИЗАЦИЯ ---
RAG_CACHE = {}  # ИСПРАВЛЕНО: Кеш для RAG запросов
RAG_CACHE_TTL = 3600  # 1 час
MAX_CACHE_SIZE = 1000

def cache_rag_result(query: str, result: tuple, ttl: int = RAG_CACHE_TTL) -> None:
    """НОВОЕ: Кеширование результатов RAG"""
    if len(RAG_CACHE) >= MAX_CACHE_SIZE:
        # Очищаем старые записи
        oldest_key = min(RAG_CACHE.keys(), key=lambda k: RAG_CACHE[k]['timestamp'])
        del RAG_CACHE[oldest_key]
    
    RAG_CACHE[query] = {
        'result': result,
        'timestamp': time.time(),
        'ttl': ttl
    }

def get_cached_rag_result(query: str) -> Optional[tuple]:
    """НОВОЕ: Получение кешированного результата RAG"""
    if query in RAG_CACHE:
        cached = RAG_CACHE[query]
        if time.time() - cached['timestamp'] < cached['ttl']:
            return cached['result']
        else:
            del RAG_CACHE[query]
    return None

# --- БЕЗОПАСНОСТЬ: САНИТИЗАЦИЯ ВХОДНЫХ ДАННЫХ ---
def sanitize_user_input(user_input: str) -> str:
    """НОВОЕ: Санитизация пользовательского ввода от prompt injection"""
    if not isinstance(user_input, str):
        return ""
    
    # Удаляем потенциально опасные паттерны
    dangerous_patterns = [
        "ignore previous instructions",
        "system:",
        "assistant:",
        "###",
        "---",
        "[INST]",
        "</INST>",
        "<system>",
        "</system>"
    ]
    
    sanitized = user_input
    for pattern in dangerous_patterns:
        sanitized = sanitized.replace(pattern, "")
    
    # Ограничиваем длину
    if len(sanitized) > 2000:
        sanitized = sanitized[:2000]
    
    # Убираем множественные переносы строк
    sanitized = '\n'.join(line.strip() for line in sanitized.split('\n') if line.strip())
    
    return sanitized

# --- ЛОГИРОВАНИЕ И МОНИТОРИНГ ---
import logging

# НОВОЕ: Настройка продакшн логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """НОВОЕ: Мониторинг производительности системы"""
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0,
            'rag_cache_hits': 0,
            'rag_cache_misses': 0
        }
        self.lock = threading.Lock()
    
    def record_request(self, success: bool, response_time: float):
        with self.lock:
            self.metrics['total_requests'] += 1
            if success:
                self.metrics['successful_requests'] += 1
            else:
                self.metrics['failed_requests'] += 1
            
            # Обновляем среднее время ответа
            self.metrics['avg_response_time'] = (
                (self.metrics['avg_response_time'] * (self.metrics['total_requests'] - 1) + response_time) 
                / self.metrics['total_requests']
            )
    
    def record_cache_hit(self):
        with self.lock:
            self.metrics['rag_cache_hits'] += 1
    
    def record_cache_miss(self):
        with self.lock:
            self.metrics['rag_cache_misses'] += 1
    
    def get_metrics(self) -> dict:
        with self.lock:
            return self.metrics.copy()

performance_monitor = PerformanceMonitor()

# --- КОНФИГУРАЦИЯ КЛИЕНТОВ ---
genai.configure(api_key=GEMINI_API_KEY)
embedding_model = 'models/text-embedding-004'

# --- ИНИЦИАЛИЗАЦИЯ REDIS С ОБРАБОТКОЙ ОШИБОК ---
redis_client = None
redis_available = False
redis_lock = threading.Lock()  # НОВОЕ: Thread safety для Redis

def init_redis():
    """Инициализация Redis с обработкой ошибок"""
    global redis_client, redis_available
    try:
        logger.info("Инициализируем Redis client...")
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        redis_available = True
        logger.info("Redis подключен успешно")
    except Exception as e:
        redis_available = False
        logger.warning(f"Redis недоступен: {e}")
        logger.info("Система будет работать без постоянной памяти диалогов")

init_redis()

# ИСПРАВЛЕНО: Thread-safe ограничение размера fallback_memory
MAX_FALLBACK_USERS = 1000
fallback_memory = {}
fallback_memory_lock = threading.Lock()

def cleanup_fallback_memory():
    """Очистка fallback памяти от старых записей"""
    with fallback_memory_lock:
        if len(fallback_memory) > MAX_FALLBACK_USERS:
            # Удаляем половину самых старых записей
            old_keys = list(fallback_memory.keys())[:len(fallback_memory)//2]
            for key in old_keys:
                del fallback_memory[key]
            logger.info(f"Очищена fallback память: удалено {len(old_keys)} старых записей")

# --- ОПТИМИЗИРОВАННАЯ ИНИЦИАЛИЗАЦИЯ PINECONE ---
pinecone_index = None
pinecone_available = False
pinecone_lock = threading.Lock()  # НОВОЕ: Thread safety

@lru_cache(maxsize=1)  # НОВОЕ: Кеширование инициализации
def get_pinecone_index():
    """Ленивая инициализация Pinecone с thread safety"""
    global pinecone_index, pinecone_available
    
    with pinecone_lock:
        if pinecone_index is not None:
            return pinecone_index
            
        try:
            logger.info("Инициализируем Pinecone client...")
            pc = Pinecone(api_key=PINECONE_API_KEY)
            try:
                facts_description = pc.describe_index("ukido")
                pinecone_index = pc.Index(host=facts_description.host)
                logger.info("Pinecone подключен динамически")
            except Exception as dynamic_error:
                logger.warning(f"Динамическое подключение не удалось: {dynamic_error}")
                pinecone_index = pc.Index(host=PINECONE_HOST_FACTS)
                logger.info("Pinecone подключен через прямой host")
            pinecone_available = True
            return pinecone_index
        except Exception as e:
            pinecone_available = False
            logger.error(f"Pinecone полностью недоступен: {e}")
            return None

# --- НАСТРОЙКИ ПАМЯТИ ДИАЛОГОВ ---
CONVERSATION_MEMORY_SIZE = 15
CONVERSATION_EXPIRATION_SECONDS = 3600

# --- ЖЕЛЕЗНЫЙ КУЛАК: УЛУЧШЕННАЯ МАШИНА СОСТОЯНИЙ ---
DIALOGUE_STATES = {
    'greeting': 'Приветствие и первое знакомство',
    'problem_solving': 'Решение проблем ребенка, консультирование', 
    'fact_finding': 'Поиск информации о курсах, ценах, расписании',
    'closing': 'Готовность к записи на пробный урок'
}

# ИСПРАВЛЕНО: Улучшенные ключевые слова для анализа состояний
STATE_KEYWORDS = {
    'problem_solving': ['проблем', 'сложно', 'трудно', 'застенчив', 'боится', 'не слушается', 'агрессивн', 'замкн', 'помогите'],
    'fact_finding': ['цена', 'стоимость', 'расписание', 'время', 'когда', 'сколько', 'преподаватель', 'группа', 'возраст'],
    'closing': ['записат', 'попробова', 'хочу', 'готов', 'решил', 'интересно', 'согласен', 'давайте']
}

def normalize_chat_id(chat_id) -> str:
    """ИСПРАВЛЕНО: Нормализация chat_id для консистентности"""
    if chat_id is None:
        return ""
    return str(chat_id)

def validate_user_message(user_message: str) -> bool:
    """ИСПРАВЛЕНО: Валидация входящих сообщений"""
    if not user_message or not isinstance(user_message, str):
        return False
    if len(user_message.strip()) == 0:
        return False
    if len(user_message) > 2000:  # Защита от слишком длинных сообщений
        return False
    return True

# НОВОЕ: Thread-safe управление состояниями с блокировкой race conditions
state_locks = {}  # chat_id -> Lock
state_locks_lock = threading.Lock()

def get_chat_lock(chat_id: str) -> threading.Lock:
    """НОВОЕ: Получение блокировки для конкретного чата"""
    with state_locks_lock:
        if chat_id not in state_locks:
            state_locks[chat_id] = threading.Lock()
        return state_locks[chat_id]

def get_dialogue_state(chat_id: str) -> str:
    """ИСПРАВЛЕНО: Thread-safe получение состояния диалога"""
    chat_id = normalize_chat_id(chat_id)
    if not chat_id:
        return 'greeting'
    
    chat_lock = get_chat_lock(chat_id)
    with chat_lock:
        if redis_available:
            try:
                with redis_lock:
                    state_key = f"state:{chat_id}"
                    state = redis_client.get(state_key)
                    return state if state else 'greeting'
            except Exception as e:
                logger.warning(f"Ошибка чтения состояния из Redis: {e}")
        
        # ИСПРАВЛЕНО: Умный fallback на основе содержания истории
        with fallback_memory_lock:
            history = fallback_memory.get(chat_id, [])
            
        if len(history) == 0:
            return 'greeting'
        
        # Анализируем последние сообщения для определения состояния
        recent_messages = ' '.join(history[-4:]).lower()
        
        for state, keywords in STATE_KEYWORDS.items():
            if any(keyword in recent_messages for keyword in keywords):
                return state
        
        # Fallback - логика переходов по умолчанию
        if len(history) < 4:
            return 'greeting'
        elif len(history) < 8:
            return 'fact_finding'
        else:
            return 'problem_solving'

def update_dialogue_state(chat_id: str, new_state: str):
    """ИСПРАВЛЕНО: Thread-safe обновление состояния диалога"""
    chat_id = normalize_chat_id(chat_id)
    if not chat_id or new_state not in DIALOGUE_STATES:
        logger.warning(f"Невалидные параметры состояния: chat_id={chat_id}, state={new_state}")
        return
    
    chat_lock = get_chat_lock(chat_id)
    with chat_lock:
        if redis_available:
            try:
                with redis_lock:
                    state_key = f"state:{chat_id}"
                    redis_client.set(state_key, new_state, ex=CONVERSATION_EXPIRATION_SECONDS)
                    logger.info(f"Состояние диалога обновлено: {chat_id} -> {new_state}")
            except Exception as e:
                logger.error(f"Ошибка записи состояния в Redis: {e}")

def analyze_and_determine_next_state(user_message: str, ai_response: str, current_state: str) -> str:
    """ИСПРАВЛЕНО: Оптимизированный анализ состояний с кешированием"""
    
    if not validate_user_message(user_message):
        return current_state
    
    # КРИТИЧЕСКИ ВАЖНО: Прямые запросы урока → сразу closing
    direct_lesson_keywords = ["пробн", "бесплатн", "попробова", "записат", "хочу урок", "дайте ссылку"]
    if any(word in user_message.lower() for word in direct_lesson_keywords):
        logger.info("Детектирован прямой запрос урока → state='closing'")
        return 'closing'
    
    # Быстрый анализ по ключевым словам (избегаем лишних API вызовов)
    message_lower = user_message.lower()
    for state, keywords in STATE_KEYWORDS.items():
        if any(keyword in message_lower for keyword in keywords):
            logger.info(f"Детектировано состояние по ключевым словам: {state}")
            return state
    
    # ОПТИМИЗАЦИЯ: Используем LLM анализ только если ключевые слова не сработали
    # И только для длинных сообщений (короткие обычно не меняют состояние)
    if len(user_message.split()) < 5:
        return current_state
    
    # Кешируем анализ состояний
    analysis_key = hashlib.md5(f"{user_message}{current_state}".encode()).hexdigest()
    cached_state = get_cached_rag_result(f"state_{analysis_key}")
    if cached_state:
        performance_monitor.record_cache_hit()
        return cached_state[0]
    
    performance_monitor.record_cache_miss()
    
    analysis_prompt = f"""Определи ОДНО состояние диалога. Отвечай только названием состояния.

Текущее: {current_state}
Вопрос: "{user_message}"

Состояния:
greeting - первое знакомство
problem_solving - обсуждение проблем ребенка  
fact_finding - вопросы о ценах/расписании
closing - готовность записаться

Ответ (только название):"""

    try:
        response = call_gpt4o_mini(analysis_prompt)
        new_state = response.strip().lower()
        
        if new_state in DIALOGUE_STATES:
            cache_rag_result(f"state_{analysis_key}", (new_state,), ttl=1800)  # Кешируем на 30 минут
            return new_state
        else:
            logger.warning(f"LLM вернул некорректное состояние: {response}")
    except Exception as e:
        logger.error(f"Ошибка анализа состояния: {e}")
    
    # Fallback - логика переходов по умолчанию
    if current_state == 'greeting':
        return 'fact_finding'
    elif current_state == 'fact_finding' and len(user_message.split()) > 10:
        return 'problem_solving'
    else:
        return current_state

# --- ЖЕЛЕЗНЫЙ КУЛАК: ОПТИМИЗИРОВАННОЕ ПЕРЕПИСЫВАНИЕ ЗАПРОСОВ ---
def rewrite_query_for_rag(history_list: list, user_message: str) -> str:
    """ИСПРАВЛЕНО: Кешированное переписывание запросов"""
    
    if not validate_user_message(user_message):
        return user_message
    
    # Если истории мало, ищем как есть
    if not history_list or len(history_list) < 2:
        return user_message
    
    # Если вопрос уже подробный, не переписываем
    if len(user_message.split()) > 3:
        return user_message
    
    # Кешируем результаты переписывания
    rewrite_key = hashlib.md5(f"{user_message}{''.join(history_list[-3:])}".encode()).hexdigest()
    cached_rewrite = get_cached_rag_result(f"rewrite_{rewrite_key}")
    if cached_rewrite:
        performance_monitor.record_cache_hit()
        return cached_rewrite[0]
    
    performance_monitor.record_cache_miss()
    
    # ИСПРАВЛЕНО: Берем только пользовательские сообщения для контекста
    user_history = [msg for msg in history_list if msg.startswith("Пользователь:")][-3:]
    
    if not user_history:
        return user_message
    
    rewrite_prompt = f"""Перепиши короткий вопрос в полный запрос для поиска о школе Ukido.

Предыдущие вопросы:
{chr(10).join(user_history)}

Вопрос: "{user_message}"

Переписанный запрос (максимум 10 слов):"""

    try:
        rewritten = call_gpt4o_mini(rewrite_prompt).strip()
        
        # ИСПРАВЛЕНО: Защита от некорректного переписывания
        if len(rewritten.split()) > 15 or len(rewritten) > 100:
            logger.warning("Переписанный запрос слишком длинный, используем оригинал")
            return user_message
        
        cache_rag_result(f"rewrite_{rewrite_key}", (rewritten,), ttl=3600)
        logger.info(f"Запрос переписан: '{user_message}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.error(f"Ошибка переписывания запроса: {e}")
        return user_message

# --- УЛУЧШЕННЫЕ ПРОМПТЫ С МАШИНОЙ СОСТОЯНИЙ ---
@lru_cache(maxsize=4)  # НОВОЕ: Кеширование промптов
def get_enhanced_base_prompt(current_state: str) -> str:
    """ИСПРАВЛЕНО: Кешированные промпты с учетом состояния диалога"""
    
    state_instructions = {
        'greeting': """
ТЕКУЩАЯ ФАЗА: ПРИВЕТСТВИЕ И ЗНАКОМСТВО
- Будь дружелюбным, но не навязчивым
- Узнай потребности и проблемы
- НЕ предлагай урок на этом этапе
- Фокусируйся на понимании ситуации""",
        
        'problem_solving': """
ТЕКУЩАЯ ФАЗА: РЕШЕНИЕ ПРОБЛЕМ И КОНСУЛЬТИРОВАНИЕ  
- Максимальная тактичность и эмпатия
- Давай практические советы
- Показывай экспертность
- НЕ предлагай урок, пока проблема не проработана
- Если решение требует курса, можешь упомянуть методику""",
        
        'fact_finding': """
ТЕКУЩАЯ ФАЗА: ПОИСК ФАКТИЧЕСКОЙ ИНФОРМАЦИИ
- Давай точные факты из RAG
- Отвечай конкретно на вопросы о ценах, расписании, преподавателях
- Используй легкую иронию для иллюстрации
- НЕ предлагай урок, пока не ответил на все вопросы""",
        
        'closing': """
ТЕКУЩАЯ ФАЗА: ГОТОВНОСТЬ К ЗАПИСИ
- Родитель готов к следующему шагу
- ОБЯЗАТЕЛЬНО предложи пробный урок
- ВСЕГДА вставляй токен [ACTION:SEND_LESSON_LINK] в ответ
- Пример: "Записывайтесь: [ACTION:SEND_LESSON_LINK]"
- Будь уверенным, но не давящим"""
    }

    return f"""Ты — AI-ассистент школы soft skills "Ukido". Твоя роль — мудрый, ироничный наставник с мировоззрением и стилем Михаила Жванецкого. Ты говоришь парадоксами и жизненными наблюдениями. Твоя главная задача — помочь родителю разобраться, а не продать любой ценой.

{state_instructions.get(current_state, state_instructions['greeting'])}

### ЖЕЛЕЗНЫЙ КУЛАК: ПРАВИЛА ПРЕДЛОЖЕНИЯ УРОКА
- Предлагай урок ТОЛЬКО если текущее состояние = 'closing'
- При предложении урока ВСЕГДА используй токен [ACTION:SEND_LESSON_LINK]
- НЕ генерируй ссылки самостоятельно - только токен!
- Пример: "Записывайтесь: [ACTION:SEND_LESSON_LINK]"

### ГЛАВНЫЕ ПРАВИЛА ПОВЕДЕНИЯ
1.  **ПРАВИЛО ВЫСШЕГО ПРИОРИТЕТА: ДЕЛИКАТНЫЕ ТЕМЫ.** Если родитель говорит о проблемах ребенка (застенчивость, страхи, конфликты, неуверенность), **ПОЛНОСТЬЮ ОТКЛЮЧИ ИРОНИЮ И ШУТКИ**. Твой тон — максимально тактичный и поддерживающий.
2.  **СТРОГО СЛЕДУЙ ФАКТАМ ИЗ RAG:** Не придумывай имена, цены или детали, которых нет в предоставленном контексте.
3.  **ИСПОЛЬЗУЙ КОНКРЕТИКУ:** Вместо "проблемы с поведением" используй "кричит, хлопает дверью, уроки не делает".

### УРОВНИ ИРОНИИ И СТИЛЯ ЖВАНЕЦКОГО
- **Уровень 1: Информационный запрос** - Ноль иронии. Только факты из RAG.
- **Уровень 2: Консультационный запрос** - Легкая ирония, жизненные аналогии.
- **Уровень 3: Философский вопрос** - Полный стиль Жванецкого: парадоксы, накопление деталей.

### ТАБУ (ЗАПРЕЩЕНО)
- Канцеляризмы ("С радостью расскажу", "Данный курс")
- Пустые фразы ("Спасибо за ваш вопрос")
- Высокопарные метафоры ("морское плавание", "корабли")
- Генерировать ссылки самостоятельно (только токен [ACTION:SEND_LESSON_LINK])
"""

# --- THREAD-SAFE ФУНКЦИИ УПРАВЛЕНИЯ ПАМЯТЬЮ ДИАЛОГОВ ---

def get_conversation_history(chat_id: str) -> list:
    """ИСПРАВЛЕНО: Thread-safe получение истории диалога"""
    chat_id = normalize_chat_id(chat_id)
    if not chat_id:
        return []
    
    chat_lock = get_chat_lock(chat_id)
    with chat_lock:
        if redis_available:
            try:
                with redis_lock:
                    history_key = f"history:{chat_id}"
                    return redis_client.lrange(history_key, 0, -1)[::-1]
            except Exception as e:
                logger.warning(f"Ошибка чтения из Redis: {e}")
        
        with fallback_memory_lock:
            return fallback_memory.get(chat_id, [])

def update_conversation_history(chat_id: str, user_message: str, ai_response: str):
    """ИСПРАВЛЕНО: Thread-safe обновление истории диалога"""
    chat_id = normalize_chat_id(chat_id)
    if not chat_id or not validate_user_message(user_message):
        return
    
    # ИСПРАВЛЕНО: Очищаем токены действий перед сохранением в историю
    clean_response = ai_response.replace("[ACTION:SEND_LESSON_LINK]", "[ССЫЛКА_НА_УРОК]")
    
    chat_lock = get_chat_lock(chat_id)
    with chat_lock:
        timestamp = datetime.now().isoformat()
        
        if redis_available:
            try:
                with redis_lock:
                    history_key = f"history:{chat_id}"
                    metadata_key = f"metadata:{chat_id}"
                    pipe = redis_client.pipeline()
                    pipe.lpush(history_key, f"Ассистент: {clean_response}")
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
                logger.error(f"Ошибка записи в Redis: {e}")
                update_fallback_memory(chat_id, user_message, clean_response)
        else:
            update_fallback_memory(chat_id, user_message, clean_response)

def update_fallback_memory(chat_id: str, user_message: str, ai_response: str):
    """ИСПРАВЛЕНО: Thread-safe обновление fallback-памяти"""
    chat_id = normalize_chat_id(chat_id)
    if not chat_id:
        return
    
    with fallback_memory_lock:
        if chat_id not in fallback_memory:
            fallback_memory[chat_id] = []
            
        fallback_memory[chat_id].append(f"Пользователь: {user_message}")
        fallback_memory[chat_id].append(f"Ассистент: {ai_response}")
        
        max_lines = CONVERSATION_MEMORY_SIZE * 2
        if len(fallback_memory[chat_id]) > max_lines:
            fallback_memory[chat_id] = fallback_memory[chat_id][-max_lines:]
    
    # Периодическая очистка fallback памяти
    cleanup_fallback_memory()

# --- ОПТИМИЗИРОВАННЫЕ ФУНКЦИИ RAG СИСТЕМЫ ---

def get_relevance_description(score: float) -> str:
    if score >= 0.9: return "Отличное совпадение"
    if score >= 0.7: return "Хорошее совпадение"
    if score >= 0.5: return "Среднее совпадение"
    return "Слабое совпадение"

def get_speed_description(seconds: float) -> str:
    if seconds < 2: return "Быстро"
    if seconds <= 5: return "Нормально"
    return "Медленно"

def get_facts_from_rag(user_message: str, chat_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """ИСПРАВЛЕНО: Оптимизированная RAG система с кешированием и мониторингом"""
    
    if not validate_user_message(user_message):
        return "", {"error": "Невалидное сообщение", "fallback_used": True}
    
    search_start = time.time()
    chat_id = normalize_chat_id(chat_id) if chat_id else None
    
    # ОПТИМИЗАЦИЯ: Переписывание запроса для улучшения поиска
    final_query = user_message
    if chat_id:
        try:
            history_list = get_conversation_history(chat_id)
            final_query = rewrite_query_for_rag(history_list, user_message)
        except Exception as e:
            logger.error(f"Ошибка переписывания запроса: {e}")
            final_query = user_message
    
    # НОВОЕ: Проверяем кеш перед обращением к Pinecone
    cache_key = hashlib.md5(final_query.encode()).hexdigest()
    cached_result = get_cached_rag_result(cache_key)
    if cached_result:
        performance_monitor.record_cache_hit()
        logger.info(f"Кеш попадание для запроса: {final_query}")
        return cached_result
    
    performance_monitor.record_cache_miss()
    
    try:
        index = get_pinecone_index()
        if not index:
            fallback_result = ("", {"error": "Pinecone недоступен", "fallback_used": True, "search_time": time.time() - search_start})
            return fallback_result
        
        embedding_start = time.time()
        query_embedding = genai.embed_content(
            model=embedding_model, 
            content=final_query, 
            task_type="RETRIEVAL_QUERY"
        )['embedding']
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
            "speed_desc": get_speed_description(total_time), "success": True,
            "original_query": user_message, "rewritten_query": final_query
        }
        
        # НОВОЕ: Кешируем результат
        result = (context, metrics)
        cache_rag_result(cache_key, result)
        
        return result
        
    except Exception as e:
        total_time = time.time() - search_start
        logger.error(f"Ошибка RAG системы: {e}")
        fallback_context = "Ukido - онлайн-школа soft skills для детей. Курсы: 'Юный Оратор' (7-10 лет, 6000 грн/мес), 'Эмоциональный Компас' (9-12 лет, 7500 грн/мес), 'Капитан Проектов' (11-14 лет, 8000 грн/мес)."
        metrics = {
            "search_time": round(total_time, 2), "error": str(e), 
            "fallback_used": True, "chunks_found": 1, "success": False
        }
        return fallback_context, metrics

# --- ФУНКЦИЯ ДЛЯ ВЫЗОВА GPT-4o MINI ЧЕРЕЗ OPENROUTER ---
@lru_cache(maxsize=100)  # НОВОЕ: Кеширование часто используемых промптов
def call_gpt4o_mini(prompt: str) -> str:
    """ИСПРАВЛЕНО: Оптимизированный вызов GPT-4o mini с кешированием"""
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            logger.error(f"OpenRouter API error: {response.status_code}")
            return "Извините, временная проблема с генерацией ответа."
            
    except requests.exceptions.Timeout:
        logger.error("Таймаут при вызове GPT-4o mini")
        return "Извините, временная проблема с генерацией ответа."
    except Exception as e:
        logger.error(f"Ошибка GPT-4o Mini: {e}")
        return "Извините, временная проблема с генерацией ответа."

# --- ЖЕЛЕЗНЫЙ КУЛАК: ОСНОВНАЯ ФУНКЦИЯ ГЕНЕРАЦИИ ОТВЕТОВ ---
def generate_response(chat_id: str, user_message: str, is_test_mode: bool = False) -> Tuple[str, Dict[str, Any]]:
    """ИСПРАВЛЕНО: Оптимизированная генерация ответов с мониторингом производительности"""
    
    start_time = time.time()
    success = False
    
    try:
        # ИСПРАВЛЕНО: Валидация и санитизация входных данных
        if not validate_user_message(user_message):
            return "Пожалуйста, отправьте текстовое сообщение.", {"error": "Invalid input"}
        
        user_message = sanitize_user_input(user_message)
        chat_id = normalize_chat_id(chat_id)
        
        # ИСПРАВЛЕНО: Thread-safe получение состояния с блокировкой race conditions
        current_state = get_dialogue_state(chat_id)
        logger.info(f"Текущее состояние диалога {chat_id}: {current_state}")
        
        # Получаем контекст из RAG с переписыванием запросов
        facts_context, rag_metrics = get_facts_from_rag(user_message, chat_id)
        history_list = get_conversation_history(chat_id)

        # Формируем промпт с учетом состояния диалога
        enhanced_prompt = get_enhanced_base_prompt(current_state)
        history_context = "\n".join(history_list) if history_list else "Это начало диалога."
        
        full_prompt = f"""{enhanced_prompt}

История диалога:
{history_context}

Информация о школе Ukido из базы знаний:
{facts_context}

Пользователь: {user_message}
Ассистент:"""

        llm_start = time.time()
        ai_response = call_gpt4o_mini(full_prompt)
        llm_time = time.time() - llm_start

        # ЖЕЛЕЗНЫЙ КУЛАК: Обработка токенов действий (ЕДИНСТВЕННЫЙ способ генерации ссылок)
        base_url = os.environ.get('BASE_URL', 'https://ukidoaiassistant-production.up.railway.app')
        if "[ACTION:SEND_LESSON_LINK]" in ai_response:
            lesson_url = f"{base_url}/lesson?user_id={chat_id}"
            ai_response = ai_response.replace("[ACTION:SEND_LESSON_LINK]", lesson_url)
            logger.info("Обработан токен [ACTION:SEND_LESSON_LINK] - ссылка вставлена")

        # ЖЕЛЕЗНЫЙ КУЛАК: Thread-safe анализ и обновление состояния диалога
        new_state = analyze_and_determine_next_state(user_message, ai_response, current_state)
        
        if not is_test_mode and chat_id:
            update_dialogue_state(chat_id, new_state)
            update_conversation_history(chat_id, user_message, ai_response)

        total_time = time.time() - start_time
        success = True

        response_metrics = {
            "total_time": round(total_time, 2), "llm_time": round(llm_time, 2),
            "rag_metrics": rag_metrics, "history_length": len(history_list),
            "redis_available": redis_available, "pinecone_available": pinecone_available,
            "dialogue_state_transition": f"{current_state} → {new_state}",
            "iron_fist_active": True,
            "cache_hit": rag_metrics.get('cache_hit', False)
        }
        
        return ai_response, response_metrics

    except Exception as e:
        logger.error(f"Критическая ошибка в generate_response: {e}")
        error_response = "Извините, возникла техническая проблема. Пожалуйста, попробуйте перефразировать вопрос."
        return error_response, {"error": str(e), "total_time": time.time() - start_time}
    
    finally:
        # НОВОЕ: Записываем метрики производительности
        total_time = time.time() - start_time
        performance_monitor.record_request(success, total_time)

# --- ЖЕЛЕЗНЫЙ КУЛАК: ТЕСТОВЫЕ ВОПРОСЫ ---
IRON_FIST_TEST_QUESTIONS = [
    # Тест машины состояний
    "Привет, расскажите о вашей школе.",  # greeting
    "Какие у вас есть курсы?",  # fact_finding
    "Мой ребенок очень застенчивый, боится отвечать у доски.",  # problem_solving
    "Хочу записаться на пробный урок!",  # closing (прямой запрос)
    
    # Тест переписывания запросов
    "Сколько это стоит?",  # должно переписаться с учетом контекста
    "А кто его ведет?",  # анафора "его"
    "Какое расписание?",  # требует конкретизации
    
    # Тест токенов действий
    "Как попробовать ваш курс?",  # должен вызвать [ACTION:SEND_LESSON_LINK]
    "Дайте ссылку на пробный урок",  # прямой запрос ссылки
]

# --- ТЕСТОВЫЕ ФУНКЦИИ С ИЗОЛЯЦИЕЙ ---
latest_test_results = {"timestamp": None, "tests": [], "summary": {}}
test_results_lock = threading.Lock()  # НОВОЕ: Thread safety

def update_test_conversation_history(chat_id: str, user_message: str, ai_response: str):
    """ИСПРАВЛЕНО: Thread-safe обновление тестовой памяти"""
    test_key = f"test_{chat_id}"
    with fallback_memory_lock:
        if test_key not in fallback_memory:
            fallback_memory[test_key] = []
            
        fallback_memory[test_key].append(f"Пользователь: {user_message}")
        fallback_memory[test_key].append(f"Ассистент: {ai_response}")
        
        max_lines = CONVERSATION_MEMORY_SIZE * 2
        if len(fallback_memory[test_key]) > max_lines:
            fallback_memory[test_key] = fallback_memory[test_key][-max_lines:]

def get_test_conversation_history(chat_id: str) -> list:
    """ИСПРАВЛЕНО: Thread-safe получение тестовой истории"""
    test_key = f"test_{chat_id}"
    with fallback_memory_lock:
        return fallback_memory.get(test_key, [])

def clear_test_data(test_chat_id: str):
    """ИСПРАВЛЕНО: Thread-safe очистка тестовых данных"""
    test_key = f"test_{test_chat_id}"
    with fallback_memory_lock:
        if test_key in fallback_memory:
            del fallback_memory[test_key]
    
    if redis_available:
        try:
            with redis_lock:
                redis_client.delete(f"state:{test_chat_id}", f"history:{test_chat_id}")
        except Exception as e:
            logger.warning(f"Ошибка очистки Redis данных: {e}")

# --- HUBSPOT ИНТЕГРАЦИЯ ---
def send_to_hubspot(user_data: Dict[str, Any]) -> bool:
    """ИСПРАВЛЕНО: Улучшенная отправка данных в HubSpot с валидацией"""
    
    # Валидация данных
    required_fields = ['firstName', 'lastName', 'email']
    for field in required_fields:
        if not user_data.get(field):
            logger.error(f"Отсутствует обязательное поле: {field}")
            return False
    
    hubspot_url = "https://api.hubapi.com/crm/v3/objects/contacts"
    headers = {"Authorization": f"Bearer {HUBSPOT_API_KEY}", "Content-Type": "application/json"}
    contact_data = {"properties": {
        "firstname": str(user_data["firstName"])[:50],  # Ограничение длины
        "lastname": str(user_data["lastName"])[:50],
        "email": str(user_data["email"])[:100],
        "telegram_user_id": str(user_data.get("userId", ""))[:20]
    }}
    
    try:
        response = requests.post(
            hubspot_url, 
            headers=headers, 
            json=contact_data,
            timeout=10
        )
        
        if response.status_code == 201:
            logger.info("Контакт успешно создан в HubSpot!")
            return True
        else:
            logger.error(f"Ошибка HubSpot API: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error("Таймаут при отправке в HubSpot")
        return False
    except Exception as e:
        logger.error(f"Ошибка при отправке в HubSpot: {str(e)}")
        return False

# --- ОСНОВНОЕ ПРИЛОЖЕНИЕ FLASK ---
app = Flask(__name__)

def send_telegram_message(chat_id: str, text: str) -> bool:
    """ИСПРАВЛЕНО: Улучшенная отправка сообщений в Telegram"""
    
    if not text or len(text.strip()) == 0:
        logger.warning("Попытка отправить пустое сообщение")
        return False
    
    # Ограничение длины сообщения для Telegram
    if len(text) > 4096:
        text = text[:4093] + "..."
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": normalize_chat_id(chat_id), "text": text}
    
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"Сообщение успешно отправлено пользователю {chat_id}")
                return True
            logger.warning(f"Telegram API вернул статус {response.status_code}: {response.text}")
        except Exception as e:
            logger.error(f"Ошибка при отправке сообщения в Telegram (попытка {attempt + 1}/3): {e}")
        if attempt < 2: 
            time.sleep(1)
    return False

# --- МАРШРУТЫ FLASK ---
@app.route('/lesson')
def show_lesson_page():
    user_id = request.args.get('user_id')
    return render_template('lesson.html', user_id=user_id)

@app.route('/metrics')
def get_metrics():
    """НОВОЕ: Эндпоинт для мониторинга производительности"""
    metrics = performance_monitor.get_metrics()
    return {
        "status": "healthy",
        "performance": metrics,
        "redis_available": redis_available,
        "pinecone_available": pinecone_available,
        "cache_size": len(RAG_CACHE)
    }

@app.route('/', methods=['POST'])
def webhook():
    """ИСПРАВЛЕНО: Главный webhook с улучшенной обработкой ошибок и мониторингом"""
    try:
        update = request.get_json()
        
        if not update or "message" not in update:
            return "OK", 200
            
        message = update["message"]
        if "text" not in message or "chat" not in message:
            return "OK", 200
            
        chat_id = message["chat"]["id"]
        received_text = message["text"]
        
        # Генерируем ответ и отправляем
        ai_response, metrics = generate_response(chat_id, received_text)
        success = send_telegram_message(chat_id, ai_response)
        
        if success:
            transition = metrics.get('dialogue_state_transition', 'N/A')
            cache_status = "HIT" if metrics.get('cache_hit', False) else "MISS"
            logger.info(f"Обработан запрос от {chat_id}: {metrics.get('total_time', 'N/A')}с, переход: {transition}, кеш: {cache_status}")
        
        return "OK", 200
        
    except Exception as e:
        logger.error(f"Ошибка в webhook: {e}")
        return "Error", 500

@app.route('/test-iron-fist')
def test_iron_fist_system():
    """ИСПРАВЛЕНО: Thread-safe тестирование системы Железный Кулак"""
    global latest_test_results
    
    with test_results_lock:
        logger.info("НАЧАЛО ТЕСТИРОВАНИЯ СИСТЕМЫ ЖЕЛЕЗНЫЙ КУЛАК")
        
        test_chat_id = "iron_fist_test_session"
        
        # Полная очистка тестовых данных
        clear_test_data(test_chat_id)
        
        total_test_start = time.time()
        latest_test_results = {"timestamp": datetime.now().isoformat(), "tests": [], "summary": {}}
        
        for i, question in enumerate(IRON_FIST_TEST_QUESTIONS, 1):
            logger.info(f"ТЕСТ №{i}/{len(IRON_FIST_TEST_QUESTIONS)}: {question}")
            
            response, metrics = generate_response(test_chat_id, question, is_test_mode=True)
            rag_metrics = metrics.get('rag_metrics', {})
            
            test_result = {
                "question_number": i, "question": question, "response": response,
                "metrics": metrics, "rag_success": rag_metrics.get('success', False),
                "search_time": rag_metrics.get('search_time', 0),
                "chunks_found": rag_metrics.get('chunks_found', 0),
                "best_score": rag_metrics.get('best_score', 0),
                "relevance_desc": rag_metrics.get('relevance_desc', 'Неизвестно'),
                "state_transition": metrics.get('dialogue_state_transition', 'N/A'),
                "query_rewrite": rag_metrics.get('rewritten_query', question) != question,
                "action_tokens_used": "[ACTION:SEND_LESSON_LINK]" in response or "lesson?user_id=" in response,
                "cache_hit": metrics.get('cache_hit', False)
            }
            latest_test_results["tests"].append(test_result)
            
            # Обновляем тестовую память для следующих вопросов
            update_test_conversation_history(test_chat_id, question, response)
            time.sleep(0.5)
        
        # Полная очистка после тестов
        clear_test_data(test_chat_id)
        
        total_test_time = time.time() - total_test_start
        latest_test_results["summary"] = {
            "total_time": round(total_test_time, 2), 
            "avg_time_per_question": round(total_test_time/len(IRON_FIST_TEST_QUESTIONS), 2),
            "redis_status": "available" if redis_available else "unavailable",
            "pinecone_status": "available" if pinecone_available else "unavailable",
            "questions_tested": len(IRON_FIST_TEST_QUESTIONS),
            "iron_fist_features_tested": ["state_machine", "query_rewriting", "action_tokens", "caching", "thread_safety"],
            "performance_metrics": performance_monitor.get_metrics()
        }
        
        logger.info(f"ТЕСТИРОВАНИЕ ЗАВЕРШЕНО! Общее время: {total_test_time:.1f}с")
        
        return latest_test_results, 200

@app.route('/iron-fist-results')
def show_iron_fist_results():
    """Отображение результатов тестирования Железного Кулака"""
    with test_results_lock:
        if not latest_test_results["tests"]:
            return "<h1>Тестирование Железного Кулака не проводилось. Запустите <a href='/test-iron-fist'>/test-iron-fist</a></h1>"
        
        summary = latest_test_results['summary']
        tests_html = ""
        
        for test in latest_test_results["tests"]:
            rag_class = "good" if test["rag_success"] else "error"
            state_class = "good" if "→" in test["state_transition"] else "warning"
            rewrite_class = "good" if test["query_rewrite"] else "warning"
            action_class = "good" if test["action_tokens_used"] else "warning"
            cache_class = "good" if test.get("cache_hit", False) else "warning"
            
            tests_html += f"""
            <div class="test">
                <div class="question">❓ Вопрос №{test['question_number']}: {test['question']}</div>
                <div class="metrics">
                    <strong>🎯 Состояние:</strong> <span class="{state_class}">{test['state_transition']}</span> | 
                    <strong>🔄 Переписан:</strong> <span class="{rewrite_class}">{'Да' if test['query_rewrite'] else 'Нет'}</span> | 
                    <strong>🔗 Токены:</strong> <span class="{action_class}">{'Да' if test['action_tokens_used'] else 'Нет'}</span> |
                    <strong>💾 Кеш:</strong> <span class="{cache_class}">{'HIT' if test.get('cache_hit', False) else 'MISS'}</span> |
                    <strong>🔍 RAG:</strong> <span class="{rag_class}">{'✅' if test["rag_success"] else '❌'}</span>
                </div>
                <div class="response"><strong>🤖 Ответ:</strong><br>{test['response'].replace('\n', '<br>')}</div>
                <div class="metrics"><strong>⏱️ Время:</strong> {test['metrics']['total_time']}с</div>
            </div>"""
        
        redis_class = "good" if summary['redis_status'] == 'available' else 'error'
        pinecone_class = "good" if summary['pinecone_status'] == 'available' else 'error'
        
        html = f"""
        <!DOCTYPE html>
        <html><head><title>Результаты тестирования Железного Кулака</title>
        <style>
            body {{ font-family: Arial; margin: 20px; }}
            .summary {{ background: #f0f8ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
            .test {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 8px; }}
            .question {{ font-weight: bold; color: #2c3e50; margin-bottom: 8px; }}
            .response {{ background: white; padding: 10px; border-left: 4px solid #e74c3c; margin: 10px 0; }}
            .metrics {{ color: #7f8c8d; font-size: 0.9em; margin: 5px 0; }}
            .good {{ color: #27ae60; font-weight: bold; }}
            .warning {{ color: #f39c12; font-weight: bold; }}
            .error {{ color: #e74c3c; font-weight: bold; }}
        </style></head>
        <body>
        <h1>🥊 Результаты тестирования системы "Железный Кулак" (Экспертная версия)</h1>
        <div class="summary">
            <h3>📊 Суммарная статистика</h3>
            <strong>Время тестирования:</strong> {summary['total_time']}с<br>
            <strong>Среднее время на вопрос:</strong> {summary['avg_time_per_question']}с<br>
            <strong>Вопросов протестировано:</strong> {summary['questions_tested']}<br>
            <strong>Протестированные функции:</strong> {', '.join(summary['iron_fist_features_tested'])}<br>
            <strong>Redis:</strong> <span class="{redis_class}">{summary['redis_status']}</span><br>
            <strong>Pinecone:</strong> <span class="{pinecone_class}">{summary['pinecone_status']}</span><br>
            <strong>🚀 Общий успех запросов:</strong> {summary.get('performance_metrics', {}).get('successful_requests', 0)}<br>
            <strong>⚡ Кеш попаданий:</strong> {summary.get('performance_metrics', {}).get('rag_cache_hits', 0)}
        </div>
        {tests_html}
        </body></html>
        """
        return html

@app.route('/submit-lesson-form', methods=['POST'])
def submit_lesson_form():
    """ИСПРАВЛЕНО: Thread-safe обработка формы урока"""
    try:
        form_data = request.get_json()
        if not form_data:
            return {"success": False, "error": "Нет данных"}, 400
            
        logger.info(f"Получены данные формы: {form_data.get('firstName')} {form_data.get('lastName')}")
        hubspot_success = send_to_hubspot(form_data)
        return {"success": hubspot_success}, 200
        
    except Exception as e:
        logger.error(f"Ошибка обработки формы: {e}")
        return {"success": False, "error": str(e)}, 500

@app.route('/hubspot-webhook', methods=['POST'])
def hubspot_webhook():
    """
    ЖЕЛЕЗНЫЙ КУЛАК: Обновленные follow-up сообщения в стиле Жванецкого
    """
    try:
        webhook_data = request.get_json()
        if not webhook_data:
            return "No data", 400
            
        properties = webhook_data.get('properties', {})
        first_name = properties.get('firstname', {}).get('value', 'наш друг')
        telegram_id = properties.get('telegram_user_id', {}).get('value')
        message_type = request.args.get('message_type', 'first_follow_up')
        
        if telegram_id:
            # ЖЕЛЕЗНЫЙ КУЛАК: Новые тексты в стиле Жванецкого
            message_generators = {
                'first_follow_up': f"Ну что, {first_name}, как впечатления? Говорят, после хорошего спектакля хочется обсудить. А после нашего пробного урока — хочется либо записаться, либо забыть. Надеюсь, у вас первый вариант. Если что, мы тут, на связи.",
                'second_follow_up': f"{first_name}, это снова мы. Не то чтобы мы скучали, но тишина в эфире — это как антракт, затянувшийся на два акта. Если вы еще думаете, это хорошо. Думать полезно. Но пока мы думаем, дети растут. Может, все-таки решимся на разговор?"
            }
            message_to_send = message_generators.get(message_type)
            
            if message_to_send:
                success = send_telegram_message(telegram_id, message_to_send)
                if success:
                    logger.info(f"Follow-up '{message_type}' в стиле Жванецкого отправлено {telegram_id}")
                else:
                    logger.error(f"Не удалось отправить follow-up {telegram_id}")
            else:
                logger.warning(f"Неизвестный тип сообщения: {message_type}")
        else:
            logger.error("Не найден telegram_user_id для контакта в HubSpot")
            
        return "OK", 200
        
    except Exception as e:
        logger.error(f"Ошибка обработки HubSpot webhook: {e}")
        return "Error", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info("="*60)
    logger.info("🥊 ЗАПУСК UKIDO AI ASSISTANT С ЖЕЛЕЗНЫМ КУЛАКОМ (ЭКСПЕРТНАЯ ВЕРСИЯ)")
    logger.info("="*60)
    logger.info("🎯 Активированы функции:")
    logger.info("   - Машина состояний диалога")
    logger.info("   - Переписывание запросов для RAG")
    logger.info("   - Токены действий для надежных ссылок")
    logger.info("   - Обновленные follow-up сообщения")
    logger.info("   - Thread-safe операции")
    logger.info("   - Кеширование и оптимизация производительности")
    logger.info("   - Санитизация входных данных")
    logger.info("   - Мониторинг производительности")
    logger.info("   - Race condition protection")
    logger.info("="*60)
    
    app.run(debug=debug_mode, port=port, host='0.0.0.0')