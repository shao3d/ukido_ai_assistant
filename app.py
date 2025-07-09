# app.py
"""
✅ ФИНАЛЬНАЯ ВЕРСИЯ v13: Улучшен контроль юмора.
"""
import logging
import time
import threading
import atexit
import os
import re
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify, render_template
from typing import Dict, Any, Optional, List
import requests

from config import config
from telegram_bot import telegram_bot
from conversation import conversation_manager
from llamaindex_rag import llama_index_rag

from llama_index.core.llms import ChatMessage, MessageRole

try:
    from rag_debug_logger import rag_debug
    DEBUG_LOGGING_ENABLED = True
except ImportError:
    DEBUG_LOGGING_ENABLED = False
    print("Debug логирование отключено - rag_debug_logger не найден")

SENSITIVE_KEYWORDS = [
    # Здоровье и трагедии
    'болезнь', 'болеет', 'больница', 'врач', 'диагноз', 'лечение', 'операция',
    'смерть', 'умер', 'похоронили', 'трагедия', 'несчастье', 'беда',
    'инвалидность', 'травма', 'сломал', 'авария', 'скорая',
    
    # Семейные кризисы  
    'развод', 'расставание', 'ушел из семьи', 'бросил', 'алименты',
    'суд', 'опека', 'лишение прав',
    
    # Психологические проблемы
    'депрессия', 'суицид', 'тревожность', 'паника', 'истерика',
    'психолог', 'психиатр', 'таблетки', 'антидепрессанты',
    
    # Насилие и буллинг
    'избили', 'издеваются', 'травля', 'буллинг', 'насилие',
    'полиция', 'заявление', 'угрозы',
    
    # Особые потребности
    'аутизм', 'синдром', 'отставание', 'задержка развития',
    'особенный ребенок', 'инклюзия'
]
FACTUAL_KEYWORDS = [
    'цена', 'стоимость', 'сколько стоит', 'расписание', 'время', 'когда', 'адрес',
    'телефон', 'контакт', 'преподавател', 'квалификация', 'опыт', 'образование'
]

# Паттерны, которые ТРЕБУЮТ юмористического ответа
HUMOR_TRIGGER_PATTERNS = [
    # Вопросы о цене
    (r'почему.*(дорого|дёшево|цена|стоит)', 
     "Ценовой скептицизм - классика жанра!"),
    
    # Недоверие и подозрительность
    (r'(не верю|сомневаюсь|подозрительно|слишком хорошо)',
     "Здоровый скептицизм - признак мудрости!"),
    
    # Сарказм про идеальность
    (r'(везде.*лучшие|всё.*правильное|слишком.*умные|идеальные)',
     "Ирония про нашу 'идеальность'"),
    
    # Прямые обвинения
    (r'(обман|развод.*деньги|впариваете|навязываете)',
     "Обвинения требуют элегантного ответа"),
    
    # Сравнения с конкурентами
    (r'(у других|в другой школе|конкуренты.*лучше)',
     "Сравнения - повод для остроумия")
]

class ProductionConnectionPool:
    def __init__(self):
        self.session = requests.Session()
        self.logger = logging.getLogger(f"{__name__}.ConnectionPool")
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=3)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        atexit.register(self.cleanup)
        self.logger.info("🔗 Production connection pool готов")
    def post(self, *args, **kwargs): return self.session.post(*args, **kwargs)
    def get(self, *args, **kwargs): return self.session.get(*args, **kwargs)
    def cleanup(self): self.session.close(); self.logger.info("🔗 Connection pool закрыт")

class ProductionFastResponseCache:
    """✅ УЛУЧШЕНО: Кэш срабатывает только на короткие сообщения."""
    def __init__(self):
        self.fast_responses = {
            'цена': "Стоимость курсов от 6000 до 8000 грн в месяц. Первый урок бесплатный!",
            'стоимость': "Стоимость курсов от 6000 до 8000 грн в месяц. Первый урок бесплатный!",
            'сколько стоит': "Стоимость курсов от 6000 до 8000 грн в месяц. Первый урок бесплатный!",
            'пробный': "Отлично! Первый урок у нас бесплатный.",
            'урок': "У нас есть курсы soft-skills для детей 7-17 лет. Первый урок бесплатный!",
            'возраст': "Курсы для детей 7-17 лет, группы: 7-9, 10-12, 13-17 лет.",
            'время': "Расписание гибкое, подстраиваемся под удобное время.",
            'записаться': "Замечательно! Давайте запишем на бесплатный пробный урок."
        }
        self.logger = logging.getLogger(f"{__name__}.FastCache")
        self.logger.info("💨 Умный fast response cache (v2) готов")

    def get_fast_response(self, message: str, chat_id: str) -> Optional[str]:
        message_lower = message.lower().strip()
        
        if len(message_lower.split()) > 3:
            return None

        for keyword, response in self.fast_responses.items():
            if keyword in message_lower:
                self.logger.info(f"⚡️ Сработал быстрый ответ по ключу '{keyword}'")
                if keyword in ['пробный', 'записаться', 'урок']:
                    return f"{response}\n\n🔗 {config.get_lesson_url(user_id=chat_id)}"
                return response
        return None

class ProductionAIService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connection_pool = ProductionConnectionPool()
        self.fast_response_cache = ProductionFastResponseCache()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="UkidoAI")
        if not llama_index_rag: raise RuntimeError("LlamaIndex RAG failed to initialize")
        self.analyzer_llm = llama_index_rag.llm
        self.logger.info("🚀 ProductionAIService (v13) готов")

    def _should_use_humor(self, user_message: str, history: List[str]) -> bool:
        message_lower = user_message.lower()
        
        # 1. ЧЕРНЫЙ СПИСОК - никогда не шутить
        if any(keyword in message_lower for keyword in SENSITIVE_KEYWORDS):
            self.logger.info("🚫 Юмор ОТКЛЮЧЕН (чувствительная тема)")
            return False
        
        # 2. БЕЛЫЙ СПИСОК - всегда шутить  
        for pattern, reason in HUMOR_TRIGGER_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                self.logger.info(f"😄 Юмор ВКЛЮЧЕН ПРИНУДИТЕЛЬНО: {reason}")
                return True
        
        # 3. Фактические запросы - без юмора
        if any(keyword in message_lower for keyword in FACTUAL_KEYWORDS):
            self.logger.info("📊 Юмор ОТКЛЮЧЕН (фактический запрос)")
            return False
        
        # 4. Все остальное - умный анализ через LLM
        self.logger.info("🤔 Юмор НЕ ОПРЕДЕЛЕН. Запускаем умную проверку через LLM...")
        try:
            history_text = "\n".join(history[-4:])
            prompt = f"""Это история диалога:\n{history_text}\n\nЭто последнее сообщение пользователя:\n"{user_message}"\n\nПроанализируй ПОСЛЕДНЕЕ СООБЩЕНИЕ в контексте всей истории. К какой категории оно относится?\nОтветь ОДНИМ словом:\n- philosophical (размышления, мнения, "что если...")\n- emotional (пользователь делится чувствами, радостью, беспокойством)\n- general_talk (общий разговор, "как дела", "а что еще интересного")\n- factual (запрос конкретного факта, не покрытый ключевыми словами)\n\nКатегория:"""
            response = self.analyzer_llm.complete(prompt)
            category = response.text.strip().lower()
            if category in ['philosophical', 'emotional', 'general_talk']:
                self.logger.info(f"✅ Юмор РАЗРЕШЕН. Категория: {category}")
                return True
            else:
                self.logger.info(f"❌ Юмор ОТКЛЮЧЕН. Категория: {category}")
                return False
        except Exception as e:
            self.logger.warning(f"⚠️ Ошибка анализа юмора: {e}. По умолчанию - БЕЗ юмора")
            return False

    def process_user_message(self, user_message: str, chat_id: str) -> str:
        start_time = time.time()
        if DEBUG_LOGGING_ENABLED: rag_debug.start_session(chat_id, user_message)
        final_response = ""
        rag_metrics = {}
        try:
            fast_response = self.fast_response_cache.get_fast_response(user_message, chat_id)
            if fast_response:
                conversation_manager.update_conversation_history(chat_id, user_message, fast_response)
                final_response = fast_response
                self.logger.info(f"⚡️ Быстрый ответ для {chat_id}")
            else:
                current_state = conversation_manager.get_dialogue_state(chat_id)
                
                conversation_history = conversation_manager.get_conversation_history(chat_id)
                
                use_humor = self._should_use_humor(user_message, conversation_history)
                
                response_text, rag_metrics = llama_index_rag.search_and_answer(
                    query=user_message,
                    conversation_history=conversation_history,
                    current_state=current_state,
                    use_humor=use_humor
                )
                
                is_error_response = "ошибка" in response_text.lower()
                if not is_error_response:
                    processed_response = self._process_action_tokens(response_text, chat_id)
                    conversation_manager.update_conversation_history(chat_id, user_message, processed_response)
                    new_state = conversation_manager.analyze_message_for_state_transition(user_message, current_state)
                    if new_state != current_state:
                        conversation_manager.set_dialogue_state(chat_id, new_state)
                    final_response = processed_response
                else:
                    self.logger.warning(f"❗️ Обнаружен ответ с ошибкой, не сохраняем в историю: '{response_text}'")
                    final_response = response_text
            self.logger.info(f"✅ Ответ готов для {chat_id} за {(time.time() - start_time):.3f}s")
        except Exception as e:
            self.logger.error(f"💥 Критическая ошибка обработки сообщения: {e}", exc_info=True)
            final_response = "Извините, произошла временная техническая проблема."
        if DEBUG_LOGGING_ENABLED: rag_debug.log_final_response(final_response, rag_metrics.get('search_time', 0))
        return final_response

    def _process_action_tokens(self, response: str, chat_id: str) -> str:
        if "[ACTION:SEND_LESSON_LINK]" in response:
            lesson_link = config.get_lesson_url(user_id=chat_id)
            response = response.replace("[ACTION:SEND_LESSON_LINK]", f"\n\nОтлично! Вот ссылка для записи на бесплатный пробный урок:\n🔗 {lesson_link}").strip()
        return response

production_ai_service = ProductionAIService()
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def handle_telegram_webhook():
    if request.method == 'GET':
        return "Ukido AI Assistant (v13) is running! 🚀", 200
    try:
        update = request.get_json()
        if 'message' in update and 'text' in update['message']:
            message = update['message']
            chat_id = str(message['chat']['id'])
            user_message = message['text']
            threading.Thread(target=process_and_send, args=(user_message, chat_id)).start()
        return "OK", 200
    except Exception as e:
        logging.error(f"Webhook error: {e}")
        return "Error", 500

def process_and_send(user_message, chat_id):
    bot_response = production_ai_service.process_user_message(user_message, chat_id)
    telegram_bot.send_message(chat_id, bot_response)

@app.route('/test-message', methods=['POST'])
def test_message_endpoint():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        user_id = data.get('user_id', 'test_user')
        start_time = time.time()
        bot_response = production_ai_service.process_user_message(user_message, user_id)
        response_time = time.time() - start_time
        return {'bot_response': bot_response, 'response_time': response_time, 'status': 'success'}, 200
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/clear-memory', methods=['POST'])
def clear_memory():
    try:
        from conversation import conversation_manager
        conversation_manager.clear_all_conversations()
        return {"status": "success", "message": "Memory cleared"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

def parse_log_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: content = f.read()
        session_data = {'id': os.path.basename(filepath).replace('.log', ''), 'question': 'N/A', 'answer': 'N/A', 'chunks': [], 'metrics': {'time': 0, 'chunks_found': 0, 'max_score': 0, 'avg_score': 0}}
        q_match = re.search(r"❓ Question: (.*?)\n", content)
        if q_match: session_data['question'] = q_match.group(1).strip()
        ans_match = re.search(r"AI Response:\n-+\n(.*?)\n-+", content, re.DOTALL)
        if ans_match: session_data['answer'] = ans_match.group(1).strip()
        time_match = re.search(r"Generation Time: ([\d.]+)s", content)
        if time_match: session_data['metrics']['time'] = float(time_match.group(1))
        chunks_found_match = re.search(r"Chunks: \d+ → (\d+)", content)
        if chunks_found_match: session_data['metrics']['chunks_found'] = int(chunks_found_match.group(1))
        max_score_match = re.search(r"Scores: MAX=([\d.]+)", content)
        if max_score_match: session_data['metrics']['max_score'] = float(max_score_match.group(1))
        avg_score_match = re.search(r"AVG=([\d.]+)", content)
        if avg_score_match: session_data['metrics']['avg_score'] = float(avg_score_match.group(1))
        chunks = re.findall(r"\d+\. \[([\d.]+)\] (.*?)\.\.\.", content)
        session_data['chunks'] = [{'score': float(s), 'content': c} for s, c in chunks]
        return session_data
    except Exception: return None

@app.route('/dashboard')
def dashboard():
    log_dir = "rag_debug_logs"
    sessions = []
    if os.path.exists(log_dir):
        log_files = sorted([os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.log')], key=os.path.getmtime, reverse=True)
        for log_file in log_files:
            parsed_data = parse_log_file(log_file)
            if parsed_data: sessions.append(parsed_data)
    return render_template('dashboard.html', sessions=sessions)

@app.route('/save-log', methods=['POST'])
def save_log():
    try:
        data = request.get_json()
        filename = data.get('filename')
        if not filename or not isinstance(filename, str):
            return jsonify({"status": "error", "message": "Filename is missing or invalid"}), 400
        if not re.match(r'^[\w\-\.]+$', filename):
             return jsonify({"status": "error", "message": "Invalid filename format"}), 400
        success = rag_debug.save_full_log_to_file(filename)
        if success:
            return jsonify({"status": "success", "message": f"Log saved to {filename}"}), 200
        else:
            return jsonify({"status": "error", "message": "Failed to save log on server"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
