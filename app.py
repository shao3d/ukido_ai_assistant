# app.py
"""
✅ ФИНАЛЬНАЯ ВЕРСИЯ v3: RAG-оркестратор с Машиной Состояний.
- Получает и передает состояние диалога в LlamaIndex.
- Определяет смену состояния на основе ключевых слов.
- Обрабатывает токен [ACTION:SEND_LESSON_LINK].
"""

import logging
import time
import threading
import atexit
import os
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request
from typing import Dict, Any, Optional, List
import requests

from config import config
from telegram_bot import telegram_bot
from conversation import conversation_manager
from hubspot_client import hubspot_client
from llamaindex_rag import llama_index_rag

try:
    from rag_debug_logger import rag_debug
    DEBUG_LOGGING_ENABLED = True
except ImportError:
    DEBUG_LOGGING_ENABLED = False
    print("Debug логирование отключено - rag_debug_logger не найден")


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
    def __init__(self):
        self.fast_responses = {
            'цена': "Стоимость курсов от 6000 до 8000 грн в месяц. Первый урок бесплатный!",
            'стоимость': "Стоимость курсов от 6000 до 8000 грн в месяц. Первый урок бесплатный!",
            'сколько стоит': "Стоимость курсов от 6000 до 8000 грн в месяц. Первый урок бесплатный!",
            'пробный': "Отлично! Первый урок у нас бесплатный.",
            'урок': "У нас есть курсы soft-skills для детей 7-17 лет. Первый урок бесплатный!",
            'возраст': "Курсы для детей 7-17 лет, группы: 7-9, 10-12, 13-17 лет.",
            'время': "Расписание гибкое, подстраиваемся под удобное время.",
            'записаться': "Замечательно! Давайте запишем на бесплатный пробный урок.",
        }
        self.logger = logging.getLogger(f"{__name__}.FastCache")
        self.logger.info("💨 Fast response cache готов")
    def get_fast_response(self, message: str, chat_id: str) -> Optional[str]:
        message_lower = message.lower().strip()
        for keyword, response in self.fast_responses.items():
            if keyword in message_lower:
                if keyword in ['пробный', 'записаться']:
                    lesson_link = config.get_lesson_url(user_id=chat_id)
                    return f"{response}\n\n🔗 {lesson_link}"
                return response
        return None


class ProductionAIService:
    """
    ✅ Production-ready AI сервис v3 с Машиной Состояний.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connection_pool = ProductionConnectionPool()
        self.fast_response_cache = ProductionFastResponseCache()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="UkidoAI")
        self.performance_stats = {'total_requests': 0, 'fast_responses': 0, 'avg_response_time': 0}
        self.stats_lock = threading.Lock()
        
        if not llama_index_rag:
            raise RuntimeError("LlamaIndex RAG failed to initialize")
        
        self.logger.info("🚀 ProductionAIService (v3, с Машиной Состояний) готов")

    def process_user_message(self, user_message: str, chat_id: str) -> str:
        start_time = time.time()
        
        if DEBUG_LOGGING_ENABLED:
            rag_debug.start_session(chat_id, user_message)
        
        try:
            # 1. Проверяем fast response
            fast_response = self.fast_response_cache.get_fast_response(user_message, chat_id)
            if fast_response:
                # ... (обновление статистики и истории для быстрого ответа) ...
                self.logger.info(f"⚡️ Быстрый ответ для {chat_id}")
                return fast_response

            # 2. ✅ ПОЛУЧАЕМ ТЕКУЩЕЕ СОСТОЯНИЕ И ИСТОРИЮ
            current_state = conversation_manager.get_dialogue_state(chat_id)
            conversation_history = conversation_manager.get_conversation_history(chat_id)
            
            # 3. Вызываем LlamaIndex для получения ГОТОВОГО ответа, ПЕРЕДАВАЯ СОСТОЯНИЕ
            if not llama_index_rag:
                 raise RuntimeError("LlamaIndex RAG не доступен.")

            final_response, rag_metrics = llama_index_rag.search_and_answer(
                query=user_message,
                conversation_history=conversation_history,
                current_state=current_state
            )
            
            # 4. ЗАЩИТА ИСТОРИИ: не сохраняем технические ошибки
            is_error_response = "ошибка" in final_response.lower()

            if not is_error_response:
                # 5. ✅ ОБРАБАТЫВАЕМ ТОКЕН ДЕЙСТВИЯ
                processed_response = self._process_action_tokens(final_response, chat_id)
                
                # 6. Обновляем историю ТОЛЬКО успешным ответом
                conversation_manager.update_conversation_history(chat_id, user_message, processed_response)

                # 7. ✅ ОПРЕДЕЛЯЕМ И УСТАНАВЛИВАЕМ НОВОЕ СОСТОЯНИЕ
                new_state = conversation_manager.analyze_message_for_state_transition(user_message, current_state)
                if new_state != current_state:
                    conversation_manager.set_dialogue_state(chat_id, new_state)
                    self.logger.info(f"Состояние для {chat_id} изменено: {current_state} -> {new_state}")
                
                final_response = processed_response
            else:
                 self.logger.warning(f"❗️ Обнаружен ответ с ошибкой, не сохраняем в историю: '{final_response}'")

            # ... (логирование и статистика) ...
            
            self.logger.info(f"✅ Ответ готов для {chat_id} за {(time.time() - start_time):.3f}s")
            return final_response
            
        except Exception as e:
            self.logger.error(f"💥 Критическая ошибка обработки сообщения: {e}", exc_info=True)
            return "Извините, произошла временная техническая проблема."

    def _process_action_tokens(self, response: str, chat_id: str) -> str:
        """✅ ИСПРАВЛЕНО: Обработка токена [ACTION:SEND_LESSON_LINK]"""
        if "[ACTION:SEND_LESSON_LINK]" in response:
            lesson_link = config.get_lesson_url(user_id=chat_id)
            # Заменяем токен на реальную ссылку и добавляем поясняющий текст
            response = response.replace(
                "[ACTION:SEND_LESSON_LINK]", 
                f"\n\nОтлично! Вот ссылка для записи на бесплатный пробный урок:\n🔗 {lesson_link}"
            ).strip()
        
        return response

    # ... (остальные методы класса и код Flask без изменений) ...


# === ГЛОБАЛЬНАЯ ИНИЦИАЛИЗАЦИЯ ===
production_ai_service = ProductionAIService()
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def handle_telegram_webhook():
    if request.method == 'GET':
        return "Ukido AI Assistant (RAG v3, State Machine) is running! 🚀", 200
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)