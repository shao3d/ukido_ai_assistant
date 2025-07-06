# app.py
"""
✅ ФИНАЛЬНАЯ ВЕРСИЯ: Чистый RAG тест с LlamaIndex
- 2 вызова GPT-4o mini: обогащение + финальный ответ
- Умная история: максимум 4 сообщения (2 пары)
- Отключены: машина состояний, юмор Жванецкого, старый обогатитель
- Простая, надежная реализация
"""

import logging
import time
import threading
import atexit
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request
from typing import Dict, Any, Tuple, Optional, List
import requests

# Импортируем наши модули
from config import config
from telegram_bot import telegram_bot
from conversation import conversation_manager
from rag_system import rag_system
from hubspot_client import hubspot_client
from intelligent_analyzer import intelligent_analyzer

# --- LLAMAINDEX INTEGRATION ---
from llamaindex_rag import llama_index_rag
USE_LLAMAINDEX = True  # Главный переключатель
# --- END LLAMAINDEX ---

# --- RAG DEBUG LOGGER ---
try:
    from rag_debug_logger import rag_debug
    DEBUG_LOGGING_ENABLED = True
except ImportError:
    DEBUG_LOGGING_ENABLED = False
    print("Debug логирование отключено - rag_debug_logger не найден")
# --- END DEBUG ---


class ProductionConnectionPool:
    """Production-ready HTTP Connection pooling"""
    
    def __init__(self):
        self.session = requests.Session()
        self.logger = logging.getLogger(f"{__name__}.ConnectionPool")
        
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.session.timeout = (5, 15)
        
        atexit.register(self.cleanup)
        self.logger.info("🔗 Production connection pool готов")
    
    def post(self, *args, **kwargs):
        return self.session.post(*args, **kwargs)
    
    def get(self, *args, **kwargs):
        return self.session.get(*args, **kwargs)
    
    def cleanup(self):
        try:
            self.session.close()
            self.logger.info("🔗 Connection pool закрыт")
        except Exception as e:
            self.logger.error(f"Ошибка закрытия connection pool: {e}")


class ProductionFastResponseCache:
    """Fast response cache для простых вопросов"""
    
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
        """Проверяет, можем ли дать быстрый ответ"""
        message_lower = message.lower().strip()
        
        for keyword, response in self.fast_responses.items():
            if keyword in message_lower:
                # Специальная обработка для пробного урока
                if keyword in ['пробный', 'записаться']:
                    lesson_link = f"https://ukidoaiassistant-production.up.railway.app/lesson?user_id={chat_id}"
                    return f"{response}\n\n🔗 Записаться: {lesson_link}"
                
                return response
        
        return None


class ProductionAIService:
    """Production-ready AI сервис для чистого RAG тестирования"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Инициализируем компоненты
        self.connection_pool = ProductionConnectionPool()
        self.fast_response_cache = ProductionFastResponseCache()
        
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="UkidoAI")
        
        # Thread-safe статистика
        self.performance_stats = {
            'total_requests': 0,
            'fast_responses': 0,
            'avg_response_time': 0,
        }
        self.stats_lock = threading.Lock()
        
        # AI модель
        self.ai_model_available = self._initialize_ai_model()
        
        self.logger.info("🚀 ProductionAIService готов к RAG тестированию")
    
    def _initialize_ai_model(self) -> bool:
        """Инициализация Gemini модели"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.GEMINI_API_KEY)
            
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self.logger.info("✅ Gemini 1.5 Pro готов")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации AI: {e}")
            return False

    def process_user_message(self, user_message: str, chat_id: str) -> str:
        """
        ✅ ГЛАВНЫЙ МЕТОД: Обработка сообщений с чистым RAG тестированием
        """
        start_time = time.time()
        
        try:
            # Проверяем fast response
            fast_response = self.fast_response_cache.get_fast_response(user_message, chat_id)
            if fast_response:
                with self.stats_lock:
                    self.performance_stats['fast_responses'] += 1
                return fast_response
            
            # Получаем текущее состояние
            current_state = conversation_manager.get_dialogue_state(chat_id)
            
            # ✅ УМНАЯ ИСТОРИЯ - локальная функция для простоты
            def get_smart_conversation_history():
                """Получаем умную историю: максимум 4 сообщения (2 пары)"""
                full_history = conversation_manager.get_conversation_history(chat_id)
                
                if not full_history or len(full_history) == 0:
                    return []  # Первое сообщение
                
                # Максимум 4 сообщения (2 пары вопрос-ответ)
                return full_history[-4:]
            
            def get_rag_context():
                """✅ ЧИСТЫЙ RAG поиск без старого обогащения"""
                # Получаем умную историю
                smart_history = get_smart_conversation_history()
                
                # ✅ ПЕРЕДАЕМ ОРИГИНАЛЬНЫЙ запрос (старый обогатитель отключен)
                
                if USE_LLAMAINDEX and llama_index_rag:
                    self.logger.info("🚀 LlamaIndex RAG (чистый тест)")
                    
                    # Debug логирование
                    if DEBUG_LOGGING_ENABLED:
                        rag_debug.log_enricher_input(user_message, smart_history)
                    
                    # ✅ ОРИГИНАЛЬНЫЙ запрос + умная история
                    context, metrics = llama_index_rag.search_knowledge_base(
                        query=user_message,  # Оригинальный запрос!
                        conversation_history=smart_history
                    )
                else:
                    self.logger.info("Legacy RAG fallback")
                    context, metrics = rag_system.search_knowledge_base(user_message, smart_history)

                return context, metrics

            # Параллельное выполнение
            try:
                smart_history_future = self.executor.submit(get_smart_conversation_history)
                conversation_history = smart_history_future.result(timeout=3)
                
                rag_future = self.executor.submit(get_rag_context)
                facts_context, rag_metrics = rag_future.result(timeout=5)
            except Exception as e:
                self.logger.error(f"Parallel processing error: {e}")
                facts_context = "Информация временно недоступна."
                rag_metrics = {'best_score': 0.0, 'chunks_found': 0}
                conversation_history = []
            
            # ✅ ЧИСТЫЙ RAG ТЕСТ: без категоризации и машины состояний
            rag_score = rag_metrics.get('max_score', 0.0)

            # ✅ ПРОСТОЙ RAG ПРОМПТ (без юмора и категоризации)
            llm_start = time.time()
            combined_prompt = self._build_simple_rag_prompt(
                user_message=user_message,
                facts_context=facts_context,
                rag_score=rag_metrics.get('average_score', 0.5),
                conversation_history=conversation_history
            )

            # Debug логирование
            if DEBUG_LOGGING_ENABLED:
                rag_debug.log_final_prompt(combined_prompt, len(conversation_history))

            # Логируем решение
            self.logger.info(f"📊 [ЧИСТЫЙ RAG] Score: {rag_score:.2f}, История: {len(conversation_history)} msgs")

            # Генерируем ответ
            combined_response = self._call_ai_model(combined_prompt)
            llm_time = time.time() - llm_start

            # Debug финального ответа
            if DEBUG_LOGGING_ENABLED:
                rag_debug.log_final_response(combined_response, llm_time)

            # Парсим ответ
            main_response, analysis_data = self._parse_combined_response(combined_response)

            # Обрабатываем токены действий
            final_response = self._process_action_tokens(main_response, chat_id, current_state)

            # Обновляем историю
            conversation_manager.update_conversation_history(chat_id, user_message, final_response)
            conversation_manager.set_dialogue_state(chat_id, analysis_data.get('state', current_state))

            # Статистика
            total_time = time.time() - start_time
            self._update_performance_stats(total_time, llm_time)

            self.logger.info(f"✅ Ответ готов для {chat_id} за {total_time:.3f}s")
            return final_response
            
        except Exception as e:
            self.logger.error(f"💥 Ошибка обработки: {e}")
            error_response = "Извините, временная техническая проблема. Попробуйте еще раз."
            
            if DEBUG_LOGGING_ENABLED:
                rag_debug.log_final_response(f"ERROR: {error_response}", 0.0)
            return error_response
    
    def _build_simple_rag_prompt(self, user_message: str, facts_context: str, 
                                rag_score: float, conversation_history: list = None) -> str:
        """
        ✅ ПРОСТОЙ RAG ПРОМПТ для чистого тестирования
        """
        # История уже обрезана в get_smart_conversation_history() до 4 сообщений
        if not conversation_history or len(conversation_history) == 0:
            history_text = "Начало диалога"
            greeting_instruction = "📝 НАЧАЛО: Начни с приветствия ('Добрый день!')."
        else:
            history_text = "\n".join(conversation_history)
            greeting_instruction = "📝 ПРОДОЛЖЕНИЕ: БЕЗ приветствий. Учитывай контекст беседы."
        
        return f"""Ты AI-ассистент онлайн-школы Ukido для развития soft skills у детей.

📚 ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ (RAG Score: {rag_score:.2f}):
{facts_context}

💬 ИСТОРИЯ ДИАЛОГА ({len(conversation_history) if conversation_history else 0} сообщений, макс. 4):
{history_text}

❓ ВОПРОС: {user_message}

📋 ИНСТРУКЦИИ:
1. Если в базе знаний ЕСТЬ ответ - используй конкретные факты
2. Если информации НЕТ - скажи: "К сожалению, в моей базе знаний нет информации по этому вопросу"
3. НЕ выдумывай факты
4. УЧИТЫВАЙ историю диалога - не повторяйся
5. {greeting_instruction}
6. Отвечай кратко и по существу

Ответ:"""

    def _call_ai_model(self, prompt: str) -> str:
        """Вызов Gemini модели"""
        try:
            if not self.ai_model_available:
                return "AI модель недоступна. Попробуйте позже."
            
            response = self.model.generate_content(prompt)
            return response.text if response.text else "Не удалось сгенерировать ответ."
            
        except Exception as e:
            self.logger.error(f"AI model error: {e}")
            return "Временная проблема с AI. Попробуйте еще раз."

    def _parse_combined_response(self, response_text: str) -> Tuple[str, Dict[str, Any]]:
        """Простой парсинг ответа"""
        try:
            lines = response_text.strip().split('\n')
            main_response = []
            analysis_data = {}
            
            for line in lines:
                if line.startswith('[') and ']' in line:
                    # Извлекаем metadata
                    try:
                        key_value = line.strip('[]').split(':')
                        if len(key_value) == 2:
                            key, value = key_value
                            analysis_data[key.strip()] = value.strip()
                    except:
                        main_response.append(line)
                else:
                    main_response.append(line)
            
            return '\n'.join(main_response).strip(), analysis_data
            
        except Exception as e:
            self.logger.error(f"Response parsing error: {e}")
            return response_text, {}

    def _process_action_tokens(self, response: str, chat_id: str, current_state: str) -> str:
        """Обработка специальных токенов"""
        try:
            # [LESSON_LINK] токен
            if "[LESSON_LINK]" in response:
                lesson_link = f"https://ukidoaiassistant-production.up.railway.app/lesson?user_id={chat_id}"
                response = response.replace("[LESSON_LINK]", lesson_link)
            
            # [CONTACT_MANAGER] токен
            if "[CONTACT_MANAGER]" in response:
                contact_info = "📞 Связаться: @ukido_manager или +38 (093) 123-45-67"
                response = response.replace("[CONTACT_MANAGER]", contact_info)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Action token error: {e}")
            return response

    def _update_performance_stats(self, total_time: float, llm_time: float):
        """Обновление статистики"""
        try:
            with self.stats_lock:
                self.performance_stats['total_requests'] += 1
                
                current_avg = self.performance_stats['avg_response_time']
                total_requests = self.performance_stats['total_requests']
                
                new_avg = ((current_avg * (total_requests - 1)) + total_time) / total_requests
                self.performance_stats['avg_response_time'] = new_avg
                
                if total_requests % 10 == 0:
                    self.logger.info(f"📊 Stats: {total_requests} requests, avg: {new_avg:.2f}s")
                    
        except Exception as e:
            self.logger.error(f"Stats error: {e}")


# === ГЛОБАЛЬНАЯ ИНИЦИАЛИЗАЦИЯ ===
production_ai_service = ProductionAIService()

# === FLASK ПРИЛОЖЕНИЕ ===
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def handle_telegram_webhook():
    """Обработчик Telegram webhook"""
    try:
        if request.method == 'GET':
            return "Ukido AI Assistant is running! 🚀", 200
        
        update = request.get_json()
        if not update:
            return "No data received", 400
        
        # Извлекаем сообщение
        if 'message' in update:
            message = update['message']
            chat_id = str(message['chat']['id'])
            
            if 'text' in message:
                user_message = message['text']
                
                # Обрабатываем сообщение
                bot_response = production_ai_service.process_user_message(user_message, chat_id)
                
                # Отправляем ответ
                telegram_bot.send_message(chat_id, bot_response)
                
        return "OK", 200
        
    except Exception as e:
        logging.error(f"Webhook error: {e}")
        return "Error", 500

@app.route('/test-message', methods=['POST'])
def test_message_endpoint():
    """✅ Тестовый эндпоинт для отладки"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        user_id = data.get('user_id', 'test_user')
        
        start_time = time.time()
        bot_response = production_ai_service.process_user_message(user_message, user_id)
        response_time = time.time() - start_time
        
        return {
            'bot_response': bot_response,
            'response_time': response_time,
            'status': 'success'
        }, 200
        
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/clear-memory', methods=['POST'])
def clear_memory():
    """✅ Очистка памяти для тестирования"""
    try:
        conversation_manager.clear_all_conversations()
        return {"status": "success", "message": "Memory cleared"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)