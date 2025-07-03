# app.py (CRITICAL FIXES - Production Ready)
"""
🚨 КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:
1. Правильная генерация ссылок с user_id
2. Убраны стилистические проблемы
3. Исправлена логика fast response
4. Улучшено форматирование ответов
"""

import logging
import time
import threading
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request
from typing import Dict, Any, Tuple, Optional
import requests
import weakref

# Импортируем наши модули
from config import config
from telegram_bot import telegram_bot
from conversation import conversation_manager
from rag_system import rag_system
from hubspot_client import hubspot_client
from intelligent_analyzer import intelligent_analyzer


class ProductionConnectionPool:
    """Production-ready HTTP Connection pooling с proper cleanup"""
    
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
        self.logger.info("🔗 Production connection pool инициализирован")
    
    def post(self, *args, **kwargs):
        return self.session.post(*args, **kwargs)
    
    def get(self, *args, **kwargs):
        return self.session.get(*args, **kwargs)
    
    def cleanup(self):
        try:
            self.session.close()
            self.logger.info("🔗 Connection pool закрыт")
        except Exception as e:
            self.logger.error(f"Ошибка при закрытии connection pool: {e}")


class ProductionFastResponseCache:
    """
    🚨 ИСПРАВЛЕНО: Fast response cache с правильной обработкой пробных уроков
    """
    
    def __init__(self):
        # ИСПРАВЛЕНО: Убрана захардкоженная ссылка для 'пробный'
        self.fast_responses = {
            'цена': "Стоимость курсов от 6000 до 8000 грн в месяц в зависимости от возраста. Первый урок бесплатный!",
            'возраст': "У нас курсы для разных возрастов: 7-10 лет (Юный оратор), 9-12 лет (Эмоциональный компас), 11-14 лет (Капитан проектов).",
            'онлайн': "Да, все занятия проходят онлайн в удобном формате с живым общением.",
            'расписание': "Занятия 2 раза в неделю по 90 минут. Расписание подбираем индивидуально под вас.",
            # УДАЛЕНО: 'пробный' - будет обрабатываться через AI для правильной генерации ссылки
        }
        
        # ДОБАВЛЕНО: Паттерны которые ДОЛЖНЫ обрабатываться через AI
        self.ai_required_patterns = [
            'пробн', 'записа', 'урок', 'бесплатн', 'попробова', 'тест'
        ]
        
        # Отслеживание использованных метафор для предотвращения повторений
        self.used_metaphors = {}
        
        self.usage_stats = {}
        self.stats_lock = threading.Lock()
        self.max_stats_entries = 1000
        
        for key in self.fast_responses:
            self.usage_stats[key] = 0
            
        self.logger = logging.getLogger(f"{__name__}.FastCache")
    
    def get_fast_response(self, user_message: str, chat_id: str = None) -> Optional[str]:
        """
        🚨 ИСПРАВЛЕНО: Пропускает запросы на пробный урок к AI для правильной генерации ссылки
        """
        message_lower = user_message.lower()
        
        # КРИТИЧНО: Проверяем, требует ли сообщение обработки AI
        for ai_pattern in self.ai_required_patterns:
            if ai_pattern in message_lower:
                self.logger.info(f"🎯 Сообщение требует AI обработки для ссылки с user_id: {ai_pattern}")
                return None  # Пропускаем к AI
        
        # Обычная логика fast response для остальных случаев
        for keyword, response in self.fast_responses.items():
            if keyword in message_lower and len(user_message.split()) <= 3:
                with self.stats_lock:
                    self.usage_stats[keyword] += 1
                    self._cleanup_stats_if_needed()
                return response
        
        return None
    
    def track_metaphor_usage(self, chat_id: str, response: str):
        """ДОБАВЛЕНО: Отслеживание использованных метафор"""
        if chat_id not in self.used_metaphors:
            self.used_metaphors[chat_id] = set()
        
        metaphor_patterns = [
            'ресторан', 'меню', 'швейцарский нож', 'горячие пирожки',
            'шведский стол', 'кулинарн', 'повар', 'блюдо'
        ]
        
        response_lower = response.lower()
        for pattern in metaphor_patterns:
            if pattern in response_lower:
                self.used_metaphors[chat_id].add(pattern)
    
    def get_metaphor_restriction(self, chat_id: str) -> str:
        """Возвращает ограничения для промпта"""
        used = self.used_metaphors.get(chat_id, set())
        if used:
            return f"\n❌ НЕ ИСПОЛЬЗУЙ уже использованные метафоры: {', '.join(used)}"
        return ""
    
    def _cleanup_stats_if_needed(self):
        """Периодическая очистка статистики"""
        if len(self.usage_stats) > self.max_stats_entries:
            old_stats = {}
            for key in self.fast_responses.keys():
                old_stats[key] = self.usage_stats.get(key, 0)
            self.usage_stats = old_stats
            self.logger.info("🧹 Stats cleanup выполнен")


class OptimizedPromptBuilder:
    """
    🚨 ИСПРАВЛЕНО: Промпты без "Ответ:", "ну", с правильным форматированием
    """
    
    @staticmethod
    def build_combined_analysis_prompt(user_message: str, current_state: str, 
                                     conversation_history: list, facts_context: str, 
                                     chat_id: str = "", metaphor_restrictions: str = "") -> str:
        """
        КРИТИЧЕСКИ ИСПРАВЛЕНО: Убраны все стилистические проблемы
        """
        
        short_history = '\n'.join(conversation_history[-4:]) if conversation_history else "Начало диалога"
        short_facts = facts_context[:1000] + "..." if len(facts_context) > 1000 else facts_context
        
        # Определяем стиль ответа
        message_lower = user_message.lower()
        detailed_keywords = ['расскажи про', 'подробнее', 'детально', 'все курсы', 'цены и условия']
        specific_keywords = ['цена', 'сколько', 'когда', 'где', 'как записаться', 'возраст']
        
        if any(keyword in message_lower for keyword in detailed_keywords):
            response_style = "развернутый"
        elif any(keyword in message_lower for keyword in specific_keywords):
            response_style = "краткий"
        else:
            response_style = "средний"
        
        return f"""Ты AI-ассистент школы Ukido.

🚨 КРИТИЧЕСКИ ВАЖНО:
1. НЕ начинай ответы словами "Ответ:", "Ну", или подобными вводными словами
2. Для пробного урока ОБЯЗАТЕЛЬНО используй токен: [ACTION:SEND_LESSON_LINK]
3. Используй форматирование: абзацы, списки, структуру
4. Рекомендуемый стиль ответа: {response_style}
{metaphor_restrictions}

АКТУАЛЬНАЯ ИНФОРМАЦИЯ О КУРСАХ:
✅ СУЩЕСТВУЮЩИЕ КУРСЫ:
• "Юный Оратор" (7-10 лет) - 6000 грн/месяц
• "Эмоциональный Компас" (9-12 лет) - 7000 грн/месяц  
• "Капитан Проектов" (11-14 лет) - 8000 грн/месяц

❌ НЕ СУЩЕСТВУЮТ:
• "Творческое мышление"
• "Иностранные языки"
• "Математика для всех"
• "Программирование"

FORMAT:
• ТОЛЬКО онлайн
• 2 раза в неделю по 90 минут
• Время: 17:00 или 19:00
• Первый урок БЕСПЛАТНЫЙ для всех курсов

ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:
{short_facts}

СТИЛЬ ОБЩЕНИЯ:
- Остроумный, как Жванецкий
- БЕЗ навязчивых вводных слов ("ну", "ответ")  
- С правильным форматированием
- Четкие, структурированные ответы

ИСТОРИЯ ДИАЛОГА:
{short_history}

ТЕКУЩЕЕ СОСТОЯНИЕ: {current_state}

ВОПРОС: "{user_message}"

БЫСТРЫЙ АНАЛИЗ + СТРУКТУРИРОВАННЫЙ ОТВЕТ:

АНАЛИЗ (одной строкой):
Категория: factual/philosophical/problem_solving/sensitive | Состояние: greeting/fact_finding/problem_solving/closing | Стиль: {response_style}

ОСНОВНОЙ ОТВЕТ:
[Твой четко структурированный ответ БЕЗ лишних вводных слов, с правильным форматированием]"""


class ProductionAIService:
    """Production-ready высокопроизводительная версия AI сервиса"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Инициализируем компоненты производительности
        self.connection_pool = ProductionConnectionPool()
        self.fast_response_cache = ProductionFastResponseCache()
        self.prompt_builder = OptimizedPromptBuilder()
        
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="UkidoAI")
        
        # Thread-safe статистика производительности
        self.performance_stats = {
            'total_requests': 0,
            'fast_responses': 0,
            'parallel_processed': 0,
            'avg_response_time': 0,
            'total_time_saved': 0
        }
        self.stats_lock = threading.Lock()
        
        atexit.register(self.cleanup)
        self._init_ai_model()
        self._setup_module_connections()
        
        self.logger.info("🚀 Production-ready AI Service инициализирован")
    
    def cleanup(self):
        """Правильная очистка всех ресурсов"""
        try:
            self.executor.shutdown(wait=True, timeout=30)
            self.connection_pool.cleanup()
            self.logger.info("🧹 AI Service cleanup завершен")
        except Exception as e:
            self.logger.error(f"AI Service cleanup error: {e}")
    
    def _init_ai_model(self):
        """Инициализация AI модели"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.ai_model_available = True
            self.logger.info("✅ Gemini AI модель инициализирована")
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации AI модели: {e}")
            self.ai_model_available = False
    
    def _setup_module_connections(self):
        """Настройка соединений между модулями"""
        try:
            telegram_bot.set_message_handler(self.handle_message)
            self.logger.info("✅ Module connections установлены")
        except Exception as e:
            self.logger.error(f"❌ Ошибка настройки connections: {e}")
    
    def handle_message(self, chat_id: str, user_message: str) -> str:
        """
        🚨 ИСПРАВЛЕНО: Основной обработчик с правильными ссылками и форматированием
        """
        start_time = time.time()
        
        try:
            with self.stats_lock:
                self.performance_stats['total_requests'] += 1
            
            # ИСПРАВЛЕНО: Проверяем fast response с передачей chat_id
            fast_response = self.fast_response_cache.get_fast_response(user_message, chat_id)
            if fast_response:
                with self.stats_lock:
                    self.performance_stats['fast_responses'] += 1
                self.logger.info(f"⚡ Fast response для {chat_id}")
                return fast_response
            
            # Получаем состояние пользователя
            current_state = conversation_manager.get_user_state(chat_id)
            
            # Parallel processing для независимых операций
            def get_rag_context():
                return rag_system.get_relevant_context(user_message)
            
            def get_conversation_history():
                return conversation_manager.get_conversation_history(chat_id)
            
            # Запускаем параллельно
            with self.executor as executor:
                rag_future = executor.submit(get_rag_context)
                history_future = executor.submit(get_conversation_history)
                
                # Собираем результаты
                facts_context = rag_future.result(timeout=5)
                conversation_history = history_future.result(timeout=3)
            
            parallel_time = time.time() - start_time
            
            # ИСПРАВЛЕНО: Добавляем ограничения на метафоры
            metaphor_restrictions = self.fast_response_cache.get_metaphor_restriction(chat_id)
            
            # Single LLM call вместо трех отдельных
            llm_start = time.time()
            combined_prompt = self.prompt_builder.build_combined_analysis_prompt(
                user_message, current_state, conversation_history, facts_context, 
                chat_id, metaphor_restrictions
            )
            
            # Генерируем ответ
            combined_response = self._call_ai_model(combined_prompt)
            llm_time = time.time() - llm_start
            
            # Парсим ответ
            main_response, analysis_data = self._parse_combined_response(combined_response)
            
            # 🚨 КРИТИЧНО: Обрабатываем токены действий с правильными ссылками
            final_response = self._process_action_tokens(main_response, chat_id, current_state)
            
            # Отслеживаем использованные метафоры
            self.fast_response_cache.track_metaphor_usage(chat_id, final_response)
            
            # Обновляем историю
            conversation_manager.update_conversation_history(chat_id, user_message, final_response)
            conversation_manager.update_user_state(chat_id, analysis_data.get('state', current_state))
            
            # Обновляем статистику
            total_time = time.time() - start_time
            self._update_performance_stats(total_time, parallel_time, llm_time)
            
            self.logger.info(f"✅ Ответ сгенерирован для {chat_id} за {total_time:.3f}s")
            return final_response
            
        except Exception as e:
            self.logger.error(f"💥 Ошибка обработки сообщения: {e}")
            return "Извините, временная техническая проблема. Попробуйте еще раз."
    
    def _call_ai_model(self, prompt: str) -> str:
        """Вызов AI модели с proper error handling"""
        try:
            if not self.ai_model_available:
                return "AI модель недоступна. Попробуйте позже."
            
            response = self.model.generate_content(prompt)
            return response.text if response.text else "Извините, не удалось сгенерировать ответ."
            
        except Exception as e:
            self.logger.error(f"AI model error: {e}")
            return "Извините, временная проблема с AI. Попробуйте еще раз."
    
    def _parse_combined_response(self, combined_response: str) -> Tuple[str, Dict[str, str]]:
        """
        🚨 ИСПРАВЛЕНО: Парсинг без артефактов "Ответ:"
        """
        try:
            lines = combined_response.split('\n')
            analysis_line = ""
            response_lines = []
            
            for line in lines:
                if line.strip().startswith('Категория:'):
                    analysis_line = line.strip()
                elif line.strip().startswith('Ответ:'):
                    # ИСПРАВЛЕНО: Пропускаем строки с "Ответ:"
                    continue
                else:
                    response_lines.append(line)
            
            # Парсим анализ
            analysis_data = {}
            if analysis_line:
                try:
                    parts = analysis_line.split('|')
                    for part in parts:
                        if 'Категория:' in part:
                            analysis_data['category'] = part.split(':')[1].strip()
                        elif 'Состояние:' in part:
                            analysis_data['state'] = part.split(':')[1].strip()
                        elif 'Стиль:' in part:
                            analysis_data['style'] = part.split(':')[1].strip()
                except:
                    pass
            
            # Основной ответ (убираем пустые строки)
            main_response = '\n'.join([line for line in response_lines if line.strip()])
            
            return main_response, analysis_data
            
        except Exception as e:
            self.logger.error(f"Parse error: {e}")
            return combined_response, {}
    
    def _process_action_tokens(self, response: str, chat_id: str, current_state: str) -> str:
        """
        🚨 КРИТИЧНО ИСПРАВЛЕНО: Правильная замена токенов с user_id и красивым форматированием
        """
        if "[ACTION:SEND_LESSON_LINK]" in response:
            lesson_url = config.get_lesson_url(chat_id)
            
            # ИСПРАВЛЕНО: Красивое форматирование ссылки
            formatted_link = f"\n\n🎓 **Первый урок бесплатный!**\n📝 Записывайтесь: {lesson_url}\n"
            
            response = response.replace("[ACTION:SEND_LESSON_LINK]", formatted_link)
            self.logger.info(f"✅ Ссылка на урок добавлена с user_id: {chat_id}")
        
        return response
    
    def _update_performance_stats(self, total_time: float, parallel_time: float, llm_time: float):
        """Thread-safe обновление статистики производительности"""
        with self.stats_lock:
            estimated_sequential_time = 9.65
            time_saved = max(0, estimated_sequential_time - total_time)
            self.performance_stats['total_time_saved'] += time_saved
            
            # Обновляем среднее время ответа
            current_avg = self.performance_stats['avg_response_time']
            total_requests = self.performance_stats['total_requests']
            new_avg = (current_avg * (total_requests - 1) + total_time) / total_requests
            self.performance_stats['avg_response_time'] = new_avg
    
    def get_system_status(self) -> Dict[str, Any]:
        """Thread-safe возвращение статуса с метриками производительности"""
        base_status = {
            "config_valid": config.validate_configuration(),
            "telegram_bot_ready": telegram_bot is not None,
            "conversation_manager_ready": conversation_manager is not None,
            "rag_system_stats": rag_system.get_stats(),
            "ai_model_available": self.ai_model_available
        }
        
        with self.stats_lock:
            performance_metrics = self.performance_stats.copy()
        
        if performance_metrics['total_requests'] > 0:
            performance_metrics['fast_response_rate'] = round(
                (performance_metrics['fast_responses'] / performance_metrics['total_requests']) * 100, 1
            )
            performance_metrics['avg_speedup'] = round(
                9.65 / max(performance_metrics['avg_response_time'], 0.1), 2
            )
        
        base_status['performance_metrics'] = performance_metrics
        return base_status


# Создаем оптимизированные компоненты
app = Flask(__name__)
ai_service = ProductionAIService()


# === ОПТИМИЗИРОВАННЫЕ МАРШРУТЫ ===

@app.route('/', methods=['POST'])
def telegram_webhook():
    """Оптимизированный webhook с минимальными накладными расходами"""
    return telegram_bot.handle_webhook()

@app.route('/lesson')
def lesson_page():
    """Страница урока с кешированием"""
    return telegram_bot.show_lesson_page()

@app.route('/submit-lesson-form', methods=['POST'])
def submit_lesson_form():
    """Production-ready обработка формы с proper error handling"""
    try:
        form_data = request.get_json()
        if not form_data:
            return {"success": False, "error": "Нет данных"}, 400
        
        # Safe асинхронная отправка в HubSpot
        def safe_hubspot_submission():
            try:
                hubspot_client.create_contact(form_data)
                logging.getLogger(__name__).info(f"HubSpot contact created for: {form_data.get('firstName', 'Unknown')}")
            except Exception as e:
                logging.getLogger(__name__).error(f"HubSpot submission error: {e}")
        
        threading.Thread(target=safe_hubspot_submission, daemon=True).start()
        
        return {"success": True, "message": "Данные получены"}, 200
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Form error: {e}")
        return {"success": False, "error": str(e)}, 500

@app.route('/hubspot-webhook', methods=['POST'])
def hubspot_webhook():
    """Production-ready HubSpot webhook с proper error handling"""
    try:
        webhook_data = request.get_json()
        message_type = request.args.get('message_type', 'first_follow_up')
        
        def safe_webhook_processing():
            try:
                hubspot_client.process_webhook(webhook_data, message_type)
                logging.getLogger(__name__).info(f"HubSpot webhook processed: {message_type}")
            except Exception as e:
                logging.getLogger(__name__).error(f"HubSpot webhook processing error: {e}")
        
        threading.Thread(target=safe_webhook_processing, daemon=True).start()
        
        return "OK", 200
        
    except Exception as e:
        logging.getLogger(__name__).error(f"HubSpot webhook error: {e}")
        return "Error", 500

@app.route('/health')
def health_check():
    """Health check с метриками производительности"""
    return ai_service.get_system_status()

@app.route('/metrics')
def metrics():
    """Detailed performance metrics"""
    return {
        "system_status": ai_service.get_system_status(),
        "rag_stats": rag_system.get_stats(),
        "performance_summary": {
            "optimization_level": "HIGH",
            "parallel_processing": "ENABLED", 
            "connection_pooling": "ENABLED",
            "fast_responses": "ENABLED",
            "estimated_speedup": "4x+"
        }
    }

@app.route('/clear-memory', methods=['POST'])
def clear_memory():
    """Production-ready очистка памяти с proper error handling"""
    try:
        def safe_memory_clear():
            try:
                conversation_manager._clear_all_memory()
                logging.getLogger(__name__).info("Memory cleared successfully")
            except Exception as e:
                logging.getLogger(__name__).error(f"Memory clear error: {e}")
        
        threading.Thread(target=safe_memory_clear, daemon=True).start()
        return {"success": True, "message": "Очистка начата"}, 200
    except Exception as e:
        return {"success": False, "error": str(e)}, 500


# === ТОЧКА ВХОДА ===

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("🚨 PRODUCTION-READY UKIDO AI ASSISTANT - CRITICAL FIXES APPLIED")
    logger.info("✅ Правильные ссылки с user_id для HubSpot интеграции")
    logger.info("✅ Убраны стилистические проблемы (ну, ответ)")
    logger.info("✅ Улучшено форматирование и структура ответов")
    logger.info("✅ Исправлена логика fast response для пробных уроков")
    logger.info("=" * 60)
    
    try:
        status = ai_service.get_system_status()
        logger.info(f"📊 Production system status: {status.get('config_valid', False)}")
    except Exception as e:
        logger.error(f"Status check error: {e}")
    
    app.run(
        debug=config.DEBUG_MODE,
        port=config.PORT,
        host='0.0.0.0',
        threaded=True,
        use_reloader=False
    )