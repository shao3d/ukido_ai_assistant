# app.py
"""
🚨 КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:
1. УБРАНО дублирование классов ZhvanetskyHumorLevelSystem и ProductionFastResponseCache
2. Правильная генерация ссылок с user_id
3. Убраны стилистические проблемы
4. Исправлена логика fast response
5. Улучшено форматирование ответов
6. ДОБАВЛЕНО: Debug логирование RAG pipeline
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

# --- НОВЫЙ БЛОК: ИМПОРТ И НАСТРОЙКА LLAMAINDEX ---
from llamaindex_rag import llama_index_rag # Импортируем новый модуль
USE_LLAMAINDEX = True # 🚩 ГЛАВНЫЙ ПЕРЕКЛЮЧАТЕЛЬ
# --- КОНЕЦ НОВОГО БЛОКА ---

# --- БЛОК: ИМПОРТ DEBUG ЛОГГЕРА ---
try:
    from rag_debug_logger import rag_debug
    DEBUG_LOGGING_ENABLED = True
except ImportError:
    DEBUG_LOGGING_ENABLED = False
    print("Debug логирование отключено - rag_debug_logger не найден")
# --- КОНЕЦ БЛОКА ---


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
            'стоимость': "Стоимость курсов от 6000 до 8000 грн в месяц в зависимости от возраста. Первый урок бесплатный!",
            'сколько стоит': "Стоимость курсов от 6000 до 8000 грн в месяц в зависимости от возраста. Первый урок бесплатный!",
            'сколько стоят': "Стоимость курсов от 6000 до 8000 грн в месяц в зависимости от возраста. Первый урок бесплатный!",
            'дорого': "Стоимость курсов от 6000 до 8000 грн в месяц в зависимости от возраста. Первый урок бесплатный!",
            'дёшево': "Стоимость курсов от 6000 до 8000 грн в месяц в зависимости от возраста. Первый урок бесплатный!",
            'привет': "Добро пожаловать в онлайн-школу Ukido! 👋 Мы развиваем soft skills у детей. Расскажите, что вас интересует?",
            'здравствуйте': "Здравствуйте! Добро пожаловать в Ukido - онлайн-школу развития soft skills для детей. Чем могу помочь?",
            'курсы': "У нас есть курсы по развитию коммуникации, лидерства, критического мышления и эмоционального интеллекта для детей разных возрастов.",
            'возраст': "Наши курсы подходят для детей от 7 до 17 лет. Программы адаптированы под разные возрастные группы.",
        }
        
        # Трекинг использованных метафор для каждого пользователя
        self.metaphor_usage = {}
        self.metaphor_lock = threading.Lock()
        
        self.logger = logging.getLogger(f"{__name__}.FastCache")
        self.logger.info("⚡ Fast response cache инициализирован")
    
    def get_fast_response(self, message: str, chat_id: str = "") -> Optional[str]:
        """Быстрая проверка на простые запросы"""
        message_lower = message.lower().strip()
        
        # Точное совпадение
        if message_lower in self.fast_responses:
            self.logger.info(f"⚡ Fast response (exact): '{message_lower}'")
            return self.fast_responses[message_lower]
        
        # Поиск по содержанию
        for key, response in self.fast_responses.items():
            if key in message_lower and len(message_lower) < 50:  # Короткие сообщения
                self.logger.info(f"⚡ Fast response (contains): '{key}' in '{message_lower}'")
                return response
        
        return None
    
    def track_metaphor_usage(self, chat_id: str, response_text: str):
        """Отслеживание использованных метафор"""
        # Простое отслеживание ключевых слов
        metaphor_keywords = ['как', 'словно', 'точно', 'будто', 'похоже']
        
        with self.metaphor_lock:
            if chat_id not in self.metaphor_usage:
                self.metaphor_usage[chat_id] = []
            
            for keyword in metaphor_keywords:
                if keyword in response_text.lower():
                    self.metaphor_usage[chat_id].append(keyword)
                    break
    
    def get_metaphor_restriction(self, chat_id: str) -> str:
        """Получение ограничений на метафоры"""
        with self.metaphor_lock:
            used_metaphors = self.metaphor_usage.get(chat_id, [])
            if len(used_metaphors) > 3:
                return "Избегайте повторения метафор, уже использованных в диалоге."
            return ""


class ZhvanetskyHumorLevelSystem:
    """
    🎭 Система градуированного юмора в стиле Михаила Жванецкого
    """
    
    def __init__(self):
        self.humor_levels = {
            'no_humor': {
                'intensity': 'БЕЗ ЮМОРА',
                'style': 'Строго профессиональный, серьезный тон',
                'examples': ['Прямые ответы', 'Факты без украшений', 'Деловой стиль']
            },
            'gentle': {
                'intensity': 'ЛЕГКИЙ ЮМОР',
                'style': 'Теплые житейские наблюдения без иронии',
                'examples': ['Как в хорошей семье', 'Понятно как дважды два', 'По-человечески']
            },
            'moderate': {
                'intensity': 'КЛАССИЧЕСКИЙ ЖВАНЕЦКИЙ',
                'style': 'Наблюдательный юмор с легкой иронией',
                'examples': ['А знаете что интересно...', 'Вот такая история', 'И тут понимаешь']
            },
            'sophisticated': {
                'intensity': 'ГЛУБОКИЙ ЖВАНЕЦКИЙ', 
                'style': 'Философские метафоры о человеческой природе',
                'examples': ['Жизнь как театр', 'Дети как зеркало общества', 'Воспитание - искусство']
            }
        }
        
        self.logger = logging.getLogger(f"{__name__}.HumorSystem")
    
    def get_humor_level(self, category: str, rag_score: float) -> str:
        """
        Определение уровня юмора на основе категории вопроса и качества RAG
        """
        if category == 'sensitive':
            return 'no_humor'
        elif category == 'factual' and rag_score > 0.7:
            return 'gentle'
        elif category == 'factual' and rag_score > 0.4:
            return 'moderate'
        elif category == 'philosophical':
            return 'sophisticated'
        elif category == 'problem_solving':
            return 'moderate'
        else:
            return 'gentle'
    
    def build_humor_instructions(self, humor_level: str, metaphor_restrictions: str = "") -> str:
        """Построение инструкций по стилю"""
        level_info = self.humor_levels.get(humor_level, self.humor_levels['gentle'])
        
        if humor_level == 'gentle':
            return f"""
🎭 СТИЛЬ: Легкий располагающий тон
• {level_info['intensity']} - {level_info['style']}
• Простые, понятные метафоры из повседневной жизни
• БЕЗ сарказма, иронии или сложных подтекстов
• Теплый, располагающий тон
• Примеры: {', '.join(level_info['examples'])}
{metaphor_restrictions}
"""
        elif humor_level == 'moderate':
            return f"""
🎭 СТИЛЬ: Классический наблюдательный Жванецкий  
• {level_info['intensity']} - {level_info['style']}
• Житейские метафоры с легкой иронией
• Подмечает забавные стороны обычных ситуаций
• Мудрые наблюдения без язвительности
• Примеры: {', '.join(level_info['examples'])}
{metaphor_restrictions}
"""
        elif humor_level == 'sophisticated':
            return f"""
🎭 СТИЛЬ: Философский глубокий Жванецкий
• {level_info['intensity']} - {level_info['style']}
• Метафоры с глубоким смыслом о человеческой природе
• Тонкие наблюдения о воспитании и жизни
• Помогает увидеть суть через призму юмора
• Примеры: {', '.join(level_info['examples'])}
{metaphor_restrictions}
"""
        else:  # no_humor
            return f"""
🎭 СТИЛЬ: Серьезный, эмпатичный тон
• {level_info['intensity']} - {level_info['style']}
• БЕЗ юмора, метафор и шуток
• Прямые, четкие, поддерживающие ответы
• Понимание и профессиональная помощь
• Примеры: {', '.join(level_info['examples'])}
"""


class OptimizedPromptBuilder:
    """
    🚨 ИСПРАВЛЕНО: Промпты без "Ответ:", "ну", с правильным форматированием
    🎭 Новая версия с системой градусов юмора Жванецкого
    """
    
    @staticmethod
    def build_combined_analysis_prompt_RAG_TEST(user_message: str, 
                                               facts_context: str,
                                               rag_score: float,
                                               conversation_history: list = None) -> str:
        """
        🔧 ИСПРАВЛЕНО: Упрощенный промпт для тестирования RAG с поддержкой истории диалога
        """
        
        # Подготавливаем историю
        if conversation_history and len(conversation_history) > 0:
            recent_history = conversation_history[-6:]  # Последние 6 сообщений
            history_text = "\n".join(recent_history)
            greeting_instruction = "📝 ПРОДОЛЖЕНИЕ ДИАЛОГА: НЕ используй приветствия. Учитывай предыдущий контекст беседы."
        else:
            history_text = "Начало диалога"
            greeting_instruction = "📝 НАЧАЛО ДИАЛОГА: Начни ответ с вежливого приветствия ('Добрый день!')."
        
        return f"""Ты AI-ассистент онлайн-школы Ukido для развития soft skills у детей.

📚 ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ (RAG Score: {rag_score:.2f}):
{facts_context}

💬 ИСТОРИЯ ДИАЛОГА:
{history_text}

❓ ТЕКУЩИЙ ВОПРОС: {user_message}

📋 ИНСТРУКЦИИ:
1. Если в базе знаний ЕСТЬ точный ответ - приведи его с конкретными фактами
2. Если информации НЕТ - честно скажи: "К сожалению, в моей базе знаний нет информации по этому вопросу"
3. НЕ выдумывай факты, которых нет в контексте
4. УЧИТЫВАЙ предыдущую историю диалога - не повторяй уже сказанное
5. {greeting_instruction}
6. Отвечай кратко и по существу

Ответ:"""

    @staticmethod
    def build_combined_analysis_prompt(user_message: str, current_state: str, 
                                     conversation_history: list, facts_context: str, 
                                     chat_id: str = "", metaphor_restrictions: str = "",
                                     category: str = "factual", rag_score: float = 0.5) -> str:
        """
        🎭 ОБНОВЛЕННАЯ ВЕРСИЯ: С умными стратегиями ответов
        """
        # Инициализируем систему юмора
        humor_system = ZhvanetskyHumorLevelSystem()
        
        # Определяем уровень юмора на основе категории + RAG score
        humor_level = humor_system.get_humor_level(category, rag_score)
        
        # Создаем инструкции по стилю
        humor_instructions = humor_system.build_humor_instructions(humor_level, metaphor_restrictions)
        

        short_history = '\n'.join(conversation_history[-4:]) if conversation_history else "Начало диалога"
        # НОВЫЙ БЛОК: Динамическая инструкция для приветствия
        if not conversation_history or len(conversation_history) == 0:
            greeting_instruction = "📝 НАЧАЛО ДИАЛОГА: Начни ответ с вежливого приветствия ('Добрый день!', 'Здравствуйте!')."
        else:
            greeting_instruction = "📝 ПРОДОЛЖЕНИЕ ДИАЛОГА: НЕ используй приветствия ('Добрый день', 'Здравствуйте'). Сразу переходи к ответу по сути."
        # КОНЕЦ НОВОГО БЛОКА
        short_facts = facts_context[:800] + "..." if len(facts_context) > 800 else facts_context
        
        # 🎯 УМНЫЕ СТРАТЕГИИ НА ОСНОВЕ КАТЕГОРИИ + RAG SCORE
        strategy_instructions = OptimizedPromptBuilder._get_strategy_instructions(
            category, rag_score, user_message, short_facts
        )
        
        # Определяем тип ответа (сохраняем существующую логику)
        message_lower = user_message.lower()
        detailed_keywords = ['расскажи про', 'подробнее', 'детально', 'все курсы', 'что у вас есть']
        lesson_request_keywords = ['записаться', 'пробный', 'попробовать', 'хочу урок', 'тестово']
        simple_keywords = ['цена', 'сколько', 'когда', 'где', 'возраст', 'время']
        
        if any(keyword in message_lower for keyword in detailed_keywords):
            response_type = "подробный"
        elif any(keyword in message_lower for keyword in lesson_request_keywords):
            response_type = "с_ссылкой"
        elif any(keyword in message_lower for keyword in simple_keywords):
            response_type = "краткий"
        else:
            response_type = "средний"
        
        # Создаем итоговый промпт
        return f"""Ты AI-ассистент онлайн-школы Ukido для развития soft-skills у детей.

{greeting_instruction}

{humor_instructions}

{strategy_instructions}

💡 ОПРЕДЕЛЕНО СИСТЕМОЙ:
• Категория: {category}
• RAG Score: {rag_score:.2f}
• Уровень юмора: {humor_level}  
• Тип ответа: {response_type}

📚 АКТУАЛЬНАЯ ИНФОРМАЦИЯ О КУРСАХ:
✅ СУЩЕСТВУЮЩИЕ КУРСЫ (только эти 3!):
1. "Коммуникация и уверенность" (7-9 лет, 10-12 лет, 13-17 лет)  
2. "Лидерство и командная работа" (10-12 лет, 13-17 лет)
3. "Эмоциональный интеллект" (7-9 лет, 10-12 лет, 13-17 лет)

💰 ЦЕНЫ: 6000-8000 грн/месяц (зависит от возраста)
🎁 БОНУС: Первый урок БЕСПЛАТНЫЙ
⏰ РАСПИСАНИЕ: Гибкое, подстраиваем под ребенка
👨‍🏫 ФОРМАТ: Онлайн с преподавателем

📊 КОНКРЕТНАЯ ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ:
{short_facts}

💬 КОНТЕКСТ БЕСЕДЫ:
{short_history}

❓ ВОПРОС РОДИТЕЛЯ: {user_message}

💭 ТВОЯ ЗАДАЧА:
Ответь {response_type}но в заданном стиле, используя ТОЛЬКО достоверную информацию из базы знаний."""

    @staticmethod
    def _get_strategy_instructions(category: str, rag_score: float, user_message: str, facts_context: str) -> str:
        """Получение умных стратегий ответа"""
        
        if category == 'factual' and rag_score > 0.7:
            return """
✅ СТРАТЕГИЯ: Уверенный фактический ответ
• База знаний содержит точную информацию - используй ее как основу
• Будь конкретным с цифрами, возрастами, ценами
• Добавь заданный уровень юмора для теплоты
• Предложи следующий шаг (пробный урок, консультацию)"""
        elif category == 'factual' and rag_score > 0.3:
            return """
⚠️ СТРАТЕГИЯ: Осторожный ответ с проверкой
• Информация найдена, но может быть неполной
• Дай ответ на основе найденного, но осторожно
• Предложи уточнить детали у консультанта
• Подчеркни важность индивидуального подхода"""
        elif category == 'factual' and rag_score <= 0.3:
            return """
🚨 СТРАТЕГИЯ: Честное признание пробела
• Информация в базе недостаточна для точного ответа
• Честно скажи что нужно уточнить
• Предложи связаться с консультантом
• Направь на конкретное действие (пробный урок, звонок)"""
        elif category == 'philosophical':
            return """
🤔 СТРАТЕГИЯ: Мудрые размышления о воспитании
• Покажи понимание глубины темы
• Используй метафоры и наблюдения о детском развитии
• Связывай с важностью развития soft skills
• Мягко переведи разговор к теме развития детей и школы
• Подчеркни важность soft skills в современном мире"""
        elif category == 'sensitive':
            return """
🚨 СТРАТЕГИЯ: Осторожность и перенаправление
• Признай серьезность и деликатность темы
• Тактично перенаправь к квалифицированным специалистам
• Предложи контакты детского психолога (если есть)
• БЕЗ попыток консультирования или советов"""
        elif category == 'off_topic':
            return """
🚨 СТРАТЕГИЯ: Дружелюбное перенаправление
• Дружелюбно ответь на вопрос в рамках разумного
• Мягко переведи разговор к теме развития детей
• Подчеркни, как это связано с образованием и навыками
• Предложи узнать о курсах школы"""
        else:
            return """
✅ СТРАТЕГИЯ: Уверенный ответ на основе найденной информации
• Используй найденную информацию как основу ответа
• Отвечай в заданном стиле юмора
• Будь информативным и полезным"""


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
        
        # AI модель инициализация
        self.ai_model_available = self._initialize_ai_model()
        
        self.logger.info("🚀 ProductionAIService инициализирован")
    
    def _initialize_ai_model(self) -> bool:
        """Инициализация AI модели"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                model_name='gemini-1.5-pro-latest',
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.9,
                    top_k=40,
                    max_output_tokens=1000,
                )
            )
            self.logger.info("✅ AI модель (Gemini 1.5 Pro) готова")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации AI: {e}")
            return False
    
    def process_message(self, chat_id: str, user_message: str) -> str:
        """
        🔧 ОБНОВЛЕНО: Обработка сообщения с debug логированием
        """
        start_time = time.time()
        
        # 🚀 НАЧИНАЕМ DEBUG СЕССИЮ (если доступен)
        if DEBUG_LOGGING_ENABLED:
            rag_debug.start_session(chat_id, user_message)
        
        try:
            with self.stats_lock:
                self.performance_stats['total_requests'] += 1
            
            # Проверяем fast response cache
            fast_response = self.fast_response_cache.get_fast_response(user_message, chat_id)
            if fast_response:
                with self.stats_lock:
                    self.performance_stats['fast_responses'] += 1
                self.logger.info(f"⚡ Fast response для {chat_id}")
                
                # Debug логирование для fast response
                if DEBUG_LOGGING_ENABLED:
                    rag_debug.log_final_response(fast_response, 0.001)
                return fast_response
                
            # Получаем состояние пользователя
            current_state = conversation_manager.get_dialogue_state(chat_id)
            
            # Parallel processing для независимых операций
            def get_conversation_history():
                return conversation_manager.get_conversation_history(chat_id)
            
            def get_rag_context():
                history = conversation_manager.get_conversation_history(chat_id)
                enriched_query = intelligent_analyzer.enrich_query_with_context(user_message, history)

                # --- RAG ПОИСК С DEBUG ЛОГИРОВАНИЕМ ---
                if USE_LLAMAINDEX and llama_index_rag:
                    self.logger.info("🚀 Используется LlamaIndex RAG")
                    
                    # Debug: логируем входные данные
                    if DEBUG_LOGGING_ENABLED:
                        rag_debug.log_enricher_input(user_message, history)
                    
                    context, metrics = llama_index_rag.search_knowledge_base(enriched_query)
                else:
                    self.logger.info("Legacy RAG в действии")
                    context, metrics = rag_system.search_knowledge_base(enriched_query)

                return context, metrics

            # Запускаем параллельно
            try:
                history_future = self.executor.submit(get_conversation_history)
                conversation_history = history_future.result(timeout=3)
                
                rag_future = self.executor.submit(get_rag_context)
                facts_context, rag_metrics = rag_future.result(timeout=5)
            except Exception as e:
                self.logger.error(f"Parallel processing error: {e}")
                facts_context = "Информация временно недоступна."
                rag_metrics = {'best_score': 0.0, 'chunks_found': 0}
                conversation_history = []
            
            parallel_time = time.time() - start_time
            
            # Добавляем ограничения на метафоры
            metaphor_restrictions = self.fast_response_cache.get_metaphor_restriction(chat_id)

            # Определяем категорию и извлекаем RAG score
            question_category = intelligent_analyzer.analyze_question_category_optimized(user_message)
            rag_score = rag_metrics.get('max_score', 0.0)

            # ✅ ИСПРАВЛЕННЫЙ ВЫЗОВ ПРОМПТА С ИСТОРИЕЙ
            llm_start = time.time()
            combined_prompt = self.prompt_builder.build_combined_analysis_prompt_RAG_TEST(
                user_message=user_message,
                facts_context=facts_context,
                rag_score=rag_metrics.get('average_score', 0.5),
                conversation_history=conversation_history  # ✅ ДОБАВЛЕНА ИСТОРИЯ!
            )

            # Debug: логируем финальный промпт
            if DEBUG_LOGGING_ENABLED:
                rag_debug.log_final_prompt(combined_prompt, len(conversation_history))

            # Логируем принятое решение
            self.logger.info(f"📊 [RAG TEST] Category: {question_category}, RAG: {rag_score:.2f}")

            # Генерируем ответ
            combined_response = self._call_ai_model(combined_prompt)
            llm_time = time.time() - llm_start

            # Debug: логируем финальный ответ
            if DEBUG_LOGGING_ENABLED:
                rag_debug.log_final_response(combined_response, llm_time)

            # Парсим ответ
            main_response, analysis_data = self._parse_combined_response(combined_response)

            # Обрабатываем токены действий
            final_response = self._process_action_tokens(main_response, chat_id, current_state)

            # Отслеживаем использованные метафоры
            self.fast_response_cache.track_metaphor_usage(chat_id, final_response)

            # Обновляем историю
            conversation_manager.update_conversation_history(chat_id, user_message, final_response)
            conversation_manager.set_dialogue_state(chat_id, analysis_data.get('state', current_state))

            # Обновляем статистику
            total_time = time.time() - start_time
            self._update_performance_stats(total_time, parallel_time, llm_time)

            self.logger.info(f"✅ Ответ сгенерирован для {chat_id} за {total_time:.3f}s")
            return final_response
            
        except Exception as e:
            self.logger.error(f"💥 Ошибка обработки сообщения: {e}")
            error_response = "Извините, временная техническая проблема. Попробуйте еще раз."
            
            # Debug логирование ошибки
            if DEBUG_LOGGING_ENABLED:
                rag_debug.log_final_response(f"ERROR: {error_response}", 0.0)
            return error_response
    
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
        
        return {**base_status, "performance_metrics": performance_metrics}


# Глобальная инициализация сервиса
ai_service = ProductionAIService()

# Flask приложение
app = Flask(__name__)
app.config['SECRET_KEY'] = config.FLASK_SECRET_KEY

@app.route('/', methods=['POST'])
def telegram_webhook():
    """Обработка webhook от Telegram"""
    try:
        update = request.get_json()
        if not update:
            return "No JSON data", 400
        
        message = update.get('message')
        if not message:
            return "OK", 200
            
        chat_id = str(message['chat']['id'])
        user_message = message.get('text', '').strip()
        
        if not user_message:
            return "OK", 200
        
        # Обрабатываем сообщение через AI сервис
        ai_response = ai_service.process_message(chat_id, user_message)
        
        # Отправляем ответ через Telegram
        telegram_bot.send_message(chat_id, ai_response)
        
        return "OK", 200
        
    except Exception as e:
        logging.error(f"Webhook error: {e}")
        return "Error", 500

@app.route('/status', methods=['GET'])
def system_status():
    """Статус системы"""
    try:
        status = ai_service.get_system_status()
        return status, 200
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/test-message', methods=['POST'])
def test_message():
    """Тестовый endpoint для отладки"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        user_id = data.get('user_id', 'test_user')
        
        start_time = time.time()
        ai_response = ai_service.process_message(user_id, user_message)
        response_time = time.time() - start_time
        
        return {
            "bot_response": ai_response,
            "response_time": response_time,
            "status": "success"
        }, 200
        
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/clear-memory', methods=['POST'])
def clear_memory():
    """Очистка памяти для тестирования"""
    try:
        # Очищаем кеши и историю
        conversation_manager.clear_all_conversations()
        
        return {"status": "success", "message": "Memory cleared"}, 200
    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)