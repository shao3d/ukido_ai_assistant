# main.py
"""
Главная точка входа для Ukido AI Assistant.

Этот файл служит центральным координатором всех модулей системы.
Здесь мы соединяем Telegram бота, систему диалогов, RAG поиск и HubSpot интеграцию
в единое целое, следуя принципам модульной архитектуры.

Архитектурная философия:
- Каждый модуль отвечает за свою область ответственности
- Зависимости инжектируются через конструкторы (Dependency Injection)
- Центральная точка конфигурации через config.py
- Graceful degradation при недоступности внешних сервисов
"""

import logging
import time
from flask import Flask, request
from typing import Dict, Any

# Импортируем наши модули
from config import config
from telegram_bot import telegram_bot
from conversation import conversation_manager
from rag_system import rag_system
from hubspot_client import hubspot_client


class AIAssistantService:
    """
    Центральный сервис, который координирует работу всех модулей.
    
    Это пример паттерна "Service Layer" - слой, который инкапсулирует
    бизнес-логику и координирует взаимодействие между различными
    компонентами системы.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Инициализируем AI модель для генерации ответов
        self._init_ai_model()
        
        # Связываем модули друг с другом
        self._setup_module_connections()
        
        self.logger.info("🚀 AI Assistant сервис инициализирован")
    
    def _init_ai_model(self):
        """
        Инициализирует AI модель для генерации ответов.
        В данном случае используем OpenRouter API с GPT-4o mini.
        """
        # Пока оставляем простую инициализацию
        # В будущем здесь можно добавить более сложную логику
        self.ai_model_available = True
        self.logger.info("🤖 AI модель готова к работе")
    
    def _setup_module_connections(self):
        """
        Устанавливает связи между модулями.
        
        Это пример паттерна "Dependency Injection" - мы явно указываем,
        какие зависимости нужны каждому модулю, вместо того чтобы
        позволить им создавать зависимости внутри себя.
        """
        # Telegram бот должен знать, как обрабатывать сообщения
        telegram_bot.set_message_handler(self.process_user_message)
        
        self.logger.info("🔗 Модули успешно связаны")
    
    def process_user_message(self, chat_id: str, user_message: str) -> str:
        """
        Центральная функция обработки сообщений пользователя.
        
        Это "сердце" нашего AI ассистента. Здесь координируется работа:
        1. Анализ состояния диалога
        2. Поиск релевантной информации через RAG
        3. Генерация персонализированного ответа
        4. Обновление состояния и истории диалога
        
        Args:
            chat_id: Идентификатор чата пользователя
            user_message: Текст сообщения от пользователя
            
        Returns:
            str: Сгенерированный ответ для пользователя
        """
        process_start = time.time()
        
        try:
            self.logger.info(f"🔄 Начинаем обработку сообщения от {chat_id}")
            
            # Шаг 1: Получаем текущее состояние диалога
            current_state = conversation_manager.get_dialogue_state(chat_id)
            self.logger.info(f"📊 Текущее состояние диалога: {current_state}")
            
            # Шаг 2: Получаем историю разговора для контекста
            conversation_history = conversation_manager.get_conversation_history(chat_id)
            
            # Шаг 3: Ищем релевантную информацию в базе знаний
            facts_context, rag_metrics = rag_system.search_knowledge_base(
                user_message, 
                conversation_history
            )
            
            self.logger.info(f"🔍 RAG поиск: найдено {rag_metrics.get('chunks_found', 0)} релевантных фрагментов")
            
            # Шаг 4: Анализируем, нужно ли изменить состояние диалога
            new_state = conversation_manager.analyze_message_for_state_transition(
                user_message, 
                current_state
            )
            
            if new_state != current_state:
                self.logger.info(f"🔄 Переход состояния: {current_state} → {new_state}")
                conversation_manager.update_dialogue_state(chat_id, new_state)
            
            # Шаг 5: Генерируем ответ на основе всей собранной информации
            ai_response = self._generate_ai_response(
                user_message=user_message,
                current_state=new_state,
                conversation_history=conversation_history,
                facts_context=facts_context,
                chat_id=chat_id
            )
            
            # Шаг 6: Сохраняем диалог в историю
            conversation_manager.update_conversation_history(
                chat_id, 
                user_message, 
                ai_response
            )
            
            processing_time = time.time() - process_start
            self.logger.info(f"✅ Обработка завершена за {processing_time:.2f}с")
            
            return ai_response
            
        except Exception as e:
            processing_time = time.time() - process_start
            self.logger.error(f"💥 Ошибка обработки сообщения от {chat_id}: {e}", exc_info=True)
            
            # Graceful degradation - возвращаем вежливое сообщение об ошибке
            return "Извините, возникла временная техническая проблема. Пожалуйста, попробуйте перефразировать вопрос или обратитесь позже."
    
    def _generate_ai_response(
        self, 
        user_message: str, 
        current_state: str,
        conversation_history: list,
        facts_context: str,
        chat_id: str
    ) -> str:
        """
        Генерирует ответ AI на основе контекста и состояния диалога.
        
        Это временная реализация, которая объединяет логику из вашего
        оригинального кода. В будущем можно вынести в отдельный модуль.
        """
        try:
            # Формируем промпт в зависимости от состояния диалога
            system_prompt = self._get_state_specific_prompt(current_state)
            
            # Подготавливаем контекст истории
            history_context = "\n".join(conversation_history) if conversation_history else "Это начало диалога."
            
            # Собираем полный промпт
            full_prompt = f"""{system_prompt}

История диалога:
{history_context}

Информация о школе Ukido из базы знаний:
{facts_context}

Пользователь: {user_message}
Ассистент:"""

            # Вызываем AI модель (используем функцию из оригинального кода)
            ai_response = self._call_ai_model(full_prompt)
            
            # Обрабатываем специальные токены для генерации ссылок
            ai_response = self._process_action_tokens(ai_response, chat_id, current_state)
            
            return ai_response
            
        except Exception as e:
            self.logger.error(f"Ошибка генерации AI ответа: {e}")
            return "Извините, не могу сформулировать ответ. Попробуйте перефразировать вопрос."
    
    def _get_state_specific_prompt(self, state: str) -> str:
        """
        Возвращает промпт, специально настроенный для текущего состояния диалога.
        
        Это ключевая часть "машины состояний" - AI ведет себя по-разному
        в зависимости от того, на каком этапе находится разговор с пользователем.
        """
        base_personality = """Ты — AI-ассистент школы soft skills "Ukido". Твоя роль — мудрый, ироничный наставник с мировоззрением и стилем Михаила Жванецкого. Ты говоришь парадоксами и жизненными наблюдениями. Твоя главная задача — помочь родителю разобраться, а не продать любой ценой."""
        
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
        
        return f"{base_personality}\n\n{state_instructions.get(state, state_instructions['greeting'])}"
    
    def _call_ai_model(self, prompt: str) -> str:
        """
        Вызывает AI модель для генерации ответа.
        
        Пока используем простую реализацию через OpenRouter.
        В будущем можно добавить retry логику, альтернативные модели и т.д.
        """
        import requests
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
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
                self.logger.error(f"OpenRouter API error: {response.status_code}")
                return "Извините, временная проблема с генерацией ответа."
                
        except Exception as e:
            self.logger.error(f"Ошибка вызова AI модели: {e}")
            return "Извините, временная проблема с генерацией ответа."
    
    def _process_action_tokens(self, response: str, chat_id: str, current_state: str) -> str:
        """
        Обрабатывает специальные токены в ответе AI и заменяет их на реальные ссылки.
        
        Например, [ACTION:SEND_LESSON_LINK] заменяется на персонализированную ссылку урока.
        """
        # Если находимся в состоянии closing, но AI забыл добавить ссылку - добавляем принудительно
        if current_state == 'closing' and "[ACTION:SEND_LESSON_LINK]" not in response and "/lesson?user_id=" not in response:
            lesson_url = config.get_lesson_url(chat_id)
            response += f"\n\nЧтобы попробовать, вот ссылка для записи на пробный урок: {lesson_url}"
            self.logger.info("Токен не был сгенерирован AI, ссылка добавлена принудительно")
        
        # Заменяем токен на реальную ссылку
        if "[ACTION:SEND_LESSON_LINK]" in response:
            lesson_url = config.get_lesson_url(chat_id)
            response = response.replace("[ACTION:SEND_LESSON_LINK]", lesson_url)
            self.logger.info("Обработан токен [ACTION:SEND_LESSON_LINK] - ссылка вставлена")
        
        return response
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Возвращает статус всех компонентов системы.
        Полезно для health checks и мониторинга.
        """
        return {
            "config_valid": config.validate_configuration(),
            "telegram_bot_ready": telegram_bot is not None,
            "conversation_manager_ready": conversation_manager is not None,
            "rag_system_stats": rag_system.get_stats(),
            "ai_model_available": self.ai_model_available
        }


# Создаем основные компоненты приложения
app = Flask(__name__)
ai_service = AIAssistantService()


# === МАРШРУТЫ FLASK ===

@app.route('/', methods=['POST'])
def telegram_webhook():
    """
    Основной webhook для получения сообщений от Telegram.
    Делегирует обработку telegram_bot модулю.
    """
    return telegram_bot.handle_webhook()


@app.route('/lesson')
def lesson_page():
    """
    Страница интерактивного урока.
    Делегирует обработку telegram_bot модулю.
    """
    return telegram_bot.show_lesson_page()


@app.route('/submit-lesson-form', methods=['POST'])
def submit_lesson_form():
    """Обработка формы урока и отправка данных в HubSpot"""
    try:
        form_data = request.get_json()
        if not form_data:
            return {"success": False, "error": "Нет данных"}, 400
        
        logger = logging.getLogger(__name__)
        logger.info(f"Получены данные формы: {form_data.get('firstName')} {form_data.get('lastName')}")
        
        # Отправляем в HubSpot
        hubspot_success = hubspot_client.create_contact(form_data)
        
        if hubspot_success:
            return {"success": True, "message": "Данные сохранены в CRM"}, 200
        else:
            return {"success": True, "message": "Данные получены"}, 200  # Graceful degradation
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка обработки формы: {e}")
        return {"success": False, "error": str(e)}, 500


@app.route('/hubspot-webhook', methods=['POST'])
def hubspot_webhook():
    """Webhook для автоматических follow-up сообщений от HubSpot"""
    try:
        webhook_data = request.get_json()
        if not webhook_data:
            return "No data", 400
        
        message_type = request.args.get('message_type', 'first_follow_up')
        success = hubspot_client.process_webhook(webhook_data, message_type)
        
        return "OK" if success else "Error", 200
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Ошибка HubSpot webhook: {e}")
        return "Error", 500


@app.route('/health')
def health_check():
    """
    Endpoint для проверки состояния системы.
    Полезно для мониторинга и автоматических health checks.
    """
    return ai_service.get_system_status()


@app.route('/metrics')
def metrics():
    """
    Endpoint для получения метрик производительности.
    Полезно для мониторинга и оптимизации.
    """
    return {
        "system_status": ai_service.get_system_status(),
        "rag_stats": rag_system.get_stats(),
    }


# === ТОЧКА ВХОДА ===

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("🚀 ЗАПУСК UKIDO AI ASSISTANT (МОДУЛЬНАЯ АРХИТЕКТУРА)")
    logger.info("=" * 60)
    logger.info("🎯 Активированные модули:")
    logger.info("   - config.py: Централизованная конфигурация")
    logger.info("   - telegram_bot.py: Чистое взаимодействие с Telegram API")
    logger.info("   - conversation.py: Управление состояниями и памятью диалогов")
    logger.info("   - rag_system.py: Поиск в базе знаний с кешированием")
    logger.info("   - main.py: Центральная координация всех компонентов")
    logger.info("=" * 60)
    
    # Проверяем готовность всех систем
    status = ai_service.get_system_status()
    logger.info(f"📊 Статус системы: {status}")
    
    # Запускаем приложение
    app.run(
        debug=config.DEBUG_MODE,
        port=config.PORT,
        host='0.0.0.0'
    )
