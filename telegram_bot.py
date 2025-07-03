# telegram_bot.py
"""
Модуль для работы с Telegram Bot API.
Отвечает ТОЛЬКО за отправку/получение сообщений и обработку webhook'ов.
Не содержит бизнес-логику - только чистое взаимодействие с Telegram.
"""

import requests
import threading
import logging
from datetime import datetime
from flask import request, render_template
from typing import Optional, Dict, Any, Callable
from config import config


class TelegramBot:
    """
    Класс для работы с Telegram Bot API.
    Следует принципу единственной ответственности - только Telegram операции.
    """
    
    def __init__(self):
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.logger = logging.getLogger(__name__)
        
        # Callback функция для обработки сообщений (будет установлена извне)
        self.message_handler: Optional[Callable] = None
        
        self.logger.info("🤖 Telegram бот инициализирован")
    
    def set_message_handler(self, handler: Callable):
        """
        Устанавливает функцию-обработчик для входящих сообщений.
        Это пример паттерна "Dependency Injection" - мы не создаем зависимости внутри класса,
        а получаем их извне. Это делает код более гибким и тестируемым.
        """
        self.message_handler = handler
        self.logger.info("📥 Обработчик сообщений установлен")
    
    def send_message(self, chat_id: str, text: str) -> bool:
        """
        Отправляет сообщение пользователю в Telegram.
        Включает retry логику и валидацию.
        
        Args:
            chat_id: ID чата пользователя
            text: Текст сообщения
            
        Returns:
            bool: True если сообщение отправлено успешно
        """
        if not text or len(text.strip()) == 0:
            self.logger.warning("Попытка отправить пустое сообщение")
            return False
        
        # Telegram имеет лимит в 4096 символов на сообщение
        if len(text) > 4096:
            text = text[:4093] + "..."
            self.logger.warning("Сообщение обрезано до лимита Telegram")
        
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": str(chat_id),
            "text": text
        }
        
        # Реализуем retry логику - 3 попытки с паузами
        for attempt in range(3):
            try:
                response = requests.post(url, json=payload, timeout=10)
                
                if response.status_code == 200:
                    self.logger.info(f"✅ Сообщение отправлено пользователю {chat_id}")
                    return True
                else:
                    self.logger.warning(
                        f"⚠️ Telegram API вернул статус {response.status_code}: {response.text}"
                    )
                    
            except requests.exceptions.Timeout:
                self.logger.error(f"⏱️ Таймаут при отправке сообщения (попытка {attempt + 1}/3)")
            except Exception as e:
                self.logger.error(
                    f"❌ Ошибка при отправке сообщения (попытка {attempt + 1}/3): {e}"
                )
            
            # Пауза перед повторной попыткой (но не после последней)
            if attempt < 2:
                import time
                time.sleep(1)
        
        self.logger.error(f"❌ Не удалось отправить сообщение пользователю {chat_id} после 3 попыток")
        return False
    
    def process_message_in_background(self, chat_id: str, message_text: str):
        """
        Обрабатывает сообщение в фоновом потоке.
        Это критически важно для предотвращения блокировки webhook'а.
        
        ПРИНЦИП: Telegram ожидает ответ 200 OK в течение 10 секунд.
        Если мы будем обрабатывать сообщение в основном потоке (с RAG, LLM и т.д.),
        то превысим лимит времени и Telegram перестанет присылать обновления.
        """
        self.logger.info(f"🔄 Начало фоновой обработки для чата {chat_id}")
        
        try:
            if not self.message_handler:
                self.logger.error("❌ Обработчик сообщений не установлен!")
                return
            
            # Вызываем внешний обработчик (он будет из conversation модуля)
            response_text = self.message_handler(chat_id, message_text)
            
            # Отправляем ответ пользователю
            success = self.send_message(chat_id, response_text)
            
            if success:
                self.logger.info(f"✅ Фоновая обработка для {chat_id} завершена успешно")
            else:
                self.logger.error(f"❌ Не удалось отправить ответ пользователю {chat_id}")
                
        except Exception as e:
            # КРИТИЧЕСКИ ВАЖНО: логируем ошибки в потоках!
            # Без этого ошибки "проглатываются" и мы не знаем, что пошло не так
            self.logger.error(
                f"💥 Критическая ошибка в фоновом потоке для чата {chat_id}: {e}", 
                exc_info=True
            )
    
    def handle_webhook(self):
        """
        Обрабатывает входящие webhook'и от Telegram.
        
        АРХИТЕКТУРНОЕ РЕШЕНИЕ: Этот метод должен работать максимально быстро.
        Мы только извлекаем данные, запускаем фоновый поток и сразу возвращаем 200 OK.
        """
        try:
            update = request.get_json()
            
            # Базовая валидация структуры webhook'а
            if not update or "message" not in update:
                self.logger.debug("Получен webhook без сообщения (возможно, служебный)")
                return "OK", 200
            
            message = update.get("message", {})
            
            # ВАЖНО: Здесь может быть источник проблемы!
            # ВРЕМЕННО ОТКЛЮЧАЕМ ДЛЯ ДИАГНОСТИКИ
            # message_date = message.get("date")
            # if message_date and message_date < config.SERVER_START_TIME:
            #     self.logger.info(
            #         f"⏭️ Игнорируем старое сообщение (дата: {message_date}, "
            #         f"старт сервера: {config.SERVER_START_TIME})"
            #     )
            #     return "OK", 200
            
            # Проверяем, что это текстовое сообщение с чатом
            if "text" not in message or "chat" not in message:
                self.logger.debug("Получено не-текстовое сообщение (стикер, фото и т.д.)")
                return "OK", 200
            
            chat_id = message["chat"]["id"]
            received_text = message["text"]
            
            self.logger.info(f"📥 Получено сообщение от {chat_id}: {received_text[:50]}...")
            
            # Запускаем обработку в отдельном потоке
            thread = threading.Thread(
                target=self.process_message_in_background,
                args=(chat_id, received_text),
                daemon=True  # Поток автоматически завершится при завершении программы
            )
            thread.start()
            
            self.logger.info(f"🚀 Фоновая обработка запущена для чата {chat_id}")
            
            # Немедленно возвращаем успех Telegram'у
            return "OK", 200
            
        except Exception as e:
            # Даже при ошибке возвращаем 200, чтобы не сломать webhook
            self.logger.error(f"💥 Ошибка в webhook (до запуска потока): {e}")
            return "Error", 500
    
    def show_lesson_page(self):
        """
        Отображает страницу урока.
        Технически это не совсем Telegram функция, но поскольку ссылка
        генерируется ботом, оставляем здесь для логической связности.
        """
        user_id = request.args.get('user_id')
        self.logger.info(f"🎓 Запрос страницы урока для пользователя {user_id}")
        return render_template('lesson.html', user_id=user_id)
    
    def set_webhook(self, webhook_url: str) -> bool:
        """
        Устанавливает webhook URL для бота.
        Полезно для программной настройки webhook'а.
        """
        url = f"{self.base_url}/setWebhook"
        payload = {"url": webhook_url}
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("ok"):
                    self.logger.info(f"✅ Webhook установлен: {webhook_url}")
                    return True
                else:
                    self.logger.error(f"❌ Ошибка установки webhook: {result}")
                    return False
            else:
                self.logger.error(f"❌ HTTP ошибка при установке webhook: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Исключение при установке webhook: {e}")
            return False
    
    def get_webhook_info(self) -> Dict[str, Any]:
        """
        Получает информацию о текущем webhook'е.
        Полезно для диагностики проблем.
        """
        url = f"{self.base_url}/getWebhookInfo"
        
        try:
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}


# Создаем глобальный экземпляр Telegram бота
telegram_bot = TelegramBot()
