# telegram_bot.py (CRITICAL CONNECTION POOLING FIX)
"""
КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Использование unified connection pooling

ИЗМЕНЕНИЯ:
1. Заменен прямой requests на unified_http_client
2. Добавлен proper error handling с retry механизмами
3. Performance metrics для Telegram API calls
4. Thread-safe operations
"""

import threading
import logging
from datetime import datetime
from flask import request, render_template
from typing import Optional, Dict, Any, Callable
from config import config

# ИСПРАВЛЕНО: Используем unified HTTP client вместо прямых requests
try:
    from unified_http_client import http_client
except ImportError:
    # Fallback если unified_http_client не найден
    import requests
    http_client = None
    logging.getLogger(__name__).warning("Unified HTTP client не найден, используется fallback")


class TelegramBot:
    """
    THREAD-SAFE версия Telegram бота с unified connection pooling
    
    КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:
    1. Unified HTTP connection pooling
    2. Proper retry mechanisms через unified client
    3. Performance metrics для мониторинга
    4. Enhanced error handling
    """
    
    def __init__(self):
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.logger = logging.getLogger(__name__)
        
        # Callback функция для обработки сообщений
        self.message_handler: Optional[Callable] = None
        
        # Performance metrics
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'api_errors': 0,
            'webhook_calls': 0,
            'avg_send_time': 0
        }
        self.metrics_lock = threading.Lock()
        
        # Проверяем доступность unified HTTP client
        self.use_unified_client = http_client is not None
        if not self.use_unified_client:
            import requests
            self.fallback_session = requests.Session()
            self.logger.warning("⚠️ Используется fallback HTTP client")
        
        self.logger.info("🤖 Thread-safe Telegram бот инициализирован")
    
    def set_message_handler(self, handler: Callable):
        """
        Устанавливает функцию-обработчик для входящих сообщений.
        """
        self.message_handler = handler
        self.logger.info("📥 Обработчик сообщений установлен")
    
    def send_message(self, chat_id: str, text: str) -> bool:
        """
        ИСПРАВЛЕНО: Отправка сообщения через unified HTTP client
        """
        if not text or not chat_id:
            self.logger.warning("Пустое сообщение или chat_id")
            return False
        
        url = f"{self.base_url}/sendMessage"
        
        # Подготавливаем данные
        data = {
            'chat_id': chat_id,
            'text': text,
            'parse_mode': 'HTML'  # Поддержка базового форматирования
        }
        
        start_time = time.time()
        
        try:
            # ИСПРАВЛЕНО: Используем unified HTTP client
            if self.use_unified_client:
                response = http_client.post(
                    url=url,
                    service_name='telegram',
                    json=data,
                    timeout=(5, 10)  # Быстрые timeouts для Telegram
                )
            else:
                # Fallback на прямые requests
                response = self.fallback_session.post(url, json=data, timeout=(5, 10))
            
            send_time = time.time() - start_time
            
            if response.status_code == 200:
                with self.metrics_lock:
                    self.metrics['messages_sent'] += 1
                    self._update_avg_send_time(send_time)
                
                self.logger.info(f"✅ Сообщение отправлено в {chat_id} ({send_time:.3f}s)")
                return True
            else:
                error_data = response.json() if response.content else {}
                self.logger.error(f"❌ Telegram API error {response.status_code}: {error_data}")
                
                with self.metrics_lock:
                    self.metrics['api_errors'] += 1
                return False
                
        except Exception as e:
            with self.metrics_lock:
                self.metrics['api_errors'] += 1
            self.logger.error(f"💥 Критическая ошибка отправки сообщения: {e}")
            return False
    
    def handle_webhook(self) -> tuple:
        """
        THREAD-SAFE обработка webhook от Telegram
        """
        try:
            with self.metrics_lock:
                self.metrics['webhook_calls'] += 1
            
            data = request.get_json()
            if not data:
                self.logger.warning("Пустые данные webhook")
                return "No data", 400
            
            # Извлекаем сообщение из обновления
            message = data.get('message')
            if not message:
                self.logger.info("Webhook без message (возможно, другой тип обновления)")
                return "OK", 200
            
            # Проверяем фильтр времени
            message_date = message.get('date', 0)
            if message_date < config.SERVER_START_TIME:
                self.logger.info(f"Пропускаем старое сообщение: {message_date}")
                return "OK", 200
            
            # Извлекаем данные сообщения
            chat_id = str(message['chat']['id'])
            text = message.get('text', '').strip()
            
            if not text:
                self.logger.info("Получено сообщение без текста")
                return "OK", 200
            
            with self.metrics_lock:
                self.metrics['messages_received'] += 1
            
            self.logger.info(f"📨 Получено сообщение от {chat_id}: {text[:50]}...")
            
            # Обрабатываем сообщение через установленный handler
            if self.message_handler:
                try:
                    # Асинхронная обработка для не блокирования webhook
                    def process_message():
                        try:
                            response = self.message_handler(chat_id, text)
                            self.send_message(chat_id, response)
                        except Exception as e:
                            self.logger.error(f"Ошибка в message_handler: {e}")
                            # Отправляем fallback сообщение
                            self.send_message(chat_id, "Извините, временная техническая проблема. Попробуйте еще раз.")
                    
                    # Запускаем обработку в отдельном потоке
                    threading.Thread(target=process_message, daemon=True).start()
                    
                except Exception as e:
                    self.logger.error(f"Ошибка запуска обработки сообщения: {e}")
                    return "Processing Error", 500
            else:
                self.logger.warning("Message handler не установлен")
                return "No Handler", 500
            
            return "OK", 200
            
        except Exception as e:
            self.logger.error(f"💥 Критическая ошибка webhook: {e}")
            return "Internal Error", 500
    
    def show_lesson_page(self) -> str:
        """
        Отображение интерактивной страницы урока с персонализацией
        """
        try:
            user_id = request.args.get('user_id', 'demo')
            
            # Базовые данные для персонализации
            lesson_data = {
                'user_id': user_id,
                'lesson_title': 'Пробный урок по развитию soft-skills',
                'current_time': datetime.now().strftime("%H:%M"),
                'current_date': datetime.now().strftime("%d.%m.%Y")
            }
            
            return render_template('lesson.html', **lesson_data)
            
        except Exception as e:
            self.logger.error(f"Ошибка отображения страницы урока: {e}")
            return f"<h1>Ошибка загрузки урока</h1><p>{e}</p>", 500
    
    def _update_avg_send_time(self, send_time: float):
        """Thread-safe обновление среднего времени отправки"""
        current_avg = self.metrics['avg_send_time']
        messages_sent = self.metrics['messages_sent']
        
        if messages_sent == 1:
            self.metrics['avg_send_time'] = send_time
        else:
            new_avg = (current_avg * (messages_sent - 1) + send_time) / messages_sent
            self.metrics['avg_send_time'] = new_avg
    
    def get_metrics(self) -> Dict[str, Any]:
        """Thread-safe получение метрик производительности"""
        with self.metrics_lock:
            metrics_copy = self.metrics.copy()
        
        # Добавляем статус unified client
        metrics_copy['unified_client_status'] = 'active' if self.use_unified_client else 'fallback'
        
        return metrics_copy
    
    def get_bot_info(self) -> Dict[str, Any]:
        """
        Получение информации о боте для диагностики
        """
        url = f"{self.base_url}/getMe"
        
        try:
            if self.use_unified_client:
                response = http_client.get(
                    url=url,
                    service_name='telegram',
                    timeout=(3, 5)
                )
            else:
                response = self.fallback_session.get(url, timeout=(3, 5))
            
            if response.status_code == 200:
                bot_data = response.json()
                self.logger.info("✅ Информация о боте получена")
                return bot_data.get('result', {})
            else:
                self.logger.error(f"❌ Ошибка получения информации о боте: {response.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"💥 Ошибка запроса информации о боте: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup ресурсов"""
        try:
            if hasattr(self, 'fallback_session'):
                self.fallback_session.close()
            self.logger.info("🤖 Telegram bot cleanup completed")
        except Exception as e:
            self.logger.error(f"Telegram bot cleanup error: {e}")


# Создаем глобальный экземпляр бота
telegram_bot = TelegramBot()

# Импортируем time для metrics
import time