# hubspot_client.py (CRITICAL CONNECTION POOLING FIX)
"""
КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Использование unified connection pooling

ИЗМЕНЕНИЯ:
1. Заменен прямой requests на unified_http_client
2. Добавлен proper error handling с retry механизмами
3. Performance metrics для HubSpot API calls
4. Thread-safe operations
5. Асинхронная обработка для не блокирования основного потока
"""

import threading
import logging
import time
from typing import Dict, Any, Optional
from config import config

# ИСПРАВЛЕНО: Используем unified HTTP client
try:
    from unified_http_client import http_client
except ImportError:
    # Fallback если unified_http_client не найден
    import requests
    http_client = None
    logging.getLogger(__name__).warning("Unified HTTP client не найден, используется fallback")

# Импортируем telegram_bot для отправки сообщений
# ИСПРАВЛЕНО: Lazy import для избежания circular dependencies
telegram_bot = None


class HubSpotClient:
    """
    THREAD-SAFE версия HubSpot клиента с unified connection pooling
    
    КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:
    1. Unified HTTP connection pooling
    2. Асинхронная обработка webhook для не блокирования
    3. Performance metrics и error tracking
    4. Enhanced error handling с graceful degradation
    """
    
    def __init__(self):
        self.api_key = config.HUBSPOT_API_KEY
        self.base_url = "https://api.hubapi.com"
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.metrics = {
            'contacts_created': 0,
            'webhooks_processed': 0,
            'api_errors': 0,
            'avg_create_time': 0,
            'follow_up_messages_sent': 0
        }
        self.metrics_lock = threading.Lock()
        
        # Проверяем доступность unified HTTP client
        self.use_unified_client = http_client is not None
        if not self.use_unified_client:
            import requests
            self.fallback_session = requests.Session()
            self.logger.warning("⚠️ Используется fallback HTTP client для HubSpot")
        
        self.logger.info("🔗 Thread-safe HubSpot клиент инициализирован")
    
    def create_contact(self, form_data: Dict[str, Any]) -> bool:
        """
        ИСПРАВЛЕНО: Создание контакта через unified HTTP client
        """
        # Валидация данных
        required_fields = ['firstName', 'lastName', 'email']
        for field in required_fields:
            if not form_data.get(field):
                self.logger.error(f"Отсутствует обязательное поле: {field}")
                return False
        
        url = f"{self.base_url}/crm/v3/objects/contacts"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        contact_data = {
            "properties": {
                "firstname": str(form_data["firstName"])[:50],
                "lastname": str(form_data["lastName"])[:50], 
                "email": str(form_data["email"])[:100],
                "telegram_user_id": str(form_data.get("userId", ""))[:20]
            }
        }
        
        start_time = time.time()
        
        try:
            # ИСПРАВЛЕНО: Используем unified HTTP client
            if self.use_unified_client:
                response = http_client.post(
                    url=url,
                    service_name='hubspot',
                    headers=headers,
                    json=contact_data,
                    timeout=(10, 30)  # Увеличенные timeouts для HubSpot API
                )
            else:
                # Fallback на прямые requests
                response = self.fallback_session.post(
                    url, headers=headers, json=contact_data, timeout=(10, 30)
                )
            
            create_time = time.time() - start_time
            
            if response.status_code == 201:
                contact_info = response.json()
                contact_id = contact_info.get('id', 'unknown')
                
                with self.metrics_lock:
                    self.metrics['contacts_created'] += 1
                    self._update_avg_create_time(create_time)
                
                self.logger.info(f"✅ Контакт создан в HubSpot: {contact_id} ({create_time:.3f}s)")
                
                # Асинхронно планируем follow-up сообщения
                self._schedule_follow_up_messages_async(form_data)
                
                return True
            else:
                error_data = response.json() if response.content else {}
                self.logger.error(f"❌ HubSpot API error {response.status_code}: {error_data}")
                
                with self.metrics_lock:
                    self.metrics['api_errors'] += 1
                return False
                
        except Exception as e:
            with self.metrics_lock:
                self.metrics['api_errors'] += 1
            self.logger.error(f"💥 Критическая ошибка создания контакта: {e}")
            return False
    
    def _schedule_follow_up_messages_async(self, form_data: Dict[str, Any]):
        """
        ИСПРАВЛЕНО: Асинхронное планирование follow-up сообщений
        """
        def schedule_messages():
            try:
                user_id = form_data.get('userId')
                if not user_id:
                    self.logger.warning("UserId отсутствует для follow-up сообщений")
                    return
                
                # Получаем telegram_bot через lazy import
                global telegram_bot
                if telegram_bot is None:
                    from telegram_bot import telegram_bot as tb
                    telegram_bot = tb
                # Получаем имя пользователя
                first_name = form_data.get('firstName', 'друг')

                # Первое сообщение через 1 минуту
                self.logger.info(f"⏰ Планируем первое follow-up сообщение для {user_id}")
                threading.Timer(60, self._send_follow_up_message, 
                              args=(user_id, 'first_follow_up', first_name)).start()

                # Второе сообщение через 2 минуты
                self.logger.info(f"⏰ Планируем второе follow-up сообщение для {user_id}")
                threading.Timer(120, self._send_follow_up_message, 
                              args=(user_id, 'second_follow_up', first_name)).start()
                
            except Exception as e:
                self.logger.error(f"Ошибка планирования follow-up сообщений: {e}")
        
        # Запускаем планирование в отдельном потоке
        threading.Thread(target=schedule_messages, daemon=True).start()
    
    def _send_follow_up_message(self, user_id: str, message_type: str, first_name: str = 'друг'):
        """
        THREAD-SAFE отправка follow-up сообщения
        """
        try:
            messages = {
                'first_follow_up': f"""Ну что, {first_name}, как впечатления? 
Говорят, после хорошего спектакля хочется обсудить. А после нашего пробного урока — хочется либо записаться, либо забыть. Надеюсь, у вас первый вариант.
Если что, мы тут, на связи.""",
                
                'second_follow_up': f"""{first_name}, это снова мы. 
Не то чтобы мы скучали, но тишина в эфире — это как антракт, затянувшийся на два акта. Если вы еще думаете, это хорошо. Думать полезно. Но пока мы думаем, дети растут.
Может, все-таки решимся на разговор?"""
            }
            
            message_text = messages.get(message_type, "Спасибо за ваш интерес к нашим курсам!")
            
            # Получаем telegram_bot через lazy import
            global telegram_bot
            if telegram_bot is None:
                from telegram_bot import telegram_bot as tb
                telegram_bot = tb
            
            success = telegram_bot.send_message(user_id, message_text)
            
            if success:
                with self.metrics_lock:
                    self.metrics['follow_up_messages_sent'] += 1
                self.logger.info(f"✅ Follow-up сообщение отправлено: {message_type} -> {user_id}")
            else:
                self.logger.error(f"❌ Ошибка отправки follow-up сообщения: {message_type} -> {user_id}")
                
        except Exception as e:
            self.logger.error(f"💥 Критическая ошибка follow-up сообщения: {e}")
    
    def process_webhook(self, webhook_data: Dict[str, Any], message_type: str = 'first_follow_up'):
        """
        THREAD-SAFE обработка webhook от HubSpot
        """
        try:
            with self.metrics_lock:
                self.metrics['webhooks_processed'] += 1
            
            self.logger.info(f"📥 Обработка HubSpot webhook: {message_type}")
            
            # Извлекаем данные контакта из webhook
            contact_data = self._extract_contact_from_webhook(webhook_data)
            
            if contact_data and contact_data.get('telegram_user_id'):
                user_id = contact_data['telegram_user_id']
                
                # Асинхронно отправляем follow-up сообщение
                def send_webhook_message():
                    self._send_follow_up_message(user_id, message_type)
                
                threading.Thread(target=send_webhook_message, daemon=True).start()
                
                self.logger.info(f"✅ Webhook обработан для пользователя: {user_id}")
            else:
                self.logger.warning("Webhook не содержит данных пользователя или telegram_user_id")
                
        except Exception as e:
            self.logger.error(f"💥 Ошибка обработки webhook: {e}")
    
    def _extract_contact_from_webhook(self, webhook_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Извлечение данных контакта из webhook данных HubSpot
        """
        try:
            # HubSpot webhook может иметь различную структуру
            # Пытаемся извлечь данные из наиболее распространенных форматов
            
            if 'objectId' in webhook_data:
                # Простой webhook с ID объекта
                contact_id = webhook_data['objectId']
                return {'contact_id': contact_id}
            
            if 'events' in webhook_data:
                # Webhook с массивом событий
                events = webhook_data['events']
                if events and len(events) > 0:
                    first_event = events[0]
                    if 'objectId' in first_event:
                        contact_id = first_event['objectId']
                        return {'contact_id': contact_id}
            
            # Можно добавить дополнительную логику извлечения данных
            self.logger.warning("Не удалось извлечь данные контакта из webhook")
            return None
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения данных из webhook: {e}")
            return None
    
    def _update_avg_create_time(self, create_time: float):
        """Thread-safe обновление среднего времени создания контакта"""
        current_avg = self.metrics['avg_create_time']
        contacts_created = self.metrics['contacts_created']
        
        if contacts_created == 1:
            self.metrics['avg_create_time'] = create_time
        else:
            new_avg = (current_avg * (contacts_created - 1) + create_time) / contacts_created
            self.metrics['avg_create_time'] = new_avg
    
    def get_metrics(self) -> Dict[str, Any]:
        """Thread-safe получение метрик производительности"""
        with self.metrics_lock:
            metrics_copy = self.metrics.copy()
        
        # Добавляем статус unified client
        metrics_copy['unified_client_status'] = 'active' if self.use_unified_client else 'fallback'
        
        # Вычисляем дополнительные метрики
        if metrics_copy['contacts_created'] > 0:
            total_api_calls = metrics_copy['contacts_created'] + metrics_copy['api_errors']
            metrics_copy['success_rate'] = round(
                (metrics_copy['contacts_created'] / total_api_calls) * 100, 1
            )
        
        return metrics_copy
    
    def test_connection(self) -> bool:
        """
        Тестирование соединения с HubSpot API
        """
        url = f"{self.base_url}/crm/v3/objects/contacts"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            if self.use_unified_client:
                response = http_client.get(
                    url=f"{url}?limit=1",
                    service_name='hubspot',
                    headers=headers,
                    timeout=(5, 10)
                )
            else:
                response = self.fallback_session.get(
                    f"{url}?limit=1", headers=headers, timeout=(5, 10)
                )
            
            if response.status_code == 200:
                self.logger.info("✅ HubSpot API соединение успешно")
                return True
            else:
                self.logger.error(f"❌ HubSpot API тест неудачен: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"💥 Ошибка тестирования HubSpot соединения: {e}")
            return False
    
    def cleanup(self):
        """Cleanup ресурсов"""
        try:
            if hasattr(self, 'fallback_session'):
                self.fallback_session.close()
            self.logger.info("🔗 HubSpot client cleanup completed")
        except Exception as e:
            self.logger.error(f"HubSpot client cleanup error: {e}")


# Создаем глобальный экземпляр HubSpot клиента
hubspot_client = HubSpotClient()