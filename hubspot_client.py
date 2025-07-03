# hubspot_client.py
"""
HubSpot CRM интеграция для Ukido AI Assistant.
Обрабатывает создание контактов и автоматические follow-up сообщения.
"""

import requests
import logging
from typing import Dict, Any, Optional
from config import config
from telegram_bot import telegram_bot


class HubSpotClient:
    """Клиент для работы с HubSpot CRM API"""
    
    def __init__(self):
        self.api_key = config.HUBSPOT_API_KEY
        self.base_url = "https://api.hubapi.com"
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔗 HubSpot клиент инициализирован")
    
    def create_contact(self, form_data: Dict[str, Any]) -> bool:
        """
        Создает контакт в HubSpot CRM
        
        Args:
            form_data: Данные формы с полями firstName, lastName, email, userId
            
        Returns:
            bool: True если контакт создан успешно
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
        
        try:
            response = requests.post(url, headers=headers, json=contact_data, timeout=10)
            
            if response.status_code == 201:
                self.logger.info("✅ Контакт создан в HubSpot")
                return True
            else:
                self.logger.error(f"❌ HubSpot API error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка создания контакта: {e}")
            return False
    
    def process_webhook(self, webhook_data: Dict[str, Any], message_type: str) -> bool:
        """
        Обрабатывает webhook от HubSpot для автоматических сообщений
        
        Args:
            webhook_data: Данные webhook от HubSpot
            message_type: Тип сообщения (first_follow_up, second_follow_up)
            
        Returns:
            bool: True если сообщение отправлено успешно
        """
        try:
            properties = webhook_data.get('properties', {})
            first_name = properties.get('firstname', {}).get('value', 'друг')
            telegram_id = properties.get('telegram_user_id', {}).get('value')
            
            if not telegram_id:
                self.logger.error("❌ Не найден telegram_user_id в webhook")
                return False
            
            # Генерируем сообщение в зависимости от типа
            message_generators = {
                'first_follow_up': f"""Ну что, {first_name}, как впечатления? 

Говорят, после хорошего спектакля хочется обсудить. А после нашего пробного урока — хочется либо записаться, либо забыть. Надеюсь, у вас первый вариант.

Если что, мы тут, на связи.""",
                
                'second_follow_up': f"""{first_name}, это снова мы. 

Не то чтобы мы скучали, но тишина в эфире — это как антракт, затянувшийся на два акта. Если вы еще думаете, это хорошо. Думать полезно. Но пока мы думаем, дети растут.

Может, все-таки решимся на разговор?"""
            }
            
            message_to_send = message_generators.get(message_type)
            
            if message_to_send:
                success = telegram_bot.send_message(telegram_id, message_to_send)
                if success:
                    self.logger.info(f"✅ Follow-up '{message_type}' отправлено {telegram_id}")
                    return True
                else:
                    self.logger.error(f"❌ Не удалось отправить follow-up {telegram_id}")
                    return False
            else:
                self.logger.warning(f"⚠️ Неизвестный тип сообщения: {message_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка обработки HubSpot webhook: {e}")
            return False


# Создаем глобальный экземпляр HubSpot клиента
hubspot_client = HubSpotClient()
