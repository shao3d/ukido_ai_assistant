# config.py
"""
Центр управления всеми настройками проекта Ukido AI Assistant.
Этот модуль отвечает за загрузку переменных окружения и предоставление
единого интерфейса для доступа к конфигурации во всех частях приложения.
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv

class Config:
    """
    Класс конфигурации, который централизованно управляет всеми настройками.
    Принцип: все настройки в одном месте, легко найти и изменить.
    """
    
    def __init__(self):
        # Загружаем переменные окружения из .env файла
        load_dotenv()
        
        # Фиксируем время старта сервера для фильтрации старых сообщений
        # ВАЖНО: Это может быть источником проблемы с Telegram ботом!
        self.SERVER_START_TIME = datetime.now().timestamp()
        
        # === ОСНОВНЫЕ API КЛЮЧИ ===
        # Проверяем наличие всех критически важных переменных
        self.TELEGRAM_BOT_TOKEN = self._get_required_env("TELEGRAM_BOT_TOKEN")
        self.GEMINI_API_KEY = self._get_required_env("GEMINI_API_KEY")
        self.OPENROUTER_API_KEY = self._get_required_env("OPENROUTER_API_KEY")
        self.PINECONE_API_KEY = self._get_required_env("PINECONE_API_KEY")
        self.PINECONE_HOST_FACTS = self._get_required_env("PINECONE_HOST_FACTS")
        self.HUBSPOT_API_KEY = self._get_required_env("HUBSPOT_API_KEY")
        
        # === ОПЦИОНАЛЬНЫЕ НАСТРОЙКИ ===
        self.REDIS_URL = os.getenv("REDIS_URL")  # Может отсутствовать
        self.BASE_URL = os.getenv('BASE_URL', 'https://ukidoaiassistant-production.up.railway.app')
        
        # === НАСТРОЙКИ СЕРВЕРА ===
        self.PORT = int(os.environ.get('PORT', 5000))
        self.DEBUG_MODE = os.environ.get('DEBUG', 'false').lower() == 'true'
        
        # === НАСТРОЙКИ ПАМЯТИ И ПРОИЗВОДИТЕЛЬНОСТИ ===
        self.CONVERSATION_MEMORY_SIZE = 15  # Количество сообщений в истории
        self.CONVERSATION_EXPIRATION_SECONDS = 3600  # Время жизни диалога (1 час)
        self.RAG_CACHE_TTL = 3600  # Время жизни кеша RAG (1 час)
        self.MAX_CACHE_SIZE = 1000  # Максимальный размер кеша
        self.MAX_FALLBACK_USERS = 1000  # Максимум пользователей в fallback памяти
        
        # === НАСТРОЙКИ МОДЕЛИ ===
        self.EMBEDDING_MODEL = 'models/text-embedding-004'  # Gemini модель для эмбеддингов
        
        # Инициализируем логирование
        self._setup_logging()
    
    def _get_required_env(self, var_name: str) -> str:
        """
        Получает обязательную переменную окружения.
        Если переменная отсутствует, выбрасывает исключение с понятным сообщением.
        """
        value = os.getenv(var_name)
        if not value:
            raise ValueError(f"Отсутствует обязательная переменная окружения: {var_name}")
        return value
    
    def _setup_logging(self):
        """
        Настраивает логирование для всего приложения.
        Использует единый формат для всех модулей.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Создаем основной логгер для конфигурации
        self.logger = logging.getLogger(__name__)
        self.logger.info("🔧 Конфигурация успешно загружена")
        self.logger.info(f"🚀 Режим отладки: {'включен' if self.DEBUG_MODE else 'отключен'}")
        self.logger.info(f"🌐 Base URL: {self.BASE_URL}")
    
    def validate_configuration(self) -> bool:
        """
        Проверяет корректность всех настроек.
        Возвращает True, если все настройки валидны.
        """
        try:
            # Проверяем, что все критические ключи присутствуют и не пустые
            critical_keys = [
                self.TELEGRAM_BOT_TOKEN,
                self.GEMINI_API_KEY, 
                self.OPENROUTER_API_KEY,
                self.PINECONE_API_KEY,
                self.PINECONE_HOST_FACTS,
                self.HUBSPOT_API_KEY
            ]
            
            for key in critical_keys:
                if not key or len(key.strip()) < 10:  # Минимальная длина для API ключей
                    return False
            
            # Проверяем числовые параметры
            if self.PORT < 1 or self.PORT > 65535:
                return False
                
            if self.CONVERSATION_MEMORY_SIZE < 1:
                return False
            
            self.logger.info("✅ Все настройки валидны")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка валидации конфигурации: {e}")
            return False
    
    def get_telegram_webhook_url(self) -> str:
        """
        Формирует полный URL для Telegram webhook.
        Полезно для настройки бота.
        """
        return f"{self.BASE_URL}/"
    
    def get_lesson_url(self, user_id: str) -> str:
        """
        Формирует персонализированную ссылку на урок для пользователя.
        """
        return f"{self.BASE_URL}/lesson?user_id={user_id}"


# Создаем глобальный экземпляр конфигурации
# Этот подход называется "Singleton pattern" - один объект на все приложение
config = Config()

# Проверяем конфигурацию при импорте модуля
if not config.validate_configuration():
    raise RuntimeError("Конфигурация содержит ошибки. Проверьте переменные окружения.")
