# unified_http_client.py (NEW MODULE)
"""
КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Unified Connection Management для всей системы

ПРОБЛЕМА: Разные модули используют разные подходы к HTTP requests:
- app.py: ProductionConnectionPool
- telegram_bot.py: requests напрямую  
- hubspot_client.py: requests напрямую

РЕШЕНИЕ: Единый connection manager для всех HTTP операций в системе
"""

import requests
import threading
import atexit
import logging
from typing import Dict, Any, Optional
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class UnifiedHTTPClient:
    """
    Единый HTTP клиент для всех модулей системы
    
    ВОЗМОЖНОСТИ:
    1. Connection pooling для всех HTTP requests
    2. Retry механизмы с exponential backoff
    3. Proper timeout configuration
    4. Thread-safe operations
    5. Resource cleanup
    6. Performance metrics
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern для единого connection pool"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Защита от повторной инициализации
        if hasattr(self, '_initialized'):
            return
            
        self.logger = logging.getLogger(__name__)
        
        # Создаем session с connection pooling
        self.session = requests.Session()
        
        # Настраиваем retry стратегию
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        # HTTP и HTTPS адаптеры с connection pooling
        adapter = HTTPAdapter(
            pool_connections=20,  # Увеличиваем для множественных модулей
            pool_maxsize=40,      # Увеличиваем pool size
            max_retries=retry_strategy,
            pool_block=False      # Не блокируем при недоступности pool
        )
        
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Настраиваем разумные timeouts
        self.default_timeout = (5, 15)  # (connect, read)
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'retry_attempts': 0,
            'avg_response_time': 0,
            'telegram_requests': 0,
            'hubspot_requests': 0,
            'gemini_requests': 0
        }
        self.metrics_lock = threading.Lock()
        
        # Регистрируем cleanup
        atexit.register(self.cleanup)
        
        self._initialized = True
        self.logger.info("🌐 Unified HTTP Client инициализирован")
    
    def make_request(self, method: str, url: str, 
                    timeout: Optional[tuple] = None,
                    service_name: str = 'unknown',
                    **kwargs) -> requests.Response:
        """
        Unified метод для всех HTTP requests в системе
        
        Args:
            method: HTTP метод (GET, POST, etc.)
            url: URL для запроса
            timeout: Tuple (connect, read) timeout
            service_name: Название сервиса для метрик
            **kwargs: Дополнительные параметры для requests
            
        Returns:
            requests.Response object
        """
        start_time = time.time()
        timeout = timeout or self.default_timeout
        
        try:
            with self.metrics_lock:
                self.metrics['total_requests'] += 1
                if service_name in ['telegram', 'hubspot', 'gemini']:
                    self.metrics[f'{service_name}_requests'] += 1
            
            # Выполняем запрос через unified session
            response = self.session.request(
                method=method,
                url=url,
                timeout=timeout,
                **kwargs
            )
            
            # Обновляем метрики успеха
            response_time = time.time() - start_time
            with self.metrics_lock:
                self.metrics['successful_requests'] += 1
                self._update_avg_response_time(response_time)
            
            self.logger.debug(f"✅ {service_name} {method} {url} - {response.status_code} ({response_time:.3f}s)")
            return response
            
        except requests.exceptions.RetryError as e:
            with self.metrics_lock:
                self.metrics['failed_requests'] += 1
                self.metrics['retry_attempts'] += 1
            self.logger.error(f"❌ {service_name} {method} {url} - Retry exhausted: {e}")
            raise
            
        except requests.exceptions.Timeout as e:
            with self.metrics_lock:
                self.metrics['failed_requests'] += 1
            self.logger.error(f"⏰ {service_name} {method} {url} - Timeout: {e}")
            raise
            
        except Exception as e:
            with self.metrics_lock:
                self.metrics['failed_requests'] += 1
            self.logger.error(f"💥 {service_name} {method} {url} - Error: {e}")
            raise
    
    def post(self, url: str, service_name: str = 'unknown', 
             timeout: Optional[tuple] = None, **kwargs) -> requests.Response:
        """Convenience method for POST requests"""
        return self.make_request('POST', url, timeout, service_name, **kwargs)
    
    def get(self, url: str, service_name: str = 'unknown',
            timeout: Optional[tuple] = None, **kwargs) -> requests.Response:
        """Convenience method for GET requests"""
        return self.make_request('GET', url, timeout, service_name, **kwargs)
    
    def put(self, url: str, service_name: str = 'unknown',
            timeout: Optional[tuple] = None, **kwargs) -> requests.Response:
        """Convenience method for PUT requests"""
        return self.make_request('PUT', url, timeout, service_name, **kwargs)
    
    def delete(self, url: str, service_name: str = 'unknown',
               timeout: Optional[tuple] = None, **kwargs) -> requests.Response:
        """Convenience method for DELETE requests"""
        return self.make_request('DELETE', url, timeout, service_name, **kwargs)
    
    def _update_avg_response_time(self, response_time: float):
        """Thread-safe обновление среднего времени ответа"""
        current_avg = self.metrics['avg_response_time']
        successful_requests = self.metrics['successful_requests']
        
        if successful_requests == 1:
            self.metrics['avg_response_time'] = response_time
        else:
            new_avg = (current_avg * (successful_requests - 1) + response_time) / successful_requests
            self.metrics['avg_response_time'] = new_avg
    
    def get_metrics(self) -> Dict[str, Any]:
        """Thread-safe получение метрик производительности"""
        with self.metrics_lock:
            metrics_copy = self.metrics.copy()
        
        # Добавляем вычисляемые метрики
        total_requests = metrics_copy['total_requests']
        if total_requests > 0:
            metrics_copy['success_rate'] = round(
                (metrics_copy['successful_requests'] / total_requests) * 100, 1
            )
            metrics_copy['failure_rate'] = round(
                (metrics_copy['failed_requests'] / total_requests) * 100, 1
            )
        
        return metrics_copy
    
    def cleanup(self):
        """Правильная очистка всех ресурсов"""
        try:
            if hasattr(self, 'session'):
                self.session.close()
                self.logger.info("🌐 Unified HTTP Client закрыт")
        except Exception as e:
            self.logger.error(f"Ошибка при закрытии HTTP client: {e}")
    
    def __del__(self):
        """Backup cleanup"""
        self.cleanup()


# Создаем глобальный экземпляр unified HTTP client
http_client = UnifiedHTTPClient()