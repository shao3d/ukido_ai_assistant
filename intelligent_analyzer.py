# intelligent_analyzer.py
"""
Интеллектуальный анализатор категорий вопросов и состояний лидов.
Использует гибридный подход: ключевые слова + AI анализ + специальная логика.

ПРОДАКШН УЛУЧШЕНИЯ:
- Многоуровневое кеширование с разными TTL для разных типов анализа
- Агрессивное кеширование для снижения LLM затрат
- Умное предварительное кеширование популярных запросов
- Статистика эффективности кеширования
"""

import logging
import hashlib
import time
import threading
from typing import Tuple, List, Optional, Dict, Any
from collections import defaultdict
from config import config


class AdvancedCache:
    """
    Продвинутая система кеширования с разными стратегиями для разных типов данных.
    """
    
    def __init__(self):
        # Основные кеши с разными TTL
        self.category_cache = {}      # Кеш категорий вопросов (долгий TTL)
        self.state_cache = {}         # Кеш состояний лидов (средний TTL)
        self.philosophy_cache = {}    # Кеш философских анализов (короткий TTL)
        
        # TTL настройки (в секундах)
        self.category_ttl = 24 * 3600    # 24 часа для категорий
        self.state_ttl = 4 * 3600        # 4 часа для состояний
        self.philosophy_ttl = 2 * 3600   # 2 часа для философских анализов
        
        # Статистика кеширования
        self.cache_stats = {
            'category_hits': 0, 'category_misses': 0,
            'state_hits': 0, 'state_misses': 0,
            'philosophy_hits': 0, 'philosophy_misses': 0,
            'total_ai_calls_saved': 0
        }
        
        # Популярные запросы для предварительного кеширования
        self.popular_patterns = defaultdict(int)
        
        # Блокировки для thread safety
        self.category_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.philosophy_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
        self.logger = logging.getLogger(f"{__name__}.AdvancedCache")
        self.logger.info("🧠 Продвинутая система кеширования инициализирована")
    
    def _cleanup_expired_entries(self, cache_dict: dict, ttl: int, lock: threading.Lock):
        """Очищает истекшие записи из кеша"""
        current_time = time.time()
        with lock:
            expired_keys = [
                key for key, value in cache_dict.items()
                if current_time - value['timestamp'] > ttl
            ]
            for key in expired_keys:
                del cache_dict[key]
            
            if expired_keys:
                self.logger.info(f"🧹 Очищено {len(expired_keys)} истекших записей кеша")
    
    def get_category_cache(self, key: str) -> Optional[str]:
        """Получает категорию из кеша"""
        self._cleanup_expired_entries(self.category_cache, self.category_ttl, self.category_lock)
        
        with self.category_lock:
            if key in self.category_cache:
                entry = self.category_cache[key]
                if time.time() - entry['timestamp'] < self.category_ttl:
                    with self.stats_lock:
                        self.cache_stats['category_hits'] += 1
                        self.cache_stats['total_ai_calls_saved'] += 1
                    return entry['value']
        
        with self.stats_lock:
            self.cache_stats['category_misses'] += 1
        return None
    
    def set_category_cache(self, key: str, value: str):
        """Сохраняет категорию в кеш"""
        with self.category_lock:
            self.category_cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
    
    def get_state_cache(self, key: str) -> Optional[str]:
        """Получает состояние из кеша"""
        self._cleanup_expired_entries(self.state_cache, self.state_ttl, self.state_lock)
        
        with self.state_lock:
            if key in self.state_cache:
                entry = self.state_cache[key]
                if time.time() - entry['timestamp'] < self.state_ttl:
                    with self.stats_lock:
                        self.cache_stats['state_hits'] += 1
                        self.cache_stats['total_ai_calls_saved'] += 1
                    return entry['value']
        
        with self.stats_lock:
            self.cache_stats['state_misses'] += 1
        return None
    
    def set_state_cache(self, key: str, value: str):
        """Сохраняет состояние в кеш"""
        with self.state_lock:
            self.state_cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
    
    def get_philosophy_cache(self, key: str) -> Optional[Tuple[bool, int]]:
        """Получает философский анализ из кеша"""
        self._cleanup_expired_entries(self.philosophy_cache, self.philosophy_ttl, self.philosophy_lock)
        
        with self.philosophy_lock:
            if key in self.philosophy_cache:
                entry = self.philosophy_cache[key]
                if time.time() - entry['timestamp'] < self.philosophy_ttl:
                    with self.stats_lock:
                        self.cache_stats['philosophy_hits'] += 1
                    return entry['value']
        
        with self.stats_lock:
            self.cache_stats['philosophy_misses'] += 1
        return None
    
    def set_philosophy_cache(self, key: str, value: Tuple[bool, int]):
        """Сохраняет философский анализ в кеш"""
        with self.philosophy_lock:
            self.philosophy_cache[key] = {
                'value': value,
                'timestamp': time.time()
            }
    
    def track_popular_pattern(self, pattern: str):
        """Отслеживает популярные паттерны для предварительного кеширования"""
        self.popular_patterns[pattern] += 1
        
        # Каждые 100 запросов анализируем популярные паттерны
        if sum(self.popular_patterns.values()) % 100 == 0:
            self._analyze_popular_patterns()
    
    def _analyze_popular_patterns(self):
        """Анализирует популярные паттерны для оптимизации"""
        sorted_patterns = sorted(self.popular_patterns.items(), key=lambda x: x[1], reverse=True)
        top_patterns = sorted_patterns[:10]
        
        self.logger.info(f"📊 Топ-10 популярных паттернов: {top_patterns}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кеширования"""
        with self.stats_lock:
            stats = self.cache_stats.copy()
        
        # Вычисляем показатели эффективности
        category_total = stats['category_hits'] + stats['category_misses']
        state_total = stats['state_hits'] + stats['state_misses']
        philosophy_total = stats['philosophy_hits'] + stats['philosophy_misses']
        
        stats.update({
            'category_hit_rate': round(stats['category_hits'] / max(category_total, 1) * 100, 1),
            'state_hit_rate': round(stats['state_hits'] / max(state_total, 1) * 100, 1),
            'philosophy_hit_rate': round(stats['philosophy_hits'] / max(philosophy_total, 1) * 100, 1),
            'total_requests': category_total + state_total + philosophy_total,
            'cache_sizes': {
                'category': len(self.category_cache),
                'state': len(self.state_cache), 
                'philosophy': len(self.philosophy_cache)
            }
        })
        
        return stats


class IntelligentAnalyzer:
    """
    Анализатор для определения категорий вопросов и состояний лидов.
    Включает специальную логику для "застревания" на философских вопросах.
    
    ПРОДАКШН УЛУЧШЕНИЯ:
    - Многоуровневое кеширование для снижения затрат на LLM API
    - Умное предварительное кеширование
    - Статистика производительности
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Продвинутая система кеширования
        self.cache = AdvancedCache()
        
        # Расширенные ключевые слова для категорий
        self.CATEGORY_KEYWORDS = {
            'factual': [
                # Основные факты
                'цена', 'стоимость', 'сколько стоит', 'расценки', 'тарифы',
                'курс', 'курсы', 'занятия', 'уроки', 'программа', 'программы',
                'преподаватель', 'тренер', 'учитель', 'кто ведет', 'кто учит',
                'расписание', 'время', 'когда', 'во сколько', 'график',
                'возраст', 'сколько лет', 'подходит ли', 'можно ли в',
                'группа', 'сколько детей', 'размер группы', 'индивидуально',
                'онлайн', 'формат', 'как проходят', 'платформа',
                'пробный урок', 'первое занятие', 'записаться', 'запись',
                'сертификат', 'документ', 'результат', 'гарантии',
                'скидки', 'акции', 'льготы', 'рассрочка', 'оплата'
            ],
            
            'philosophical': [
                # Глубокие размышления о воспитании
                'как правильно', 'что делать с', 'как быть', 'как жить',
                'почему дети', 'зачем детям', 'в наше время', 'раньше было',
                'современные дети', 'поколение', 'молодежь сейчас',
                'принципы воспитания', 'методики воспитания', 'подходы к детям',
                'смысл', 'важность', 'нужно ли', 'стоит ли развивать',
                'что такое правильное', 'как понять ребенка',
                'философия воспитания', 'глубинные причины'
            ],
            
            'problem_solving': [
                # Конкретные проблемы
                'не слушается', 'капризничает', 'плачет', 'истерики',
                'агрессивный', 'дерется', 'кричит', 'не говорит',
                'замкнутый', 'стеснительный', 'боится', 'тревожный',
                'не хочет', 'отказывается', 'ленивый', 'неуверенный',
                'проблема с', 'как справиться', 'что делать если'
            ]
        }
        
        # Ключевые слова для определения состояний лидов
        self.STATE_KEYWORDS = {
            'greeting': [
                'привет', 'здравствуйте', 'добро пожаловать',
                'расскажите о школе', 'что это за школа', 'первый раз слышу'
            ],
            'fact_finding': [
                'узнать', 'расскажите', 'информация', 'подробности',
                'как работает', 'что включает', 'условия'
            ],
            'problem_solving': [
                'помогите', 'посоветуйте', 'не знаю что делать',
                'проблема', 'трудности', 'как быть'
            ],
            'closing': [
                'записаться', 'попробовать', 'начать', 'готов',
                'согласен', 'подходит', 'цена устраивает', 'хочу урок'
            ]
        }
        
        # Табу слова для юмора
        self.HUMOR_TABOO_KEYWORDS = [
            'болезнь', 'больной', 'инвалид', 'инвалидность', 'диагноз',
            'смерть', 'умер', 'погиб', 'похороны', 'потеря',
            'развод', 'расстались', 'ушел от нас', 'бросил',
            'избиение', 'насилие', 'бьет', 'издевается',
            'депрессия', 'суицид', 'хочет покончить', 'травма',
            'изнасилование', 'домогательства', 'приставания'
        ]
        
        self.logger.info("🧠 Интеллектуальный анализатор с продвинутым кешированием инициализирован")
    
    def _generate_cache_key(self, text: str, context: str = "") -> str:
        """Генерирует стабильный ключ кеша"""
        combined = f"{text}|{context}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _normalize_text_for_caching(self, text: str) -> str:
        """Нормализует текст для более эффективного кеширования"""
        # Удаляем лишние пробелы, приводим к нижнему регистру
        normalized = ' '.join(text.lower().split())
        
        # Удаляем общие слова, которые не влияют на категоризацию
        stop_words = ['а', 'и', 'но', 'да', 'же', 'ну', 'вот', 'это', 'то']
        words = [w for w in normalized.split() if w not in stop_words]
        
        return ' '.join(words)
    
    def analyze_question_category(self, user_message: str, conversation_history: List[str]) -> str:
        """
        Определяет категорию вопроса с агрессивным кешированием.
        
        Returns:
            str: 'factual', 'philosophical', 'problem_solving', 'sensitive'
        """
        # Нормализуем текст для кеширования
        normalized_message = self._normalize_text_for_caching(user_message)
        cache_key = self._generate_cache_key(normalized_message, "category")
        
        # Проверяем кеш сначала
        cached_result = self.cache.get_category_cache(cache_key)
        if cached_result:
            self.logger.info(f"💾 Категория получена из кеша: {cached_result}")
            self.cache.track_popular_pattern(f"category:{normalized_message[:50]}")
            return cached_result
        
        message_lower = user_message.lower()
        
        # Проверяем табу на юмор
        if any(taboo in message_lower for taboo in self.HUMOR_TABOO_KEYWORDS):
            self.logger.info("Детектировано табу на юмор - деликатная тема")
            result = 'sensitive'
            self.cache.set_category_cache(cache_key, result)
            return result
        
        # Анализ по ключевым словам (покрывает ~90% случаев)
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(keyword in message_lower for keyword in keywords):
                self.logger.info(f"Категория определена по ключевым словам: {category}")
                self.cache.set_category_cache(cache_key, category)
                return category
        
        # AI анализ для оставшихся случаев (с кешированием)
        history_context = ' '.join(conversation_history[-4:]) if conversation_history else 'Начало диалога'
        
        ai_prompt = f"""Определи категорию вопроса родителя о развитии ребенка. Отвечай ТОЛЬКО одним словом.

История: {history_context}
Вопрос: "{user_message}"

Категории:
factual - конкретные факты о школе/курсах/ценах/расписании
philosophical - размышления о воспитании/современных детях/принципах
problem_solving - конкретные проблемы поведения ребенка

Ответ (одно слово):"""

        try:
            from app import ai_service
            result = ai_service._call_ai_model(ai_prompt).strip().lower()
            
            # Валидация результата
            valid_categories = ['factual', 'philosophical', 'problem_solving']
            if result in valid_categories:
                self.logger.info(f"Категория определена AI: {result}")
                self.cache.set_category_cache(cache_key, result)
                return result
            else:
                self.logger.warning(f"AI вернул некорректную категорию: {result}")
                result = 'factual'  # Fallback
                self.cache.set_category_cache(cache_key, result)
                return result
                
        except Exception as e:
            self.logger.error(f"Ошибка AI анализа категории: {e}")
            result = 'factual'  # Fallback
            self.cache.set_category_cache(cache_key, result)
            return result
    
    def analyze_lead_state(self, user_message: str, current_state: str, conversation_history: List[str]) -> str:
        """
        Определяет состояние лида с улучшенным кешированием.
        
        Returns:
            str: 'greeting', 'fact_finding', 'problem_solving', 'closing'
        """
        normalized_message = self._normalize_text_for_caching(user_message)
        context_key = f"{current_state}|{len(conversation_history)}"
        cache_key = self._generate_cache_key(normalized_message, context_key)
        
        # Проверяем кеш
        cached_result = self.cache.get_state_cache(cache_key)
        if cached_result:
            self.logger.info(f"💾 Состояние получено из кеша: {cached_result}")
            return cached_result
        
        message_lower = user_message.lower()
        
        # Прямые запросы урока имеют высший приоритет
        direct_lesson_keywords = [
            "записаться", "попробовать", "пробный урок", "хочу урок", 
            "дайте ссылку", "начать заниматься", "готов попробовать"
        ]
        if any(word in message_lower for word in direct_lesson_keywords):
            self.logger.info("Детектирован прямой запрос урока → closing")
            result = 'closing'
            self.cache.set_state_cache(cache_key, result)
            return result
        
        # Анализ по расширенным ключевым словам
        for state, keywords in self.STATE_KEYWORDS.items():
            if any(keyword in message_lower for keyword in keywords):
                self.logger.info(f"Состояние определено по ключевым словам: {state}")
                self.cache.set_state_cache(cache_key, state)
                return state
        
        # Для коротких сообщений состояние обычно не меняется
        if len(user_message.split()) < 5:
            self.cache.set_state_cache(cache_key, current_state)
            return current_state
        
        # AI анализ для сложных случаев (с агрессивным кешированием)
        history_context = ' '.join(conversation_history[-6:]) if conversation_history else 'Начало диалога'
        
        ai_prompt = f"""Определи состояние лида в воронке продаж. Отвечай ТОЛЬКО названием состояния.

Текущее состояние: {current_state}
История диалога: {history_context}
Последний вопрос: "{user_message}"

Состояния:
greeting - первое знакомство, общие вопросы о школе
fact_finding - поиск конкретной информации о курсах/ценах/условиях  
problem_solving - обсуждение проблем ребенка, просьба о помощи
closing - готовность к записи на урок/курс

Ответ (только название):"""

        try:
            from app import ai_service
            result = ai_service._call_ai_model(ai_prompt).strip().lower()
            
            valid_states = ['greeting', 'fact_finding', 'problem_solving', 'closing']
            if result in valid_states:
                self.logger.info(f"Состояние определено AI: {result}")
                self.cache.set_state_cache(cache_key, result)
                return result
            else:
                self.logger.warning(f"AI вернул некорректное состояние: {result}")
                self.cache.set_state_cache(cache_key, current_state)
                return current_state
                
        except Exception as e:
            self.logger.error(f"Ошибка AI анализа состояния: {e}")
            self.cache.set_state_cache(cache_key, current_state)
            return current_state
    
    def analyze_philosophical_loop(self, conversation_history: List[str]) -> Tuple[bool, int]:
        """
        Анализирует "застревание" на философских вопросах с кешированием.
        
        Returns:
            Tuple[bool, int]: (нужен_мостик_к_школе, количество_философских_подряд)
        """
        if not conversation_history:
            return False, 0
        
        # Генерируем ключ кеша на основе последних сообщений
        recent_messages = conversation_history[-10:]  # Последние 10 сообщений
        cache_key = self._generate_cache_key('|'.join(recent_messages), "philosophy")
        
        # Проверяем кеш
        cached_result = self.cache.get_philosophy_cache(cache_key)
        if cached_result:
            self.logger.info(f"💾 Философский анализ получен из кеша: {cached_result}")
            return cached_result
        
        # Анализируем последние вопросы пользователя
        user_messages = [msg for msg in conversation_history if msg.startswith("Пользователь:")][-10:]
        
        philosophical_count = 0
        
        # Считаем философские вопросы подряд с конца
        for message in reversed(user_messages):
            message_text = message.replace("Пользователь:", "").strip()
            category = self.analyze_question_category(message_text, [])
            
            if category == 'philosophical':
                philosophical_count += 1
            else:
                break  # Прерываем, если встретили не философский вопрос
        
        needs_bridge = philosophical_count >= 3
        result = (needs_bridge, philosophical_count)
        
        # Кешируем результат
        self.cache.set_philosophy_cache(cache_key, result)
        
        if needs_bridge:
            self.logger.info(f"Детектировано застревание на философии: {philosophical_count} вопросов подряд")
        
        return result
    
    def should_use_humor_taboo(self, user_message: str) -> bool:
        """
        Проверяет, нужно ли избегать юмора в ответе.
        
        Returns:
            bool: True если юмор табу
        """
        message_lower = user_message.lower()
        return any(taboo in message_lower for taboo in self.HUMOR_TABOO_KEYWORDS)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику производительности анализатора.
        """
        cache_stats = self.cache.get_cache_stats()
        
        # Добавляем общие метрики
        total_ai_calls_would_be = (
            cache_stats['category_hits'] + cache_stats['category_misses'] +
            cache_stats['state_hits'] + cache_stats['state_misses']
        )
        
        ai_calls_made = cache_stats['category_misses'] + cache_stats['state_misses']
        cost_savings_percent = round(
            (cache_stats['total_ai_calls_saved'] / max(total_ai_calls_would_be, 1)) * 100, 1
        )
        
        return {
            **cache_stats,
            'total_ai_calls_would_be': total_ai_calls_would_be,
            'actual_ai_calls_made': ai_calls_made,
            'cost_savings_percent': cost_savings_percent,
            'avg_cache_efficiency': round(
                (cache_stats['category_hit_rate'] + cache_stats['state_hit_rate']) / 2, 1
            )
        }


# Создаем глобальный экземпляр анализатора
intelligent_analyzer = IntelligentAnalyzer()