# intelligent_analyzer.py
"""
Интеллектуальный анализатор категорий вопросов и состояний лидов.
Использует гибридный подход: ключевые слова + AI анализ + специальная логика.
"""

import logging
import hashlib
import time
from typing import Tuple, List, Optional
from config import config


class IntelligentAnalyzer:
    """
    Анализатор для определения категорий вопросов и состояний лидов.
    Включает специальную логику для "застревания" на философских вопросах.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Кеш только для анализа состояний лидов (более сложный анализ)
        self.state_analysis_cache = {}
        self.cache_ttl = 1800  # 30 минут
        
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
                'родительство', 'быть родителем', 'воспитание детей',
                'границы', 'дисциплина', 'свобода или контроль',
                'гаджеты и дети', 'технологии', 'экранное время',
                'друзья ребенка', 'социализация', 'общение со сверстниками'
            ],
            
            'problem_solving': [
                # Конкретные проблемы детей
                'проблема', 'проблемы', 'трудности', 'сложности',
                'не слушается', 'не слушает', 'игнорирует', 'делает назло',
                'капризы', 'истерики', 'плачет', 'кричит', 'злится',
                'агрессивный', 'дерется', 'кусается', 'толкает',
                'застенчивый', 'стеснительный', 'боится', 'тревожный',
                'замкнутый', 'молчит', 'не общается', 'избегает',
                'гиперактивный', 'не сидит на месте', 'невнимательный',
                'не умеет', 'не может', 'отказывается', 'не хочет',
                'плохо спит', 'кошмары', 'страхи', 'фобии',
                'ревность', 'конкуренция', 'соперничество между детьми',
                'школьные проблемы', 'не хочет учиться', 'плохие оценки',
                'конфликты с учителями', 'проблемы с одноклассниками'
            ]
        }
        
        # Расширенные ключевые слова для состояний лида
        self.STATE_KEYWORDS = {
            'greeting': [
                'привет', 'здравствуйте', 'добрый', 'расскажите о школе',
                'что это за школа', 'впервые слышу', 'не знаю что это',
                'можете рассказать', 'интересует ваша школа'
            ],
            
            'fact_finding': [
                # Поиск информации
                'цена', 'стоимость', 'сколько стоит', 'расценки',
                'курсы', 'программы', 'что изучают', 'чему учат',
                'расписание', 'время', 'когда занятия', 'график',
                'возраст', 'подходит ли моему', 'можно ли в',
                'преподаватели', 'кто ведет', 'опыт тренеров',
                'формат', 'онлайн', 'как проходят', 'продолжительность',
                'группы', 'сколько детей', 'индивидуально',
                'результаты', 'эффективность', 'гарантии',
                'отзывы', 'рекомендации', 'репутация'
            ],
            
            'problem_solving': [
                # Решение проблем
                'проблема', 'проблемы', 'трудности', 'не знаю что делать',
                'помогите', 'посоветуйте', 'как быть', 'что делать',
                'не слушается', 'капризы', 'истерики', 'агрессия',
                'застенчивость', 'страхи', 'тревожность', 'замкнутость',
                'не умеет общаться', 'конфликты', 'ссоры',
                'школьные проблемы', 'не хочет учиться',
                'гиперактивность', 'невнимательность', 'рассеянность',
                'низкая самооценка', 'неуверенность', 'комплексы'
            ],
            
            'closing': [
                # Готовность к действию
                'хочу записаться', 'готов попробовать', 'давайте начнем',
                'как записаться', 'где записаться', 'можно записать',
                'пробный урок', 'попробовать', 'начать заниматься',
                'убедили', 'решили', 'подходит нам', 'интересно',
                'когда можем начать', 'есть места', 'свободное время',
                'оплатить', 'стоимость устраивает', 'цена подходит'
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
        
        self.logger.info("🧠 Интеллектуальный анализатор инициализирован")
    
    def _get_cache_key_for_state(self, data: str) -> str:
        """Генерирует ключ кеша для анализа состояний"""
        return hashlib.md5(data.encode()).hexdigest()
    
    def _get_cached_state_result(self, cache_key: str) -> Optional[str]:
        """Получает результат анализа состояния из кеша"""
        if cache_key in self.state_analysis_cache:
            entry = self.state_analysis_cache[cache_key]
            if time.time() - entry['timestamp'] < self.cache_ttl:
                return entry['result']
            else:
                del self.state_analysis_cache[cache_key]
        return None
    
    def _cache_state_result(self, cache_key: str, result: str):
        """Сохраняет результат анализа состояния в кеш"""
        self.state_analysis_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def analyze_question_category(self, user_message: str, conversation_history: List[str]) -> str:
        """
        Определяет категорию вопроса для выбора стиля Жванецкого.
        Использует ключевые слова + AI анализ (без кеширования для простоты).
        
        Returns:
            str: 'factual', 'philosophical', 'problem_solving', 'sensitive'
        """
        message_lower = user_message.lower()
        
        # Проверяем табу на юмор
        if any(taboo in message_lower for taboo in self.HUMOR_TABOO_KEYWORDS):
            self.logger.info("Детектировано табу на юмор - деликатная тема")
            return 'sensitive'
        
        # Анализ по ключевым словам (покрывает ~90% случаев)
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(keyword in message_lower for keyword in keywords):
                self.logger.info(f"Категория определена по ключевым словам: {category}")
                return category
        
        # AI анализ для оставшихся случаев (без кеширования)
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
                return result
            else:
                self.logger.warning(f"AI вернул некорректную категорию: {result}")
                return 'factual'  # Fallback
                
        except Exception as e:
            self.logger.error(f"Ошибка AI анализа категории: {e}")
            return 'factual'  # Fallback
    
    def analyze_lead_state(self, user_message: str, current_state: str, conversation_history: List[str]) -> str:
        """
        Определяет состояние лида с улучшенной логикой.
        
        Returns:
            str: 'greeting', 'fact_finding', 'problem_solving', 'closing'
        """
        message_lower = user_message.lower()
        
        # Прямые запросы урока имеют высший приоритет
        direct_lesson_keywords = [
            "записаться", "попробовать", "пробный урок", "хочу урок", 
            "дайте ссылку", "начать заниматься", "готов попробовать"
        ]
        if any(word in message_lower for word in direct_lesson_keywords):
            self.logger.info("Детектирован прямой запрос урока → closing")
            return 'closing'
        
        # Анализ по расширенным ключевым словам
        for state, keywords in self.STATE_KEYWORDS.items():
            if any(keyword in message_lower for keyword in keywords):
                self.logger.info(f"Состояние определено по ключевым словам: {state}")
                return state
        
        # Для коротких сообщений состояние обычно не меняется
        if len(user_message.split()) < 5:
            return current_state
        
        # AI анализ для сложных случаев (с кешированием для состояний)
        cache_key = self._get_cache_key_for_state(f"{user_message}{current_state}")
        cached_result = self._get_cached_state_result(cache_key)
        
        if cached_result:
            return cached_result
        
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
                self._cache_state_result(cache_key, result)
                self.logger.info(f"Состояние определено AI: {result}")
                return result
            else:
                self.logger.warning(f"AI вернул некорректное состояние: {result}")
                return current_state
                
        except Exception as e:
            self.logger.error(f"Ошибка AI анализа состояния: {e}")
            return current_state
    
    def analyze_philosophical_loop(self, conversation_history: List[str]) -> Tuple[bool, int]:
        """
        Анализирует "застревание" на философских вопросах.
        
        Returns:
            Tuple[bool, int]: (нужен_мостик_к_школе, количество_философских_подряд)
        """
        if not conversation_history:
            return False, 0
        
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
        
        if needs_bridge:
            self.logger.info(f"Детектировано застревание на философии: {philosophical_count} вопросов подряд")
        
        return needs_bridge, philosophical_count
    
    def should_use_humor_taboo(self, user_message: str) -> bool:
        """
        Проверяет, нужно ли избегать юмора в ответе.
        
        Returns:
            bool: True если юмор табу
        """
        message_lower = user_message.lower()
        return any(taboo in message_lower for taboo in self.HUMOR_TABOO_KEYWORDS)


# Создаем глобальный экземпляр анализатора
intelligent_analyzer = IntelligentAnalyzer()
