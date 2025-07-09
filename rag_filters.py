# rag_filters.py
from typing import Dict, List
import logging

class SmartQueryFilter:
    """Умная фильтрация для RAG на основе анализа запроса"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_query_intent(self, query: str) -> Dict:
        """Быстрый анализ намерения через keywords"""
        query_lower = query.lower()

        # Категории вопросов с расширенными ключевыми словами
        if any(word in query_lower for word in ['цена', 'стоим', 'скидк', 'оплат', 'тариф', 'деньг', 'грн']):
            return {'category': 'pricing', 'needs_specific': True}

        if any(word in query_lower for word in ['преподават', 'учител', 'педагог', 'тренер', 'наставник']):
            # Более детальный анализ для преподавателей
            if any(word in query_lower for word in ['опыт', 'стаж', 'лет работ', 'квалификац']):
                return {'category': 'teacher_experience', 'needs_specific': True}
            return {'category': 'teachers', 'needs_specific': True}

        if any(word in query_lower for word in ['болезн', 'диабет', 'аутизм', 'сдвг', 'особенност', 'инвалид']):
            return {'category': 'special_needs', 'needs_specific': True}

        if any(word in query_lower for word in ['капитан проект', 'юный оратор', 'эмоциональн']):
            return {'category': 'courses', 'needs_specific': True}

        if any(word in query_lower for word in ['linux', 'windows', 'техническ', 'компьютер', 'интернет', '4g']):
            return {'category': 'technical', 'needs_specific': True}

        return {'category': 'general', 'needs_specific': False}

    def get_metadata_filters(self, intent: Dict, query: str) -> Dict:
        """Генерирует LlamaIndex-совместимые фильтры"""

        # К сожалению, LlamaIndex + Pinecone имеют ограниченную поддержку фильтров
        # Поэтому пока возвращаем None, но логику оставляем для будущего
        self.logger.info(f"📎 Анализ запроса: категория={intent['category']}")

        # TODO: Когда LlamaIndex улучшит поддержку фильтров с Pinecone,
        # раскомментировать и адаптировать под их API
        return None
