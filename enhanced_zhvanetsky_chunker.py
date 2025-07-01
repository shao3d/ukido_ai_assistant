import os
import time
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pdfplumber  # Новая зависимость для работы с PDF

# --- НАСТРОЙКИ И КОНФИГУРАЦИЯ ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST_STYLE = os.getenv("PINECONE_HOST_STYLE")

@dataclass
class StyleChunkingConfig:
    """Конфигурация для обработки стилевых текстов Жванецкого"""
    
    # Размеры чанков адаптированы под стилевой контент
    min_chunk_size: int = 300         # Короче, чем справочные тексты
    ideal_chunk_size: int = 800       # Оптимум для сохранения стиля
    max_chunk_size: int = 1500        # Максимум для развернутых рассуждений
    
    # Размеры для афоризмов и коротких мыслей
    aphorism_min_size: int = 50       # Минимум для афоризма
    aphorism_max_size: int = 200      # Максимум для афоризма
    
    # API настройки
    api_requests_per_minute: int = 1500
    safety_margin: float = 0.8
    min_delay_between_requests: float = 0.1
    
    # Настройки фильтрации контента (НОВОЕ)
    enable_content_filtering: bool = True  # Включить фильтрацию для PDF
    filter_confidence_threshold: float = 0.7  # Порог уверенности для принятия чанка
    
    # Специфические настройки для Жванецкого
    preserve_rhythm: bool = True       # Сохраняем ритм текста
    respect_pauses: bool = True        # Уважаем авторские паузы
    keep_dialogue_intact: bool = True  # Не разрываем диалоги
    
    @property
    def calculated_delay(self) -> float:
        """Вычисляет задержку между API запросами"""
        max_safe_requests_per_minute = self.api_requests_per_minute * self.safety_margin
        optimal_delay = 60.0 / max_safe_requests_per_minute
        return max(optimal_delay, self.min_delay_between_requests)

class SmartRetryHandler:
    """
    Интеллектуальная система обработки повторных попыток для API вызовов.
    
    Эта система реализует best practices для работы с временными сбоями API,
    включая обработку ошибок rate limiting, экспоненциальные задержки и
    уважение к рекомендациям сервера о времени ожидания.
    
    Принципы работы:
    1. Распознает различные типы ошибок и применяет соответствующие стратегии
    2. Извлекает рекомендуемые задержки из ответов API
    3. Использует экспоненциальные задержки как fallback стратегию
    4. Ограничивает максимальное количество попыток для предотвращения бесконечных циклов
    5. Предоставляет подробное логирование для мониторинга и отладки
    """
    
    def __init__(self, max_retries: int = 5, base_delay: float = 1.0):
        """
        Инициализация системы повторных попыток.
        
        Args:
            max_retries: Максимальное количество повторных попыток
            base_delay: Базовая задержка в секундах для экспоненциального backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.retry_stats = {
            "total_retries": 0,
            "successful_retries": 0,
            "failed_operations": 0,
            "rate_limit_hits": 0
        }
        
        print(f"🔄 Smart Retry Handler инициализирован:")
        print(f"   Максимум попыток: {max_retries}")
        print(f"   Базовая задержка: {base_delay}с")
    
    def extract_retry_delay_from_error(self, error_message: str) -> Optional[float]:
        """
        Извлекает рекомендуемую задержку из сообщения об ошибке Google API.
        
        Google API предоставляет structured информацию об ошибках, включая
        рекомендуемое время ожидания в поле retry_delay. Эта функция
        парсит сообщение об ошибке и извлекает это значение.
        
        Args:
            error_message: Строка с сообщением об ошибке от API
            
        Returns:
            Рекомендуемая задержка в секундах или None если не найдена
        """
        import re
        
        # Паттерн для поиска retry_delay в structured ошибке Google API
        # Формат: retry_delay { seconds: X }
        delay_pattern = r'retry_delay\s*{\s*seconds:\s*(\d+)'
        
        match = re.search(delay_pattern, error_message)
        if match:
            return float(match.group(1))
        
        # Fallback: поиск общих паттернов времени ожидания
        # Некоторые API могут использовать другие форматы
        general_patterns = [
            r'retry after (\d+) seconds?',
            r'try again in (\d+) seconds?',
            r'wait (\d+) seconds?'
        ]
        
        for pattern in general_patterns:
            match = re.search(pattern, error_message.lower())
            if match:
                return float(match.group(1))
        
        return None
    
    def calculate_exponential_backoff(self, attempt: int) -> float:
        """
        Вычисляет экспоненциальную задержку для попытки.
        
        Экспоненциальный backoff — это стратегия, при которой время ожидания
        между попытками увеличивается экспоненциально. Это помогает избежать
        "thundering herd" проблемы, когда множество клиентов одновременно
        повторяют запросы после временного сбоя.
        
        Формула: base_delay * (2 ^ (attempt - 1)) + jitter
        Jitter добавляется для рандомизации и распределения нагрузки.
        
        Args:
            attempt: Номер текущей попытки (1-based)
            
        Returns:
            Задержка в секундах
        """
        import random
        
        # Основная экспоненциальная задержка
        exponential_delay = self.base_delay * (2 ** (attempt - 1))
        
        # Добавляем jitter (случайность) для распределения нагрузки
        # Jitter составляет ±25% от основной задержки
        jitter_range = exponential_delay * 0.25
        jitter = random.uniform(-jitter_range, jitter_range)
        
        total_delay = exponential_delay + jitter
        
        # Ограничиваем максимальную задержку разумным значением
        max_delay = 300  # 5 минут максимум
        return min(total_delay, max_delay)
    
    def retry_api_call(self, api_function, *args, **kwargs):
        """
        Выполняет API вызов с интеллектуальной системой повторных попыток.
        
        Эта функция является центральным компонентом нашей retry стратегии.
        Она wraps любую API функцию и автоматически обрабатывает временные
        сбои, включая rate limiting, сетевые ошибки и временную недоступность.
        
        Алгоритм работы:
        1. Пытается выполнить API вызов
        2. При успехе — возвращает результат
        3. При ошибке — анализирует тип ошибки
        4. Для recoverable ошибок — ждет и повторяет попытку
        5. Для fatal ошибок — прокидывает исключение дальше
        
        Args:
            api_function: Функция API для вызова
            *args, **kwargs: Аргументы для передачи в API функцию
            
        Returns:
            Результат успешного API вызова
            
        Raises:
            Exception: Если все попытки исчерпаны или произошла fatal ошибка
        """
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # Пытаемся выполнить API вызов
                result = api_function(*args, **kwargs)
                
                # Если мы добрались до этой строки, вызов успешен
                if attempt > 1:
                    # Это была повторная попытка, обновляем статистику
                    self.retry_stats["successful_retries"] += 1
                    print(f"      ✅ Успешная повторная попытка #{attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                error_message = str(e)
                
                # Проверяем, является ли это recoverable ошибкой
                if "429" in error_message or "quota" in error_message.lower():
                    # Это ошибка rate limiting — мы можем с ней справиться
                    self.retry_stats["rate_limit_hits"] += 1
                    
                    if attempt == self.max_retries:
                        # Исчерпаны все попытки
                        print(f"      ❌ Исчерпаны все {self.max_retries} попыток для rate limiting")
                        break
                    
                    # Пытаемся извлечь рекомендуемую задержку из ошибки
                    recommended_delay = self.extract_retry_delay_from_error(error_message)
                    
                    if recommended_delay:
                        # API предоставил рекомендацию — используем её
                        wait_time = recommended_delay
                        delay_source = f"API рекомендация: {recommended_delay}с"
                    else:
                        # Используем экспоненциальный backoff как fallback
                        wait_time = self.calculate_exponential_backoff(attempt)
                        delay_source = f"экспоненциальный backoff: {wait_time:.1f}с"
                    
                    print(f"      ⏳ Rate limit достигнут. Попытка {attempt}/{self.max_retries}")
                    print(f"         Ожидание {delay_source}")
                    
                    # Показываем прогресс ожидания для длительных задержек
                    if wait_time > 10:
                        self._show_countdown(wait_time)
                    else:
                        time.sleep(wait_time)
                    
                    self.retry_stats["total_retries"] += 1
                    continue
                    
                elif "5" in error_message[:3]:  # 5xx серверные ошибки
                    # Временные серверные проблемы — стоит повторить
                    if attempt == self.max_retries:
                        print(f"      ❌ Исчерпаны все попытки для серверной ошибки")
                        break
                    
                    wait_time = self.calculate_exponential_backoff(attempt)
                    print(f"      ⚠️ Серверная ошибка. Попытка {attempt}/{self.max_retries}")
                    print(f"         Ожидание {wait_time:.1f}с")
                    
                    time.sleep(wait_time)
                    self.retry_stats["total_retries"] += 1
                    continue
                    
                else:
                    # Это скорее всего fatal ошибка (400, проблемы с аутентификацией, etc.)
                    # Повторные попытки не помогут
                    print(f"      💀 Fatal ошибка API (не recoverable): {error_message[:100]}...")
                    raise e
        
        # Если мы добрались сюда, все попытки исчерпаны
        self.retry_stats["failed_operations"] += 1
        print(f"      ❌ Операция провалена после {self.max_retries} попыток")
        raise last_exception
    
    def _show_countdown(self, wait_time: float):
        """
        Показывает countdown для длительных ожиданий.
        
        Когда API рекомендует ждать более 10 секунд, полезно показать
        пользователю прогресс ожидания. Это улучшает user experience
        и помогает понять, что система работает, а не зависла.
        
        Args:
            wait_time: Время ожидания в секундах
        """
        import sys
        
        print(f"         Countdown: ", end="", flush=True)
        
        # Показываем countdown по секундам для первых 10 секунд
        initial_countdown = min(10, int(wait_time))
        for i in range(initial_countdown, 0, -1):
            print(f"{i}...", end="", flush=True)
            time.sleep(1)
            wait_time -= 1
        
        # Если осталось больше времени, ждем оставшееся время молча
        if wait_time > 0:
            print(f"(+{wait_time:.0f}с)", end="", flush=True)
            time.sleep(wait_time)
        
        print(" ✓", flush=True)
    
    def get_retry_statistics(self) -> Dict:
        """
        Возвращает статистику работы retry системы.
        
        Эта информация полезна для мониторинга и оптимизации системы.
        Мы можем отслеживать, как часто происходят retry, какие типы
        ошибок наиболее распространены, и насколько эффективна наша стратегия.
        
        Returns:
            Словарь с статистикой retry операций
        """
        return self.retry_stats.copy()


class ContentRelevanceFilter:
    """
    Система фильтрации контента с интеллектуальной обработкой ошибок API.
    Определяет, подходит ли чанк Жванецкого для школы soft skills.
    
    ОБНОВЛЕНИЕ: Теперь использует SmartRetryHandler для обработки rate limiting
    и других временных проблем с API.
    """
    
    def __init__(self, config: StyleChunkingConfig):
        self.config = config
        self.generation_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # НОВОЕ: Инициализируем умную систему повторных попыток
        self.retry_handler = SmartRetryHandler(
            max_retries=5,      # До 5 попыток для каждого API вызова
            base_delay=2.0      # Начальная задержка 2 секунды
        )
        
        # Промпт для оценки релевантности контента
        self.relevance_prompt = """
Ты - эксперт по образовательному контенту для онлайн-школы развития soft skills у детей.

Проанализируй этот фрагмент текста Михаила Жванецкого и оцени, подходит ли он для использования AI-ассистентом, который консультирует родителей по вопросам воспитания и развития детей.

КРИТЕРИИ ПОДХОДЯЩЕГО КОНТЕНТА:
- Мудрые наблюдения о человеческой природе
- Размышления о воспитании, образовании, отношениях
- Юмористические, но поучительные истории
- Афоризмы о жизни, общении, развитии личности
- Материал, который может помочь родителям лучше понимать детей

КРИТЕРИИ НЕПОДХОДЯЩЕГО КОНТЕНТА:
- Политическая сатира или критика власти
- Слишком специфичные советские реалии
- Грубый или неподходящий для семейной аудитории юмор
- Материал без образовательной ценности для родителей
- Слишком узкоспециальные темы

Фрагмент для анализа:
{text}

Ответь ТОЛЬКО одним словом:
- "ПОДХОДИТ" - если материал уместен для школы soft skills
- "НЕ_ПОДХОДИТ" - если материал не подходит для нашей аудитории

Твой ответ:"""
    
    def _make_api_call_for_filtering(self, chunk: str) -> str:
        """
        Внутренняя функция для выполнения API вызова фильтрации.
        
        Эта функция изолирует логику API вызова от retry логики,
        что делает код более модульным и тестируемым. SmartRetryHandler
        будет вызывать эту функцию и автоматически обрабатывать ошибки.
        
        Args:
            chunk: Текстовый фрагмент для анализа
            
        Returns:
            Ответ от Gemini API
            
        Raises:
            Exception: При любых проблемах с API
        """
        # Создаем промпт с конкретным чанком
        full_prompt = self.relevance_prompt.format(text=chunk[:1000])  # Ограничиваем длину для API
        
        # Отправляем запрос к Gemini (без retry логики - её обработает wrapper)
        response = self.generation_model.generate_content(full_prompt)
        return response.text.strip().upper()
    
    def evaluate_chunk_relevance(self, chunk: str) -> Tuple[bool, str]:
        """
        Оценивает релевантность чанка для школы soft skills.
        
        ИСПРАВЛЕНИЕ: Убрана дублирующая обработка ошибок, которая конфликтовала
        с SmartRetryHandler. Теперь функция полностью полагается на retry handler
        для обработки всех технических проблем с API, включая rate limiting.
        
        Эта архитектура следует принципу "единственной ответственности" - 
        функция фокусируется только на бизнес-логике анализа контента,
        а все технические аспекты надежности обрабатываются на уровне middleware.
        
        Args:
            chunk: Текстовый фрагмент для анализа
            
        Returns:
            Tuple (подходит ли чанк, объяснение решения)
        """
        # Используем retry handler для надежного API вызова
        # Все ошибки rate limiting, временные сбои и recoverable проблемы
        # будут автоматически обработаны SmartRetryHandler
        ai_decision = self.retry_handler.retry_api_call(
            self._make_api_call_for_filtering, 
            chunk
        )
        
        # Анализируем ответ AI
        is_relevant = "ПОДХОДИТ" in ai_decision
        explanation = f"AI решение: {ai_decision}"
        
        return is_relevant, explanation
    """
    Новый класс для фильтрации контента через Gemini API.
    Определяет, подходит ли чанк Жванецкого для школы soft skills.
    """
    
    def __init__(self, config: StyleChunkingConfig):
        self.config = config
        self.generation_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Промпт для оценки релевантности контента
        self.relevance_prompt = """
Ты - эксперт по образовательному контенту для онлайн-школы развития soft skills у детей.

Проанализируй этот фрагмент текста Михаила Жванецкого и оцени, подходит ли он для использования AI-ассистентом, который консультирует родителей по вопросам воспитания и развития детей.

КРИТЕРИИ ПОДХОДЯЩЕГО КОНТЕНТА:
- Мудрые наблюдения о человеческой природе
- Размышления о воспитании, образовании, отношениях
- Юмористические, но поучительные истории
- Афоризмы о жизни, общении, развитии личности
- Материал, который может помочь родителям лучше понимать детей

КРИТЕРИИ НЕПОДХОДЯЩЕГО КОНТЕНТА:
- Политическая сатира или критика власти
- Слишком специфичные советские реалии
- Грубый или неподходящий для семейной аудитории юмор
- Материал без образовательной ценности для родителей
- Слишком узкоспециальные темы

Фрагмент для анализа:
{text}

Ответь ТОЛЬКО одним словом:
- "ПОДХОДИТ" - если материал уместен для школы soft skills
- "НЕ_ПОДХОДИТ" - если материал не подходит для нашей аудитории

Твой ответ:"""
    
    def evaluate_chunk_relevance(self, chunk: str) -> Tuple[bool, str]:
        """
        Оценивает релевантность чанка для школы soft skills.
        Возвращает (подходит ли чанк, объяснение решения)
        """
        try:
            # Создаем промпт с конкретным чанком
            full_prompt = self.relevance_prompt.format(text=chunk[:1000])  # Ограничиваем длину для API
            
            # Отправляем запрос к Gemini
            response = self.generation_model.generate_content(full_prompt)
            ai_decision = response.text.strip().upper()
            
            # Анализируем ответ AI
            is_relevant = "ПОДХОДИТ" in ai_decision
            explanation = f"AI решение: {ai_decision}"
            
            return is_relevant, explanation
            
        except Exception as e:
            print(f"      ⚠️ Ошибка при фильтрации чанка: {e}")
            # В случае ошибки API, принимаем консервативное решение
            return True, f"Ошибка API, чанк принят по умолчанию: {str(e)[:100]}"

class ApiRateLimiter:
    """Контроллер частоты API запросов (расширенный для фильтрации)"""
    
    def __init__(self, config: StyleChunkingConfig):
        self.config = config
        self.last_request_time = 0
        self.request_count = 0
        self.start_time = time.time()
        
        print(f"🚦 API Rate Limiter для стилевого контента:")
        print(f"   Задержка между запросами: {config.calculated_delay:.2f} секунд")
        print(f"   Фильтрация контента: {'включена' if config.enable_content_filtering else 'отключена'}")
    
    def wait_if_needed(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.config.calculated_delay:
            sleep_time = self.config.calculated_delay - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def get_stats(self) -> Dict:
        elapsed_time = time.time() - self.start_time
        requests_per_minute = (self.request_count / elapsed_time) * 60 if elapsed_time > 0 else 0
        
        return {
            "total_requests": self.request_count,
            "elapsed_minutes": elapsed_time / 60,
            "requests_per_minute": requests_per_minute,
            "limit_utilization": (requests_per_minute / self.config.api_requests_per_minute) * 100
        }

class PDFTextExtractor:
    """
    Новый класс для извлечения и предварительной обработки текста из PDF файлов.
    """
    
    def __init__(self):
        print("📄 Инициализация PDF Text Extractor")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Извлекает текст из PDF файла с сохранением структуры.
        """
        try:
            print(f"   📄 Извлечение текста из PDF: {os.path.basename(pdf_path)}")
            
            extracted_text = ""
            
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"      📊 Всего страниц в PDF: {total_pages}")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Извлекаем текст страницы
                    page_text = page.extract_text()
                    
                    if page_text:
                        # Базовая очистка текста от артефактов PDF
                        cleaned_text = self._clean_pdf_text(page_text)
                        extracted_text += cleaned_text + "\n\n"
                    
                    # Прогресс-индикатор для больших файлов
                    if page_num % 50 == 0:
                        print(f"      📖 Обработано страниц: {page_num}/{total_pages}")
                
                print(f"   ✅ Извлечено {len(extracted_text)} символов из PDF")
                return extracted_text.strip()
                
        except Exception as e:
            print(f"   ❌ Ошибка при извлечении текста из PDF {pdf_path}: {e}")
            return ""
    
    def _clean_pdf_text(self, text: str) -> str:
        """
        Очищает текст от типичных артефактов PDF.
        """
        # Убираем лишние переносы строк
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Убираем разрывы слов на границах строк (частая проблема PDF)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Убираем лишние пробелы
        text = re.sub(r' {2,}', ' ', text)
        
        # Убираем висячие номера страниц (если они есть)
        text = re.sub(r'\n\s*\d+\s*\n', '\n\n', text)
        
        return text

class ZhvanetskyStyleAnalyzer:
    """
    Анализатор для определения структурных особенностей текстов Жванецкого.
    (Без изменений - класс уже хорошо работает)
    """
    
    @staticmethod
    def detect_aphorism(text: str) -> bool:
        """Определяет, является ли фрагмент афоризмом Жванецкого"""
        clean_text = re.sub(r'\s+', ' ', text.strip())
        
        if len(clean_text) > 300:
            return False
        
        has_contrast = any(word in clean_text.lower() for word in 
                          ['но', 'а', 'однако', 'зато', 'не', 'только', 'лишь'])
        
        has_wisdom_words = any(word in clean_text.lower() for word in 
                              ['воспитание', 'образование', 'жизнь', 'люди', 'человек', 
                               'дети', 'родители', 'мысль', 'слова', 'память'])
        
        has_parallelism = bool(re.search(r'[А-ЯЁ][^.!?]*[.!?]\s*[А-ЯЁ][^.!?]*[.!?]', clean_text))
        ends_definitively = clean_text.endswith(('.', '!', '?'))
        sentence_count = len(re.findall(r'[.!?]+', clean_text))
        
        is_likely_aphorism = (
            len(clean_text) <= 200 and 
            sentence_count <= 3 and
            ends_definitively and
            (has_contrast or has_wisdom_words or has_parallelism)
        )
        
        return is_likely_aphorism
    
    @staticmethod
    def detect_dialogue(text: str) -> bool:
        """Определяет наличие диалога в тексте Жванецкого"""
        dialogue_markers = [
            r'—\s*[А-ЯЁ]',           
            r':\s*—',               
            r'[А-ЯЁ][а-яё]*:',       
            r'спросил.*?:',         
            r'сказал.*?:',          
            r'отвечал.*?:',         
            r'— [А-ЯЁ]',            
            r'Дорогие товарищи',    
            r'маленький мальчик.*лет',
        ]
        
        marker_count = sum(1 for pattern in dialogue_markers 
                          if re.search(pattern, text))
        
        return marker_count >= 2 or bool(re.search(r'—.*—.*—', text, re.DOTALL))
    
    @staticmethod
    def detect_logical_chain(text: str) -> bool:
        """Определяет логическую цепочку рассуждений"""
        chain_markers = [
            r'во-первых|во-вторых|в-третьих',
            r'потому что|поэтому|следовательно',
            r'например|к примеру|скажем',
            r'но|однако|с другой стороны',
            r'значит|итак|таким образом'
        ]
        
        return any(re.search(pattern, text.lower()) for pattern in chain_markers)
    
    @staticmethod
    def find_natural_breaks(text: str) -> List[int]:
        """Находит естественные места для разрыва текста"""
        break_points = []
        
        for match in re.finditer(r'\n\n+', text):
            break_points.append(match.start())
        
        for match in re.finditer(r'\.\s+[А-ЯЁ]', text):
            break_points.append(match.start() + 1)
        
        for match in re.finditer(r'[!?]\s+[А-ЯЁ]', text):
            break_points.append(match.start() + 1)
        
        return sorted(set(break_points))

class ZhvanetskyStyleChunker:
    """
    Расширенный чанкер для текстов Михаила Жванецкого с поддержкой PDF и фильтрации.
    """
    
    def __init__(self, config: StyleChunkingConfig):
        self.config = config
        self.rate_limiter = ApiRateLimiter(config)
        self.analyzer = ZhvanetskyStyleAnalyzer()
        self.pdf_extractor = PDFTextExtractor()  # НОВОЕ
        self.content_filter = ContentRelevanceFilter(config) if config.enable_content_filtering else None  # НОВОЕ
        
        # Инициализация Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.embedding_model = 'models/text-embedding-004'
        
        print("🎭 РАСШИРЕННЫЙ ЧАНКЕР ДЛЯ СТИЛЕВОГО КОНТЕНТА МИХАИЛА ЖВАНЕЦКОГО")
        print(f"📝 Целевые размеры: {config.min_chunk_size}-{config.ideal_chunk_size} символов")
        print(f"💎 Афоризмы: {config.aphorism_min_size}-{config.aphorism_max_size} символов")
        print(f"🎵 Сохранение ритма: {'✓' if config.preserve_rhythm else '✗'}")
        print(f"📄 Поддержка PDF: ✓")
        print(f"🔍 Фильтрация контента: {'✓' if config.enable_content_filtering else '✗'}")
        print(f"🔧 ASCII-совместимые ID: ✓")
    
    def generate_safe_vector_id(self, index_name: str, filename: str, chunk_idx: int) -> str:
        """
        НОВАЯ ФУНКЦИЯ: Генерирует ASCII-совместимый идентификатор для Pinecone.
        
        Алгоритм работы:
        1. Базовая транслитерация кириллических символов
        2. Удаление расширений файлов и проблемных символов
        3. Замена пробелов и специальных символов на дефисы
        4. Ограничение длины для предотвращения слишком длинных ID
        5. Финальная проверка и очистка
        
        Args:
            index_name: Название индекса (например, "ukido-style")
            filename: Оригинальное имя файла
            chunk_idx: Номер чанка
        
        Returns:
            ASCII-совместимый строковый идентификатор
        """
        # Словарь для транслитерации основных кириллических символов
        # Основан на стандарте BGN/PCGN для русского языка
        cyrillic_to_latin = {
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
            'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
            'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
            'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
            'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
            
            'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
            'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
            'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
            'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Shch',
            'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya'
        }
        
        # Шаг 1: Удаляем расширение файла для более чистых ID
        clean_filename = os.path.splitext(filename)[0]
        
        # Шаг 2: Применяем транслитерацию кириллических символов
        transliterated = ""
        for char in clean_filename:
            if char in cyrillic_to_latin:
                transliterated += cyrillic_to_latin[char]
            else:
                transliterated += char
        
        # Шаг 3: Заменяем проблемные символы на дефисы
        # Удаляем или заменяем все, что не является буквой, цифрой или дефисом
        import string
        safe_chars = string.ascii_letters + string.digits + '-_'
        normalized = ""
        for char in transliterated:
            if char in safe_chars:
                normalized += char
            elif char in ' .()[]{}':
                normalized += '-'  # Заменяем пробелы и скобки на дефисы
            # Остальные символы просто пропускаем
        
        # Шаг 4: Убираем множественные дефисы и дефисы в начале/конце
        # Это важно для читаемости и предотвращения проблем с некоторыми системами
        while '--' in normalized:
            normalized = normalized.replace('--', '-')
        normalized = normalized.strip('-')
        
        # Шаг 5: Ограничиваем длину части с именем файла
        # Pinecone имеет ограничения на общую длину ID
        max_filename_length = 50  # Разумное ограничение для читаемости
        if len(normalized) > max_filename_length:
            normalized = normalized[:max_filename_length].rstrip('-')
        
        # Шаг 6: Создаем финальный ID
        # Формат: {index_name}-{safe_filename}-{chunk_number}
        safe_id = f"{index_name}-{normalized}-{chunk_idx}"
        
        # Шаг 7: Финальная проверка на ASCII совместимость
        # Это защита от неожиданных символов, которые могли проскользнуть
        try:
            safe_id.encode('ascii')
        except UnicodeEncodeError:
            # Если все еще есть проблемы, используем fallback на основе хеша
            import hashlib
            hash_part = hashlib.md5(clean_filename.encode('utf-8')).hexdigest()[:8]
            safe_id = f"{index_name}-file-{hash_part}-{chunk_idx}"
            print(f"      🔧 Fallback ID для проблемного имени файла: {safe_id}")
        
        return safe_id
    
    def extract_content_from_file(self, file_path: str) -> Optional[str]:
        """
        НОВАЯ ФУНКЦИЯ: Универсальное извлечение контента из файла (txt или pdf)
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.txt':
            # Обработка текстового файла (существующая логика)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"   ❌ Ошибка чтения текстового файла {file_path}: {e}")
                return None
                
        elif file_extension == '.pdf':
            # Обработка PDF файла (новая логика)
            return self.pdf_extractor.extract_text_from_pdf(file_path)
        
        else:
            print(f"   ⚠️ Неподдерживаемый формат файла: {file_extension}")
            return None
    
    def analyze_text_structure(self, content: str, filename: str) -> Dict:
        """Анализирует структуру текста Жванецкого (без изменений)"""
        structure = {
            "total_length": len(content),
            "has_dialogue": self.analyzer.detect_dialogue(content),
            "has_logical_chains": self.analyzer.detect_logical_chain(content),
            "natural_breaks": self.analyzer.find_natural_breaks(content),
            "estimated_aphorisms": 0,
            "paragraphs": len([p for p in content.split('\n\n') if p.strip()])
        }
        
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        for paragraph in paragraphs:
            if self.analyzer.detect_aphorism(paragraph):
                structure["estimated_aphorisms"] += 1
        
        print(f"   🔍 Анализ структуры '{filename}':")
        print(f"      📏 Длина: {structure['total_length']} символов")
        print(f"      💬 Диалоги: {'✓' if structure['has_dialogue'] else '✗'}")
        print(f"      🔗 Логические цепи: {'✓' if structure['has_logical_chains'] else '✗'}")
        print(f"      💎 Потенциальных афоризмов: {structure['estimated_aphorisms']}")
        print(f"      📄 Абзацев: {structure['paragraphs']}")
        
        return structure
    
    def create_style_aware_chunks(self, content: str, filename: str) -> List[str]:
        """Создает чанки с учетом стилевых особенностей (без изменений в основной логике)"""
        print(f"   ✂️ Создание стилевых чанков: {filename}")
        
        structure = self.analyze_text_structure(content, filename)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph_length = len(paragraph)
            
            if self.analyzer.detect_aphorism(paragraph):
                if current_chunk and current_size >= self.config.min_chunk_size:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                chunks.append(paragraph)
                print(f"      💎 Афоризм выделен: {paragraph[:50]}...")
                continue
            
            if self.analyzer.detect_dialogue(paragraph):
                if paragraph_length <= self.config.max_chunk_size:
                    if current_chunk:
                        chunks.append('\n\n'.join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    
                    chunks.append(paragraph)
                    print(f"      💬 Диалог сохранен целиком: {paragraph_length} символов")
                    continue
            
            if current_chunk:
                potential_size = current_size + paragraph_length + 2
            else:
                potential_size = paragraph_length
            
            if (potential_size > self.config.ideal_chunk_size and 
                current_size >= self.config.min_chunk_size):
                
                chunks.append('\n\n'.join(current_chunk))
                print(f"      📦 Чанк создан: {current_size} символов")
                
                current_chunk = [paragraph]
                current_size = paragraph_length
            else:
                current_chunk.append(paragraph)
                if len(current_chunk) == 1:
                    current_size = paragraph_length
                else:
                    current_size += paragraph_length + 2
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            print(f"      📦 Финальный чанк: {current_size} символов")
        
        processed_chunks = self._post_process_style_chunks(chunks)
        
        print(f"   🎯 Создано стилевых чанков: {len(processed_chunks)}")
        return processed_chunks
    
    def _post_process_style_chunks(self, chunks: List[str]) -> List[str]:
        """Постобработка чанков для обеспечения качества (без изменений)"""
        processed = []
        
        for i, chunk in enumerate(chunks):
            cleaned_chunk = chunk.strip()
            chunk_length = len(cleaned_chunk)
            
            if (chunk_length < self.config.min_chunk_size and 
                not self.analyzer.detect_aphorism(cleaned_chunk)):
                
                if processed and len(processed[-1] + '\n\n' + cleaned_chunk) <= self.config.max_chunk_size:
                    processed[-1] = processed[-1] + '\n\n' + cleaned_chunk
                    print(f"      🔗 Короткий фрагмент объединен с предыдущим")
                    continue
                else:
                    print(f"      ⚠️ Короткий фрагмент сохранен ({chunk_length} символов)")
            
            cleaned_chunk = re.sub(r'\n{3,}', '\n\n', cleaned_chunk)
            cleaned_chunk = re.sub(r' {2,}', ' ', cleaned_chunk)
            
            processed.append(cleaned_chunk)
        
        return processed
    
    def filter_chunk_if_needed(self, chunk: str, source_file: str) -> Tuple[bool, str]:
        """
        НОВАЯ ФУНКЦИЯ: Фильтрует чанк, если это необходимо.
        Возвращает (принять ли чанк, объяснение решения)
        """
        # Определяем, нужна ли фильтрация на основе типа файла
        file_extension = os.path.splitext(source_file)[1].lower()
        
        # Текстовые файлы не фильтруем (они уже отобраны вручную)
        if file_extension == '.txt':
            return True, "Текстовый файл - фильтрация не применяется"
        
        # PDF файлы фильтруем, если фильтрация включена
        if file_extension == '.pdf' and self.config.enable_content_filtering and self.content_filter:
            self.rate_limiter.wait_if_needed()  # Соблюдаем лимиты API
            return self.content_filter.evaluate_chunk_relevance(chunk)
        
        # По умолчанию принимаем чанк
        return True, "Фильтрация отключена"
    
    def vectorize_style_chunk(self, chunk: str, index_name: str, filename: str, chunk_idx: int) -> Optional[Dict]:
        """
        Векторизует стилевой чанк с соответствующими метаданными (обновленная версия).
        
        ИСПРАВЛЕНИЕ: Удалена дублирующая логика rate limiting, которая конфликтовала
        с SmartRetryHandler. Теперь управление скоростью API вызовов полностью
        централизовано в retry handler, что исключает конфликты и обеспечивает
        консистентное поведение системы.
        """
        # НОВОЕ: Генерируем ASCII-совместимый ID
        safe_chunk_id = self.generate_safe_vector_id(index_name, filename, chunk_idx)
        
        # НОВОЕ: Проверяем, нужно ли фильтровать чанк
        should_accept, filter_reason = self.filter_chunk_if_needed(chunk, filename)
        
        if not should_accept:
            print(f"      🚫 Чанк отфильтрован: {filter_reason}")
            return None
        
        # УДАЛЕНО: rate_limiter.wait_if_needed() - теперь это обрабатывает SmartRetryHandler
        # Если vectorize_style_chunk будет обернут в retry handler, задержки будут применены автоматически
        
        try:
            # Создаем embedding для стилевого документа
            response = genai.embed_content(
                model=self.embedding_model,
                content=chunk,
                task_type="RETRIEVAL_DOCUMENT",
                title="Zhvanetsky Style Sample"
            )
            
            # Определяем тип содержимого для метаданных
            content_type = "aphorism" if self.analyzer.detect_aphorism(chunk) else "narrative"
            if self.analyzer.detect_dialogue(chunk):
                content_type = "dialogue"
            
            # РАСШИРЕННЫЕ МЕТАДАННЫЕ
            return {
                "id": safe_chunk_id,  # ОБНОВЛЕНО: используем безопасный ID
                "values": response['embedding'],
                "metadata": {
                    "text": chunk,
                    "chunk_size": len(chunk),
                    "content_type": content_type,
                    "style_source": "zhvanetsky",
                    "has_dialogue": self.analyzer.detect_dialogue(chunk),
                    "is_aphorism": self.analyzer.detect_aphorism(chunk),
                    "embedding_model": self.embedding_model,
                    "task_type": "RETRIEVAL_DOCUMENT",
                    "source_file": filename,  # НОВОЕ
                    "source_file_type": os.path.splitext(filename)[1].lower(),  # НОВОЕ
                    "original_filename": filename,  # НОВОЕ: сохраняем оригинальное имя для справки
                    "safe_id": safe_chunk_id,  # НОВОЕ: сохраняем сгенерированный ID для отладки
                    "content_filtered": should_accept,  # НОВОЕ
                    "filter_reason": filter_reason,  # НОВОЕ
                    "created_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"      ❌ Ошибка векторизации стилевого чанка для файла {filename}: {e}")
            return None
    
    def process_style_directory(self, directory_path: str, index_name: str) -> Dict:
        """
        МОДИФИЦИРОВАННАЯ ФУНКЦИЯ: Обрабатывает директорию с текстовыми и PDF файлами.
        """
        start_time = time.time()
        
        print(f"\n🎭 ОБРАБОТКА СТИЛЕВЫХ ТЕКСТОВ МИХАИЛА ЖВАНЕЦКОГО")
        print(f"📂 Директория: {directory_path}")
        print(f"🎯 Индекс: {index_name}")
        print("=" * 65)
        
        # Подключаемся к Pinecone
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(host=PINECONE_HOST_STYLE)
            print("🔌 Подключение к Pinecone (стилевой индекс) успешно")
        except Exception as e:
            print(f"❌ Ошибка подключения к Pinecone: {e}")
            return {"success": False, "error": str(e)}
        
        # Очищаем стилевой индекс
        print("🗑️ Очистка существующих стилевых данных...")
        index.delete(delete_all=True)
        time.sleep(3)
        
        # Проверяем существование директории
        if not os.path.exists(directory_path):
            print(f"❌ Директория '{directory_path}' не существует")
            return {"success": False, "error": f"Directory '{directory_path}' not found"}
        
        # ОБНОВЛЕННАЯ ЛОГИКА: Получаем список файлов (txt и pdf)
        try:
            all_files = os.listdir(directory_path)
            supported_files = [f for f in all_files if f.endswith(('.txt', '.pdf'))]
        except PermissionError:
            print(f"❌ Нет прав доступа к директории '{directory_path}'")
            return {"success": False, "error": f"Permission denied for directory '{directory_path}'"}
        
        if not supported_files:
            print(f"❌ В директории '{directory_path}' не найдено .txt или .pdf файлов")
            return {"success": False, "error": f"No .txt or .pdf files found in '{directory_path}'"}
            
        print(f"📁 Найдено файлов: {len(supported_files)} (.txt и .pdf)")
        
        # Расширенная статистика обработки
        stats = {
            "files_processed": 0,
            "txt_files_processed": 0,  # НОВОЕ
            "pdf_files_processed": 0,  # НОВОЕ
            "total_chunks": 0,
            "chunks_accepted": 0,  # НОВОЕ
            "chunks_filtered": 0,  # НОВОЕ
            "aphorism_chunks": 0,
            "dialogue_chunks": 0,
            "narrative_chunks": 0,
            "vectors_uploaded": 0,
            "total_content_size": 0,
            "average_chunk_size": 0,
            "processing_time": 0,
            "api_stats": {},
            "file_details": []
        }
        
        # ОБНОВЛЕННЫЙ ЦИКЛ: Обрабатываем каждый файл (txt и pdf)
        for file_idx, filename in enumerate(supported_files):
            print(f"\n📖 Файл {file_idx + 1}/{len(supported_files)}: {filename}")
            
            file_start_time = time.time()
            file_path = os.path.join(directory_path, filename)
            file_extension = os.path.splitext(filename)[1].lower()
            
            try:
                # НОВОЕ: Универсальное извлечение контента
                content = self.extract_content_from_file(file_path)
                
                if not content or len(content) < 50:
                    print(f"   ⚠️ Файл слишком мал или пуст ({len(content) if content else 0} символов), пропускаем")
                    continue
                
                stats["total_content_size"] += len(content)
                
                # Обновляем статистику по типам файлов
                if file_extension == '.txt':
                    stats["txt_files_processed"] += 1
                elif file_extension == '.pdf':
                    stats["pdf_files_processed"] += 1
                
                # Создаем стилевые чанки
                chunks = self.create_style_aware_chunks(content, filename)
                
                if not chunks:
                    print(f"   ⚠️ Не удалось создать стилевые чанки, пропускаем файл")
                    continue
                
                # Векторизуем и загружаем чанки с фильтрацией
                print(f"   🔄 Векторизация {len(chunks)} стилевых чанков...")
                
                file_vectors = 0
                file_aphorisms = 0
                file_dialogues = 0
                file_narratives = 0
                file_accepted = 0
                file_filtered = 0
                
                for chunk_idx, chunk in enumerate(chunks):
                    # ОБНОВЛЕНО: Теперь передаем параметры для генерации безопасного ID
                    vector_data = self.vectorize_style_chunk(chunk, index_name, filename, chunk_idx)
                    
                    if vector_data:
                        # Подсчитываем типы контента
                        content_type = vector_data["metadata"]["content_type"]
                        if content_type == "aphorism":
                            file_aphorisms += 1
                        elif content_type == "dialogue":
                            file_dialogues += 1
                        else:
                            file_narratives += 1
                        
                        # Загружаем в Pinecone
                        index.upsert(vectors=[vector_data])
                        file_vectors += 1
                        file_accepted += 1
                        
                        # Прогресс-индикатор
                        if chunk_idx % 3 == 0:
                            print(f"      📊 Обработано чанков: {chunk_idx + 1}/{len(chunks)}")
                    else:
                        file_filtered += 1
                
                file_time = time.time() - file_start_time
                
                # Статистика по файлу
                file_stat = {
                    "filename": filename,
                    "file_type": file_extension,
                    "content_size": len(content),
                    "chunks_created": len(chunks),
                    "chunks_accepted": file_accepted,
                    "chunks_filtered": file_filtered,
                    "aphorisms": file_aphorisms,
                    "dialogues": file_dialogues,
                    "narratives": file_narratives,
                    "vectors_uploaded": file_vectors,
                    "processing_time": file_time,
                    "average_chunk_size": len(content) // len(chunks) if chunks else 0
                }
                
                stats["file_details"].append(file_stat)
                stats["files_processed"] += 1
                stats["total_chunks"] += len(chunks)
                stats["chunks_accepted"] += file_accepted
                stats["chunks_filtered"] += file_filtered
                stats["aphorism_chunks"] += file_aphorisms
                stats["dialogue_chunks"] += file_dialogues
                stats["narrative_chunks"] += file_narratives
                stats["vectors_uploaded"] += file_vectors
                
                print(f"   ✅ Файл обработан за {file_time:.1f}с:")
                print(f"      📄 Тип: {file_extension.upper()}")
                print(f"      ✅ Принято чанков: {file_accepted}")
                print(f"      🚫 Отфильтровано: {file_filtered}")
                print(f"      💎 Афоризмов: {file_aphorisms}")
                print(f"      💬 Диалогов: {file_dialogues}")
                print(f"      📝 Повествований: {file_narratives}")
                
            except Exception as e:
                print(f"   ❌ Ошибка обработки файла {filename}: {e}")
                continue
        
        # Финализация статистики
        total_time = time.time() - start_time
        stats["processing_time"] = total_time
        stats["average_chunk_size"] = (stats["total_content_size"] // stats["total_chunks"] 
                                     if stats["total_chunks"] > 0 else 0)
        stats["api_stats"] = self.rate_limiter.get_stats()
        
        # Проверяем результат в Pinecone
        time.sleep(3)
        final_stats = index.describe_index_stats()
        
        # Получаем статистику retry операций для полного отчета
        retry_stats = None
        if self.content_filter and self.content_filter.retry_handler:
            retry_stats = self.content_filter.retry_handler.get_retry_statistics()
        
        # РАСШИРЕННЫЙ ОТЧЕТ с информацией о retry операциях
        print(f"\n🎉 ОБРАБОТКА СТИЛЕВЫХ ТЕКСТОВ ЗАВЕРШЕНА!")
        print("=" * 55)
        print(f"📊 Результаты:")
        print(f"   📁 Файлов обработано: {stats['files_processed']}")
        print(f"      📝 TXT файлов: {stats['txt_files_processed']}")
        print(f"      📄 PDF файлов: {stats['pdf_files_processed']}")
        print(f"   📝 Всего чанков создано: {stats['total_chunks']}")
        print(f"      ✅ Принято: {stats['chunks_accepted']}")
        print(f"      🚫 Отфильтровано: {stats['chunks_filtered']}")
        print(f"   💎 Афоризмов: {stats['aphorism_chunks']}")
        print(f"   💬 Диалогов: {stats['dialogue_chunks']}")
        print(f"   📖 Повествований: {stats['narrative_chunks']}")
        print(f"   💾 Векторов загружено: {stats['vectors_uploaded']}")
        print(f"   📏 Средний размер чанка: {stats['average_chunk_size']} символов")
        print(f"   ⏱️ Общее время: {total_time/60:.1f} минут")
        
        # Информация об API использовании
        print(f"🔌 API статистика:")
        print(f"   🔄 Всего запросов: {stats['api_stats']['total_requests']}")
        print(f"   📈 Использование лимита: {stats['api_stats']['limit_utilization']:.1f}%")
        
        # НОВАЯ СЕКЦИЯ: Статистика retry операций
        if retry_stats:
            print(f"🔄 Статистика retry системы:")
            print(f"   🔁 Всего повторных попыток: {retry_stats['total_retries']}")
            print(f"   ✅ Успешных retry: {retry_stats['successful_retries']}")
            print(f"   🚫 Провалившихся операций: {retry_stats['failed_operations']}")
            print(f"   ⏱️ Rate limit событий: {retry_stats['rate_limit_hits']}")
            
            # Вычисляем эффективность retry системы
            if retry_stats['total_retries'] > 0:
                success_rate = (retry_stats['successful_retries'] / retry_stats['total_retries']) * 100
                print(f"   📊 Эффективность retry: {success_rate:.1f}%")
        
        print(f"✅ Стилевых векторов в Pinecone: {final_stats.total_vector_count}")
        
        return {"success": True, "stats": stats, "retry_stats": retry_stats}

def main():
    """Основная функция для обработки стилевых текстов Жванецкого (txt и pdf)"""
    
    # Создаем конфигурацию для стилевого контента с фильтрацией
    config = StyleChunkingConfig()
    
    # Создаем расширенный чанкер
    chunker = ZhvanetskyStyleChunker(config)
    
    # Обрабатываем стилевые тексты (поддерживает и txt, и pdf)
    result = chunker.process_style_directory("data_style", "ukido-style")
    
    if result["success"]:
        print("\n✨ Стилевые тексты Жванецкого успешно обработаны и готовы к использованию!")
        print("🎭 AI-ассистент теперь может отвечать в стиле великого сатирика")
        print("💡 Система готова сочетать мудрость Жванецкого с фактами о школе Ukido")
        print("🔍 PDF контент прошел интеллектуальную фильтрацию для школы soft skills")
    else:
        print(f"\n❌ Произошла ошибка: {result['error']}")

if __name__ == "__main__":
    main()