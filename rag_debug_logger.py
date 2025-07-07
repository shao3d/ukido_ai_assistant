# rag_debug_logger.py
"""
🔍 RAG Debug Logger v2: Логирует в консоль И сохраняет сессии в файлы.
"""
import logging
import time
import os
from datetime import datetime
from typing import List

class RAGDebugLogger:
    """
    Логгер для отладки RAG системы. Создает отдельный файл лога для каждой сессии.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("RAG_DEBUG")
        self.log_dir = "rag_debug_logs"
        self.session_id = None
        self.start_time = None
        self.current_session_logs = [] # Буфер для хранения логов текущей сессии

        # Настройка консольного логгера
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)
        
        # Создаем папку для логов
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _log(self, message: str, level=logging.INFO):
        """Внутренний метод: логирует в консоль И в буфер сессии."""
        self.logger.log(level, message)
        
        if self.session_id is not None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            log_entry = f"{timestamp} | {message}"
            self.current_session_logs.append(log_entry)

    def start_session(self, chat_id: str, user_message: str):
        """Начинаем новую сессию и инициализируем буфер."""
        self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{chat_id}"
        self.start_time = time.time()
        self.current_session_logs = [] # Очищаем буфер для новой сессии

        self._log("=" * 60)
        self._log(f"🚀 NEW SESSION: {self.session_id}")
        self._log(f"👤 Chat: {chat_id}")
        self._log(f"❓ Question: {user_message}")
        self._log("=" * 60)
    
    def log_enricher_input(self, original_query: str, history: List[str]):
        self._log("🔍 STAGE 1: ENRICHER INPUT")
        self._log(f"Original Query: {original_query}")
        self._log(f"History Messages: {len(history)}")
        if history:
            self._log("Recent History:")
            for i, msg in enumerate(history[-3:], 1):
                preview = msg[:80] + "..." if len(msg) > 80 else msg
                self._log(f"  {i}. {preview}")
    
    def log_enricher_prompt(self, prompt: str):
        self._log("🤖 STAGE 2: DYNAMIC PROMPT (SYSTEM INSTRUCTION)")
        self._log(f"Prompt Length: {len(prompt)} chars")
        self._log("Full Prompt:")
        self._log("-" * 40)
        self._log(prompt)
        self._log("-" * 40)
    
    def log_retrieval_results(self, chunks: List[str], scores: List[float], 
                            time_taken: float, total_before_rerank: int):
        self._log("🎯 STAGE 3: RETRIEVAL & RERANKING RESULTS")
        self._log(f"Time: {time_taken:.3f}s")
        self._log(f"Chunks: {total_before_rerank} → {len(chunks)} (after rerank)")
        
        if scores:
            avg_score = sum(scores) / len(scores)
            self._log(f"Scores: MAX={max(scores):.3f} AVG={avg_score:.3f} MIN={min(scores):.3f}")
        
        self._log("Found Chunks:")
        for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
            preview = chunk[:100].replace('\n', ' ')
            self._log(f"  {i}. [{score:.3f}] {preview}...")
    
    # У нас больше нет отдельного финального промпта в app.py, поэтому эти методы не нужны,
    # но оставим их пустыми для совместимости, если они где-то вызывались.
    def log_final_prompt(self, prompt: str, history_count: int): pass
    def log_enricher_output(self, output: str, time_taken: float): pass


    def log_final_response(self, response: str, time_taken: float):
        """Логируем финальный ответ и СОХРАНЯЕМ ЛОГ СЕССИИ В ФАЙЛ."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        self._log("🎉 STAGE 4: FINAL RESPONSE")
        self._log(f"Generation Time: {time_taken:.3f}s")
        self._log(f"Total Session Time: {total_time:.3f}s")
        self._log(f"Response Length: {len(response)} chars")
        self._log("AI Response:")
        self._log("-" * 40)
        self._log(response)
        self._log("-" * 40)
        
        # ✅ СОХРАНЕНИЕ В ФАЙЛ
        if self.session_id and self.current_session_logs:
            log_filename = os.path.join(self.log_dir, f"{self.session_id}.log")
            try:
                with open(log_filename, 'w', encoding='utf-8') as f:
                    f.write("\n".join(self.current_session_logs))
                
                self._log("📊 SESSION COMPLETE & SAVED")
                self.logger.info(f"📝 Log file saved: {log_filename}")

            except IOError as e:
                self.logger.error(f"❌ Failed to save session log file {log_filename}: {e}")
        
        self._log("=" * 60)
        
        # Сброс состояния для следующей сессии
        self.session_id = None
        self.current_session_logs = []
        self.start_time = None


# Глобальный экземпляр
rag_debug = RAGDebugLogger()