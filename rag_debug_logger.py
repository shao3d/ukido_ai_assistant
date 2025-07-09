# rag_debug_logger.py
"""
🔍 RAG Debug Logger v3: Логирует в консоль и накапливает все логи в память
для последующего сохранения в ОДИН файл.
"""
import logging
import time
import os
from datetime import datetime
from typing import List

class RAGDebugLogger:
    def __init__(self):
        self.logger = logging.getLogger("RAG_DEBUG")
        self.log_dir = "rag_debug_logs"
        self.full_session_logs = []  # ✅ Единый буфер для всех логов

        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)
        
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _log(self, message: str, level=logging.INFO):
        self.logger.log(level, message)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_entry = f"{timestamp} | {message}"
        self.full_session_logs.append(log_entry)

    # Методы start_session, log_enricher_input и т.д. теперь просто вызывают _log
    def start_session(self, chat_id: str, user_message: str):
        self._log("=" * 80)
        self._log(f"🚀 NEW SESSION | Chat: {chat_id}")
        self._log(f"❓ Question: {user_message}")
    
    def log_enricher_prompt(self, prompt: str):
        self._log("🤖 DYNAMIC SYSTEM PROMPT:")
        self._log(prompt)
    
    def log_retrieval_results(self, chunks: List[str], scores: List[float], time_taken: float, total_before_rerank: int):
        self._log(f"🎯 RETRIEVAL & RERANKING | Time: {time_taken:.3f}s | Chunks: {total_before_rerank} -> {len(chunks)}")
        if scores: self._log(f"Scores: MAX={max(scores):.3f} AVG={sum(scores) / len(scores):.3f}")
        for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
            self._log(f"  Chunk {i}. [{score:.3f}] {chunk[:100].replace(chr(10), ' ')}...")
    
    def log_final_response(self, response: str, time_taken: float):
        self._log("🎉 FINAL RESPONSE")
        self._log(f"AI Response: {response}")
        self._log("-" * 80)

    # ✅ Новый метод для сохранения всего лога
    def save_full_log_to_file(self, filename: str) -> bool:
        """Сохраняет все накопленные логи в один указанный файл."""
        if not self.full_session_logs:
            self.logger.warning("Нет логов для сохранения.")
            return False
            
        log_filepath = os.path.join(self.log_dir, filename)
        try:
            with open(log_filepath, 'w', encoding='utf-8') as f:
                f.write("\n".join(self.full_session_logs))
            self.logger.info(f"✅ Полный лог тестирования сохранен в файл: {log_filepath}")
            self.full_session_logs = [] # Очищаем буфер после сохранения
            return True
        except IOError as e:
            self.logger.error(f"❌ Не удалось сохранить файл лога {log_filepath}: {e}")
            return False

rag_debug = RAGDebugLogger()