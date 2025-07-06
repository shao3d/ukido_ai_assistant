# rag_debug_logger.py
"""
🔍 Простой RAG Debug Logger для отладки pipeline
"""
import logging
import time
import json
import os
from datetime import datetime
from typing import List

class RAGDebugLogger:
    """
    Простой логгер для отладки RAG системы.
    Показывает каждый этап обработки запроса.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("RAG_DEBUG")
        self.session_id = None
        self.start_time = None
        
        # Простая настройка логгера
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)
        
        # Создаем папку для логов
        os.makedirs("rag_debug_logs", exist_ok=True)
    
    def start_session(self, chat_id: str, user_message: str):
        """Начинаем новую сессию"""
        self.session_id = f"{chat_id}_{int(time.time())}"
        self.start_time = time.time()
        
        self.logger.info("=" * 60)
        self.logger.info(f"🚀 NEW SESSION: {self.session_id}")
        self.logger.info(f"👤 Chat: {chat_id}")
        self.logger.info(f"❓ Question: {user_message}")
        self.logger.info("=" * 60)
    
    def log_enricher_input(self, original_query: str, history: List[str]):
        """Логируем вход в обогатитель"""
        self.logger.info("🔍 STAGE 1: ENRICHER INPUT")
        self.logger.info(f"Original Query: {original_query}")
        self.logger.info(f"History Messages: {len(history)}")
        
        if history:
            self.logger.info("Recent History:")
            for i, msg in enumerate(history[-3:], 1):
                preview = msg[:80] + "..." if len(msg) > 80 else msg
                self.logger.info(f"  {i}. {preview}")
    
    def log_enricher_prompt(self, prompt: str):
        """Логируем промпт обогатителя"""
        self.logger.info("🤖 STAGE 2: ENRICHER PROMPT")
        self.logger.info(f"Prompt Length: {len(prompt)} chars")
        self.logger.info("Full Prompt:")
        self.logger.info("-" * 40)
        self.logger.info(prompt)
        self.logger.info("-" * 40)
    
    def log_enricher_output(self, output: str, time_taken: float):
        """Логируем результат обогащения"""
        self.logger.info("✨ STAGE 3: ENRICHER OUTPUT")
        self.logger.info(f"Time: {time_taken:.3f}s")
        self.logger.info(f"Output: {output}")
    
    def log_retrieval_results(self, chunks: List[str], scores: List[float], 
                            time_taken: float, total_before_rerank: int):
        """Логируем результаты поиска"""
        self.logger.info("🎯 STAGE 4: RETRIEVAL RESULTS")
        self.logger.info(f"Time: {time_taken:.3f}s")
        self.logger.info(f"Chunks: {total_before_rerank} → {len(chunks)} (after rerank)")
        
        if scores:
            avg_score = sum(scores) / len(scores)
            self.logger.info(f"Scores: MAX={max(scores):.3f} AVG={avg_score:.3f} MIN={min(scores):.3f}")
        
        self.logger.info("Found Chunks:")
        for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
            preview = chunk[:100].replace('\n', ' ')
            self.logger.info(f"  {i}. [{score:.3f}] {preview}...")
    
    def log_final_prompt(self, prompt: str, history_count: int):
        """Логируем финальный промпт"""
        self.logger.info("🎭 STAGE 5: FINAL PROMPT")
        self.logger.info(f"Prompt Length: {len(prompt)} chars")
        self.logger.info(f"History Included: {history_count} messages")
        
        # Подсчет примерных токенов
        estimated_tokens = len(prompt) // 4
        self.logger.info(f"Estimated Tokens: ~{estimated_tokens}")
        
        self.logger.info("Complete Final Prompt:")
        self.logger.info("=" * 60)
        self.logger.info(prompt)
        self.logger.info("=" * 60)
    
    def log_final_response(self, response: str, time_taken: float):
        """Логируем финальный ответ"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        self.logger.info("🎉 STAGE 6: FINAL RESPONSE")
        self.logger.info(f"Generation Time: {time_taken:.3f}s")
        self.logger.info(f"Total Session Time: {total_time:.3f}s")
        self.logger.info(f"Response Length: {len(response)} chars")
        self.logger.info("AI Response:")
        self.logger.info("-" * 40)
        self.logger.info(response)
        self.logger.info("-" * 40)
        
        # Простая статистика
        self.logger.info("📊 SESSION COMPLETE")
        self.logger.info(f"Session ID: {self.session_id}")
        self.logger.info(f"Total Time: {total_time:.3f}s")
        self.logger.info("=" * 60)

# Глобальный экземпляр
rag_debug = RAGDebugLogger()