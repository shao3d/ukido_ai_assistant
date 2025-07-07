# llamaindex_rag.py
"""
✅ ФИНАЛЬНАЯ ВЕРСИЯ v6: Исправлен недостающий импорт 'ChatMessage'.
"""
import logging
import time
from typing import Tuple, Dict, Any, List

import pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
# ✅ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Добавляем недостающие импорты
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

try:
    from rag_debug_logger import rag_debug
except ImportError:
    class DummyDebug:
        def log_enricher_prompt(self, *args): pass
        def log_retrieval_results(self, *args): pass
        def log_final_response(self, *args, **kwargs): pass
    rag_debug = DummyDebug()

try:
    from config import config
except ImportError:
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import config

class LlamaIndexRAG:
    """
    ✅ ФИНАЛЬНАЯ ВЕРСИЯ v6: RAG-система с динамическим созданием движка и юмором.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pinecone_index_name = "ukido"
        
        self.index = None
        self.reranker = None
        self.llm = None
        
        try:
            self.llm = OpenRouter(
                api_key=config.OPENROUTER_API_KEY, 
                model="openai/gpt-4o-mini",
                temperature=0.7,
                max_tokens=1024
            )
            Settings.llm = self.llm
            Settings.embed_model = GeminiEmbedding(
                model_name=config.EMBEDDING_MODEL, 
                api_key=config.GEMINI_API_KEY
            )

            pc = pinecone.Pinecone(api_key=config.PINECONE_API_KEY)
            pinecone_index = pc.Index(self.pinecone_index_name)
            vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
            self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

            self.reranker = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
                top_n=4
            )

            self.logger.info("✅ Компоненты LlamaIndexRAG (v6) успешно инициализированы.")

        except Exception as e:
            self.logger.error(f"❌ Критическая ошибка инициализации компонентов LlamaIndexRAG: {e}", exc_info=True)
            raise

    def _build_dynamic_system_prompt(self, current_state: str, use_humor: bool) -> str:
        base_prompt = """Ты — AI-ассистент онлайн-школы Ukido.
Твоя задача — отвечать на вопросы, используя предоставленный ниже контекст из базы знаний.
- НЕ выдумывай факты.
- Если в контексте нет ответа на вопрос, честно скажи: "К сожалению, в моей базе знаний нет информации по этому вопросу."
"""
        verbosity_instruction = "Отвечай полно и дружелюбно, но по существу." if use_humor else "Будь кратким и отвечай по существу."
        
        state_instructions = {
            'greeting': "Это начало диалога. Начни с короткого дружелюбного приветствия.",
            'fact_finding': "Сосредоточься на предоставлении точных фактов из контекста. Будь кратким и четким.",
            'problem_solving': "Прояви эмпатию к проблеме пользователя. Используй найденные факты, чтобы предложить решение или совет.",
            'closing': """ПРИОРИТЕТНАЯ ЗАДАЧА: Пользователь хочет записаться на урок. Твоя единственная цель — сгенерировать токен [ACTION:SEND_LESSON_LINK]. Не отвечай на другие части вопроса. Просто подтверди его намерение короткой фразой (например, "Отлично, с удовольствием помогу!") и сразу же добавь токен."""
        }
        
        humor_instruction = """
ИНСТРУКЦИЯ ПО СТИЛЮ: Твой стиль общения — легкий, интеллигентный юмор в духе Михаила Жванецкого. Используй меткие наблюдения, иронию и афористичные фразы. Твоя шутка не должна заслонять суть ответа, а элегантно обрамлять ее.
"""
        
        instruction = state_instructions.get(current_state, state_instructions['fact_finding'])
        final_prompt = f"{base_prompt}\n{verbosity_instruction}\nИНСТРУКЦИЯ ПО СИТУАЦИИ: {instruction}"
        
        if use_humor:
            final_prompt += humor_instruction
            
        return final_prompt

    def _prepare_chat_history(self, conversation_history: List[str] = None) -> List[ChatMessage]:
        if not conversation_history: return []
        smart_history = conversation_history[-4:]
        chat_messages = []
        for msg in smart_history:
            try:
                role_str, content = msg.split(': ', 1)
                role = MessageRole.ASSISTANT if "ассистент" in role_str.lower() else MessageRole.USER
                chat_messages.append(ChatMessage(role=role, content=content))
            except ValueError:
                self.logger.warning(f"Некорректный формат сообщения в истории: '{msg}'")
                continue
        return chat_messages

    def search_and_answer(self, query: str, conversation_history: List[str] = None, current_state: str = 'fact_finding', use_humor: bool = False) -> Tuple[str, Dict[str, Any]]:
        search_start = time.time()
        
        if not all([self.index, self.reranker, self.llm]):
            self.logger.error("Компоненты RAG не готовы")
            return "Ошибка: RAG-система не готова.", {}

        try:
            system_prompt = self._build_dynamic_system_prompt(current_state, use_humor)
            rag_debug.log_enricher_prompt(f"DYNAMIC SYSTEM PROMPT (Humor: {use_humor}):\n{system_prompt}")

            chat_engine = ContextChatEngine.from_defaults(
                retriever=self.index.as_retriever(similarity_top_k=15, node_postprocessors=[self.reranker]),
                llm=self.llm,
                system_prompt=system_prompt,
                memory=ChatMemoryBuffer.from_defaults(token_limit=16384)
            )

            chat_history = self._prepare_chat_history(conversation_history)
            history_len = len(chat_history)
            self.logger.info(f"🔍 Запрос в LlamaIndex: '{query}' | Состояние: {current_state} | История: {history_len}")
            
            response = chat_engine.chat(query, chat_history=chat_history)
            
            final_answer = response.response
            search_time = time.time() - search_start
            
            source_nodes = response.source_nodes or []
            context_chunks = [node.get_content() for node in source_nodes]
            scores = [getattr(node, 'score', 0.5) for node in source_nodes]
            average_score = sum(scores) / len(scores) if scores else 0.0

            rag_debug.log_retrieval_results(chunks=context_chunks, scores=scores, time_taken=search_time, total_before_rerank=15)

            metrics = {'search_time': search_time, 'chunks_found': len(context_chunks), 'average_score': average_score, 'max_score': max(scores) if scores else 0.0, 'history_used': history_len}

            self.logger.info(f"✅ LlamaIndex сгенерировал ответ за {search_time:.2f}s (состояние: {current_state})")
            return final_answer, metrics

        except Exception as e:
            self.logger.error(f"❌ Ошибка в RAG search_and_answer: {e}", exc_info=True)
            return "К сожалению, произошла внутренняя ошибка при поиске информации.", {}

try:
    llama_index_rag = LlamaIndexRAG()
except Exception as e:
    llama_index_rag = None
    logging.getLogger(__name__).error(f"Не удалось создать LlamaIndexRAG: {e}")