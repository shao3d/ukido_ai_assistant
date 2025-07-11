# llamaindex_rag.py
"""
✅ ВЕРСИЯ v12: Интеграция SmartQueryFilter и разнообразные ответы при отсутствии информации.
"""
import logging
import time
import random
from typing import Tuple, Dict, Any, List

import pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor.types import BaseNodePostprocessor

# НОВЫЙ ИМПОРТ
from rag_filters import SmartQueryFilter

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

class MetadataBoostRetriever(BaseRetriever):
    """Custom retriever that applies metadata-based score boosting"""
    
    def __init__(self, base_retriever, boost_function, query_intent, original_query):
        super().__init__()
        self.base_retriever = base_retriever
        self.boost_function = boost_function
        self.query_intent = query_intent
        self.original_query = original_query
        
    def _retrieve(self, query_bundle):
        # Получаем nodes от базового retriever
        nodes = self.base_retriever.retrieve(query_bundle)
        # Применяем boost
        boosted_nodes = self.boost_function(nodes, self.query_intent, self.original_query)
        return boosted_nodes

class MetadataBoostPostProcessor(BaseNodePostprocessor):
    """Post-processor that applies metadata-based score boosting after reranking"""
    query_intent: dict
    original_query: str
    final_top_k: int = 4
    
    def __init__(self, query_intent, original_query, final_top_k=4):
        super().__init__(
            query_intent=query_intent,
            original_query=original_query,
            final_top_k=final_top_k
        )
        
    def _postprocess_nodes(self, nodes, query_bundle=None):
        """Apply metadata boost and return top-k nodes"""
        import logging
        logger = logging.getLogger(__name__)
        
        boosted_nodes = []
        
        for node in nodes:
            boost_factor = 1.0
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            
            # Boost для ценовых запросов
            if self.query_intent['category'] == 'pricing' and metadata.get('has_pricing', False):
                boost_factor *= 1.5
                
            # Boost для особых потребностей
            elif self.query_intent['category'] == 'special_needs' and metadata.get('has_special_needs_info', False):
                boost_factor *= 1.6
                
            # Boost для курсов
            courses = metadata.get('courses_offered', [])
            if courses:
                # Проверяем упоминание программирования
                if any(word in self.original_query.lower() for word in ['программ', 'проект', 'технолог', 'компьютер']):
                    if 'Капитан Проектов' in courses:
                        boost_factor *= 1.4
                        
                # Общая проверка для всех курсов
                if self.query_intent['category'] == 'courses':
                    # Проверяем упоминание любого курса из списка в запросе
                    query_lower = self.original_query.lower()
                    for course in courses:
                        if course.lower() in query_lower:
                            boost_factor *= 1.4
                            break
                            
            # Boost для возрастных групп
            if metadata.get('age_groups_mentioned') and any(word in self.original_query.lower() for word in ['лет', 'возраст', 'класс', 'ребенок', 'ребёнок']):
                boost_factor *= 1.3
                
            # Boost для расписания
            if self.query_intent['category'] == 'schedule' and metadata.get('schedule_mentioned', False):
                boost_factor *= 1.4
                
            # Boost для учителей
            if any(word in self.original_query.lower() for word in ['учител', 'преподавател', 'педагог']) and metadata.get('teachers_mentioned', False):
                boost_factor *= 1.3
                
            # Применяем boost
            if hasattr(node, 'score'):
                original_score = node.score
                node.score = node.score * boost_factor
                
                # Логирование для отладки
                if boost_factor > 1.0:
                    logger.info(f"🚀 PostProcessor boosted chunk by {boost_factor}x - Score: {original_score:.3f} -> {node.score:.3f}")
                    
            boosted_nodes.append(node)
        
        # Сортируем по score в убывающем порядке
        boosted_nodes.sort(key=lambda x: getattr(x, 'score', 0.0), reverse=True)
        
        # Обрезаем до final_top_k
        return boosted_nodes[:self.final_top_k]

class LlamaIndexRAG:
    """
    ✅ ВЕРСИЯ v12: RAG-система с умной фильтрацией и вариативными ответами.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pinecone_index_name = "ukido"
        
        self.index = None
        self.reranker = None
        self.llm = None
        
        # НОВЫЙ КОМПОНЕНТ
        self.query_filter = SmartQueryFilter()
        
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
                top_n=10
            )

            self.logger.info("✅ Компоненты LlamaIndexRAG (v12) успешно инициализированы.")
            self.logger.info("✅ SmartQueryFilter интегрирован.")

        except Exception as e:
            self.logger.error(f"❌ Критическая ошибка инициализации компонентов LlamaIndexRAG: {e}", exc_info=True)
            raise

    def _build_dynamic_system_prompt(self, current_state: str, use_humor: bool) -> str:
        # НОВОЕ: Вместо скучной фразы используем варианты
        no_info_phrases = [
            "Хм, кажется, этот вопрос выходит за рамки моих знаний о школе. Может, спросите что-то другое?",
            "Интересный вопрос! Но у меня пока нет на него ответа. Давайте поговорим о наших курсах?",
            "Вы меня озадачили! Этой информации у меня нет, но я с радостью расскажу о наших программах.",
            "Надо же, не могу найти ответ на этот вопрос! Зато могу рассказать массу интересного о школе Ukido.",
            "Упс, здесь у меня пробел в знаниях! Спросите лучше про наши курсы или преподавателей."
        ]
        no_info_phrase = random.choice(no_info_phrases)

        base_prompt = f"""Ты — AI-ассистент онлайн-школы Ukido.
Твоя задача — отвечать на вопросы, используя предоставленный ниже контекст из базы знаний.
- НЕ выдумывай факты.
- Если в контексте нет ответа на вопрос, честно скажи: "{no_info_phrase}"
"""
        verbosity_instruction = "Отвечай полно и дружелюбно, но по существу." if use_humor else "Будь кратким и отвечай по существу."
        
        state_instructions = {
            'greeting': "Это начало диалога. Начни с короткого дружелюбного приветствия.",
            'fact_finding': "Сосредоточься на предоставлении точных фактов из контекста. Будь кратким и четким.",
            'problem_solving': "Прояви эмпатию к проблеме пользователя. Используй найденные факты, чтобы предложить решение или совет.",
            'closing': """ПРИОРИТЕТНАЯ ЗАДАЧА: Пользователь хочет записаться на урок. Твоя единственная цель — сгенерировать токен [ACTION:SEND_LESSON_LINK].
- Не отвечай на другие части вопроса, даже если они есть.
- Игнорируй найденный контекст, если он не помогает с записью.
- Твой ответ должен состоять ТОЛЬКО из короткой подтверждающей фразы (например, "Отлично, с удовольствием помогу!") и сразу после нее токена [ACTION:SEND_LESSON_LINK]."""
        }
        
        humor_instruction = """
ИНСТРУКЦИЯ ПО СТИЛЮ: Твой стиль общения — легкий, интеллигентный юмор в духе Михаила Жванецкого. Используй меткие наблюдения, иронию и афористичные фразы. Твоя шутка не должна заслонять суть ответа, а элегантно обрамлять ее.
"""
        
        instruction = state_instructions.get(current_state, state_instructions['fact_finding'])
        final_prompt = f"{base_prompt}\n{verbosity_instruction}\nИНСТРУКЦИЯ ПО СИТУАЦИИ: {instruction}"
        
        if use_humor:
            final_prompt += humor_instruction
            
        return final_prompt
    
    def _boost_scores_by_metadata(self, nodes, query_intent, query):
        """Повышает scores для чанков с релевантными метаданными"""
        boosted_nodes = []
        
        for node in nodes:
            boost_factor = 1.0
            metadata = node.metadata if hasattr(node, 'metadata') else {}
            
            # Boost для ценовых запросов
            if query_intent['category'] == 'pricing' and metadata.get('has_pricing', False):
                boost_factor *= 1.5
                
            # Boost для особых потребностей
            elif query_intent['category'] == 'special_needs' and metadata.get('has_special_needs_info', False):
                boost_factor *= 1.6
                
            # Boost для курсов
            courses = metadata.get('courses_offered', [])
            if courses:
                # Проверяем упоминание программирования
                if any(word in query.lower() for word in ['программ', 'проект', 'технолог', 'компьютер']):
                    if 'Капитан Проектов' in courses:
                        boost_factor *= 1.4
                        
                # Общая проверка для всех курсов
                if query_intent['category'] == 'courses':
                    # Проверяем упоминание любого курса из списка в запросе
                    query_lower = query.lower()
                    for course in courses:
                        if course.lower() in query_lower:
                            boost_factor *= 1.4
                            break
                            
            # Boost для возрастных групп
            if metadata.get('age_groups_mentioned') and any(word in query.lower() for word in ['лет', 'возраст', 'класс', 'ребенок', 'ребёнок']):
                boost_factor *= 1.3
                
            # Boost для расписания
            if query_intent['category'] == 'schedule' and metadata.get('schedule_mentioned', False):
                boost_factor *= 1.4
                
            # Boost для учителей
            if any(word in query.lower() for word in ['учител', 'преподавател', 'педагог']) and metadata.get('teachers_mentioned', False):
                boost_factor *= 1.3
                        
            # Применяем boost
            if hasattr(node, 'score'):
                original_score = node.score
                node.score = node.score * boost_factor
                
                # Логирование для отладки
                if boost_factor > 1.0:
                    self.logger.info(f"🚀 Boosted chunk by {boost_factor}x - Score: {original_score:.3f} -> {node.score:.3f} - has_pricing={metadata.get('has_pricing')}, courses={courses}")
                
            boosted_nodes.append(node)
        
        return boosted_nodes
    
    def _prepare_chat_history(self, conversation_history: List[str] = None) -> List[ChatMessage]:
        """
        Принимает список строк и конвертирует в ChatMessage.
        """
        if not conversation_history: return []
        
        smart_history = conversation_history[-4:]
        chat_messages = []
        for msg_str in smart_history:
            try:
                role_str, content = msg_str.split(': ', 1)
                role = MessageRole.ASSISTANT if "ассистент" in role_str.lower() else MessageRole.USER
                chat_messages.append(ChatMessage(role=role, content=content))
            except ValueError:
                self.logger.warning(f"Некорректный формат сообщения в истории, пропускаем: '{msg_str}'")
                continue
        return chat_messages

    def search_and_answer(self, query: str, conversation_history: List[str] = None, current_state: str = 'fact_finding', use_humor: bool = False) -> Tuple[str, Dict[str, Any]]:
        search_start = time.time()
        
        # НОВОЕ: Анализируем запрос для логирования и будущей фильтрации
        intent = self.query_filter.analyze_query_intent(query)
        self.logger.info(f"🎯 Категория запроса: {intent['category']}")
        
        if not all([self.index, self.reranker, self.llm]):
            self.logger.error("Компоненты RAG не готовы")
            return "Ошибка: RAG-система не готова.", {}

        try:
            system_prompt = self._build_dynamic_system_prompt(current_state, use_humor)
            rag_debug.log_enricher_prompt(f"DYNAMIC SYSTEM PROMPT (Humor: {use_humor}):\n{system_prompt}")

            chat_history_messages = self._prepare_chat_history(conversation_history)
            
            # Создаем базовый retriever
            base_retriever = self.index.as_retriever(similarity_top_k=15)
            
            # Оборачиваем его в наш booster
            boosted_retriever = MetadataBoostRetriever(
                base_retriever=base_retriever,
                boost_function=self._boost_scores_by_metadata,
                query_intent=intent,
                original_query=query
            )
            
            # Создаем metadata boost post-processor
            metadata_boost_processor = MetadataBoostPostProcessor(
                query_intent=intent,
                original_query=query,
                final_top_k=4
            )
            
            # Используем boosted_retriever в chat_engine
            chat_engine = ContextChatEngine.from_defaults(
                retriever=boosted_retriever,
                llm=self.llm,
                system_prompt=system_prompt,
                memory=ChatMemoryBuffer.from_defaults(token_limit=16384, chat_history=chat_history_messages),
                node_postprocessors=[self.reranker, metadata_boost_processor]
            )

            history_len = len(chat_history_messages)
            self.logger.info(f"🔍 Запрос в LlamaIndex: '{query}' | Состояние: {current_state} | История: {history_len}")
            
            response = chat_engine.chat(query)
            
            final_answer = response.response
            search_time = time.time() - search_start
            
            source_nodes = response.source_nodes or []
            
            # ВРЕМЕННОЕ ЛОГИРОВАНИЕ МЕТАДАННЫХ для отладки
            for i, node in enumerate(source_nodes[:4]):  # Только топ-4 после реранкера
                if hasattr(node, 'metadata'):
                    md = node.metadata
                    self.logger.info(f"🏷️ Chunk {i+1} metadata: "
                                   f"pricing={md.get('has_pricing', '?')}, "
                                   f"courses={md.get('courses_offered', '?')}, "
                                   f"special={md.get('has_special_needs_info', '?')}, "
                                   f"category={md.get('content_category', '?')}")
            
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
