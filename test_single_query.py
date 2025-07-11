# test_single_query.py
"""
Быстрый тест одного запроса для проверки RAG с новыми метаданными
"""
import logging
from llamaindex_rag import llama_index_rag

# Настройка логирования для просмотра метаданных
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_query(query: str):
    """Тестирует один запрос и показывает метрики"""
    print(f"\n{'='*60}")
    print(f"🔍 ТЕСТИРУЕМ ЗАПРОС: '{query}'")
    print(f"{'='*60}\n")
    
    try:
        # Вызываем RAG напрямую без обогащения
        response, metrics = llama_index_rag.search_and_answer(
            query=query,
            current_state='fact_finding',
            use_humor=False
        )
        
        # Показываем метрики
        print(f"📊 МЕТРИКИ ПОИСКА:")
        print(f"   • Max score: {metrics.get('max_score', 0):.3f}")
        print(f"   • Avg score: {metrics.get('average_score', 0):.3f}")
        print(f"   • Chunks found: {metrics.get('chunks_found', 0)}")
        print(f"   • Search time: {metrics.get('search_time', 0):.2f}s")
        
        # Показываем ответ
        print(f"\n💬 ОТВЕТ RAG:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    # Тестируем три проблемных запроса по очереди
    test_queries = [
        "А какие-то суперскидки есть?",
        "У моего сына диагностировали диабет",
        "Мой сын увлекается программированием"
    ]
    
    print("🚀 ЗАПУСК ТЕСТА RAG С НОВЫМИ МЕТАДАННЫМИ")
    print(f"📝 Будем тестировать {len(test_queries)} запроса\n")
    
    for query in test_queries:
        test_query(query)
        print("\n" + "="*80 + "\n")
