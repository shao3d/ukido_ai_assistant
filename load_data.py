import os
import time
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
import re

# --- НАСТРОЙКИ И ЗАГРУЗКА КЛЮЧЕЙ ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST_FACTS = os.getenv("PINECONE_HOST_FACTS")
PINECONE_HOST_STYLE = os.getenv("PINECONE_HOST_STYLE")

if not all([GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_HOST_FACTS, PINECONE_HOST_STYLE]):
    raise ValueError("Необходимо задать все ключи и хосты в файле .env")

# --- КОНФИГУРАЦИЯ КЛИЕНТОВ ---
genai.configure(api_key=GEMINI_API_KEY)
embedding_model = 'models/text-embedding-004'
pc = Pinecone(api_key=PINECONE_API_KEY)


def process_and_upload(directory_path, pinecone_index, index_name):
    """
    Читает все .txt файлы из директории, разбивает на чанки,
    создает для них эмбеддинги и загружает в указанный индекс Pinecone.
    """
    print(f"\n--- Начинаю обработку директории: {directory_path} ---")
    vector_id_counter = 0
    
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            print(f"Читаю файл: {filename}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = re.split(r'\n---\n|\n\n', content)
            
            for i, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if not chunk:
                    continue

                print(f"  - Создаю вектор для чанка №{i+1}...")
                try:
                    embedding = genai.embed_content(
                        model=embedding_model,
                        content=chunk,
                        task_type="RETRIEVAL_DOCUMENT",
                        title=f"Chunk from {filename}" 
                    )
                    
                    vector_to_upsert = {
                        "id": f"{index_name}-{vector_id_counter}",
                        "values": embedding['embedding'],
                        "metadata": {"text": chunk, "source": filename}
                    }
                    
                    pinecone_index.upsert(vectors=[vector_to_upsert])
                    print(f"    ... Вектор {vector_to_upsert['id']} успешно загружен.")
                    
                    vector_id_counter += 1
                    time.sleep(1) 

                except Exception as e:
                    print(f"    !!! Произошла ошибка при обработке чанка: {e}")
                    print(f"    !!! Проблемный чанк: {chunk[:100]}...")

    print(f"--- Обработка директории {directory_path} завершена. Всего загружено векторов: {vector_id_counter} ---")


# --- ОСНОВНОЙ КОД СКРИПТА ---
if __name__ == "__main__":
    try:
        print("Подключаюсь к индексам Pinecone...")
        index_facts = pc.Index(host=PINECONE_HOST_FACTS)
        index_style = pc.Index(host=PINECONE_HOST_STYLE)
        print("Подключение успешно.")
        
        # Запускаем обработку и загрузку
        process_and_upload("data_facts", index_facts, "ukido")
        process_and_upload("data_style", index_style, "ukido-style")
        
        print("\n\n✅ Все данные успешно обработаны и загружены в Pinecone!")

    except Exception as e:
        print(f"\n❌ Произошла критическая ошибка: {e}")
