"""
DETERMINISTIC BUSINESS-CRITICAL CHUNKER FOR UKIDO RAG SYSTEM
============================================================

Детерминистичные правила чанкования на основе анализа всех 9 документов.
Максимальная надежность для business-critical Telegram бота.
"""

import os
import time
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import List, Dict
from datetime import datetime
import re

# --- КОНФИГУРАЦИЯ ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST_FACTS = os.getenv("PINECONE_HOST_FACTS")

if not all([GEMINI_API_KEY, PINECONE_API_KEY, PINECONE_HOST_FACTS]):
    raise ValueError("Отсутствуют необходимые переменные окружения")

genai.configure(api_key=GEMINI_API_KEY)
embedding_model = 'models/text-embedding-004'

class DeterministicBusinessChunker:
    """
    Детерминистичные правила чанкования для каждого документа
    """
    
    def __init__(self):
        print("🎯 DETERMINISTIC BUSINESS-CRITICAL CHUNKER")
        print("📋 Детерминистичные правила для каждого документа")
        print("=" * 55)

    def create_strategic_overview_chunks(self) -> List[Dict]:
        """
        Стратегические чанки для ключевых бизнес-вопросов
        """
        chunks = []
        
        # 1. ГЛАВНЫЙ ЧАНК - Обзор всех курсов для "Какие курсы?"
        courses_main = """ОСНОВНЫЕ КУРСЫ ШКОЛЫ UKIDO

Школа Ukido предлагает три основных курса для развития soft skills у детей разного возраста:

1. КУРС "ЮНЫЙ ОРАТОР" (7-10 лет)
Цель: Убрать страх публичных выступлений, научить четко излагать мысли
Длительность: 3 месяца (24 занятия), 2 раза в неделю по 90 минут
Группы: до 8 детей
Стоимость: 6000 грн в месяц
Преподаватель: Анна Коваленко (8 лет опыта, автор методики "Бесстрашный оратор")
Результат: 94% детей избавляются от страха публичных выступлений

2. КУРС "ЭМОЦИОНАЛЬНЫЙ КОМПАС" (9-12 лет)
Цель: Научить распознавать, понимать и управлять эмоциями
Длительность: 4 месяца (32 занятия), 2 раза в неделю по 90 минут  
Группы: до 6 детей
Стоимость: 7500 грн в месяц
Преподаватель: Дмитрий Петров (PhD, автор книги "EQ для детей")
Результат: Снижение конфликтности на 76%, повышение эмпатии на 82%

3. КУРС "КАПИТАН ПРОЕКТОВ" (11-14 лет)
Цель: Развитие лидерских качеств и умения работать в команде
Длительность: 5 месяцев (40 занятий), 2 раза в неделю по 90 минут
Группы: проектные команды 4-5 человек  
Стоимость: 8000 грн в месяц
Преподаватель: Елена Сидорова (MBA, топ-10 бизнес-тренеров Украины)
Результат: 85% выпускников становятся лидерами в классах

Первый урок любого курса бесплатный и пробный. Все занятия проходят онлайн."""
        
        chunks.append({"text": courses_main, "type": "courses_overview", "priority": "critical"})
        
        # 2. ЦЕНООБРАЗОВАНИЕ
        pricing_main = """СТОИМОСТЬ КУРСОВ И СКИДКИ UKIDO

БАЗОВЫЕ ЦЕНЫ ПО КУРСАМ:
• "Юный Оратор" (7-10 лет): 6000 грн/месяц × 3 месяца = 18000 грн
• "Эмоциональный Компас" (9-12 лет): 7500 грн/месяц × 4 месяца = 30000 грн
• "Капитан Проектов" (11-14 лет): 8000 грн/месяц × 5 месяцев = 40000 грн

ВАРИАНТЫ ОПЛАТЫ И СКИДКИ:
• Помесячная оплата: стандартная цена, без комиссий
• Поквартальная оплата: скидка 5% (для "Юный Оратор" = 17100 грн вместо 18000 грн)
• Оплата полного курса: скидка 10% (для "Юный Оратор" = 16200 грн вместо 18000 грн)
• Семейная скидка: 15% при обучении 2+ детей из одной семьи
• Скидка за друга: 1000 грн при рекомендации
• Рассрочка: на 3 месяца без процентов

СЕЗОННЫЕ АКЦИИ:
• Летняя скидка: 20% (июнь-август)
• Новогодняя: 15% (декабрь-январь)  
• День знаний: +1 месяц бесплатно (сентябрь)

ГАРАНТИИ: 100% возврат за 7 дней, 50% за месяц. Пробный урок бесплатно."""
        
        chunks.append({"text": pricing_main, "type": "pricing_overview", "priority": "critical"})
        
        return chunks

    def chunk_courses_detailed(self, content: str) -> List[Dict]:
        """
        Специальные правила для courses_detailed.txt
        """
        chunks = []
        
        # Разделяем по курсам
        if "КУРС \"ЮНЫЙ ОРАТОР\"" in content:
            orator_match = re.search(r'КУРС "ЮНЫЙ ОРАТОР".*?(?=КУРС "|$)', content, re.DOTALL)
            if orator_match:
                chunks.append({
                    "text": orator_match.group(0).strip(),
                    "type": "course_detail",
                    "course": "young_orator"
                })
        
        if "КУРС \"ЭМОЦИОНАЛЬНЫЙ КОМПАС\"" in content:
            compass_match = re.search(r'КУРС "ЭМОЦИОНАЛЬНЫЙ КОМПАС".*?(?=КУРС "|$)', content, re.DOTALL)
            if compass_match:
                chunks.append({
                    "text": compass_match.group(0).strip(), 
                    "type": "course_detail",
                    "course": "emotional_compass"
                })
        
        if "КУРС \"КАПИТАН ПРОЕКТОВ\"" in content:
            captain_match = re.search(r'КУРС "КАПИТАН ПРОЕКТОВ".*?$', content, re.DOTALL)
            if captain_match:
                chunks.append({
                    "text": captain_match.group(0).strip(),
                    "type": "course_detail", 
                    "course": "project_captain"
                })
        
        return chunks

    def chunk_teachers_team(self, content: str) -> List[Dict]:
        """
        Специальные правила для teachers_team.txt - связь преподаватель+курс
        """
        chunks = []
        
        # Анна Коваленко + Юный Оратор
        anna_match = re.search(r'АННА КОВАЛЕНКО.*?(?=ДМИТРИЙ ПЕТРОВ|$)', content, re.DOTALL)
        if anna_match:
            anna_text = anna_match.group(0).strip()
            enhanced_anna = f"""ПРЕПОДАВАТЕЛЬ КУРСА "ЮНЫЙ ОРАТОР"

{anna_text}

Курс "Юный Оратор" (7-10 лет) - 6000 грн/месяц, 3 месяца, группы до 8 детей.
Результат: 94% детей избавляются от страха публичных выступлений."""
            
            chunks.append({
                "text": enhanced_anna,
                "type": "teacher_course_link",
                "teacher": "anna_kovalenko",
                "course": "young_orator"
            })
        
        # Дмитрий Петров + Эмоциональный Компас  
        dmitry_match = re.search(r'ДМИТРИЙ ПЕТРОВ.*?(?=ЕЛЕНА СИДОРОВА|$)', content, re.DOTALL)
        if dmitry_match:
            dmitry_text = dmitry_match.group(0).strip()
            enhanced_dmitry = f"""ПРЕПОДАВАТЕЛЬ КУРСА "ЭМОЦИОНАЛЬНЫЙ КОМПАС"

{dmitry_text}

Курс "Эмоциональный Компас" (9-12 лет) - 7500 грн/месяц, 4 месяца, группы до 6 детей.
Результат: Снижение конфликтности на 76%, повышение эмпатии на 82%."""
            
            chunks.append({
                "text": enhanced_dmitry,
                "type": "teacher_course_link", 
                "teacher": "dmitry_petrov",
                "course": "emotional_compass"
            })
        
        # Елена Сидорова + Капитан Проектов
        elena_match = re.search(r'ЕЛЕНА СИДОРОВА.*?(?=ОЛЬГА МИРНАЯ|ПРИНЦИПЫ РАБОТЫ|$)', content, re.DOTALL)
        if elena_match:
            elena_text = elena_match.group(0).strip()
            enhanced_elena = f"""ПРЕПОДАВАТЕЛЬ КУРСА "КАПИТАН ПРОЕКТОВ"

{elena_text}

Курс "Капитан Проектов" (11-14 лет) - 8000 грн/месяц, 5 месяцев, проектные команды 4-5 человек.
Результат: 85% выпускников становятся лидерами в классах."""
            
            chunks.append({
                "text": enhanced_elena,
                "type": "teacher_course_link",
                "teacher": "elena_sidorova", 
                "course": "project_captain"
            })
        
        return chunks

    def chunk_faq_detailed(self, content: str) -> List[Dict]:
        """
        Правила для faq_detailed.txt - группировка по темам
        """
        chunks = []
        
        # Разбиваем по основным разделам FAQ
        sections = re.split(r'\n---\n', content)
        
        for section in sections:
            if len(section.strip()) < 200:
                continue
                
            section = section.strip()
            
            # Определяем тип раздела по заголовку
            if "ОБЩИЕ ВОПРОСЫ" in section:
                chunks.append({"text": section, "type": "faq_general"})
            elif "РЕЗУЛЬТАТАХ И ЭФФЕКТИВНОСТИ" in section:
                chunks.append({"text": section, "type": "faq_results"})
            elif "ТЕХНИЧЕСКИЕ И ОРГАНИЗАЦИОННЫЕ" in section:
                chunks.append({"text": section, "type": "faq_technical"})
            elif "СЕРТИФИКАЦИИ И ДОКУМЕНТАХ" in section:
                chunks.append({"text": section, "type": "faq_certificates"})
            elif "ФИНАНСОВЫЕ ВОПРОСЫ" in section:
                chunks.append({"text": section, "type": "faq_financial"})
            elif "ПРЕПОДАВАТЕЛЯХ И МЕТОДИКАХ" in section:
                chunks.append({"text": section, "type": "faq_methodology"})
            else:
                chunks.append({"text": section, "type": "faq_other"})
        
        return chunks

    def chunk_methodology_approach(self, content: str) -> List[Dict]:
        """
        Правила для methodology_approach.txt
        """
        chunks = []
        
        # Разбиваем по ключевым разделам
        sections = re.split(r'\n---\n', content)
        
        for section in sections:
            section = section.strip()
            if len(section) < 300:
                continue
                
            if "ПРАКТИКА + ИГРА + РЕФЛЕКСИЯ" in section:
                chunks.append({"text": section, "type": "methodology_core"})
            elif "ИНДИВИДУАЛЬНЫЙ ПОДХОД" in section:
                chunks.append({"text": section, "type": "methodology_individual"})
            elif "ТЕХНОЛОГИЧЕСКАЯ ПОДДЕРЖКА" in section:
                chunks.append({"text": section, "type": "methodology_tech"})
            elif "ГЕЙМИФИКАЦИЯ" in section:
                chunks.append({"text": section, "type": "methodology_gamification"})
            elif "НАУЧНАЯ ОСНОВА" in section:
                chunks.append({"text": section, "type": "methodology_science"})
            elif "ВОЗРАСТНЫЕ ОСОБЕННОСТИ" in section:
                chunks.append({"text": section, "type": "methodology_age"})
            else:
                chunks.append({"text": section, "type": "methodology_other"})
        
        return chunks

    def chunk_standard_document(self, content: str, doc_type: str) -> List[Dict]:
        """
        Стандартное чанкование для остальных документов
        """
        chunks = []
        
        # Разбиваем по разделам с ---
        sections = re.split(r'\n---\n', content)
        
        for section in sections:
            section = section.strip()
            if len(section) < 400:
                continue
                
            if len(section) > 1200:
                # Разбиваем длинную секцию по абзацам
                paragraphs = re.split(r'\n\n', section)
                current_chunk = ""
                
                for paragraph in paragraphs:
                    potential = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
                    
                    if len(potential) > 1000 and current_chunk:
                        chunks.append({"text": current_chunk.strip(), "type": doc_type})
                        current_chunk = paragraph
                    else:
                        current_chunk = potential
                
                if current_chunk:
                    chunks.append({"text": current_chunk.strip(), "type": doc_type})
            else:
                chunks.append({"text": section, "type": doc_type})
        
        return chunks

    def process_all_documents(self, directory_path: str) -> List[Dict]:
        """
        Обработка всех документов с индивидуальными правилами
        """
        print(f"\n📂 Обработка документов из: {directory_path}")
        
        all_chunks = []
        chunk_id = 0
        
        # 1. Стратегические чанки
        print("🎯 Создание стратегических чанков...")
        strategic_chunks = self.create_strategic_overview_chunks()
        
        for chunk in strategic_chunks:
            all_chunks.append({
                "id": f"ukido-strategic-{chunk_id}",
                "text": chunk["text"],
                "metadata": {
                    "source": "strategic_overview",
                    "chunk_type": chunk["type"],
                    "priority": chunk.get("priority", "normal"),
                    "chunk_length": len(chunk["text"])
                }
            })
            chunk_id += 1
        
        # 2. Обработка документов
        print("📚 Обработка документов с индивидуальными правилами...")
        
        document_rules = {
            "courses_detailed.txt": self.chunk_courses_detailed,
            "teachers_team.txt": self.chunk_teachers_team, 
            "faq_detailed.txt": self.chunk_faq_detailed,
            "methodology_approach.txt": self.chunk_methodology_approach
        }
        
        files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        
        for filename in files:
            print(f"📄 {filename}")
            
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                continue
            
            # Применяем специальные правила или стандартное чанкование
            if filename in document_rules:
                doc_chunks = document_rules[filename](content)
            else:
                doc_type = filename.replace('.txt', '').replace('_', '-')
                doc_chunks = self.chunk_standard_document(content, doc_type)
            
            # Добавляем чанки
            for chunk in doc_chunks:
                all_chunks.append({
                    "id": f"ukido-{chunk_id}",
                    "text": chunk["text"],
                    "metadata": {
                        "source": filename,
                        "chunk_type": chunk["type"],
                        "chunk_length": len(chunk["text"]),
                        **{k: v for k, v in chunk.items() if k not in ["text", "type"]}
                    }
                })
                chunk_id += 1
            
            print(f"   ✅ {len(doc_chunks)} чанков")
        
        print(f"\n📊 ИТОГО: {len(all_chunks)} чанков")
        return all_chunks

    def vectorize_and_upload(self, chunks: List[Dict]) -> bool:
        """
        Векторизация и загрузка в Pinecone
        """
        print(f"\n🔄 Векторизация {len(chunks)} чанков...")
        
        vectors = []
        
        for i, chunk_data in enumerate(chunks):
            try:
                embedding = genai.embed_content(
                    model=embedding_model,
                    content=chunk_data['text'],
                    task_type="RETRIEVAL_DOCUMENT"
                )
                
                vectors.append({
                    "id": chunk_data['id'],
                    "values": embedding['embedding'],
                    "metadata": {
                        "text": chunk_data['text'],
                        **chunk_data['metadata']
                    }
                })
                
                if (i + 1) % 10 == 0:
                    print(f"   📊 {i + 1}/{len(chunks)}")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"   ❌ Ошибка {chunk_data['id']}: {e}")
                continue
        
        # Загрузка в Pinecone
        print(f"\n☁️ Загрузка {len(vectors)} векторов в Pinecone...")
        
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(host=PINECONE_HOST_FACTS)
            
            # Очистка
            print("   🗑️ Очистка индекса...")
            index.delete(delete_all=True)
            time.sleep(5)
            
            # Загрузка батчами
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                index.upsert(vectors=batch)
                print(f"   📦 Батч {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")
                time.sleep(1)
            
            # Проверка
            time.sleep(3)
            stats = index.describe_index_stats()
            print(f"   ✅ Загружено: {stats.total_vector_count} векторов")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Ошибка Pinecone: {e}")
            return False

def main():
    """
    Основная функция
    """
    print("🚀 DETERMINISTIC BUSINESS-CRITICAL RECHUNKING")
    print("🎯 Максимальная надежность для Telegram бота")
    print("=" * 50)
    
    chunker = DeterministicBusinessChunker()
    
    try:
        # Обработка документов
        chunks = chunker.process_all_documents("data_facts")
        
        if not chunks:
            print("❌ Нет чанков для обработки")
            return False
        
        # Векторизация и загрузка
        success = chunker.vectorize_and_upload(chunks)
        
        if success:
            print("\n🎉 RECHUNKING ЗАВЕРШЕН!")
            print("🧪 Тестируй: https://ukidoaiassistant-production.up.railway.app/test-rag")
            return True
        else:
            print("\n❌ Ошибка загрузки")
            return False
            
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ RAG оптимизирован для максимальной конверсии!")
    else:
        print("\n⚠️ Требуется устранение ошибок")
