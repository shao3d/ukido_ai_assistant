import re
import json
from typing import Dict, List, Any

def extract_metadata(text: str) -> Dict[str, Any]:
    """
    Извлекает ключевые метаданные из текста для RAG системы школы Ukido.
    Возвращает плоские метаданные без вложенных словарей.
    
    ТОП-10 самых важных полей для решения проблемных запросов:
    1. pricing_and_discounts - для запросов "Скидки есть?"
    2. special_needs - для запросов "Для моего сына с диабетом"
    3. skills_and_competencies - для запросов "Сын увлекается программированием"
    4. age_groups - критично для рекомендаций курсов
    5. time_parameters - для планирования обучения
    6. courses_offered - основной продукт
    7. content_category - для правильной категоризации
    8. technical_requirements - для технической поддержки
    9. support_and_safety - для вопросов о безопасности
    10. achievements_statistics - для доказательства эффективности
    """
    
    text_lower = text.lower()
    metadata = {}
    
    # 1. PRICING_AND_DISCOUNTS - для "Скидки есть?"
    pricing_info = _extract_pricing_info(text, text_lower)
    metadata.update({
        "has_pricing": pricing_info["has_pricing"],
        "prices_mentioned": pricing_info["prices_mentioned"],
        "discount_types": pricing_info["discount_types"],
        "payment_methods": pricing_info["payment_methods"],
        "refund_conditions": pricing_info["refund_conditions"]
    })
    
    # 2. SPECIAL_NEEDS - для "Для моего сына с диабетом"
    special_needs = _extract_special_needs(text, text_lower)
    metadata.update({
        "has_special_needs_info": special_needs["has_special_needs_info"],
        "conditions_supported": special_needs["conditions_supported"],
        "adaptations": special_needs["adaptations"],
        "learning_styles": special_needs["learning_styles"]
    })
    
    # 3. SKILLS_AND_COMPETENCIES - для "Сын увлекается программированием"
    skills_info = _extract_skills(text, text_lower)
    metadata.update({
        "primary_skills": skills_info["primary_skills"],
        "skills_courses_offered": skills_info["courses_offered"],
        "soft_skills_categories": skills_info["soft_skills_categories"]
    })
    
    # 4. AGE_GROUPS - критично для рекомендаций
    age_info = _extract_age_groups(text, text_lower)
    metadata.update({
        "min_age": str(age_info["min_age"]) if age_info["min_age"] is not None else "",
        "max_age": str(age_info["max_age"]) if age_info["max_age"] is not None else "",
        "age_groups_mentioned": age_info["age_groups_mentioned"]
    })
    # Добавляем courses_by_age как плоские поля
    for age_range, course in age_info["courses_by_age"].items():
        metadata[f"course_for_age_{age_range}"] = course
    
    # 5. TIME_PARAMETERS - для планирования
    time_info = _extract_time_parameters(text, text_lower)
    metadata.update({
        "lesson_duration": str(time_info["lesson_duration"]) if time_info["lesson_duration"] is not None else "",
        "lessons_per_week": str(time_info["lessons_per_week"]) if time_info["lessons_per_week"] is not None else "",
        "course_duration_months": [str(m) for m in time_info["course_duration_months"]],
        "schedule_times": time_info["schedule_times"],
        "homework_time": time_info["homework_time"] if time_info["homework_time"] is not None else "",
        "group_size_mentioned": time_info["group_size_mentioned"]
    })
    
    # 6. COURSES_OFFERED - основной продукт
    metadata["courses_offered"] = _extract_courses(text, text_lower)
    
    # 7. CONTENT_CATEGORY - для категоризации
    metadata["content_category"] = _extract_content_category(text, text_lower)
    
    # 8. TECHNICAL_REQUIREMENTS - для техподдержки
    tech_info = _extract_tech_requirements(text, text_lower)
    metadata.update({
        "has_tech_requirements": tech_info["has_tech_requirements"],
        "platforms_mentioned": tech_info["platforms_mentioned"],
        "internet_speed": tech_info["internet_speed"] if tech_info["internet_speed"] is not None else "",
        "devices": tech_info["devices"]
    })
    
    # 9. SUPPORT_AND_SAFETY - для безопасности
    safety_info = _extract_safety_info(text, text_lower)
    metadata.update({
        "has_safety_info": safety_info["has_safety_info"],
        "safety_measures": safety_info["safety_measures"],
        "data_protection": safety_info["data_protection"]
    })
    
    # 10. ACHIEVEMENTS_STATISTICS - для эффективности
    achievements = _extract_achievements(text, text_lower)
    metadata.update({
        "has_statistics": achievements["has_statistics"],
        "success_rates": achievements["success_rates"],
        "student_numbers": achievements["student_numbers"]
    })
    
    return metadata


def _extract_pricing_info(text: str, text_lower: str) -> Dict[str, Any]:
    """
    Извлекает информацию о ценах и скидках.
    Примеры поиска:
    - "6,000 грн" -> prices_mentioned: ["6000"]
    - "скидка 15%" -> discount_types: ["семейная_скидка_15%"]
    - "рассрочка" -> payment_methods: ["рассрочка"]
    """
    pricing_info = {
        "has_pricing": False,
        "prices_mentioned": [],
        "discount_types": [],
        "payment_methods": [],
        "refund_conditions": []
    }
    
    price_pattern = r'(\d[\d,\s]*\d)\s*(?:грн|гривен)'
    
    matches = re.findall(price_pattern, text_lower)
    for match in matches:
        price = re.sub(r'[,\s]', '', match)
        if price not in pricing_info["prices_mentioned"]:
            pricing_info["prices_mentioned"].append(price)
            
    if pricing_info["prices_mentioned"]:
        pricing_info["has_pricing"] = True
    
    discount_keywords = {
        "поквартальная": "поквартальная_оплата_5%",
        "полная оплата": "полная_оплата_курса_10%", 
        "семейная": "семейная_скидка_15%",
        "рекомендацию": "скидка_за_рекомендацию_1000грн",
        "социальная": "социальная_скидка_20%",
        "многодетн": "социальная_скидка_20%",
        "ато": "социальная_скидка_20%",
        "oos": "социальная_скидка_20%",
        "особыми потребностями": "социальная_скидка_20%",
        "стипенди": "стипендии",
        "скидка 5%": "поквартальная_оплата_5%",
        "скидка 10%": "полная_оплата_курса_10%",
        "скидка 15%": "семейная_скидка_15%",
        "скидка 20%": "социальная_скидка_20%"
    }
    
    for keyword, discount_type in discount_keywords.items():
        if keyword in text_lower:
            if discount_type not in pricing_info["discount_types"]:
                pricing_info["discount_types"].append(discount_type)
    
    payment_keywords = {
        "рассрочка": "рассрочка",
        "банковская карта": "банковская_карта",
        "visa": "банковская_карта",
        "mastercard": "банковская_карта",
        "беспроцентная": "рассрочка_банк_3мес",
        "первый взнос": "внутренняя_рассрочка"
    }
    
    for keyword, payment_method in payment_keywords.items():
        if keyword in text_lower:
            if payment_method not in pricing_info["payment_methods"]:
                pricing_info["payment_methods"].append(payment_method)
    
    refund_keywords = {
        "7 дней": "7дней_100%",
        "первый месяц": "1месяц_70%", 
        "второй месяц": "2месяц_50%",
        "медицинские показания": "мед_показания_100%",
        "100% возврат": "100%_возврат",
        "70%": "1месяц_70%",
        "50%": "2месяц_50%"
    }
    
    for keyword, refund_condition in refund_keywords.items():
        if keyword in text_lower:
            if refund_condition not in pricing_info["refund_conditions"]:
                pricing_info["refund_conditions"].append(refund_condition)
    
    return pricing_info


def _extract_special_needs(text: str, text_lower: str) -> Dict[str, Any]:
    """
    Извлекает информацию об особых потребностях.
    Примеры поиска:
    - "СДВГ" -> conditions_supported: ["СДВГ"]
    - "короткие блоки 5 минут" -> adaptations: ["короткие_блоки_5-7мин"]
    - "визуальные подсказки" -> adaptations: ["визуальные_подсказки"]
    """
    special_needs = {
        "has_special_needs_info": False,
        "conditions_supported": [],
        "adaptations": [],
        "learning_styles": []
    }
    
    conditions_keywords = {
        "сдвг": "СДВГ",
        "рас": "РАС", 
        "аутизм": "аутизм",
        "тревожн": "тревожность",
        "застенчив": "застенчивость",
        "эмоциональные нарушения": "эмоциональные_нарушения",
        "логопедические": "логопедические_проблемы",
        "нарушения слуха": "нарушения_слуха",
        "нарушения зрения": "нарушения_зрения",
        "речевые нарушения": "речевые_нарушения",
        "диабет": "медицинские_особенности",
        "аллерг": "медицинские_особенности",
        "особыми потребностями": "особые_потребности_общие"
    }
    
    for keyword, condition in conditions_keywords.items():
        if keyword in text_lower:
            if condition not in special_needs["conditions_supported"]:
                special_needs["conditions_supported"].append(condition)
                special_needs["has_special_needs_info"] = True
    
    adaptations_keywords = {
        "короткие блоки": "короткие_блоки_5-7мин",
        "3-5 минут": "короткие_блоки_3-5мин",
        "5-7 минут": "короткие_блоки_5-7мин",
        "частая смена": "частая_смена_деятельности",
        "предсказуемая структура": "предсказуемая_структура",
        "визуальные подсказки": "визуальные_подсказки",
        "камера выключена": "камера_выключена_при_перегрузке",
        "индивидуальные задания": "индивидуальные_задания",
        "техники релаксации": "техники_релаксации",
        "малых группах": "работа_в_малых_группах_2-3чел",
        "субтитры": "автоматические_субтитры",
        "аудио описания": "аудио_описания",
        "тактильные материалы": "тактильные_материалы",
        "дополнительное время": "дополнительное_время_на_ответы"
    }
    
    for keyword, adaptation in adaptations_keywords.items():
        if keyword in text_lower:
            if adaptation not in special_needs["adaptations"]:
                special_needs["adaptations"].append(adaptation)
                special_needs["has_special_needs_info"] = True
    
    learning_styles_keywords = {
        "визуал": "визуалы_35%",
        "аудиал": "аудиалы_25%", 
        "кинестетик": "кинестетики_40%",
        "холерик": "холерики",
        "сангвиник": "сангвиники",
        "флегматик": "флегматики",
        "меланхолик": "меланхолики"
    }
    
    for keyword, style in learning_styles_keywords.items():
        if keyword in text_lower:
            if style not in special_needs["learning_styles"]:
                special_needs["learning_styles"].append(style)
    
    return special_needs


def _extract_skills(text: str, text_lower: str) -> Dict[str, Any]:
    """
    Извлекает информацию о навыках и компетенциях.
    Примеры поиска:
    - "программирование" -> primary_skills: ["проектное_управление"] (ближайший навык)
    - "лидерство" -> primary_skills: ["лидерство"]
    - "Капитан Проектов" -> courses_offered: ["Капитан Проектов"]
    """
    skills_info = {
        "primary_skills": [],
        "courses_offered": [],
        "soft_skills_categories": []
    }
    
    skills_keywords = {
        "публичные выступления": "публичные_выступления",
        "выступления": "публичные_выступления",
        "ораторск": "публичные_выступления",
        "эмоциональная регуляция": "эмоциональная_регуляция",
        "эмоциональн": "эмоциональная_регуляция",
        "лидерство": "лидерство", 
        "лидер": "лидерство",
        "проектное управление": "проектное_управление",
        "проект": "проектное_управление",
        "программирование": "проектное_управление",
        "коммуникац": "коммуникация",
        "общение": "коммуникация",
        "эмпатия": "эмпатия",
        "уверенность": "уверенность_в_себе",
        "конфликт": "разрешение_конфликтов",
        "командная работа": "командная_работа",
        "команда": "командная_работа",
        "креативность": "креативность",
        "творчество": "креативность"
    }
    
    for keyword, skill in skills_keywords.items():
        if keyword in text_lower:
            if skill not in skills_info["primary_skills"]:
                skills_info["primary_skills"].append(skill)
    
    courses_keywords = {
        "юный оратор": "Юный Оратор",
        "эмоциональный компас": "Эмоциональный Компас", 
        "капитан проектов": "Капитан Проектов",
        "профессии будущего": "Профессии будущего"
    }
    
    for keyword, course in courses_keywords.items():
        if keyword in text_lower:
            if course not in skills_info["courses_offered"]:
                skills_info["courses_offered"].append(course)
    
    categories_keywords = {
        "коммуникативн": "коммуникативные",
        "эмоциональн": "эмоциональные",
        "лидерск": "лидерские", 
        "проектн": "проектные",
        "социальн": "социальные"
    }
    
    for keyword, category in categories_keywords.items():
        if keyword in text_lower:
            if category not in skills_info["soft_skills_categories"]:
                skills_info["soft_skills_categories"].append(category)
    
    return skills_info


def _extract_age_groups(text: str, text_lower: str) -> Dict[str, Any]:
    """
    Извлекает информацию о возрастных группах.
    Примеры поиска:
    - "7-10 лет" -> age_groups_mentioned: ["7-10"]
    - "9 лет" -> min_age: 9
    """
    age_info = {
        "min_age": None,
        "max_age": None,
        "age_groups_mentioned": [],
        "courses_by_age": {}
    }
    
    age_range_patterns = [
        r'(\d+)-(\d+)\s*(?:лет|года?)',
        r'(\d+)\s*-\s*(\d+)',
    ]
    
    ages_found = []
    
    for pattern in age_range_patterns:
        matches = re.findall(pattern, text_lower)
        for min_age, max_age in matches:
            min_age, max_age = int(min_age), int(max_age)
            if min_age <= 18 and max_age <= 18:
                age_range = f"{min_age}-{max_age}"
                if age_range not in age_info["age_groups_mentioned"]:
                    age_info["age_groups_mentioned"].append(age_range)
                ages_found.extend([min_age, max_age])
    
    single_age_pattern = r'(\d+)\s*(?:лет|года?|летн)'
    matches = re.findall(single_age_pattern, text_lower)
    for age_str in matches:
        age = int(age_str)
        if 6 <= age <= 18:
            ages_found.append(age)
    
    if ages_found:
        age_info["min_age"] = min(ages_found)
        age_info["max_age"] = max(ages_found)
    
    course_age_mapping = {
        "юный оратор": "7-10",
        "эмоциональный компас": "9-12",
        "капитан проектов": "11-14"
    }
    
    for course_name, age_range in course_age_mapping.items():
        if course_name in text_lower:
            age_info["courses_by_age"][age_range] = course_name.title()
    
    return age_info


def _extract_time_parameters(text: str, text_lower: str) -> Dict[str, Any]:
    """
    Извлекает временные параметры.
    """
    time_info = {
        "lesson_duration": None,
        "lessons_per_week": None,
        "course_duration_months": [],  # <<< ИЗМЕНЕНИЕ: Тип изменен на список
        "schedule_times": [],
        "homework_time": None,
        "group_size_mentioned": []  # <<< НОВОЕ ПОЛЕ: Для размеров групп
    }
    
    # Длительность занятия
    duration_patterns = [r'(\d+)\s*минут', r'(\d+)\s*мин']
    for pattern in duration_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            duration = int(matches[0])
            if 15 <= duration <= 180:
                time_info["lesson_duration"] = duration
                break
    
    # Количество занятий в неделю
    if "2 раза в неделю" in text_lower or "дважды в неделю" in text_lower:
        time_info["lessons_per_week"] = 2
    elif "раз в неделю" in text_lower:
        time_info["lessons_per_week"] = 1
    elif "3 раза в неделю" in text_lower:
        time_info["lessons_per_week"] = 3
    
    # Время расписания
    schedule_pattern = r'(\d{1,2}:\d{2}(?:-\d{1,2}:\d{2})?)'
    matches = re.findall(schedule_pattern, text)
    for time_slot in matches:
        if time_slot not in time_info["schedule_times"]:
            time_info["schedule_times"].append(time_slot)
    
    # Продолжительность курса
    duration_course_patterns = [r'(\d+)\s*месяц', r'(\d+)\s*мес']
    for pattern in duration_course_patterns:
        matches = re.findall(pattern, text_lower)
        for months_str in matches:
            months = int(months_str)
            if 1 <= months <= 12 and months not in time_info["course_duration_months"]:
                time_info["course_duration_months"].append(months)
    # <<< ИЗМЕНЕНИЕ: `break` удален для сбора всех значений
    
    # Время домашних заданий
    homework_patterns = [r'(\d+-\d+)\s*минут', r'(\d+-\d+)\s*мин']
    for pattern in homework_patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            time_info["homework_time"] = f"{matches[0]}мин"
            break
            
    # <<< НОВЫЙ БЛОК: Извлечение размера группы >>>
    # Это организационный параметр, поэтому его место здесь, а не в "достижениях".
    group_size_pattern = r'(?:до|размер группы: до|команды по)\s*(\d+(?:-\d+)?)\s*(?:детей|человек)'
    matches = re.findall(group_size_pattern, text_lower)
    for size in matches:
        if size not in time_info["group_size_mentioned"]:
            time_info["group_size_mentioned"].append(size)

    return time_info


def _extract_courses(text: str, text_lower: str) -> List[str]:
    """Извлекает упоминания курсов с учетом разных вариантов написания"""
    courses = []
    
    # Паттерны для поиска курсов с разными кавычками и форматами
    course_patterns = {
        "Юный Оратор": [
            r'"юный оратор"',      # "Юный Оратор"
            r"'юный оратор'",      # 'Юный Оратор'  
            r'«юный оратор»',      # «Юный Оратор»
            r'юный оратор',        # Юный Оратор (без кавычек)
            r'курс\s+"?юный оратор"?',  # курс "Юный Оратор" или курс Юный Оратор
        ],
        "Эмоциональный Компас": [
            r'"эмоциональный компас"',
            r"'эмоциональный компас'",
            r'«эмоциональный компас»',
            r'эмоциональный компас',
            r'курс\s+"?эмоциональный компас"?',
        ],
        "Капитан Проектов": [
            r'"капитан проектов"',
            r"'капитан проектов'",
            r'«капитан проектов»',
            r'капитан проектов',
            r'курс\s+"?капитан проектов"?',
        ],
        "Профессии будущего": [
            r'"профессии будущего"',
            r"'профессии будущего'",
            r'«профессии будущего»',
            r'профессии будущего',
            r'курс\s+"?профессии будущего"?',
        ]
    }
    
    # Ищем каждый курс по всем паттернам
    for course_name, patterns in course_patterns.items():
        found = False
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                found = True
                break
        
        if found and course_name not in courses:
            courses.append(course_name)
    
    return courses


def _extract_content_category(text: str, text_lower: str) -> str:
    """Определяет категорию контента"""
    category_keywords = {
        "условия_обучения": ["расписание", "занятий", "структура", "организация", "условия"],
        "ценообразование": ["цена", "стоимость", "грн", "скидка", "оплата", "рассрочка"],
        "курсы": ["курс", "юный оратор", "эмоциональный компас", "капитан проектов"],
        "FAQ": ["вопрос", "ответ", "faq", "часто задаваемые"],
        "методология": ["методология", "подход", "принцип", "метод"],
        "безопасность_и_доверие": ["безопасность", "защита", "доверие", "конфиденциальность"],
        "команда_преподавателей": ["преподаватель", "учитель", "тренер", "команда"],
        "результаты_достижения": ["результат", "достижение", "статистика", "успех"]
    }
    
    for category, keywords in category_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    
    return "общая_информация"


def _extract_tech_requirements(text: str, text_lower: str) -> Dict[str, Any]:
    """Извлекает технические требования"""
    tech_info = {
        "has_tech_requirements": False,
        "platforms_mentioned": [],
        "internet_speed": None,
        "devices": []
    }
    
    platforms = ["zoom", "miro", "kahoot", "padlet", "trello", "figma", "canva", "slack"]
    for platform in platforms:
        if platform in text_lower:
            if platform not in tech_info["platforms_mentioned"]:
                tech_info["platforms_mentioned"].append(platform)
                tech_info["has_tech_requirements"] = True
    
    if "мбит" in text_lower:
        speed_match = re.search(r'(\d+)[\+\s]*мбит', text_lower)
        if speed_match:
            tech_info["internet_speed"] = f"{speed_match.group(1)}+ Мбит/с"
            tech_info["has_tech_requirements"] = True
    
    device_keywords = ["компьютер", "ноутбук", "планшет", "ipad", "windows", "macos"]
    for device in device_keywords:
        if device in text_lower:
            if device not in tech_info["devices"]:
                tech_info["devices"].append(device)
                tech_info["has_tech_requirements"] = True
    
    return tech_info


def _extract_safety_info(text: str, text_lower: str) -> Dict[str, Any]:
    """Извлекает информацию о безопасности"""
    safety_info = {
        "has_safety_info": False,
        "safety_measures": [],
        "data_protection": []
    }
    
    safety_keywords = {
        "пароль": "уникальные_пароли",
        "камера": "обязательные_веб_камеры",
        "согласие": "родительское_согласие",
        "модерация": "модерация_активности"
    }
    
    for keyword, measure in safety_keywords.items():
        if keyword in text_lower:
            if measure not in safety_info["safety_measures"]:
                safety_info["safety_measures"].append(measure)
                safety_info["has_safety_info"] = True
    
    data_keywords = {
        "gdpr": "GDPR_соблюдение",
        "конфиденциальность": "защита_персональных_данных",
        "шифрование": "шифрование_данных"
    }
    
    for keyword, protection in data_keywords.items():
        if keyword in text_lower:
            if protection not in safety_info["data_protection"]:
                safety_info["data_protection"].append(protection)
                safety_info["has_safety_info"] = True
    
    return safety_info


def _extract_achievements(text: str, text_lower: str) -> Dict[str, Any]:
    """Извлекает статистику и достижения"""
    achievements = {
        "has_statistics": False,
        "success_rates": [],
        "student_numbers": []
    }
    
    # <<< ИЗМЕНЕНИЕ: Паттерн стал точнее >>>
    # Статистика успеха в процентах
    percentage_pattern = r'(\d+)%\s*(?:детей|выпускников|родителей|снижается|возрастает)'
    matches = re.findall(percentage_pattern, text_lower)
    for percentage in matches:
        rate_str = f"{percentage}%"
        if rate_str not in achievements["success_rates"]:
            achievements["success_rates"].append(rate_str)
            achievements["has_statistics"] = True
    
    # <<< ИЗМЕНЕНИЕ: Паттерн стал точнее, чтобы не захватывать размер группы >>>
    # Количество учеников/выпускников (общие цифры)
    numbers_pattern = r'(\d+)\s*(?:выпускник|учеников прошло|реализована|запущено)'
    matches = re.findall(numbers_pattern, text_lower)
    for number in matches:
        student_count_str = f"{number}_total"
        if student_count_str not in achievements["student_numbers"]:
            achievements["student_numbers"].append(student_count_str)
            achievements["has_statistics"] = True
    
    return achievements


# Пример использования функции
if __name__ == "__main__":
    test_text_full_doc = """
    # КУРС "ЮНЫЙ ОРАТОР" (7-10 лет)
    - Длительность: 3 месяца (24 занятия)
    - Частота: 2 раза в неделю по 90 минут
    - Размер группы: до 8 детей
    - Стоимость: 6000 грн в месяц
    После курса 94% детей не боятся выступать.

    # КУРС "ЭМОЦИОНАЛЬНЫЙ КОМПАС" (9-12 лет)
    - Длительность: 4 месяца (32 занятия)
    - Размер группы: до 6 детей
    - у выпускников курса на 76% снижается количество конфликтов
    
    # КУРС "КАПИТАН ПРОЕКТОВ" (11-14 лет)
    - Длительность: 5 месяцев (40 занятий)
    - Размер группы: проектные команды по 4-5 человек
    - 85% выпускников становятся лидерами
    Примеры успешных проектов: экологическая инициатива "Чистый двор" (реализована в 3 школах Киева)
    """
    
    print("=== ТЕСТ: Полный документ ===")
    result = extract_metadata(test_text_full_doc)
    
    print("\n--- ИЗВЛЕЧЕННЫЕ ВРЕМЕННЫЕ ПАРАМЕТРЫ ---")
    print(f"Длительности курсов (в месяцах): {result['course_duration_months']}")
    print(f"Размеры групп: {result['group_size_mentioned']}")
    
    print("\n--- ИЗВЛЕЧЕННАЯ СТАТИСТИКА ---")
    print(f"Показатели успеха (%): {result['success_rates']}")
    print(f"Количество учеников (общие цифры): {result['student_numbers']}")

    print("\nПолный JSON:")
    print(json.dumps(result, indent=4, ensure_ascii=False))