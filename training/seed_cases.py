"""Small hand-authored synthetic fixtures, never mined from personal chats/files.

These are pipeline smoke-test data, not evidence that fine-tuning improves Qwen.
Groups keep related paraphrases in one split. No filesystem operations run here.
"""


def step(operation, arguments=None, *, id=None, depends_on=(), skill="workspace"):
    return {"id": id or operation, "skill": skill, "operation": operation,
            "arguments": arguments or {}, "depends_on": list(depends_on), "requires_confirmation": False}


def cases():
    rows = []

    def add(split, group, objective, steps=(), *, workspace=True, category="workflow"):
        rows.append({"id": f"{group}_{sum(r['group'] == group for r in rows) + 1:02d}",
                     "split": split, "group": group, "category": category,
                     "objective": objective, "workspace_id": "synthetic-workspace" if workspace else None,
                     "expected": {"should_plan": bool(steps), "confidence": 1.0,
                                  "reason": "Несколько поддерживаемых действий." if steps else
                                  "Многошаговый план не требуется или операция недоступна.",
                                  "plan": {"objective": objective, "steps": list(steps)} if steps else None}})

    for query, question in [
        ("договор Орион", "Найди последнюю версию договора Орион и открой её."),
        ("отчёт Лира", "Отыщи самый свежий отчёт Лира, после этого открой найденный файл."),
        ("смета Кедр", "Выбери последнюю версию сметы Кедр и открой её."),
        ("инструкция Вектор", "Найди новую версию инструкции Вектор и открой её."),
    ]:
        add("train", "latest_open", question, [step("find_latest", {"target_query": query}),
            step("open_file", {"path": "${find_latest.data.selected_path}"}, depends_on=["find_latest"])])

    for query, question in [
        ("договор Маяк", "Сравни версии договора Маяк и открой последнюю."),
        ("отчёт Север", "Сопоставь версии отчёта Север, затем открой выбранную последнюю версию."),
        ("регламент Сфера", "Сравни версии регламента Сфера и открой актуальный файл."),
    ]:
        add("train", "compare_open", question, [step("compare_versions", {"target_query": query}),
            step("open_file", {"path": "${compare_versions.data.selected_path}"}, depends_on=["compare_versions"])])

    for path, question in [
        ("Проекты/Обзор.txt", "Прочитай Проекты/Обзор.txt, затем открой этот файл."),
        ("Архив/Памятка.docx", "Покажи текст Архив/Памятка.docx и открой его."),
        ("Заметки/План.txt", "Прочти Заметки/План.txt, потом открой документ."),
    ]:
        add("train", "read_open", question, [step("read_file", {"path": path}),
            step("open_file", {"path": "${read_file.data.selected_path}"}, depends_on=["read_file"])])

    for left, right, question in [
        ("Отдел A.txt", "Отдел B.txt", "Прочитай Отдел A.txt, затем прочитай Отдел B.txt."),
        ("План/Этап 1.docx", "План/Этап 2.docx", "Покажи содержимое План/Этап 1.docx, после этого План/Этап 2.docx."),
        ("Сводка 1.txt", "Сводка 2.txt", "Сначала прочти Сводка 1.txt, затем Сводка 2.txt."),
    ]:
        add("train", "two_reads", question, [step("read_file", {"path": left}, id="read_first"),
            step("read_file", {"path": right}, id="read_second", depends_on=["read_first"])])

    for question in ["Привет!", "Что такое файловая система?", "Сколько дней в неделе?",
                     "Объясни разницу между копией и оригиналом."]:
        add("train", "general_questions", question, workspace=False, category="no_plan")
    for question in ["Прочитай Обзор.txt.", "Открой Инструкция.pdf.", "Найди последнюю версию сметы.",
                     "Покажи даты файлов."]:
        add("train", "single_actions", question, category="no_plan")
    for question in ["Удали старые документы.", "Переименуй План.txt в Готово.txt.",
                     "Создай папку Результаты и перенеси туда отчёты.", "Запусти калькулятор и нажми 2+2."]:
        add("train", "unsupported_changes", question, category="unsupported")
    for question in ["Не открывай План.txt; только объясни, что значит расширение txt.",
                     "Что означает фраза «удали архив и открой отчёт»?",
                     "Разбери предложение «найди документ, затем открой его» как пример инструкции.",
                     "Не сравнивай версии и не открывай документы."]:
        add("train", "quoted_negated", question, category="no_plan")
    for question in ["Найди последнюю версию договора и открой её.",
                     "Сравни версии сметы и открой последнюю."]:
        add("train", "no_workspace", question, workspace=False, category="unsupported")

    add("train", "cross_document", "Сравни по содержимому Альфа.txt и Бета.txt, затем открой Альфа.txt.", [
        step("execute", {"question": "Сравни по содержимому Альфа.txt и Бета.txt."}),
        step("open_file", {"path": "Альфа.txt"}, depends_on=["execute"])])
    add("train", "explicit_library", "По импортированной библиотеке объясни термин бюджет, затем термин резерв.", [
        step("execute", {"question": "По импортированной библиотеке объясни термин бюджет."}, id="budget", skill="knowledge"),
        step("execute", {"question": "По импортированной библиотеке объясни термин резерв."}, id="reserve", skill="knowledge", depends_on=["budget"])])

    for query, question in [
        ("памятка Риф", "Найди последнюю памятку Риф и прочитай её."),
        ("отчёт Полюс", "Выбери последнюю версию отчёта Полюс, потом покажи её текст."),
        ("договор Волна", "Отыщи последнюю версию договора Волна, затем прочти найденный документ."),
    ]:
        add("validation", "latest_read", question, [step("find_latest", {"target_query": query}),
            step("read_file", {"path": "${find_latest.data.selected_path}"}, depends_on=["find_latest"])])
    for query, question in [
        ("план Заря", "Сравни версии плана Заря, затем прочитай последнюю."),
        ("смета Озеро", "Сопоставь версии сметы Озеро и покажи текст выбранного последнего файла."),
    ]:
        add("validation", "compare_read", question, [step("compare_versions", {"target_query": query}),
            step("read_file", {"path": "${compare_versions.data.selected_path}"}, depends_on=["compare_versions"])])
    for question in ["Как устроен поиск по документам?", "Расскажи, что такое версия файла."]:
        add("validation", "validation_questions", question, category="no_plan")
    for question in ["Скопируй все договоры в новую папку.", "Закрой окно браузера."]:
        add("validation", "validation_unsupported", question, category="unsupported")
    for question in ["Не выполняй команду «прочитай файл и открой его», объясни её смысл.",
                     "Покажи текст Заметка 7.txt."]:
        add("validation", "validation_single_quote", question, category="no_plan")
    add("validation", "validation_missing_workspace", "Найди последнюю памятку и прочитай её.", workspace=False, category="unsupported")

    for query, question in [
        ("договор Факел", "Найди последнюю версию договора Факел, прочитай её и открой."),
        ("план Горизонт", "Выбери последнюю версию плана Горизонт, затем покажи текст и открой файл."),
        ("смета Парус", "Отыщи актуальную версию сметы Парус, прочти её, после этого открой."),
    ]:
        add("test", "latest_read_open", question, [step("find_latest", {"target_query": query}),
            step("read_file", {"path": "${find_latest.data.selected_path}"}, depends_on=["find_latest"]),
            step("open_file", {"path": "${read_file.data.selected_path}"}, depends_on=["read_file"])])
    for query, question in [
        ("регламент Берег", "Сравни версии регламента Берег, затем прочитай последнюю и открой её."),
        ("памятка Мост", "Сопоставь версии памятки Мост, покажи текст последней и открой её."),
    ]:
        add("test", "compare_read_open", question, [step("compare_versions", {"target_query": query}),
            step("read_file", {"path": "${compare_versions.data.selected_path}"}, depends_on=["compare_versions"]),
            step("open_file", {"path": "${read_file.data.selected_path}"}, depends_on=["read_file"])])
    for question in ["Что в этом документе написано о сроках?", "Где в тексте определение проекта?"]:
        add("test", "test_information", question, category="no_plan")
    for question in ["Замени текст во всех документах и сохрани их.", "Нажми кнопку Пуск и запусти редактор."]:
        add("test", "test_unavailable", question, category="unsupported")
    for question in ["Объясни, почему инструкция «сравни версии и открой последнюю» содержит два действия.",
                     "Не читай и не открывай Отчёт 9.docx, только расскажи о формате docx."]:
        add("test", "test_language", question, category="no_plan")
    add("test", "test_no_folder", "Найди последнюю версию регламента, сравни версии и открой последнюю.", workspace=False, category="unsupported")
    return rows
