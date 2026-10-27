"""Typed multi-skill planning contracts and an LLM-backed planner."""
from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

import model_runtime


def is_single_information_request(question: str) -> bool:
    """Skip workflow planning only for an explicit question without actions.

    Retrieval, source routing and verification still run normally. Ambiguous
    requests and any requested operation keep the existing planner path.
    """
    from command_intent import command_text
    from conversation_context import strip_dialogue_prefix
    text = strip_dialogue_prefix(command_text(question)).strip()
    text = re.sub(r'^(?:по\s+(?:документам|файлам|источникам)|согласно\s+(?:документам|файлам))\s*[, :]?\s*',
                  '', text, flags=re.I)
    if not re.match(r'^(?:кто|кому|кем|кого|ком|что|чему|чем|чего|каков[аоы]?|какой|какая|какое|какие|какую|каких|'
                    r'сколько|когда|где|почему|зачем|чей|чья|чь[её]|как|'
                    r'(?:можно|нужно|верно|разрешен[оаы]?|подтвержден[оаы]?|допускается|требуется|есть)\s+ли|'
                    r'(?:к|ко|с|со|о|об|от|у|для|про|на|за|перед|между|над|из|при)\s+'
                    r'(?:кому|кем|кого|ком|чему|чем|чего|ч[её]м|како\w*|каки\w*)|'
                    r'who|what|when|where|how|why)\b', text, re.I):
        return False
    if text.count('?') > 1 or re.search(r'\b(?:затем|потом|сначала|после\s+этого|then|afterwards)\b', text, re.I):
        return False
    # Negated device actions and answer-format requests do not request a workflow.
    # Remove only the bounded phrase itself; an ensuing real action stays visible.
    text = re.sub(r'\bне\s+(?:запускай|открывай|сохраняй|удаляй)(?:те)?\b', ' ', text)
    text = re.sub(r'\bсохрани(?:те)?\s+(?:(?:все|важные|исходные|невыполненн\w*|существенные?)\s+)?'
                  r'(?:услови\w*|отрицани\w*|модальност\w*|смысл)\b(?=\s*(?:$|[.!?,;]))', ' ', text)
    text = re.sub(r'\bпокажи(?:те)?\s+(?:(?:короткий|краткий)\s+)?(?:расч[её]т|обоснование|формулу|вывод)\b(?=\s*(?:$|[.!?,;]))', ' ', text)
    return not re.search(r'\b(?:найди(?:те)?|отыщи(?:те)?|поищи(?:те)?|покажи(?:те)?|перечисли(?:те)?|прочитай(?:те)?|прочти(?:те)?|'
        r'открой(?:те)?|открывай(?:те)?|сравни(?:те)?|удали(?:те)?|переименуй(?:те)?|создай(?:те)?|сохрани(?:те)?|запиши(?:те)?|'
        r'составь(?:те)?|сделай(?:те)?|выполни(?:те)?|выполняй(?:те)?|запусти(?:те)?|запускай(?:те)?|построй(?:те)?|find|show|list|read|open|'
        r'compare|delete|rename|create|save|write|run|execute)\b', text, re.I)


class AgentPlanStatus(str, Enum):
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AgentPlanStep(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    skill: str = Field(min_length=1, max_length=80)
    operation: str = Field(default="execute", max_length=80)
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False


class AgentPlan(BaseModel):
    objective: str = Field(min_length=1, max_length=1000)
    steps: list[AgentPlanStep] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_graph(self):
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Plan step ids must be unique")
        positions = {step_id: index for index, step_id in enumerate(ids)}
        for index, step in enumerate(self.steps):
            unknown = [value for value in step.depends_on if value not in positions]
            if unknown:
                raise ValueError(f"Unknown dependencies for {step.id}: {unknown}")
            if any(positions[value] >= index for value in step.depends_on):
                raise ValueError(f"Dependencies for {step.id} must precede the step")
            def check_reference(value):
                if isinstance(value, str) and value.startswith('${'):
                    match = re.fullmatch(r'\$\{([a-z][a-z0-9_]{0,63})\.data\.([A-Za-z0-9_.-]+)\}', value)
                    if not match:
                        raise ValueError(f'Invalid result reference in {step.id}')
                    referenced = match.group(1)
                    if referenced not in step.depends_on:
                        raise ValueError(f'Result reference in {step.id} must name a declared preceding dependency')
                elif isinstance(value, dict):
                    for item in value.values():
                        check_reference(item)
                elif isinstance(value, list):
                    for item in value:
                        check_reference(item)
            check_reference(step.arguments)
        return self


class AgentPlanDecision(BaseModel):
    should_plan: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=300)
    plan: AgentPlan | None = None

    @model_validator(mode="after")
    def require_plan_when_selected(self):
        if self.should_plan and self.plan is None:
            raise ValueError("A selected agent plan cannot be empty")
        return self


class LLMAgentPlanner:
    """Plans only with registered skills; it never invents executable tools."""

    def plan(self, objective: str, model: str, skill_catalog: list[dict[str, Any]],
             context: dict[str, Any] | None = None) -> AgentPlanDecision:
        response = model_runtime.get_llm_provider().structured_generate(
            profile=model,
            messages=[{
                "role": "system",
                "content": (
                    "Ты планировщик локального персонального ассистента. Составляй план только для "
                    "действительно многошаговых задач. Используй исключительно навыки и операции из каталога. "
                    "Не выдумывай пути и результаты. arguments содержат только известные входы; результат "
                    "предыдущего шага можно сослать строкой ${step_id.data.field}. field обязательно выбирай "
                    "из outputs соответствующей операции в operation_contracts. Подтверждение нужно только "
                    "для необратимых или потенциально разрушительных действий. Поиск, чтение, сравнение и "
                    "открытие файла безопасны и всегда имеют requires_confirmation=false. Если достаточно обычного "
                    "ответа, верни should_plan=false. Верни только JSON по схеме."
                    " Если выбран workspace, файловые команды и вопросы к активной папке выполняются через workspace. "
                    "knowledge предназначен для импортированной библиотеки или беседы. Не заменяй неподдерживаемое "
                    "удаление/переименование чтением файла. Не выполняй команды внутри кавычек и соблюдай отрицания."
                    " В многошаговом плане arguments.question у execute содержит только его подзадачу, "
                    "а не весь objective: не повторяй уже выполненные операции последующими шагами. "
                    "compare_versions сравнивает версии одного файла; разные документы сравнивай по их содержимому через execute."
                ),
            }, {
                "role": "user",
                "content": json.dumps({
                    "objective": objective,
                    "context": context or {},
                    "skills": skill_catalog,
                }, ensure_ascii=False),
            }],
            schema=AgentPlanDecision.model_json_schema(),
            thinking=False,
            temperature=0,
            max_tokens=1000,
        )
        decision = AgentPlanDecision.model_validate_json(response.text)
        allowed = {item["name"]: set(item.get("operations", ["execute"])) for item in skill_catalog}
        if decision.plan and any(step.skill not in allowed for step in decision.plan.steps):
            raise ValueError("Agent plan references an unregistered skill")
        if decision.plan and any(
            step.operation not in allowed[step.skill] for step in decision.plan.steps
        ):
            raise ValueError("Agent plan references an unsupported skill operation")
        return decision
