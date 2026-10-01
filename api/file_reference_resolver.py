"""LLM-assisted typo correction constrained to real workspace file candidates."""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

import model_runtime


class FileReferenceDecision(BaseModel):
    selected_path: str = ""
    corrected_name: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=240)


class FileReferenceResolver:
    def resolve(self, requested: str, candidates: list[str], model: str) -> FileReferenceDecision:
        response = model_runtime.get_llm_provider().structured_generate(
            profile=model,
            messages=[{
                "role": "system",
                "content": (
                    "Разреши естественное описание файла и выбери подходящий путь только из предоставленного "
                    "списка. Учитывай вложенную папку, неполное имя, исправление опечаток и условия вроде "
                    "«имя начинается на букву Ф». Не выбирай файл лишь по общей теме содержимого. Цифры в имени "
                    "должны совпадать с цифрами кандидата. Если соответствие не однозначно, верни пустой "
                    "selected_path. Верни только JSON по схеме."
                ),
            }, {
                "role": "user",
                "content": json.dumps({"requested": requested, "candidates": candidates}, ensure_ascii=False),
            }],
            schema=FileReferenceDecision.model_json_schema(),
            thinking=False,
            temperature=0,
            max_tokens=180,
        )
        return FileReferenceDecision.model_validate_json(response.text)
