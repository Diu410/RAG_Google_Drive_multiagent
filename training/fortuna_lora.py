"""Portable dataset validation and conservative planner scoring; no PC actions."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

REFERENCE = re.compile(r"^\$\{([a-z][a-z0-9_]{0,63})\.data\.([A-Za-z0-9_.-]+)\}$")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path).name}:{number}: invalid JSON") from exc
    return rows


def render_examples(rows: list[dict], tokenizer, max_length: int, max_new_tokens: int) -> list[dict]:
    """Render the native template; reject truncation and prompt/completion drift."""
    rendered = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        full = tokenizer.apply_chat_template(row["messages"] + [{
            "role": "assistant", "content": json.dumps(row["expected"], ensure_ascii=False),
        }], tokenize=False, add_generation_prompt=False, enable_thinking=False)
        if not full.startswith(prompt) or len(full) <= len(prompt):
            raise ValueError(f"{row['id']}: chat template changed the assistant prefix")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(full, add_special_tokens=False)
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(f"{row['id']}: token boundary drift at the completion")
        if len(full_ids) > max_length or len(prompt_ids) + max_new_tokens > max_length:
            raise ValueError(f"{row['id']}: context too small; nothing was truncated")
        rendered.append({"prompt": prompt, "completion": full[len(prompt):]})
    return rendered


def messages_for(objective: str, workspace_id: str | None, contract: dict) -> list[dict]:
    return [
        {"role": "system", "content": contract["system"]},
        {"role": "user", "content": json.dumps({
            "objective": objective,
            "context": {"workspace_id": workspace_id},
            "skills": [s for s in contract["skills"] if s["name"] != "workspace" or workspace_id],
        }, ensure_ascii=False)},
    ]


def validate_decision(decision: dict, messages: list[dict], contract: dict) -> None:
    # JSON Schema alone cannot express dependency order or allowed reference outputs.
    from jsonschema import Draft202012Validator
    Draft202012Validator(contract["schema"]).validate(decision)
    if set(decision) != {"should_plan", "confidence", "reason", "plan"}:
        raise ValueError("Unexpected or missing decision fields")
    if not decision["should_plan"]:
        if decision["plan"] is not None:
            raise ValueError("An unselected plan must be null")
        return
    plan = decision["plan"]
    if not plan or set(plan) != {"objective", "steps"}:
        raise ValueError("Selected plan must contain objective and steps")
    catalog = {s["name"]: s for s in json.loads(messages[1]["content"])["skills"]}
    seen: dict[str, dict] = {}
    for step in plan["steps"]:
        if set(step) != {"id", "skill", "operation", "arguments", "depends_on", "requires_confirmation"}:
            raise ValueError("Unexpected or missing step fields")
        if step["id"] in seen:
            raise ValueError("Duplicate step id")
        if len(set(step["depends_on"])) != len(step["depends_on"]):
            raise ValueError("Duplicate dependency")
        if any(dep not in seen for dep in step["depends_on"]):
            raise ValueError("Dependencies must precede their consumer")
        skill = catalog.get(step["skill"])
        if not skill or step["operation"] not in skill["operations"]:
            raise ValueError("Unknown skill or operation")
        operation = skill.get("operation_contracts", {}).get(step["operation"], {})
        allowed_inputs = operation.get("inputs", ["question"])
        if set(step["arguments"]) - set(allowed_inputs):
            raise ValueError("Unexpected operation arguments")
        if set(allowed_inputs) - set(step["arguments"]):
            raise ValueError("Missing operation arguments")

        def check(value: Any) -> None:
            if isinstance(value, str) and "${" in value:
                match = REFERENCE.fullmatch(value)
                if not match or match[1] not in step["depends_on"]:
                    raise ValueError("Invalid or undeclared result reference")
                source = seen[match[1]]
                outputs = catalog[source["skill"]].get("operation_contracts", {}).get(
                    source["operation"], {}).get("outputs", [])
                if match[2] not in outputs:
                    raise ValueError("Reference names an undeclared output")
            elif isinstance(value, dict):
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
        check(step["arguments"])
        seen[step["id"]] = step


def validate_splits(splits: dict[str, list[dict]], contract: dict) -> dict[str, int]:
    ids, prompts, groups = set(), set(), {}
    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"Empty split: {split}")
        for row in rows:
            if row["id"] in ids:
                raise ValueError("Duplicate example id")
            ids.add(row["id"])
            if groups.setdefault(row["group"], split) != split:
                raise ValueError("Scenario group crosses split boundaries")
            if row["provenance"] != "synthetic_seed":
                raise ValueError("This starter pack accepts synthetic fixtures only")
            user = json.loads(row["messages"][1]["content"])
            expected_messages = messages_for(user["objective"], user["context"]["workspace_id"], contract)
            if row["messages"] != expected_messages:
                raise ValueError("Messages differ from the exported runtime prompt/catalog")
            # Catch identical requests even if their synthetic workspace UUID changes.
            prompt_key = user["objective"].strip().casefold()
            if prompt_key in prompts:
                raise ValueError("Duplicate objective across the dataset")
            prompts.add(prompt_key)
            validate_decision(row["expected"], row["messages"], contract)
    return {split: len(rows) for split, rows in splits.items()}


def _signature(decision: dict) -> tuple:
    if not decision["should_plan"]:
        return (False,)
    positions = {step["id"]: i for i, step in enumerate(decision["plan"]["steps"])}

    def normalize(value):
        if isinstance(value, str) and (match := REFERENCE.fullmatch(value)):
            return f"${{{positions[match[1]]}.data.{match[2]}}}"
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return value
    return (True, tuple((
        s["skill"], s["operation"], json.dumps(normalize(s["arguments"]), ensure_ascii=False, sort_keys=True),
        tuple(sorted(positions[d] for d in s["depends_on"])), s["requires_confirmation"],
    ) for s in decision["plan"]["steps"]))


def score_prediction(text: str, row: dict, contract: dict) -> dict:
    score = {"json_valid": False, "contract_valid": False, "decision_match": False, "plan_match": False}
    try:
        def reject_constant(value):
            raise ValueError(f"Non-JSON numeric constant: {value}")
        decision = json.loads(text, parse_constant=reject_constant)
        score["json_valid"] = True
        validate_decision(decision, row["messages"], contract)
        score["contract_valid"] = True
        score["decision_match"] = decision["should_plan"] == row["expected"]["should_plan"]
        score["plan_match"] = _signature(decision) == _signature(row["expected"])
    except (ValueError, TypeError, KeyError, IndexError):
        pass
    except Exception as exc:
        # jsonschema's ValidationError is not a ValueError. Do not hide runtime faults.
        from jsonschema import ValidationError
        if not isinstance(exc, ValidationError):
            raise
    return score


def summarize(results: list[dict]) -> dict:
    keys = ("json_valid", "contract_valid", "decision_match", "plan_match")
    summary = {"count": len(results), **{
        key: sum(bool(r[key]) for r in results) / len(results) if results else 0.0 for key in keys
    }}
    categories = sorted({r["category"] for r in results})
    summary["by_category"] = {c: {
        "count": sum(r["category"] == c for r in results),
        "plan_match": sum(r["plan_match"] for r in results if r["category"] == c)
        / sum(r["category"] == c for r in results),
    } for c in categories}
    return summary


def verify_bundle(root: str | Path) -> tuple[dict, dict]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    required = {"contract.json", "fortuna_lora.py", "README.md", "train.jsonl", "validation.jsonl", "test.jsonl"}
    if set(manifest["sha256"]) != required:
        raise ValueError("Manifest must cover every bundle payload")
    for name, expected_hash in manifest["sha256"].items():
        if Path(name).name != name:
            raise ValueError("Bundle filenames must be flat")
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    splits = {s: read_jsonl(root / f"{s}.jsonl") for s in ("train", "validation", "test")}
    counts = validate_splits(splits, contract)
    if counts != manifest["counts"]:
        raise ValueError("Manifest counts differ from the data")
    return contract, splits
