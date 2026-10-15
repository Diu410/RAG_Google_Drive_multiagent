"""Build a synthetic-only Colab bundle from the current planner contracts.

Run with the project's backend venv. Does not start models or read chat/DB/files.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from fortuna_lora import messages_for, validate_splits, verify_bundle
from seed_cases import cases


def class_literals(path: Path, name: str) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    result = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def export_contract() -> dict:
    # Runtime imports are lazy: preparing this bundle never requests a provider.
    sys.path.insert(0, str(ROOT / "api"))
    from agent_planner import AgentPlanDecision
    from pydantic_models import QueryInput

    tree = ast.parse((ROOT / "api/agent_planner.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LLMAgentPlanner")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "plan")
    call = next(n for n in ast.walk(method) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "structured_generate")
    messages = next(k.value for k in call.keywords if k.arg == "messages")
    system = next(v for k, v in zip(messages.elts[0].keys, messages.elts[0].values)
                  if ast.literal_eval(k) == "content")
    skills = []
    for filename, name in [("knowledge_skill.py", "KnowledgeSkill"), ("workspace.py", "WorkspaceSkill")]:
        literals = class_literals(ROOT / "api" / filename, name)
        skills.append({
            "name": literals["name"], "description": literals["description"],
            "input_schema": QueryInput.model_json_schema() if name == "KnowledgeSkill" else literals["input_schema"],
            "permissions": list(literals["permissions"]),
            "operations": list(literals.get("operations", ("execute",))),
            "operation_contracts": literals.get("operation_contracts", {}),
        })
    return {
        "version": 1, "purpose": "planner_only_synthetic_pilot",
        "system": ast.literal_eval(system), "skills": skills,
        "schema": AgentPlanDecision.model_json_schema(),
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "runtime_max_new_tokens": 1000,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "api/agent_planner.py", "api/workspace.py", "api/knowledge_skill.py", "api/pydantic_models.py")},
    }


def build(output: Path) -> dict:
    contract = export_contract()
    splits = {name: [] for name in ("train", "validation", "test")}
    from agent_planner import AgentPlanDecision
    for seed in cases():
        AgentPlanDecision.model_validate(seed["expected"])
        splits[seed["split"]].append({
            "id": seed["id"], "group": seed["group"], "category": seed["category"],
            "provenance": "synthetic_seed",
            "messages": messages_for(seed["objective"], seed["workspace_id"], contract),
            "expected": seed["expected"],
        })
    counts = validate_splits(splits, contract)
    files = {
        "contract.json": json.dumps(contract, ensure_ascii=False, indent=2).encode("utf-8"),
        "fortuna_lora.py": (ROOT / "training/fortuna_lora.py").read_bytes(),
        "README.md": (ROOT / "training/README.md").read_bytes(),
    }
    for name, rows in splits.items():
        files[f"{name}.jsonl"] = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
    manifest = {
        "version": 1, "counts": counts, "synthetic_only": True,
        "scope": "Planner JSON selection. No actions are executed; this is not a PC task benchmark.",
        "sha256": {name: hashlib.sha256(body).hexdigest() for name, body in files.items()},
    }
    files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    output.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (output / name).write_bytes(body)
    verify_bundle(output)
    archive = output / "fortuna-colab.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, body in files.items():
            zipped.writestr(name, body)
    print(json.dumps({"bundle": str(archive), "counts": counts, "synthetic_only": True}, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".training-output")
    build(parser.parse_args().output.resolve())
