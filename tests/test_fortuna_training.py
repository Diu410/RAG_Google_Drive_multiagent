"""Meaningful offline checks for data leakage, broken plans and lost labels."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from fortuna_lora import render_examples, score_prediction, validate_decision, validate_splits, verify_bundle

spec = importlib.util.spec_from_file_location("prepare_fortuna_colab", ROOT / "scripts/prepare_fortuna_colab.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class SimpleTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        text = "".join(f"<{m['role']}>" + m["content"] + "</end>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))


class FortunaTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.tmp.name)
        builder.build(cls.directory)
        cls.contract, cls.splits = verify_bundle(cls.directory)
        cls.positive = next(r for r in cls.splits["train"] if r["expected"]["should_plan"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exported_prompt_matches_runtime_call(self):
        # Capture the real planner's messages, without a model/server or filesystem tool.
        from unittest.mock import patch
        from types import SimpleNamespace
        from agent_planner import LLMAgentPlanner
        captured = {}
        class Provider:
            def structured_generate(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(text=json.dumps(self_decision))
        self_decision = self.positive["expected"]
        user = json.loads(self.positive["messages"][1]["content"])
        with patch("agent_planner.model_runtime.get_llm_provider", return_value=Provider()):
            LLMAgentPlanner().plan(user["objective"], "FAST", user["skills"], user["context"])
        self.assertEqual(captured["messages"], self.positive["messages"])
        self.assertEqual(captured["schema"], self.contract["schema"])

    def test_bundle_passes_real_runtime_schema(self):
        from agent_planner import AgentPlanDecision
        for rows in self.splits.values():
            for row in rows:
                AgentPlanDecision.model_validate(row["expected"])
                self.assertTrue(score_prediction(json.dumps(row["expected"]), row, self.contract)["plan_match"])

    def test_changed_operations_are_not_mistaken_for_correct_plan(self):
        row = self.positive
        changed = deepcopy(row["expected"])
        changed["plan"]["steps"][-1]["operation"] = "read_file"
        score = score_prediction(json.dumps(changed), row, self.contract)
        self.assertTrue(score["contract_valid"])
        self.assertFalse(score["plan_match"])

    def test_renamed_step_ids_are_equivalent(self):
        row = self.positive
        changed = deepcopy(row["expected"])
        original = changed["plan"]["steps"][0]["id"]
        changed["plan"]["steps"][0]["id"] = "lookup"
        changed["plan"]["steps"][1]["depends_on"] = ["lookup"]
        changed["plan"]["steps"][1]["arguments"]["path"] = "${lookup.data.selected_path}"
        self.assertNotEqual(original, "lookup")
        self.assertTrue(score_prediction(json.dumps(changed), row, self.contract)["plan_match"])

    def test_unsupported_tools_and_unknown_outputs_fail(self):
        for operation, reference in [("delete_file", "${find_latest.data.selected_path}"),
                                     ("open_file", "${find_latest.data.nonexistent}"),
                                     ("open_file", "${future.data.selected_path}")]:
            changed = deepcopy(self.positive["expected"])
            changed["plan"]["steps"][1]["operation"] = operation
            changed["plan"]["steps"][1]["arguments"]["path"] = reference
            self.assertFalse(score_prediction(json.dumps(changed), self.positive, self.contract)["contract_valid"])

    def test_extra_arguments_and_missing_plan_fail(self):
        changed = deepcopy(self.positive["expected"])
        changed["plan"]["steps"][0]["arguments"]["workspace_id"] = "other"
        self.assertFalse(score_prediction(json.dumps(changed), self.positive, self.contract)["contract_valid"])
        changed["plan"] = None
        self.assertFalse(score_prediction(json.dumps(changed), self.positive, self.contract)["contract_valid"])

    def test_cross_split_group_and_prompt_leaks_fail(self):
        for field in ("group", "messages"):
            splits = deepcopy(self.splits)
            splits["test"][0][field] = deepcopy(splits["train"][0][field])
            with self.assertRaises(ValueError):
                validate_splits(splits, self.contract)

    def test_non_json_and_non_finite_numbers_fail(self):
        for text in ("```json\n{}\n```", "{", "[]", '{"confidence": NaN}', "null"):
            score = score_prediction(text, self.positive, self.contract)
            self.assertFalse(score["contract_valid"])

    def test_completion_is_retained_and_context_overflow_rejected(self):
        tokenizer = SimpleTokenizer()
        result = render_examples([self.positive], tokenizer, 100000, 1000)[0]
        self.assertIn('"should_plan": true', result["completion"])
        with self.assertRaisesRegex(ValueError, "context too small"):
            render_examples([self.positive], tokenizer, 100, 1000)

    def test_template_prefix_drift_rejected(self):
        class ChangedTokenizer(SimpleTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                result = super().apply_chat_template(messages, **kwargs)
                return result + "unexpected" if kwargs["add_generation_prompt"] else result
        with self.assertRaisesRegex(ValueError, "assistant prefix"):
            render_examples([self.positive], ChangedTokenizer(), 100000, 1000)

    def test_bundle_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            from shutil import copytree
            target = Path(directory) / "bundle"
            copytree(self.directory, target)
            (target / "train.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_bundle(target)

    def test_manifest_cannot_omit_a_payload_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            from shutil import copytree
            target = Path(directory) / "bundle"
            copytree(self.directory, target)
            manifest_path = target / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["sha256"]["fortuna_lora.py"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "every bundle payload"):
                verify_bundle(target)

    def test_notebook_is_valid_python_and_has_no_saved_outputs(self):
        notebook = json.loads((ROOT / "training/Fortuna_Qwen35_4B_LoRA.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])


if __name__ == "__main__":
    unittest.main()
