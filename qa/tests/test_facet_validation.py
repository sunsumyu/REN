import unittest

from core.pipeline_workflow import validate_facet_label
from models import FacetPlan
from services.llm_service import LLMService


def _candidate(label: str) -> dict:
    return {
        "label": label,
        "category": "other_medical",
        "answer_scope": "围绕该医学视角组织回答",
        "why_relevant": "该视角与主问题直接相关",
        "risk_level": "low",
    }


class FacetValidationTests(unittest.TestCase):
    def test_facet_plan_rejects_placeholder_label(self):
        with self.assertRaises(ValueError):
            FacetPlan.model_validate({"facets": [_candidate("示例视角2"), _candidate("功效主治")]})

    def test_facet_plan_rejects_schema_prompt_leakage(self):
        with self.assertRaises(ValueError):
            FacetPlan.model_validate({
                "facets": [
                    _candidate("You are a rigorous data processing API"),
                    _candidate("功效主治"),
                ]
            })

    def test_runtime_facet_gate_rejects_bad_labels(self):
        for label in ["示例视角1", "提示：缺少医疗问题", "You are a rigorous data processing API"]:
            with self.subTest(label=label):
                ok, reason = validate_facet_label(label)
                self.assertFalse(ok)
                self.assertTrue(reason)

    def test_runtime_facet_gate_accepts_medical_labels(self):
        for label in ["成分构成", "功效主治", "药代动力学"]:
            with self.subTest(label=label):
                ok, reason = validate_facet_label(label)
                self.assertTrue(ok)
                self.assertEqual(reason, "")

    def test_structured_output_guard_rejects_schema_leakage_in_values(self):
        obj = FacetPlan.model_validate({"facets": [_candidate("成分构成"), _candidate("功效主治")]})
        obj.facets[0].answer_scope = '包含 "$defs": 和 "properties": 的 schema 泄漏文本'

        with self.assertRaises(ValueError):
            LLMService._assert_no_structured_prompt_leak(obj, FacetPlan)


if __name__ == "__main__":
    unittest.main()
