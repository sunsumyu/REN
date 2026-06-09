import unittest

from core.evidence_contract import (
    build_evidence_contract,
    detect_forbidden_expansion,
    render_evidence_contract_prompt,
)


class EvidenceContractTests(unittest.TestCase):
    def test_sparse_liver_refs_block_pk_hallucinations(self):
        refs = [
            {
                "source": "refs:咪唑斯汀说明书",
                "context": "主要不良反应包括肝转氨酶升高和 Q-T 间期延长。",
                "metadata": {"category": "不良反应"},
            }
        ]
        routed = {"CORE": [], "BOUNDARY": refs, "BLOCKED": [], "UNUSED": []}

        contract = build_evidence_contract("轻中度肝功能不全患者使用咪唑斯汀时应如何评估风险？", refs, routed)

        self.assertEqual(contract["evidence_status"], "insufficient")
        self.assertEqual(contract["allowed_fact_count"], 1)
        self.assertTrue(any("CYP/AUC" in item for item in contract["forbidden_expansions"]))

    def test_no_refs_are_insufficient(self):
        contract = build_evidence_contract("没有证据的问题", [])

        self.assertEqual(contract["evidence_status"], "insufficient")
        self.assertEqual(contract["allowed_fact_count"], 0)

    def test_core_and_boundary_refs_are_sufficient(self):
        refs = [
            {
                "source": "refs:药品说明书",
                "context": "该药用于过敏性结膜炎。",
                "metadata": {"category": "适应症"},
            },
            {
                "source": "refs:药品说明书",
                "context": "哺乳期妇女使用时应权衡风险。",
                "metadata": {"category": "禁忌"},
            },
        ]
        routed = {"CORE": [refs[0]], "BOUNDARY": [refs[1]], "BLOCKED": [], "UNUSED": []}

        contract = build_evidence_contract("哺乳期患者能否使用该药？", refs, routed)

        self.assertEqual(contract["evidence_status"], "sufficient")
        self.assertEqual(contract["core_fact_count"], 1)
        self.assertEqual(contract["boundary_fact_count"], 1)

    def test_render_contract_prompt_contains_allowed_and_forbidden_rules(self):
        refs = [
            {
                "source": "refs:说明书",
                "context": "主要不良反应包括肝转氨酶升高。",
                "metadata": {"category": "不良反应"},
            }
        ]
        contract = build_evidence_contract("肝功能异常患者如何评估？", refs)

        prompt = render_evidence_contract_prompt(contract)

        self.assertIn("证据契约硬约束", prompt)
        self.assertIn("允许事实清单", prompt)
        self.assertIn("F001", prompt)
        self.assertIn("禁止外推清单", prompt)
        self.assertIn("CYP/AUC", prompt)

    def test_detects_positive_forbidden_pk_expansion(self):
        refs = [
            {
                "source": "refs:说明书",
                "context": "主要不良反应包括肝转氨酶升高。",
                "metadata": {"category": "不良反应"},
            }
        ]
        contract = build_evidence_contract("肝功能异常患者如何评估？", refs)

        violations = detect_forbidden_expansion("该药经 CYP3A4 代谢，肝损伤会导致 AUC 升高。", contract)

        self.assertTrue(violations)
        self.assertEqual(violations[0]["family"], "pharmacokinetics")

    def test_allows_evidence_boundary_expression(self):
        refs = [
            {
                "source": "refs:说明书",
                "context": "主要不良反应包括肝转氨酶升高。",
                "metadata": {"category": "不良反应"},
            }
        ]
        contract = build_evidence_contract("肝功能异常患者如何评估？", refs)

        violations = detect_forbidden_expansion("现有证据未提供 CYP 或 AUC 信息，因此不能判断药代变化。", contract)

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
