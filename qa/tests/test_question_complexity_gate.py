import unittest

from core.pipeline_workflow import is_fact_retrieval_question


class QuestionComplexityGateTests(unittest.TestCase):
    def test_rejects_recent_single_hop_failure_patterns(self):
        bad_questions = [
            "盐酸曲美他嗪片在成年患者中的推荐用法用量是什么？",
            "安非他酮的主要不良反应有哪些？",
            "在年龄为13~17岁的儿科患者中，阿立哌唑用于治疗精神分裂症的安全性和有效性是通过什么类型的临床试验确定的？",
        ]

        for question in bad_questions:
            with self.subTest(question=question):
                self.assertTrue(is_fact_retrieval_question(question))

    def test_allows_contextual_clinical_reasoning_question(self):
        question = (
            "一名伴中度肾功能损害的老年心绞痛患者正在考虑使用盐酸曲美他嗪片，"
            "应如何结合肾功能、给药频次和3个月疗效评估来权衡继续用药风险？"
        )

        self.assertFalse(is_fact_retrieval_question(question))

    def test_allows_soft_lookup_wording_when_clinical_friction_is_present(self):
        question = (
            "一名合并肾功能下降且正在服用多种心血管药物的老年患者使用该药后，"
            "有哪些用药风险需要结合禁忌边界和疗效监测进行权衡？"
        )

        self.assertFalse(is_fact_retrieval_question(question))


if __name__ == "__main__":
    unittest.main()
