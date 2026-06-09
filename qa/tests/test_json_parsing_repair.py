import unittest

import prompts
from pipeline import parse_json_safely


class JsonParsingRepairTests(unittest.TestCase):
    def test_repairs_missing_final_object_brace(self):
        raw = (
            '{\n'
            '  "think": "构建一个需要多因素权衡的临床决策难题。",\n'
            '  "questions": ["一名患者需要结合禁忌、剂量规则和妊娠计划综合评估治疗方案。"]\n'
        )

        parsed = parse_json_safely(raw, {})

        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["think"], "构建一个需要多因素权衡的临床决策难题。")
        self.assertEqual(len(parsed["questions"]), 1)

    def test_salvages_questions_when_discarded_think_has_unescaped_quotes(self):
        raw = (
            '{\n'
            '  "think": "问题字数自检："肾衰持续血液透析患者，因带状疱疹口服阿昔洛韦，下一步给药策略是什么？"共38字。",\n'
            '  "questions": [\n'
            '    "肾衰持续血液透析患者，因带状疱疹口服阿昔洛韦，现肌肉疼痛伴血肌酐升高，下一步给药策略是什么？"\n'
            '  ]\n'
            '}'
        )

        parsed = parse_json_safely(raw, {})

        self.assertEqual(parsed, {
            "questions": [
                "肾衰持续血液透析患者，因带状疱疹口服阿昔洛韦，现肌肉疼痛伴血肌酐升高，下一步给药策略是什么？"
            ]
        })

    def test_question_creator_prompt_no_longer_requires_think_output(self):
        template = prompts._BOOTSTRAP_QUESTION_CREATOR_TEMPLATE

        self.assertIn('严格输出只包含 "questions" 的 JSON 对象', template)
        self.assertNotIn('"think":', template)


if __name__ == "__main__":
    unittest.main()
