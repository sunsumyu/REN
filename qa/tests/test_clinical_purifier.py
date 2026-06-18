# -*- coding: utf-8 -*-
import unittest
import sys
import os

# 把工作目录加入路径以方便导入 utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.clinical_purifier import ClinicalPathwayPurifier

class TestClinicalPathwayPurifier(unittest.TestCase):
    
    def setUp(self):
        # 准备典型的临床路径模拟文本
        self.mock_document = (
            "**寻常型天疱疮临床路径**\n"
            "\n"
            "**（2019年版）**\n"
            "\n"
            "一、寻常型天疱疮临床路径标准住院流程\n"
            "\n"
            "**（一）适用对象**\n"
            "\n"
            "第一诊断为寻常型天疱疮（ICD-10：L10.0）。\n"
            "\n"
            "**（二）诊断依据**\n"
            "一些诊断依据说明文字。\n"
            "\n"
            "**（四）标准住院日为21～28天**\n"
            "\n"
            "**（五）进入路径标准**\n"
            "进入标准描述文字。\n"
            "\n"
            "二、寻常型天疱疮临床路径表单\n"
            "\n"
            "适用对象：第一诊断为寻常型天疱疮（ICD-10：L10.0）\n"
            "\n"
            "| 时间 | 住院第1天 | 住院第2～6天 |\n"
            "| --- | --- | --- |\n"
            "| 主要诊疗工作 | * 询问病史 * 查体 | * 医生查房 |\n"
        )

    def test_metadata_extraction(self):
        # 测试元数据提取
        metadata = ClinicalPathwayPurifier.extract_metadata(self.mock_document, "寻常型天疱疮临床路径（2019年版）.docx")
        
        self.assertEqual(metadata["disease_name"], "寻常型天疱疮")
        self.assertEqual(metadata["icd_code"], "L10.0")
        self.assertEqual(metadata["standard_days"], "21～28天")

    def test_metadata_extraction_fallback_filename(self):
        # 测试文件名兜底
        text_without_title = "一些正文，没有标题在前面。"
        metadata = ClinicalPathwayPurifier.extract_metadata(text_without_title, "自发性气胸（呼吸内科）临床路径 (2019年版).docx")
        
        self.assertEqual(metadata["disease_name"], "自发性气胸")

    def test_boilerplate_truncation_by_header(self):
        # 测试通过标题截断行政表单
        purified = ClinicalPathwayPurifier.truncate_boilerplate(self.mock_document)
        
        # 验证“二、...表单”及以下的内容均被成功截断
        self.assertNotIn("二、寻常型天疱疮临床路径表单", purified)
        self.assertNotIn("适用对象：第一诊断为寻常型", purified)
        self.assertNotIn("| 时间 | 住院第1天 |", purified)
        # 验证保留了住院流程正文
        self.assertIn("一、寻常型天疱疮临床路径标准住院流程", purified)
        self.assertIn("（一）适用对象", purified)
        self.assertIn("寻常型天疱疮（ICD-10：L10.0）", purified)

    def test_boilerplate_truncation_by_table_fallback(self):
        # 测试当标题丢失时，使用 Markdown 格式特征分隔线兜底截断
        text_without_header = (
            "一、急性呼吸窘迫综合征临床路径标准住院流程\n"
            "\n"
            "**（一）适用对象**\n"
            "第一诊断为 ARDS。\n"
            "\n"
            "| 时间 | 住院第1天 |\n"
            "| --- | --- |\n"
            "| 诊疗 | 完成病历 |\n"
        )
        purified = ClinicalPathwayPurifier.truncate_boilerplate(text_without_header)
        
        self.assertNotIn("| 时间 |", purified)
        self.assertIn("一、急性呼吸窘迫综合征临床路径标准住院流程", purified)
        self.assertIn("第一诊断为 ARDS。", purified)

    def test_markdown_normalization_keeps_newlines(self):
        # 测试排版清洗保留换行
        raw_text = "<p>正文第1段</p>\n\n\n\n\n\n  正文第2段  \n\n\n\n"
        normalized = ClinicalPathwayPurifier.normalize_markdown(raw_text)
        
        # 验证 HTML 被剥离，多余空行被折叠为双换行，且前后空格被清除
        self.assertEqual(normalized, "正文第1段\n\n正文第2段")

    def test_enrich_chunks(self):
        # 测试语义块元数据头部注入
        chunks = ["这是第1个文本切片块内容。", "这是第2个文本切片块内容。"]
        metadata = {
            "disease_name": "寻常型天疱疮",
            "icd_code": "L10.0",
            "standard_days": "21～28天"
        }
        
        enriched = ClinicalPathwayPurifier.enrich_chunks(chunks, metadata)
        
        self.assertEqual(len(enriched), 2)
        for chunk in enriched:
            self.assertTrue(chunk.startswith("【临床路径知识库】"))
            self.assertIn("关联病种: 寻常型天疱疮", chunk)
            self.assertIn("疾病编码 (ICD-10): L10.0", chunk)
            self.assertIn("标准住院日: 21～28天", chunk)
            self.assertTrue(chunk.endswith("内容。"))


if __name__ == "__main__":
    unittest.main()
