# -*- coding: utf-8 -*-
"""
企业级临床路径数据净化与预处理工具模块。
提供元数据解析、表单核对项剔除、文本排版规范化以及语义块上下文增强等功能。
"""

import re
import os

class ClinicalPathwayPurifier:
    """
    临床路径文档净化处理器，用于过滤行政噪音并提取医学核心知识。
    """
    
    @staticmethod
    def clean_disease_name(name: str) -> str:
        """
        去除病种名称中的干扰后缀、版本号、科室说明等，返回纯净的疾病名称。
        """
        if not name:
            return "未知"
        # 移除可能存在的后缀名
        name = re.sub(r'\.doc[x]?$', '', name, flags=re.IGNORECASE)
        # 移除“临床路径”字样
        name = re.sub(r'临床路径', '', name)
        # 移除年份版本限制描述（如 2019年版，2019版），支持中英文括号
        name = re.sub(r'[（\(]?\d{4}年版[）\)]?|[（\(]?\d{4}版[）\)]?', '', name)
        # 移除括号及其里面的科室辅助信息（如 （呼吸内科），(内科)，包括空括号 ()）
        name = re.sub(r'（[^）]*）|\([^)]*\)', '', name)
        return name.strip()

    @classmethod
    def extract_metadata(cls, text: str, filename: str = "") -> dict:
        """
        从文本内容与文件名中提取关键元数据。
        包括：病种名称、ICD-10 编码、标准住院日。
        """
        metadata = {
            "disease_name": "未知",
            "icd_code": "未知",
            "standard_days": "未知"
        }
        
        if not text:
            return metadata
            
        # 1. 提取病种标题 (通常在文本前 5 行)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        title = ""
        for line in lines[:5]:
            clean_line = line.replace("*", "").replace("_", "").strip()
            if "临床路径" in clean_line and "表单" not in clean_line:
                title = clean_line
                break
        
        # 提取疾病纯净名称
        if title:
            metadata["disease_name"] = cls.clean_disease_name(title)
        elif filename:
            # 如果从文本中没找到，则使用文件名兜底
            metadata["disease_name"] = cls.clean_disease_name(filename)
            
        # 截断以确保名称不会过长
        metadata["disease_name"] = metadata["disease_name"][:30]

        # 2. 提取 ICD-10 编码
        # 常见格式如: （ICD-10：L10.0） 或 （ICD–10：E03.802） 或 ICD-10：J93.0-J93.1
        icd_pattern = re.compile(r'(?i)ICD[-–]?10\s*[:：]\s*([A-Z0-9\.\-\u2013/]+)')
        icd_match = icd_pattern.search(text)
        if icd_match:
            metadata["icd_code"] = icd_match.group(1).strip()

        # 3. 提取标准住院日
        # 常见格式如: 标准住院日为21～28天 或 标准住院日为10-14天
        days_pattern = re.compile(r'标准住院日(?:为|:|：)?\s*([0-9\uff5e\-\u2013a-zA-Z\u4e00-\u9fa5\s]+天)')
        days_match = days_pattern.search(text)
        if days_match:
            metadata["standard_days"] = days_match.group(1).strip()
            
        return metadata

    @staticmethod
    def truncate_boilerplate(text: str) -> str:
        """
        结构化截断：识别并剔除“二、xxxx临床路径表单”及其之下的所有非文本表格内容。
        采用双轨制防漏设计：
        1. 优先匹配标题拆分标志；
        2. 其次匹配 Markdown 表格排版标志作为兜底。
        """
        if not text:
            return ""
            
        # 轨道一：匹配分卷标题 “二、xxx临床路径表单” 或 “二、临床路径表单” 或 “临床路径表单” 独占行
        split_pattern = re.compile(
            r'(?im)^\s*(?:#{1,6}\s+)?(?:二\s*[、.]\s*.*(?:表单|临床路径表单)|(?:临床路径表单|路径表单)\s*$)'
        )
        match = split_pattern.search(text)
        if match:
            return text[:match.start()].strip()
            
        # 轨道二：如果未匹配到标题，寻找 Markdown 表格的特征分隔线（例如 | --- | --- |）
        # 正常临床路径的第一段（标准住院流程）完全是文本说明，不含此类表格，因此这是极其安全的降级处理
        table_pattern = re.compile(r'(?m)^\s*\|\s*[:-]+\s*\|')
        table_match = table_pattern.search(text)
        if table_match:
            # 去除末尾的换行以防 split('\n') 产生多余空行，导致回溯查找表头时提前 break
            lines = text[:table_match.start()].rstrip().split('\n')
            # 向上追溯，找到该 Markdown 表格第一行（通常是以 | 开头的表头行）
            truncate_idx = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith('|'):
                    truncate_idx = i
                else:
                    break
            return '\n'.join(lines[:truncate_idx]).strip()
            
        return text.strip()

    @staticmethod
    def normalize_markdown(text: str) -> str:
        """
        清洗 HTML 标签，规范化空格与多余空行，但严格保留 Markdown 的行折行排版结构。
        """
        if not text:
            return ""
            
        # 移除 HTML 标签，如 <p>, <br> 等
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 逐行清洗行内多余空格并过滤尾部空格，但保持换行
        lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # 将三行及以上的多余空行坍缩为双换行 (标准 Markdown 段落分界)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    @classmethod
    def purify(cls, text: str, filename: str = "") -> tuple[str, dict]:
        """
        净化流水线统一入口。
        输入原始文本与文件名，返回净化后的文本以及解析到的结构化元数据。
        """
        metadata = cls.extract_metadata(text, filename)
        truncated_text = cls.truncate_boilerplate(text)
        normalized_text = cls.normalize_markdown(truncated_text)
        return normalized_text, metadata

    @staticmethod
    def enrich_chunks(chunks: list[str], metadata: dict) -> list[str]:
        """
        上下文注入：在每个切片块的头部注入关联的结构化元数据，
        提升大模型在 RAG 检索时的语境识别和问答精准度。
        """
        enriched = []
        header = (
            f"【临床路径知识库】\n"
            f"- 关联病种: {metadata.get('disease_name', '未知')}\n"
            f"- 疾病编码 (ICD-10): {metadata.get('icd_code', '未知')}\n"
            f"- 标准住院日: {metadata.get('standard_days', '未知')}\n"
            f"----------------------------------------\n"
        )
        for chunk in chunks:
            if chunk.strip():
                enriched.append(header + chunk.strip())
        return enriched
