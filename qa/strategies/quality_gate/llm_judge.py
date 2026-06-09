# -*- coding: utf-8 -*-
import json
import logging
import re
from typing import Dict, Any, List
from services.llm_service import ILLMService
import core.purification_helper as purifier_module

logger = logging.getLogger("MedicalQA.QualityGate")

from abc import ABC, abstractmethod

class IEvaluationStrategy(ABC):
    @abstractmethod
    async def evaluate(self, q: str, planner: str, raw_think: str, purified_think: str, line_num: int = None, refs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        pass

class LLMJudgeStrategy(IEvaluationStrategy):
    def __init__(self, llm_service: ILLMService):
        self.llm_service = llm_service

    async def evaluate(self, q: str, planner: str, raw_think: str, purified_think: str, line_num: int = None, refs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        基于大语言模型对净化后的思维链进行严格的三维质量门控评估。

        该函数构建包含问题、视角、原始事实依据及思维链对比的提示词，调用LLM进行评分，
        并对返回结果进行结构化解析、字段完整性校验及异常容错处理。

        Args:
            q (str): 用户提出的医疗相关问题。
            planner (str): 当前使用的规划器或切面视角标识。
            raw_think (str): 包含噪声的原始思维链内容。
            purified_think (str): 经过净化重写后的思维链内容。
            line_num (int, optional): 当前处理的行号，用于日志追踪。默认为 None。
            refs (List[Dict[str, Any]], optional): 参考的循证医学事实列表，用于构建Ground Truth。默认为 None。

        Returns:
            Dict[str, Any]: 包含三维评分（语义纯净度、医疗严谨性、逻辑深度）、事实错误列表、
                            冲突检测结果及改进建议的结构化字典。若评估失败，则返回默认的低分阻断结果。
        """
        # 格式化原始数据源（剔除工程标签，保留纯净的循证事实供 Judge 核验）
        cleaned_facts_text = ""
        if refs:
            cleaned_facts_text = "\n### 原始循证医学事实依据 (Ground Truth Cleaned Facts):\n"
            for idx, r in enumerate(refs, start=1):
                if isinstance(r, dict):
                    ctx = r.get('context', 'N/A')
                    # 清理可能携带的 RAG 抱怨和工程词头
                    clean_ctx = ctx.replace("【互联网权威医疗站快讯】:", "").replace("【互联网权威医疗数据通报】:", "").strip()
                    cleaned_facts_text += f"- fact_{idx:03d}: {clean_ctx}\n"

        prompt = f"""问题: {q}
切面视角: {planner}

{cleaned_facts_text}

原始思维链 (包含噪声):
\"\"\"
{raw_think}
\"\"\"

净化重写后的思维链:
\"\"\"
{purified_think}
\"\"\"

请严格按照质检准则对净化后的思维链进行三维评分，并直接输出规范 of JSON 数据。"""
        try:
            stage_prefix = f"[{line_num}行] " if line_num else ""
            response = await self.llm_service.call_llm(prompt, system_prompt=purifier_module.JUDGE_SYSTEM_PROMPT, model_pool="judge", stage=f"{stage_prefix}思维链三维质检 - {planner}")
            json_str = purifier_module.extract_json_block(response)
            scores = json.loads(json_str)
            
            if not isinstance(scores, dict):
                raise ValueError("Parsed output is not a JSON object")
                
            required_keys = [
                "semantic_purity_score", "medical_rigor_score", "logical_depth_score", 
                "factual_errors", "reason", "improvement_suggestions",
                "conflict_detected", "conflict_description", "conflict_details"
            ]
            # 确保返回结果包含所有必需字段，缺失字段根据类型设置默认值
            for key in required_keys:
                if key not in scores:
                    if key == "factual_errors":
                        scores[key] = []
                    elif key == "reason":
                        scores[key] = "No explanation provided"
                    elif key == "improvement_suggestions":
                        scores[key] = "No suggestions provided"
                    elif key == "conflict_detected":
                        scores[key] = False
                    elif key == "conflict_description":
                        scores[key] = ""
                    elif key == "conflict_details":
                        scores[key] = None
                    else:
                        scores[key] = 90
                        
            # 确保 factual_errors 字段为字符串列表格式
            if not isinstance(scores.get("factual_errors"), list):
                if isinstance(scores.get("factual_errors"), str):
                    scores["factual_errors"] = [scores["factual_errors"]]
                else:
                    scores["factual_errors"] = []
                    
            return scores
        except Exception as e:
            logger.warning(f"Judge LLM evaluation failed: {e}. Returning blocking low scores to force rollback.")
            return {
                "semantic_purity_score": 0,
                "medical_rigor_score": 0,
                "logical_depth_score": 0,
                "reason": f"Evaluation error: {e}",
                "is_passed": False,
                "conflict_detected": False,
                "conflict_description": "",
                "conflict_details": None,
            }
