# -*- coding: utf-8 -*-
import json
import logging
import re
from typing import Dict, Any
from services.llm_service import ILLMService
import core.purification_helper as purifier_module

logger = logging.getLogger("MedicalQA.QualityGate")

from abc import ABC, abstractmethod

class IEvaluationStrategy(ABC):
    @abstractmethod
    async def evaluate(self, q: str, planner: str, raw_think: str, purified_think: str, line_num: int = None) -> Dict[str, Any]:
        pass

class LLMJudgeStrategy(IEvaluationStrategy):
    def __init__(self, llm_service: ILLMService):
        self.llm_service = llm_service

    async def evaluate(self, q: str, planner: str, raw_think: str, purified_think: str, line_num: int = None) -> Dict[str, Any]:
        """
        Evaluate the purified thought chain under strict three-dimensional quality gates.
        """
        prompt = f"""问题: {q}
切面视角: {planner}
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
                
            required_keys = ["semantic_purity_score", "medical_rigor_score", "logical_depth_score", "factual_errors", "reason"]
            for key in required_keys:
                if key not in scores:
                    if key == "factual_errors":
                        scores[key] = []
                    elif key == "reason":
                        scores[key] = "No explanation provided"
                    else:
                        scores[key] = 90
                        
            # Ensure factual_errors is a list of strings
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
            }
