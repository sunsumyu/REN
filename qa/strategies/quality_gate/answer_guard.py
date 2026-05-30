# -*- coding: utf-8 -*-
import re
import logging
from typing import Tuple

logger = logging.getLogger("MedicalQA.AnswerGuard")

def check_answer_quality(answer_body: str, reasoning_content: str = "") -> Tuple[bool, str]:
    """
    Runs quality guardrails to filter out refusals, lazy reasoning, and prompt pollutions.
    """
    refusal_pattern = re.compile(r"(抱歉|无法协助|不符合安全规定|作为一个AI|不能回答|作为AI|未获得授权)")
    if refusal_pattern.search(answer_body):
        return False, "safety refusal"
        
    pollution_pattern = re.compile(r"(Step A|推理链|证据清单|法律合规|供应链|财务审计|few-shot|提示词|模型生成)")
    if pollution_pattern.search(answer_body):
        return False, "prompt pollution"
        
    if reasoning_content:
        cleaned_reasoning = reasoning_content.replace("<think>", "").replace("</think>", "").strip()
        if len(cleaned_reasoning) < 120:
            return False, f"lazy reasoning (too short: {len(cleaned_reasoning)} chars)"
        
    return True, ""
