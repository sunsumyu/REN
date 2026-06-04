# -*- coding: utf-8 -*-
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field
from services.llm_service import ILLMService

logger = logging.getLogger("MedicalQA.FacetStrategy")

class FacetGovernanceDecision(BaseModel):
    intent: str = Field(description="主问题的分类意图，如 DOSAGE_LIMIT, COMPONENT, CONTRAINDICATION 等")
    facet_action: Literal["KEEP", "RENAME", "DROP", "REDIRECT_SIMPLE"] = Field(
        description="针对切面的治理动作：KEEP(保持), RENAME(重命名并修复), DROP(废弃/脏切面), REDIRECT_SIMPLE(重定向至极简模式并重命名为规范切面)"
    )
    target_facet: str = Field(description="如果是 RENAME 或 REDIRECT_SIMPLE，则提供目标规范切面名（如：剂量用法、成分构成、禁忌人群、贮藏条件、临床事实核验等）；否则保持空或原名")
    compatibility: Literal["COMPATIBLE", "COMPATIBLE_SIMPLE", "FORCED_SKIP"] = Field(
        description="兼容性评估结果：COMPATIBLE(强兼容), COMPATIBLE_SIMPLE(简单兼容，极简推理), FORCED_SKIP(强套偏题，直接舍弃)"
    )
    reason: str = Field(description="治理决策的医学/逻辑判定理由")


class FacetGovernanceStrategy(ABC):
    """医学切面治理策略基类"""
    @abstractmethod
    async def apply(self, q: str, raw_facet: str, context: Dict[str, Any]) -> Dict[str, Any]:
        pass


class DropDirtyFacetStrategy(FacetGovernanceStrategy):
    """直接删除彻底偏题的脏切面策略"""
    async def apply(self, q: str, raw_facet: str, context: Dict[str, Any]) -> Dict[str, Any]:
        context.setdefault("audit_log", []).append(
            f"Pruned facet '{raw_facet}' due to incompatibility with Q: '{q}'"
        )
        return {"action": "drop", "facet": None}


class RenameAndRepairStrategy(FacetGovernanceStrategy):
    """偏题但可修复切面：重命名为更宽泛的兼容切面"""
    def __init__(self, target_facet: str):
        self.target_facet = target_facet

    async def apply(self, q: str, raw_facet: str, context: Dict[str, Any]) -> Dict[str, Any]:
        context.setdefault("audit_log", []).append(
            f"Repaired facet '{raw_facet}' -> '{self.target_facet}' for Q: '{q}'"
        )
        return {"action": "repair", "facet": self.target_facet}


class RedirectToSimpleStrategy(FacetGovernanceStrategy):
    """简单事实题：强制改名为标准规范切面，并重定向至极简推理模式，防止模型过度演绎"""
    def __init__(self, canonical_facet: str):
        self.canonical_facet = canonical_facet

    async def apply(self, q: str, raw_facet: str, context: Dict[str, Any]) -> Dict[str, Any]:
        context["simplify"] = True
        context["compatibility"] = "COMPATIBLE_SIMPLE"
        context.setdefault("audit_log", []).append(
            f"Redirected simple Q: '{q}' with facet '{raw_facet}' -> '{self.canonical_facet}' (COMPATIBLE_SIMPLE)"
        )
        return {"action": "simplify", "facet": self.canonical_facet, "simplify": True}


def classify_intent_by_rule(q: str) -> str:
    """利用正则及匹配规则对 Q 进行快速意图分流，识别基础意图"""
    q_low = (q or "").lower()
    if any(k in q_low for k in ["剂量", "极量", "最大量", "多大剂量", "服用量", "吃多少", "用量", "单次", "一日"]):
        return "DOSAGE_LIMIT"
    if any(k in q_low for k in ["成分", "组成", "配方", "包含哪些", "辅料", "有什么成分", "主要化学", "配料"]):
        return "COMPONENT"
    if any(k in q_low for k in ["包装", "规格", "每盒", "袋", "装量", "片数", "瓶"]):
        return "PACKAGING"
    if any(k in q_low for k in ["贮藏", "储存", "存放", "保存", "阴凉", "常温"]):
        return "STORAGE"
    if any(k in q_low for k in ["禁忌", "禁用", "慎用", "不能吃", "不能用", "不宜"]):
        return "CONTRAINDICATION"
    return "GENERAL_MEDICAL"


class FacetGovernanceFilter:
    """前置 Q-Facet 兼容性治理器"""
    def __init__(self, llm_service: ILLMService):
        self.llm_service = llm_service

    async def evaluate_compatibility(self, query: str, raw_facet: str) -> FacetGovernanceDecision:
        # 1. 第一步：规则分类
        intent = classify_intent_by_rule(query)
        
        # 2. 第二步：LLM 结构化验证决策
        system_prompt = (
            "你是一个极其严格的医疗问答视角语义匹配与治理网关。\n"
            "你的唯一任务是评估给定的【主问题】与【分析视角/切面】之间的语义适配性，以防范数据生成/清洗过程中的‘视角强套’与‘虚假因果关系脑补’。\n\n"
            "### ⚖️ 评估判定规则：\n"
            "1. 【DROP】(绝对禁止强套)：\n"
            "   - 当主问题是一个简单的事实查找（如具体的药品剂量、包装规格、贮藏条件等），却被强行匹配了完全无关或会迫使虚构病理因果的视角（如强套“病理生理机制”），请决策为 DROP。\n"
            "2. 【REDIRECT_SIMPLE】(简单事实极简推理)：\n"
            "   - 当主问题是相对直接的临床/事实问题，但匹配了有关联的视角时（如剂量题匹配“用药方案”、“禁忌人群”）。应决策为 REDIRECT_SIMPLE，并重命名 target_facet 为对应的标准规范切面，如 '剂量用法'、'成分构成' 等。\n"
            "3. 【RENAME】(偏题修复)：\n"
            "   - 视角与问题弱相关，但可以通过重命名为更贴切的医学视角来修复它。\n"
            "4. 【KEEP】(强兼容保持)：\n"
            "   - 问题与视角高度契合，需要呈现深度复杂的药理/临床推理。"
        )
        
        user_prompt = (
            f"主问题: '{query}'\n"
            f"规则识别意图: '{intent}'\n"
            f"待评估分析视角: '{raw_facet}'"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            decision: FacetGovernanceDecision = await self.llm_service.call_llm_structured(
                messages,
                FacetGovernanceDecision,
                model_pool="lightweight",
                stage=f"Q-Facet兼容治理判定 - {raw_facet}"
            )
            return decision
        except Exception as e:
            logger.error(f"Structured facet compatibility check failed: {e}. Defaulting to KEEP.")
            # Default fallback strategy is KEEP
            action = "KEEP"
            compatibility = "COMPATIBLE"
            # If the rule detected a simple factual intent, fallback to REDIRECT_SIMPLE to be safe
            if intent in ["DOSAGE_LIMIT", "COMPONENT", "PACKAGING", "STORAGE"]:
                action = "REDIRECT_SIMPLE"
                compatibility = "COMPATIBLE_SIMPLE"
            
            canonical_mapping = {
                "DOSAGE_LIMIT": "剂量用法",
                "COMPONENT": "成分构成",
                "PACKAGING": "临床事实核验",
                "STORAGE": "贮藏条件",
                "CONTRAINDICATION": "禁忌人群"
            }
            target = canonical_mapping.get(intent, raw_facet)
            
            return FacetGovernanceDecision(
                intent=intent,
                facet_action=action,
                target_facet=target,
                compatibility=compatibility,
                reason=f"Structured check call exception: {e}"
            )
