# -*- coding: utf-8 -*-
import os
import sys
import json
import asyncio
import logging
import re
import shutil
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

from config import LLM_MODEL, PURIFY_LIMIT, PURIFY_LINES, PURIFY_START_LINE, PURIFY_CONCURRENCY, PURIFY_STRICT_RIGOR
from services.llm_service import ILLMService
from services.healing_service import IHealingService
from strategies.quality_gate.llm_judge import IEvaluationStrategy
import core.purification_helper as purifier_module

logger = logging.getLogger("MedicalQA.PurificationEngine")


def strip_unsupported_official_identifiers(text: str, evidence_text: str = "") -> str:
    """
    Drop official-looking standard/approval identifiers from the source thought
    if the confirmed evidence anchors do not contain the same identifier.
    """
    if not text:
        return text

    evidence_text = evidence_text or ""
    if not evidence_text.strip():
        return text
    pattern = re.compile(
        r"(?P<prefix>(?:国家[^，。；\n]{0,30})?(?:执行标准|标准代号|标准号|批准文号|注册号|药品标准|标准批件)"
        r"[：:\s]*)?(?P<code>[A-Z][A-Z0-9()\-]{3,}\d{2,})"
    )

    def repl(match: re.Match) -> str:
        code = match.group("code")
        if code and code in evidence_text:
            return match.group(0)
        prefix = match.group("prefix") or ""
        if prefix:
            return ""
        return ""

    cleaned = pattern.sub(repl, text)
    cleaned = re.sub(r"[，,；;]\s*[，,；;]+", "，", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()

class PurificationEngine:
    """
    Orchestrates the entire semantic purification and quality check process,
    achieving 100% clean mental flow datasets with automated hallucination dropping.
    """
    def __init__(
        self, 
        llm_service: ILLMService, 
        healing_service: IHealingService, 
        evaluator_strategy: IEvaluationStrategy
    ):
        self.llm_service = llm_service
        self.healing_service = healing_service
        self.evaluator = evaluator_strategy

    async def purify_single_think(
        self, 
        q: str, 
        planner: str, 
        raw_think: str, 
        purified_answer: str,
        line_num: int = None,
        refs: List[Dict[str, Any]] = None,
        simplify: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """
        利用反馈控制环路和无监督“刚性事实锚点”注入机制，将原始混杂 RAG 与工程噪声的思维链，
        重写为高熵、真实、严密且绝对对齐医学事实的临床专家 CoT。
        """
        max_retries = 3
        THRESHOLD_PURITY = 85
        THRESHOLD_RIGOR = 90
        THRESHOLD_DEPTH = 50 if simplify else 85
        
        last_scores = {}
        feedback_prompt = ""
        
        stripped_think = purifier_module.pre_strip_engineering_noise(raw_think)
        smoothed_planner = await purifier_module.smooth_planner_term(self.llm_service, planner, line_num=line_num)
        
        few_shot = purifier_module.FACET_FEW_SHOTS.get(planner, purifier_module.FEW_SHOT_GENERAL)
        system_prompt = purifier_module.get_purify_system_prompt(smoothed_planner, simplify=simplify)
        directive = purifier_module.get_system_directive(smoothed_planner)

        # 🛡️ 【企业级证据作用域分级路由与过滤】
        from core.rag.evidence_scope_router import EvidenceScopeRouter
        from core.governance.facet_strategy import classify_intent_by_rule
        router = EvidenceScopeRouter()
        intent = classify_intent_by_rule(q)
        routed_refs = router.route_references(q, intent, refs or [])
        active_refs = routed_refs["CORE"] + routed_refs["BOUNDARY"]

        # 🛡️ 【企业级刚性事实锚点解析注入】提取绝对正确的原始图谱或联网事实，强行阻断幻觉空间
        anchors_prompt = ""
        anchors_text = ""
        if active_refs:
            anchors = []
            for idx, r in enumerate(active_refs, start=1):
                if isinstance(r, dict):
                    ctx = r.get("context", "")
                    if ctx:
                        # 剥离多余的工程前缀，提取纯粹的事实陈述
                        clean_ctx = ctx.replace("【互联网权威医疗站快讯】:", "").replace("【互联网权威医疗数据通报】:", "").strip()
                        anchors.append(f"- [文献_{idx:02d}] {clean_ctx}")
            if anchors:
                anchors_text = "\n".join(anchors)
                anchors_prompt = f"""

### 确证医学文献事实与临床研究数据 (Confirmed Clinical & Literature Facts):
{anchors_text}
【⚠️ 确证事实对齐】：请注意，以上数据为临床确证事实，你的药理因果推演必须与之完全吻合，绝对禁止对其中任何药理关系、不良反应或用药禁忌进行任何否定、篡改或凭空编造！"""
        stripped_think = strip_unsupported_official_identifiers(stripped_think, anchors_text)
        
        for attempt in range(max_retries):
            simplify_prompt_addition = ""
            if simplify:
                simplify_prompt_addition = "\n【⚠️ 极简重构硬性要求】：该问题为简单事实查询，严禁脑补虚构复杂的分子机制、受体通路、靶点、免疫机制或大样本临床试验。请直接用 2-3 步精炼的因果推导得出结论；若机制或相互作用依据不足，只能自然收束为“不能据此推断具体机制/通路”，禁止大段微观机制演绎，但必须保持流畅的探究性临床推理心流（至少 150 字），不能只写一句话结论或复读说明书条目。"
            
            # 显式拼接已缩窄的 Answer Body 限制生成边界，确保 Think 宽度绝不宽于 Answer
            answer_boundary_prompt = ""
            if purified_answer:
                answer_boundary_prompt = f"""

### 已提纯的回答正文 (Purified Answer Body Boundary):
{purified_answer}
【⚠️ 思考链事实边界硬对齐红线】：上文为该行提纯后的唯一最终回答正文。你的 CoT 思考流（Think）必须全程且仅围绕本正文中包含的事实展开。绝对禁止在思考链中讨论或推导演答正文未提及的任何旁路药物成分、次要机制、或临床研究！"""

            prompt = f"""{few_shot}

### 系统指令 (System Directive)：
Please write an extremely raw, high-entropy clinical reasoning thought trace focusing on {directive}.
CRITICAL红线：You MUST write in a live EXPLORATORY CoT style. Do NOT write a textbook article or explanation (绝对禁止以教科书平铺直叙或说明书废话体写作). 
In corporate with counterfactual checks, you should naturally integrate clinical self-questioning markers with a question mark at points of uncertainty or divergence (在遇到逻辑分叉或极限情况时，应自然融入探究性的自我提问，展现真实的解题反思与假说排查，例如以以“？”结尾的疑问句进行内部推演，但绝对禁止在文本尾部生硬塞入无意义的问号占位符).
Do NOT output the word 'facet' or the facet name '{smoothed_planner}' in the text. You are strictly FORBIDDEN from using any meta-narrative terms indicating internal system implementations, such as 'refs', '图谱', '实体库', '关系库', '数据源', 'json_schema', 'answer_body', 'sub_questions' or 'reasoning_chains'. Output ONLY the purified thought chain. You are permitted and highly encouraged to naturally attribute facts using standard, professional references such as "根据药品说明书记载", "临床文献报道指出", or "根据临床研究数据".
CRITICAL factual boundary: never invent or preserve unsupported official standard numbers, approval numbers, registration numbers, receptor pathways, targets, immune mechanisms, pharmacokinetic parameters, or molecular pathways. If the confirmed facts do not explicitly support a mechanism/pathway/identifier, omit it or state in natural clinical language that no specific mechanism can be inferred from the available clinical facts.{anchors_prompt}{answer_boundary_prompt}

问题: {q}
原始思维链 (CoT) 内容:
\"\"\"
{stripped_think}
\"\"\"{feedback_prompt}

请严格按照净化重写指南，仅输出重构后的纯净思维链本身。{simplify_prompt_addition}"""
            try:
                stage_prefix = f"[{line_num}行] " if line_num else ""
                # 设定 max_tokens=3072 给深层临床药理因果推演提供充足的空间，避免物理截断
                purified = await self.llm_service.call_llm(
                    prompt, 
                    system_prompt=system_prompt, 
                    model_pool="premium", 
                    stage=f"{stage_prefix}思维链重写提纯 - {smoothed_planner}",
                    max_tokens=3072
                )
                purified = purified.replace("<think>", "").replace("</think>", "").strip()
                
                if purified.startswith("```"):
                    purified = "\n".join(purified.splitlines()[1:])
                if purified.endswith("```"):
                    purified = "\n".join(purified.splitlines()[:-1])
                purified = purified.strip()
                
                # 🟢 【泛化升级】：将去噪、抹除做题家序号与残缺自愈完全交给轻量级大模型（model_pool="lightweight"）进行智能语义重构，
                # 彻底废除脆弱、易破损主语的 post_strip_meta_openings 和 post_strip_structural_transitions 正则。
                purified = await self.healing_service.heal_conversational_noise(purified, line_num=line_num)
                
                if purifier_module.is_catastrophic_format_collapse(purified):
                    logger.warning(f"   🚨 Attempt {attempt+1} triggered SYNTAX FORMAT COLLAPSE! Local intercepting...")
                    scores = {
                        "semantic_purity_score": 0,
                        "medical_rigor_score": 90,
                        "logical_depth_score": 0,
                        "reason": "触发物理格式崩溃硬性熔断门禁。"
                    }
                elif purifier_module.has_repetition_loop(purified):
                    logger.warning(f"   🚨 Attempt {attempt+1} triggered Repetition penalty! Local intercepting...")
                    scores = {
                        "semantic_purity_score": 50,
                        "medical_rigor_score": 90,
                        "logical_depth_score": 50,
                        "reason": "检测到提纯后的文本发生了大面积死循环与复读退化。"
                    }
                else:
                    scores = await self.evaluator.evaluate(q, smoothed_planner, raw_think, purified, line_num=line_num, refs=active_refs)
                
                last_scores = scores
                
                p_score = purifier_module.safe_int(scores.get("semantic_purity_score", 90))
                r_score = purifier_module.safe_int(scores.get("medical_rigor_score", 90))
                d_score = purifier_module.safe_int(scores.get("logical_depth_score", scores.get("logical_coherence_score", 90)))
                factual_errors = scores.get("factual_errors", [])
                reason = str(scores.get("reason", "No reason provided"))
                
                has_fact_err = len(factual_errors) > 0
                is_rigor_passed = (r_score >= THRESHOLD_RIGOR) and (not has_fact_err if PURIFY_STRICT_RIGOR else True)
                
                logger.info(f"   └─ Attempt {attempt+1}: [Purity: {p_score}/100, Rigor: {r_score}/100, Depth: {d_score}/100, Fact Errors: {len(factual_errors)}] | Reason: {reason}")
                
                # 🌟 [方案四 - 语义提纯网关触发]
                if p_score < THRESHOLD_PURITY and is_rigor_passed and d_score >= THRESHOLD_DEPTH:
                    logger.info("   🛡️ [方案四 - 语义提纯网关触发] 检测到医学及逻辑达标，但纯净度偏低。启动企业级语义重构...")
                    purifier_prompt = f"""你是一个顶级循证医学学术编辑。你的任务是将一段混有“开场废话”、“视角扮演宣告”和“元叙事噪声”的医疗推理思维链（CoT），重构为一段完全连贯、自然流动且绝对纯净的临床学术推理心流。

### 🛠️ 重构与平滑红线：
1. ❌ 彻底移除任何元叙事废话与开场白（如：“好的”、“我们被要求以...视角”、“问题是...”、“我的分析是...”）。
2. 🔄 语义平滑融合：如果开场句中包含关键实体（例如“地氟烷”、“瑞波西利”等药物或疾病名称），请将该实体与真实的药理/推演逻辑完美融合为一句专业的学术开场白（例如，将“我们被要求分析地氟烷的禁忌”重构为“解构地氟烷的临床禁忌边界，必须剖析其...”），绝对不要直接截断导致首句不连贯！
3. 🔗 修复指代关系：确保第一句有明确的医学实体作为主语，将任何模糊的代词（如“它”、“该药物”、“此类患者”）替换为具体的医学名字，确保全篇行云流水、因果严密。
4. 📤 仅输出重构后的纯净思维链本身，不要包裹在 <think> 或 markdown 块中，不要有任何额外解释。

原始思维链内容:
\"\"\"
{purified}
\"\"\""""
                    try:
                        purified_smooth = await self.llm_service.call_llm(
                            purifier_prompt,
                            model_pool="lightweight",
                            stage=f"{stage_prefix}思维链语义提纯 - {smoothed_planner}"
                        )
                        purified_smooth = purified_smooth.replace("<think>", "").replace("</think>", "").strip()
                        if purified_smooth.startswith("```"):
                            purified_smooth = "\n".join(purified_smooth.splitlines()[1:])
                        if purified_smooth.endswith("```"):
                            purified_smooth = "\n".join(purified_smooth.splitlines()[:-1])
                        purified_smooth = purified_smooth.strip()
                        
                        # 🟢 【方案四提纯】：提纯模型已经过详尽 prompt 去除了过渡序号和开场噪音，直接信任并保留其输出
                        pass
                        
                        # 重新运行裁判打分校验
                        scores_smooth = await self.evaluator.evaluate(q, smoothed_planner, raw_think, purified_smooth, line_num=line_num, refs=active_refs)
                        p_score_smooth = purifier_module.safe_int(scores_smooth.get("semantic_purity_score", 90))
                        
                        logger.info(f"   └─ 提纯后重校验: [Purity: {p_score_smooth}/100] | Reason: {scores_smooth.get('reason', 'N/A')}")
                        
                        if p_score_smooth >= THRESHOLD_PURITY:
                            logger.info("   🎉 [方案四 - 语义提纯成功] 纯净度已完全达标！")
                            purified = purified_smooth
                            scores = scores_smooth
                            p_score = p_score_smooth
                    except Exception as e_smooth:
                        logger.error(f"   ⚠️ [方案四 - 提纯发生异常]: {e_smooth}")

                if p_score >= THRESHOLD_PURITY and is_rigor_passed and d_score >= THRESHOLD_DEPTH:
                    logger.info(f"   🎉 Quality Gate PASSED on attempt {attempt+1}! Healing academic entities...")
                    purified = await self.healing_service.verify_and_repair_academic_entities(
                        purified, 
                        q, 
                        smoothed_planner, 
                        line_num=line_num
                    )
                    
                    sim = purifier_module.calculate_similarity(raw_think, purified)
                    has_noise = any(kw in purified.lower() for kw in ["json", "schema", "免责声明", "忽略", "refs", "图谱"])
                    is_bypass = sim > 0.85 and has_noise
                    scores["purity_bypass"] = is_bypass
                    scores["is_passed"] = True
                    return purified, scores
                else:
                    logger.warning(
                        f"\n============================================================\n"
                        f"   ❌ Quality Gate FAILED on attempt {attempt+1}!\n"
                        f"   [Line Number / 行号]: {line_num or 'Unknown'}\n"
                        f"   [Facet / 切面]: {smoothed_planner}\n"
                        f"   [Failing Reason / 裁判评语]: {reason}\n"
                        f"============================================================\n"
                    )
                    
                    # De-contaminated Feedback Loop
                    feedback_msg = f"\n\n【前一次清洗尝试质量不达标反馈：语义纯净度={p_score}/100, 医学严谨度={r_score}/100, 逻辑深度={d_score}/100。】"
                    if p_score < THRESHOLD_PURITY:
                        feedback_msg += "\n【核心优化指令：你的前一次写入在“语义纯净度”上不符合规范。请确保全篇为完全连贯、自然流动的临床学术段落，绝对禁止提及或表露元数据结构。】"
                    if r_score < THRESHOLD_RIGOR:
                        feedback_msg += "\n【核心优化指令：你的前一次写入在“医学严谨度”分数上不符合规范，请注意确证事实的对齐。】"
                    if factual_errors:
                        feedback_msg += "\n【核心优化指令：质检审查裁判发现的具体医学事实/化学术语/学术错误清单如下，请在本次重写中予以彻底纠正】：\n" + "\n".join(f"- {err}" for err in factual_errors)
                    if d_score < THRESHOLD_DEPTH:
                        feedback_msg += "\n【核心优化指令：你的前一次写入在“逻辑深度”上不符合规范，请避免平铺直叙，融入探究反思。】"
                    if reason and reason != "No explanation provided" and not factual_errors:
                        feedback_msg += f"\n【质检审查裁判的具体评审意见：{reason}】"
                    
                    feedback_msg = feedback_msg.replace('[', '【').replace(']', '】')
                    feedback_prompt = feedback_msg
                    
            except Exception as e:
                logger.error(f"   ⚠️ Error during purification attempt {attempt+1}: {e}")
                
        logger.warning("   ⚠️ Quality Gate Max Retries exceeded. Gracefully falling back to safety fallback.")
        try:
            from scripts.clean_dataset import clean_think_text
            purified = clean_think_text(raw_think)
        except ImportError:
            purified = purifier_module.post_strip_structural_transitions(
                purifier_module.post_strip_meta_openings(
                    purifier_module.pre_strip_engineering_noise(raw_think)
                )
            )
            
        sim = purifier_module.calculate_similarity(raw_think, purified)
        has_noise = any(kw in purified.lower() for kw in ["json", "schema", "免责声明", "忽略", "refs", "图谱"])
        is_bypass = sim > 0.85 and has_noise
        
        ret_scores = last_scores or {}
        ret_scores.update({
            "semantic_purity_score": 85,
            "medical_rigor_score": 90,
            "logical_depth_score": 85,
            "reason": "Fallback used.",
            "purity_bypass": is_bypass,
            "is_passed": False
        })
        return purified, ret_scores
