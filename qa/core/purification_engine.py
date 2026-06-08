# -*- coding: utf-8 -*-
import os
import sys
import json
import asyncio
import logging
import re
import shutil
import datetime
import sqlite3
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


    # ─────────────────────────────────────────────────────────────────
    # ① 私有辅助：证据路由 + 刚性事实锚点 Prompt 构建
    # ─────────────────────────────────────────────────────────────────
    async def _route_evidence(
        self,
        q: str,
        refs: List[Dict[str, Any]],
    ) -> Tuple[List, str, str]:
        """
        职责：调用 EvidenceScopeRouter 过滤 refs，并将 active_refs 转化为
        可直接嵌入 Prompt 的刚性事实锚点文本块。
        返回: (active_refs, anchors_text, anchors_prompt)
        """
        from core.rag.evidence_scope_router import EvidenceScopeRouter
        from core.governance.facet_strategy import classify_intent_by_rule
        router = EvidenceScopeRouter()
        intent = classify_intent_by_rule(q)
        routed_refs = await router.route_references(q, intent, refs or [])
        active_refs = routed_refs["CORE"] + routed_refs["BOUNDARY"]

        anchors_text = ""
        anchors_prompt = ""
        if active_refs:
            anchors = []
            for idx, r in enumerate(active_refs, start=1):
                if isinstance(r, dict):
                    ctx = r.get("context", "")
                    if ctx:
                        clean_ctx = (
                            ctx.replace("【互联网权威医疗站快讯】:", "")
                               .replace("【互联网权威医疗数据通报】:", "")
                               .strip()
                        )
                        anchors.append(f"- [文献_{idx:02d}] {clean_ctx}")
            if anchors:
                anchors_text = "\n".join(anchors)
                anchors_prompt = f"""

### 确证医学文献事实与临床研究数据 (Confirmed Clinical & Literature Facts):
{anchors_text}
【⚠️ 确证事实对齐】：请注意，以上数据为临床确证事实，你的药理因果推演必须与之完全吻合，绝对禁止对其中任何药理关系、不良反应或用药禁忌进行任何否定、篡改或凭空编造！"""
        return active_refs, anchors_text, anchors_prompt

    # ─────────────────────────────────────────────────────────────────
    # ② 私有辅助：Prompt 纯函数装配
    # ─────────────────────────────────────────────────────────────────
    def _build_purify_prompt(
        self,
        q: str,
        stripped_think: str,
        smoothed_planner: str,
        few_shot: str,
        directive: str,
        anchors_prompt: str,
        purified_answer: str,
        simplify: bool,
        feedback_prompt: str,
    ) -> Tuple[str, str]:
        """
        职责：将各独立提示组件拼接为最终的 system_prompt + user prompt（纯函数，无 I/O）。
        返回: (prompt, system_prompt)
        """
        system_prompt = purifier_module.get_purify_system_prompt(smoothed_planner, simplify=simplify)

        simplify_prompt_addition = ""
        if simplify:
            simplify_prompt_addition = (
                "\n【⚠️ 极简重构硬性要求】：该问题为简单事实查询，严禁脑补虚构复杂的分子机制、受体通路、靶点、免疫机制或大样本临床试验。"
                "请直接用 2-3 步精炼的因果推导得出结论；若机制或相互作用依据不足，只能自然收束为'不能据此推断具体机制/通路'，"
                "禁止大段微观机制演绎，但必须保持流畅的探究性临床推理心流（至少 150 字），不能只写一句话结论或复读说明书条目。"
            )


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
In corporate with counterfactual checks, you should naturally integrate clinical self-questioning markers with a question mark at points of uncertainty or divergence (在遇到逻辑分叉或极限情况时，应自然融入探究性的自我提问，展现真实的解题反思与假说排查，例如以以"？"结尾的疑问句进行内部推演，但绝对禁止在文本尾部生硬塞入无意义的问号占位符).
Do NOT output the word 'facet' or the facet name '{smoothed_planner}' in the text. You are strictly FORBIDDEN from using any meta-narrative terms indicating internal system implementations, such as 'refs', '图谱', '实体库', '关系库', '数据源', 'json_schema', 'answer_body', 'sub_questions' or 'reasoning_chains'. Output ONLY the purified thought chain. You are permitted and highly encouraged to naturally attribute facts using standard, professional references such as "根据药品说明书记载", "临床文献报道指出", or "根据临床研究数据".
CRITICAL factual boundary: never invent or preserve unsupported official standard numbers, approval numbers, registration numbers, receptor pathways, targets, immune mechanisms, pharmacokinetic parameters, or molecular pathways. If the confirmed facts do not explicitly support a mechanism/pathway/identifier, omit it or state in natural clinical language that no specific mechanism can be inferred from the available clinical facts.{anchors_prompt}{answer_boundary_prompt}

问题: {q}
原始思维链 (CoT) 内容:
\"\"\"
{stripped_think}
\"\"\"{feedback_prompt}

请严格按照净化重写指南，仅输出重构后的纯净思维链本身。{simplify_prompt_addition}"""
        return prompt, system_prompt

    # ─────────────────────────────────────────────────────────────────
    # ③ 私有辅助：LLM 调用 + 输出后处理 + 本地快速结构/重复校验
    # ─────────────────────────────────────────────────────────────────
    async def _call_and_heal(
        self,
        prompt: str,
        system_prompt: str,
        smoothed_planner: str,
        stage_prefix: str,
        line_num: int,
    ) -> Tuple[str, Dict]:
        """
        职责：调用高级 LLM，清理输出格式（think标签/markdown围栏），语义自愈，
        并执行本地快速结构塌陷与重复环路校验（短路求值，命中则跳过远端裁判）。
        返回: (purified_text, fast_scores_or_None)
        """
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

        # 语义自愈：轻量级大模型去除做题家序号与元叙事噪声
        purified = await self.healing_service.heal_conversational_noise(purified, line_num=line_num)

        if purifier_module.is_catastrophic_format_collapse(purified):
            logger.warning("   🚨 Local fast-check: SYNTAX FORMAT COLLAPSE detected. Intercepting...")
            return purified, {
                "semantic_purity_score": 0,
                "medical_rigor_score": 90,
                "logical_depth_score": 0,
                "reason": "触发物理格式崩溃硬性熔断门禁。"
            }
        if purifier_module.has_repetition_loop(purified):
            logger.warning("   🚨 Local fast-check: Repetition loop detected. Intercepting...")
            return purified, {
                "semantic_purity_score": 50,
                "medical_rigor_score": 90,
                "logical_depth_score": 50,
                "reason": "检测到提纯后的文本发生了大面积死循环与复读退化。"
            }
        return purified, None

    # ─────────────────────────────────────────────────────────────────
    # ④ 私有静态辅助：冲突事件持久化日志
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _log_conflict_to_registry(entry: Dict) -> None:
        """职责：将单次冲突事件以 JSONL 格式追加写入 factual_conflicts_registry.jsonl。"""
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        path = os.path.join(logs_dir, "factual_conflicts_registry.jsonl")
        try:
            with open(path, "a", encoding="utf-8", newline="\n") as f_reg:
                f_reg.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as ex:
            logger.error(f"Failed to log conflict to registry: {ex}")

    # ─────────────────────────────────────────────────────────────────
    # ⑤ 私有辅助：FactAuditingGateway 二审 + DB Safety Gate 协调
    # ─────────────────────────────────────────────────────────────────
    async def _run_conflict_audit(
        self,
        q: str,
        planner: str,
        smoothed_planner: str,
        purified: str,
        scores: Dict,
        active_refs: List,
        refs: List,
        line_num: int,
    ) -> Tuple[Dict, str, str]:
        """
        职责：协调事实冲突二审（FactAuditingGateway）与 Safety Gate（DBHotpatchManager）。
        返回: (updated_scores, updated_anchors_text, updated_anchors_prompt)
          - anchors_text/anchors_prompt 仅在热修复成功后有值；被阻断则返回空字符串。
        """
        import datetime
        from services.fact_correction_service import FactAuditingGateway

        conflict_desc = scores.get("conflict_description", "")
        conflict_details = scores.get("conflict_details") or {}

        registry_entry = {
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "line_num": line_num,
            "question": q,
            "planner": planner,
            "purified_think": purified,
            "conflict_description": conflict_desc,
            "conflict_details": conflict_details,
            "status": "DETECTED",
            "verdict": None
        }
        self._log_conflict_to_registry(registry_entry)

        updated_anchors_text, updated_anchors_prompt = "", ""

        try:
            gateway = FactAuditingGateway(self.llm_service)
            audit_result = await gateway.audit_conflict(
                q=q,
                planner=smoothed_planner,
                purified_think=purified,
                refs=active_refs,
                first_stage_conflict_description=conflict_desc,
                first_stage_conflict_details=conflict_details
            )
            verdict = audit_result.get("verdict")
            confidence = audit_result.get("confidence", 0.0)
            logger.info(f"   ├─ Stage-2 Audit Verdict: '{verdict}' (Confidence: {confidence})")

            registry_entry.update({
                "verdict": verdict,
                "status": "AUDITED",
                "audit_confidence": confidence,
                "audit_reason": audit_result.get("reason"),
            })
            self._log_conflict_to_registry(registry_entry)

            if verdict == "RAG_ERROR":
                updated_anchors_text, updated_anchors_prompt = await self._attempt_db_hotpatch(
                    audit_result, conflict_details, active_refs, refs
                )
                if not updated_anchors_text:
                    scores["is_passed"] = False
                    scores["requires_human_review"] = True
            else:
                logger.warning(f"🚫 Stage-2 Audit verdict '{verdict}' is not RAG_ERROR. Blocked DB writing.")
                scores["is_passed"] = False
                scores["requires_human_review"] = True

        except Exception as audit_ex:
            logger.error(f"⚠️ Exception during FactAuditingGateway workflow: {audit_ex}")

        return scores, updated_anchors_text, updated_anchors_prompt

    # ─────────────────────────────────────────────────────────────────
    # ⑥ 私有辅助：DB 热修复 Safety Gate
    # ─────────────────────────────────────────────────────────────────
    async def _attempt_db_hotpatch(
        self,
        audit_result: Dict,
        conflict_details: Dict,
        active_refs: List,
        refs: List,
    ) -> Tuple[str, str]:
        """
        职责：在 Safety Gate 框架内尝试将修复提案写入 local_rag.db。
        当前 Safety Gate 已启用，物理写入被阻断，提案进入 PENDING_APPROVAL 队列。
        返回: (updated_anchors_text, updated_anchors_prompt)，失败返回 ("", "")。
        """
        import time
        import random
        from services.fact_correction_service import FactCorrectionProposal, DBHotpatchManager
        from retrieval.local_rag import LocalRAGService

        confidence = audit_result.get("confidence", 0.0)
        db_path = "local_rag.db"
        if not os.path.exists(db_path):
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "local_rag.db")

        evidence_text = conflict_details.get("evidence") or ""

        def find_db_record(path: str, text: str):
            if not os.path.exists(path) or not text:
                return None
            try:
                conn_local = sqlite3.connect(path)
                cur = conn_local.cursor()
                cur.execute("SELECT id, context FROM local_rag_index WHERE context = ?", (text,))
                row = cur.fetchone()
                if not row:
                    cur.execute("SELECT id, context FROM local_rag_index WHERE context LIKE ?", (f"%{text}%",))
                    row = cur.fetchone()
                conn_local.close()
                return row
            except Exception as ex:
                logger.error(f"Error querying local DB: {ex}")
            return None

        db_info = None
        if evidence_text:
            db_info = await asyncio.to_thread(find_db_record, db_path, evidence_text)
        if not db_info:
            for ref in active_refs:
                ref_ctx = ref.get("context") if isinstance(ref, dict) else ""
                if ref_ctx:
                    db_info = await asyncio.to_thread(find_db_record, db_path, ref_ctx)
                    if db_info:
                        break

        if not db_info:
            logger.error("❌ Could not match RAG evidence to database record for patching.")
            return "", ""

        db_id, db_context = db_info
        conflict_original = conflict_details.get("original_value") or ""
        conflict_corrected = conflict_details.get("corrected_value") or ""

        if not (conflict_original and conflict_original in db_context):
            return "", ""

        corrected_context = db_context.replace(conflict_original, conflict_corrected)
        proposal = FactCorrectionProposal(
            proposal_id=f"prop_{int(time.time())}_{random.randint(1000, 9999)}",
            db_type="local_rag.db",
            table_name="local_rag_index",
            target_id=str(db_id),
            field_name="context",
            corrected_value=corrected_context,
            clinical_evidence=audit_result.get("reason", "Heterogeneous audit verdict"),
            confidence_score=confidence,
            original_value=conflict_original,
            verdict="RAG_ERROR"
        )

        patch_manager = DBHotpatchManager(db_path)
        # [Safety Guard] 禁用物理 DB 写入，等待人工审批；如需启用，取消下行注释并注释 applied = False
        # applied = patch_manager.execute_patch(proposal)
        applied = False

        if applied:
            LocalRAGService.clear_all_caches()
            for r in active_refs + (refs or []):
                if isinstance(r, dict) and r.get("context") == db_context:
                    r["context"] = corrected_context

            anchors = []
            for idx, r in enumerate(active_refs, start=1):
                if isinstance(r, dict):
                    ctx = r.get("context", "")
                    if ctx:
                        clean_ctx = (
                            ctx.replace("【互联网权威医疗站快讯】:", "")
                               .replace("【互联网权威医疗数据通报】:", "")
                               .strip()
                        )
                        anchors.append(f"- [文献_{idx:02d}] {clean_ctx}")
            if anchors:
                updated_text = "\n".join(anchors)
                updated_prompt = f"""

### 确证医学文献事实与临床研究数据 (Confirmed Clinical & Literature Facts):
{updated_text}
【⚠️ 确证事实对齐】：请注意，以上数据为临床确证事实，你的药理因果推演必须与之完全吻合，绝对禁止对其中任何药理关系、不良反应或用药禁忌进行任何否定、篡改或凭空编造！"""
                logger.info("🎉 Hotpatched local database and synchronized active references in memory. Forcing self-healing retry.")
                return updated_text, updated_prompt
            return "", ""
        else:
            logger.warning("🚫 Audit blocked by PENDING_APPROVAL Safety Gate. No database change applied.")
            return "", ""

    # ─────────────────────────────────────────────────────────────────
    # ⑦ 私有辅助：方案四——语义提纯网关
    # ─────────────────────────────────────────────────────────────────
    async def _apply_semantic_purifier(
        self,
        q: str,
        smoothed_planner: str,
        raw_think: str,
        purified: str,
        active_refs: List,
        stage_prefix: str,
        line_num: int,
        threshold_purity: int,
    ) -> Tuple[str, Dict]:
        """
        职责：方案四——当医学严谨度/逻辑深度均已达标但语义纯净度偏低时，
        启动轻量级大模型对元叙事噪声进行二次精准提纯，并重新评分验证。
        返回: (best_purified, best_scores)；提纯未能达标则返回 (原始 purified, {})。
        """
        purifier_prompt = f"""你是一个顶级循证医学学术编辑。你的任务是将一段混有"开场废话"、"视角扮演宣告"和"元叙事噪声"的医疗推理思维链（CoT），重构为一段完全连贯、自然流动且绝对纯净的临床学术推理心流。

### 🛠️ 重构与平滑红线：
1. ❌ 彻底移除任何元叙事废话与开场白（如："好的"、"我们被要求以...视角"、"问题是..."、"我的分析是..."）。
2. 🔄 语义平滑融合：如果开场句中包含关键实体（例如"地氟烷"、"瑞波西利"等药物或疾病名称），请将该实体与真实的药理/推演逻辑完美融合为一句专业的学术开场白（例如，将"我们被要求分析地氟烷的禁忌"重构为"解构地氟烷的临床禁忌边界，必须剖析其..."），绝对不要直接截断导致首句不连贯！
3. 🔗 修复指代关系：确保第一句有明确的医学实体作为主语，将任何模糊的代词（如"它"、"该药物"、"此类患者"）替换为具体的医学名字，确保全篇行云流水、因果严密。
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

            scores_smooth = await self.evaluator.evaluate(
                q, smoothed_planner, raw_think, purified_smooth,
                line_num=line_num, refs=active_refs
            )
            p_score_smooth = purifier_module.safe_int(scores_smooth.get("semantic_purity_score", 90))
            logger.info(
                f"   └─ 提纯后重校验: [Purity: {p_score_smooth}/100] | "
                f"Reason: {scores_smooth.get('reason', 'N/A')}"
            )
            if p_score_smooth >= threshold_purity:
                logger.info("   🎉 [方案四 - 语义提纯成功] 纯净度已完全达标！")
                return purified_smooth, scores_smooth

        except Exception as e_smooth:
            logger.error(f"   ⚠️ [方案四 - 提纯发生异常]: {e_smooth}")

        # 提纯未能达标或发生异常，返回原始版本（空 scores 告知调用方未更新）
        return purified, {}

    # ─────────────────────────────────────────────────────────────────
    # ⑧ 私有静态辅助：反馈指令构建（纯函数）
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_feedback_prompt(
        scores: Dict,
        p_score: int,
        r_score: int,
        d_score: int,
        factual_errors: List,
        reason: str,
        threshold_purity: int,
        threshold_rigor: int,
        threshold_depth: int,
    ) -> str:
        """
        职责：根据本轮质量评分，纯函数式构建下一轮重试的反馈指令字符串（无任何 I/O）。
        """
        msg = (
            f"\n\n【前一次清洗尝试质量不达标反馈："
            f"语义纯净度={p_score}/100, 医学严谨度={r_score}/100, 逻辑深度={d_score}/100。】"
        )
        if p_score < threshold_purity:
            msg += "\n【核心优化指令：你的前一次写入在'语义纯净度'上不符合规范。请确保全篇为完全连贯、自然流动的临床学术段落，绝对禁止提及或表露元数据结构。】"
        if r_score < threshold_rigor:
            msg += "\n【核心优化指令：你的前一次写入在'医学严谨度'分数上不符合规范，请注意确证事实的对齐。】"
        if factual_errors:
            msg += "\n【核心优化指令：质检审查裁判发现的具体医学事实/化学术语/学术错误清单如下，请在本次重写中予以彻底纠正】：\n"
            msg += "\n".join(f"- {err}" for err in factual_errors)
        if d_score < threshold_depth:
            msg += "\n【核心优化指令：你的前一次写入在'逻辑深度'上不符合规范，请避免平铺直叙，融入探究反思。】"
        if reason and reason != "No explanation provided" and not factual_errors:
            msg += f"\n【质检审查裁判的具体评审意见：{reason}】"
        suggestions = scores.get("improvement_suggestions", "")
        if suggestions and suggestions != "No suggestions provided":
            msg += f"\n【质检审查裁判给出的具体改进建议：{suggestions}】"
        return msg.replace('[', '【').replace(']', '】')

    # ─────────────────────────────────────────────────────────────────
    # Public entry：协调入口（对外接口签名保持完全不变）
    # ─────────────────────────────────────────────────────────────────
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
        利用反馈控制环路和无监督"刚性事实锚点"注入机制，将原始混杂 RAG 与工程噪声的思维链，
        重写为高熵、真实、严密且绝对对齐医学事实的临床专家 CoT。
        """
        THRESHOLD_PURITY = 85
        THRESHOLD_RIGOR = 90
        THRESHOLD_DEPTH = 50 if simplify else 85
        MAX_RETRIES = 3

        last_scores: Dict = {}
        feedback_prompt = ""
        stage_prefix = f"[{line_num}行] " if line_num else ""

        # ① 证据路由 + 事实锚点
        active_refs, anchors_text, anchors_prompt = await self._route_evidence(q, refs)
        stripped_think = strip_unsupported_official_identifiers(
            purifier_module.pre_strip_engineering_noise(raw_think), anchors_text
        )
        smoothed_planner = await purifier_module.smooth_planner_term(
            self.llm_service, planner, line_num=line_num
        )
        few_shot = purifier_module.FACET_FEW_SHOTS.get(planner, purifier_module.FEW_SHOT_GENERAL)
        directive = purifier_module.get_system_directive(smoothed_planner)

        for attempt in range(MAX_RETRIES):
            # ② Prompt 装配
            prompt, system_prompt = self._build_purify_prompt(
                q, stripped_think, smoothed_planner, few_shot, directive,
                anchors_prompt, purified_answer, simplify, feedback_prompt
            )
            try:
                # ③ LLM 调用 + 后处理 + 本地快速校验
                purified, fast_scores = await self._call_and_heal(
                    prompt, system_prompt, smoothed_planner, stage_prefix, line_num
                )

                # ④ 远端裁判评分（本地未拦截时执行）
                scores = fast_scores if fast_scores is not None else await self.evaluator.evaluate(
                    q, smoothed_planner, raw_think, purified, line_num=line_num, refs=active_refs
                )

                # ⑤ 事实冲突二审（FactAuditingGateway）
                if scores.get("conflict_detected"):
                    logger.warning(
                        "   🚨 [Conflict Detected] Stage-1 Judge flagged a conflict. Triggering FactAuditingGateway..."
                    )
                    scores, hot_text, hot_prompt = await self._run_conflict_audit(
                        q, planner, smoothed_planner, purified, scores, active_refs, refs, line_num
                    )
                    if hot_text:  # 热修复成功，更新 anchors 供后续重试
                        anchors_prompt = hot_prompt

                last_scores = scores
                p_score = purifier_module.safe_int(scores.get("semantic_purity_score", 90))
                r_score = purifier_module.safe_int(scores.get("medical_rigor_score", 90))
                d_score = purifier_module.safe_int(
                    scores.get("logical_depth_score", scores.get("logical_coherence_score", 90))
                )
                factual_errors = scores.get("factual_errors", [])
                reason = str(scores.get("reason", "No reason provided"))
                is_rigor_passed = (r_score >= THRESHOLD_RIGOR) and (
                    not factual_errors if PURIFY_STRICT_RIGOR else True
                )

                logger.info(
                    f"   └─ Attempt {attempt+1}: [Purity: {p_score}/100, Rigor: {r_score}/100, "
                    f"Depth: {d_score}/100, Fact Errors: {len(factual_errors)}] | Reason: {reason}"
                )

                # ⑥ 方案四：语义提纯网关（医学/逻辑达标但纯净度不足时触发）
                if p_score < THRESHOLD_PURITY and is_rigor_passed and d_score >= THRESHOLD_DEPTH:
                    logger.info(
                        "   🛡️ [方案四 - 语义提纯网关触发] 检测到医学及逻辑达标，但纯净度偏低。启动企业级语义重构..."
                    )
                    purified_candidate, scores_candidate = await self._apply_semantic_purifier(
                        q, smoothed_planner, raw_think, purified,
                        active_refs, stage_prefix, line_num, THRESHOLD_PURITY
                    )
                    if scores_candidate:  # 提纯成功，使用新结果
                        purified, scores = purified_candidate, scores_candidate
                        p_score = purifier_module.safe_int(scores.get("semantic_purity_score", 90))

                # ⑦ 最终质量门禁
                if p_score >= THRESHOLD_PURITY and is_rigor_passed and d_score >= THRESHOLD_DEPTH:
                    logger.info(f"   🎉 Quality Gate PASSED on attempt {attempt+1}! Healing academic entities...")
                    purified = await self.healing_service.verify_and_repair_academic_entities(
                        purified, q, smoothed_planner, line_num=line_num
                    )
                    sim = purifier_module.calculate_similarity(raw_think, purified)
                    has_noise = any(
                        kw in purified.lower()
                        for kw in ["json", "schema", "免责声明", "忽略", "refs", "图谱"]
                    )
                    scores["purity_bypass"] = sim > 0.85 and has_noise
                    scores["is_passed"] = True
                    return purified, scores

                # ⑧ 质量未达标 → 构建反馈指令进入下一轮重试
                logger.warning(
                    f"\n============================================================\n"
                    f"   ❌ Quality Gate FAILED on attempt {attempt+1}!\n"
                    f"   [Line Number / 行号]: {line_num or 'Unknown'}\n"
                    f"   [Facet / 切面]: {smoothed_planner}\n"
                    f"   [Failing Reason / 裁判评语]: {reason}\n"
                    f"============================================================\n"
                )
                feedback_prompt = self._build_feedback_prompt(
                    scores, p_score, r_score, d_score, factual_errors, reason,
                    THRESHOLD_PURITY, THRESHOLD_RIGOR, THRESHOLD_DEPTH
                )

            except Exception as e:
                logger.error(f"   ⚠️ Error during purification attempt {attempt+1}: {e}")

        # ⑨ 兜底 Fallback：超出最大重试次数后的降级清洗
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
        ret_scores = last_scores or {}
        ret_scores.update({
            "semantic_purity_score": 85,
            "medical_rigor_score": 90,
            "logical_depth_score": 85,
            "reason": "Fallback used.",
            "purity_bypass": sim > 0.85 and has_noise,
            "is_passed": False
        })
        return purified, ret_scores
