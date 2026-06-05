# -*- coding: utf-8 -*-
import asyncio
import datetime
import json
import random
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import config
import prompts
from models import FacetPlan, FacetQAOutput
from services.llm_service import ILLMService
from services.graph_service import IGraphService
from strategies.redundancy_filter.llm_filter import IRedundancyFilterStrategy
from core.prompt_renderer import PromptRenderer

logger = logging.getLogger("MedicalQA.PipelineWorkflow")
GENERATION_AUDIT_PATH = Path(__file__).resolve().parent.parent / "logs" / "generation_audit.jsonl"


class SampleQuarantineException(Exception):
    """
    当样本因前置治理过滤、切面不足或质量不合格而需被正常隔离（quarantine）时的业务级非致命异常。
    """
    pass


FACET_FORBIDDEN_PATTERNS = [
    r"示例\s*视角",
    r"缺少医疗问题",
    r"请提供",
    r"请输入",
    r"数据不足",
    r"无法规划",
    r"rigorous\s+data\s+processing\s+api",
    r"json\s*schema",
    r"\bschema\b",
    r"\bjson\b",
    r"\bapi\b",
    r"\bsystem\b",
    r"\bassistant\b",
    r"\buser\b",
    r"```",
    r"\{|\}|\[|\]",
]


def validate_facet_label(facet: str) -> Tuple[bool, str]:
    label = (facet or "").strip()
    if not label:
        return False, "empty facet"
    if len(label) < 2 or len(label) > 16:
        return False, "facet length must be 2-16 chars"
    if "\n" in label or "\r" in label or "\t" in label:
        return False, "facet must be one short phrase"
    if len(re.findall(r"[A-Za-z]", label)) > 8:
        return False, "facet contains too much English text"
    if any(ch in label for ch in [":", "：", "。", "？", "?", "！", "!", "，", ","]):
        return False, "facet contains sentence punctuation"
    lowered = label.lower()
    for pattern in FACET_FORBIDDEN_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return False, f"facet contains forbidden pattern: {pattern}"
    return True, ""


def filter_valid_facets(facets: List[str]) -> Tuple[List[str], List[Dict[str, str]]]:
    valid = []
    invalid = []
    for facet in facets or []:
        label = str(facet).strip()
        ok, reason = validate_facet_label(label)
        if ok and label not in valid:
            valid.append(label)
        else:
            invalid.append({"facet": label, "reason": reason or "duplicate facet"})
    return valid, invalid


def record_generation_audit(event: Dict[str, Any]) -> None:
    try:
        event.setdefault("time", datetime.datetime.now().isoformat(timespec="seconds"))
        GENERATION_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GENERATION_AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write generation audit event: {e}")

class PipelineWorkflow:
    """
    核心多轮生成编排器的工作流架构类。
    负责调度知识图谱检索、大模型推理、质量把控及冗余过滤等环节，完成多轮医疗问答数据的生成。
    """
    def __init__(
        self, 
        llm_service: ILLMService, 
        graph_service: IGraphService, 
        redundancy_filter: IRedundancyFilterStrategy
    ):
        """
        初始化 PipelineWorkflow 实例。
        
        :param llm_service: 大语言模型服务接口，用于调用各类大模型生成任务
        :param graph_service: 知识图谱服务接口，用于获取图谱实体和关系数据
        :param redundancy_filter: 冗余过滤策略接口，用于去除生成内容中的冗余切面
        """
        self.llm_service = llm_service
        self.graph_service = graph_service
        self.redundancy_filter = redundancy_filter

    async def _prepare_context_and_refs(self, graph_data: Dict[str, Any], query: str = "") -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        将知识图谱响应数据转换为上下文列表(context_list)和参考引用块(refs)。
        包含第四阶段的指南依据注入。
        
        :param graph_data: 包含实体和关系数据的字典，来源于知识图谱服务
        :param query: 当前查询问题，用于分层检索匹配参考依据
        :return: 元组，包含(上下文列表, 参考文献列表)
        """
        # 提取实体和关系，若为空则默认空列表
        entities = graph_data.get("entities", []) or []
        relationships = graph_data.get("relationships", []) or []
        
        context_list = []
        refs = []
        
        # 延迟导入检索管理器
        from retrieval.retrieval_manager import RetrievalManager
        retrieval_mgr = RetrievalManager(llm_service=self.llm_service)
        
        # 格式化实体数据
        for entity in entities:
            name = entity.get("name", "未命名实体")
            ent_type = entity.get("type", "普通实体")
            description = entity.get("description", "暂无描述").strip()
            ent_id = entity.get("id", "")
            
            # 构造上下文字符串和来源标识
            context_str = f"医疗实体【{name}】（类型: {ent_type}）：{description}"
            source_str = f"《图谱实体集:{name}》"
            
            item = {
                "context": context_str,
                "source": source_str,
                "entity_id": str(ent_id)
            }
            context_list.append(item)
            refs.append({
                "context": f"概念定义: {name} (类型: {ent_type}) - {description}",
                "source": f"refs:《实体库:{name}》"
            })
            
            # 尝试获取分层检索依据，注入到参考引用中
            try:
                tiered_refs, tier_label = await retrieval_mgr.get_grounding_references(query, name)
                logger.info(f"Tiered Grounding match found for '{name}' via [{tier_label}]: injecting {len(tiered_refs)} reference items.")
                for r_item in tiered_refs:
                    refs.append(r_item)
            except Exception as e:
                logger.error(f"Failed to fetch tiered retrieval grounding for entity '{name}': {e}")
            
        # 格式化关系数据
        for rel in relationships:
            src = rel.get("sourceName", "")
            tgt = rel.get("targetName", "")
            relation = rel.get("relationship", "")
            strength = rel.get("relationshipStrength", 10)
            
            # 构造关系上下文字符串和来源标识
            context_str = f"关联关系：【{src}】与【{tgt}】之间存在关联【{relation}】（强度: {strength}）"
            source_str = f"《图谱关系集:{src}-{tgt}》"
            
            item = {
                "context": context_str,
                "source": source_str
            }
            context_list.append(item)
            refs.append({
                "context": f"知识关联: 【{src}】 --({relation})--> 【{tgt}】",
                "source": f"refs:《图谱关系:{src}-{tgt}》"
            })
            
        # 关闭检索管理器释放资源
        retrieval_mgr.close()
        return context_list, refs

    async def generate_initial_question(self, context_list: List[Dict[str, str]], task_id_label: str = "") -> str:
        """
        根据上下文列表生成多个问题，并随机选择其中一个作为初始问题。
        路由至轻量级大模型执行。
        
        :param context_list: 上下文信息列表
        :param task_id_label: 任务ID标签，用于日志追踪
        :return: 被随机选中的初始问题字符串
        :raises Exception: 如果未能从上下文中生成任何问题则抛出异常
        """
        # 渲染问题生成提示词
        prompt = PromptRenderer.render(prompts.QUESTION_CREATOR_TEMPLATE, context_list=context_list)
        stage_prefix = f"[{task_id_label}] " if task_id_label else ""
        # 调用轻量级大模型
        response = await self.llm_service.call_llm(prompt, model_pool="lightweight", stage=f"{stage_prefix}初始问题生成")
        
        # 安全解析JSON响应
        from pipeline import parse_json_safely
        questions = parse_json_safely(response, [])
        if not questions:
            raise Exception("Failed to generate questions from context.")
            
        # 随机选择一个问题
        selected_q = random.choice(questions)
        logger.info(f"Generated {len(questions)} questions. Selected: '{selected_q}'")
        return selected_q

    async def plan_facets(self, query: str, task_id_label: str = "") -> List[str]:
        """
        使用轻量级结构化约束规划回答视角（切面/facets）。
        
        :param query: 当前查询问题
        :param task_id_label: 任务ID标签，用于日志追踪
        :return: 规划出的切面列表；若多次校验失败则返回空列表，由上游隔离该样本
        """
        # 渲染切面规划提示词
        stage_prefix = f"[{task_id_label}] " if task_id_label else ""
        audit_attempts = []

        for attempt in range(3):
            prompt = PromptRenderer.render(prompts.FACET_PLANNER_TEMPLATE, query=query)
            if audit_attempts:
                invalid_notes = "\n".join(
                    f"- attempt {a['attempt']}: {a['reason']}" for a in audit_attempts
                )
                prompt += (
                    "\n\n<previous_validation_failures>\n"
                    f"{invalid_notes}\n"
                    "请重新规划。不要输出占位符、提示语、Schema/System/API/JSON 文本；"
                    "必须输出合法的医学 facet 候选对象。\n"
                    "</previous_validation_failures>"
                )
            messages = [{"role": "user", "content": prompt}]

            try:
                result: FacetPlan = await self.llm_service.call_llm_structured(
                    messages,
                    FacetPlan,
                    model_pool="lightweight",
                    stage=f"{stage_prefix}视角切面规划 attempt-{attempt + 1}"
                )
                facets = []
                invalid = []
                for candidate in result.facets:
                    label = candidate.label.strip()
                    ok, reason = validate_facet_label(label)
                    if ok and label not in facets:
                        facets.append(label)
                    else:
                        invalid.append({"label": label, "reason": reason or "duplicate facet"})

                if len(facets) >= 2:
                    logger.info(f"Planned validated facets: {facets}")
                    record_generation_audit({
                        "stage": "facet_planning",
                        "status": "success",
                        "task_id": task_id_label,
                        "query": query,
                        "attempt": attempt + 1,
                        "facets": facets,
                    })
                    return facets

                reason = f"validated facet count < 2; invalid={invalid}"
                audit_attempts.append({"attempt": attempt + 1, "reason": reason})
                logger.warning(f"Facet plan validation failed: {reason}")
            except Exception as e:
                reason = str(e)
                audit_attempts.append({"attempt": attempt + 1, "reason": reason})
                logger.error(f"Failed to plan facets on attempt {attempt + 1}: {e}")

        logger.critical(f"Facet planning failed after retries for query '{query}'. Attempts: {audit_attempts}")
        record_generation_audit({
            "stage": "facet_planning",
            "status": "failed",
            "task_id": task_id_label,
            "query": query,
            "attempts": audit_attempts,
        })
        return []

    async def preprocess_facets(self, query: str, facets: List[str], task_id_label: str = "") -> List[str]:
        """
        对切面进行预处理（路由至轻量级大模型）。
        根据切面数量执行不同的策略：>8则缩减，2~8则扩充，其余保持原样。
        
        :param query: 当前查询问题
        :param facets: 原始切面列表
        :param task_id_label: 任务ID标签，用于日志追踪
        :return: 预处理后的切面列表
        """
        facets, invalid_initial = filter_valid_facets(facets)
        if invalid_initial:
            logger.warning(f"Invalid facets removed before preprocessing: {invalid_initial}")
            record_generation_audit({
                "stage": "facet_preprocess",
                "status": "invalid_removed",
                "task_id": task_id_label,
                "query": query,
                "invalid_facets": invalid_initial,
            })
        if len(facets) < 2:
            logger.critical(f"Facet preprocessing aborted: fewer than 2 valid facets for query '{query}'.")
            return []

        count = len(facets)
        stage_prefix = f"[{task_id_label}] " if task_id_label else ""
        
        from pipeline import parse_json_safely
        if count > 8:
            # 切面数量大于8，执行缩减操作
            logger.info(f"Facet count {count} > 8. Running Facet Reducer...")
            prompt = PromptRenderer.render(prompts.FACET_REDUCER_TEMPLATE, query=query, facets=facets)
            response = await self.llm_service.call_llm(prompt, model_pool="lightweight", stage=f"{stage_prefix}切面视角缩减过滤")
            reduced_facets = parse_json_safely(response, [])
            reduced_facets, invalid_reduced = filter_valid_facets(reduced_facets)
            if invalid_reduced:
                logger.warning(f"Invalid facets removed from reducer output: {invalid_reduced}")
                record_generation_audit({
                    "stage": "facet_reducer",
                    "status": "invalid_removed",
                    "task_id": task_id_label,
                    "query": query,
                    "invalid_facets": invalid_reduced,
                })
            # 如果缩减成功且为8个则返回，否则强行截断前8个
            if len(reduced_facets) == 8:
                return reduced_facets
            else:
                return facets[:8]
        elif 2 < count < 8:
            # 切面数量在2到8之间，执行扩充操作
            logger.info(f"Facet count {count} is between 2 and 8. Running Facet Expander...")
            prompt = PromptRenderer.render(prompts.FACET_EXPANDER_TEMPLATE, query=query, facets=facets)
            response = await self.llm_service.call_llm(prompt, model_pool="lightweight", stage=f"{stage_prefix}切面视角丰富扩充")
            expanded_new = parse_json_safely(response, [])
            expanded_new, invalid_expanded = filter_valid_facets(expanded_new)
            if invalid_expanded:
                logger.warning(f"Invalid facets removed from expander output: {invalid_expanded}")
                record_generation_audit({
                    "stage": "facet_expander",
                    "status": "invalid_removed",
                    "task_id": task_id_label,
                    "query": query,
                    "invalid_facets": invalid_expanded,
                })
            
            # 合并扩充的切面，去重
            combined = list(facets)
            for f in expanded_new:
                if f not in combined:
                    combined.append(f)
            return combined
        else:
            # 切面数量为2，直接返回两个最贴切且互补的切面
            return facets

    async def answer_single_facet(self, query: str, facet: str, refs: List[Dict[str, str]], semaphore: asyncio.Semaphore, simplify: bool = False, boundary_refs: List[Dict[str, str]] = None, task_id_label: str = "") -> Tuple[str, str]:
        """
        为单个视角（切面）调用图问答智能体进行深度问答。
        核心生成步骤：必须路由至高级模型执行。
        包含质量门控检查和降级容错机制。
        
        :param query: 当前查询问题
        :param facet: 当前切面名称
        :param refs: 参考文献列表
        :param semaphore: 异步并发信号量，控制并发数
        :param simplify: 是否启用极简推理模式
        :param boundary_refs: 边界限制的参考文献列表
        :param task_id_label: 任务ID标签，用于日志追踪
        :return: 元组，包含(切面名称, 生成的回答内容)，若质量把控失败则回答内容为None
        """
        from strategies.quality_gate.answer_guard import check_answer_quality

        is_valid_facet, invalid_reason = validate_facet_label(facet)
        if not is_valid_facet:
            logger.critical(f"Facet hard gate rejected facet '{facet}' before answer generation: {invalid_reason}")
            record_generation_audit({
                "stage": "answer_facet_gate",
                "status": "rejected",
                "task_id": task_id_label,
                "query": query,
                "facet": facet,
                "reason": invalid_reason,
            })
            return facet, None
        
        async with semaphore:
            # 构造系统提示词 and 多层级的用户提示词
            system_prompt = PromptRenderer.get_l1_meta()
            task_prompt = PromptRenderer.get_l2_execution(facet)
            context_prompt = PromptRenderer.get_l3_context(query, refs, [])
            user_prompt = f"{task_prompt}\n\n{context_prompt}"
            
            if simplify:
                user_prompt += "\n\n【⚠️ 极简推理特别指令】：当前问题属于简单事实查询。你必须极度简化推理和回答。在 evidences、reasoning_chains 和 answer_body 中，严禁脑补复杂的生化机制、受体或分子通路，只列出直接相关的临床证据，进行 1-2 步极简因果推导即可。"
                
            if boundary_refs:
                boundary_texts = "\n".join([f"- [{r.get('source')}] {r.get('context')}" for r in boundary_refs])
                user_prompt += f"\n\n【⚠️ 边界限制事实对齐边界】：\n{boundary_texts}\n【⚠️ 边界限制硬性约束】：上述限制事实仅限在思考过程（Think）与最终答案（Answer）中各用最多一句话简要提及，严禁针对其展开长篇大论或虚构衍生逻辑分支！"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            max_quality_attempts = 2
            stage_prefix = f"[{task_id_label}] " if task_id_label else ""
            
            try:
                result = None
                is_passed = False
                reason = ""
                
                # 结构化输出重试循环：最多尝试 max_quality_attempts + 1 次
                for q_attempt in range(max_quality_attempts + 1):
                    # 调用高级大模型获取结构化问答输出
                    result: FacetQAOutput = await self.llm_service.call_llm_structured(
                        messages, 
                        FacetQAOutput, 
                        model_pool="premium", 
                        stage=f"{stage_prefix}切面深度问答 - {facet}"
                    )
                    
                    # 检查回答质量
                    is_passed, reason = check_answer_quality(
                        result.answer_body, 
                        getattr(result, "_reasoning_content", ""),
                        simplify=simplify
                    )
                    if is_passed:
                        break
                    logger.warning(f"Quality Guardrail FAILED on structured QA attempt {q_attempt} for facet '{facet}': {reason}. Retrying...")
                    
                # 如果多次尝试仍未通过质量检查，抛出异常进入降级流程
                if not is_passed:
                    raise ValueError(f"Quality guardrail failed repeatedly for structured output: {reason}")
                
                # 处理推理过程内容
                reasoning_content = getattr(result, "_reasoning_content", "")
                if reasoning_content:
                    cleaned_reasoning = reasoning_content.replace("<think>", "").replace("</think>", "").strip()
                    think_block = f"<think><facet = {facet}>\n{cleaned_reasoning}\n</think>\n"
                else:
                    # 如果没有原生推理内容，则从结构化输出中提取证据、推理链等构造思考块
                    evidences_str = "\n".join([
                        f"[证据R{i+1}：来源={e.source}，定位={e.location}，要点={e.summary}]"
                        for i, e in enumerate(result.evidences)
                    ])
                    reasoning_str = "\n".join([
                        f"- {step.step_id}: {step.logic}"
                        for step in result.reasoning_chains
                    ])
                    sub_questions_str = "\n".join([f"- {q}" for q in result.sub_questions])
                    
                    think_block = (
                        f"<think><facet = {facet}>\n"
                        f"问题拆解：\n{sub_questions_str}\n"
                        f"证据清单：\n{evidences_str}\n"
                        f"推理链：\n{reasoning_str}\n"
                        f"最终结论摘要：\n- {result.final_conclusion_summary}\n"
                        f"</think>\n"
                    )
                
                # 组装最终响应：思考块 + 回答主体
                response = think_block + result.answer_body
                logger.info(f"Successfully generated structured and layered QA output for facet: {facet}")
                
            except Exception as e:
                # 结构化输出失败，触发降级方案：使用普通文本生成模式
                logger.error(f"Structured QA call failed for facet '{facet}': {e}. Triggering fallback.")
                fallback_prompt = f"{system_prompt}\n\n{user_prompt}"
                raw_response = ""
                reasoning_content = ""
                is_passed = False
                reason = ""
                
                # 降级生成重试循环
                for fb_attempt in range(max_quality_attempts + 1):
                    # 调用带推理输出的高级大模型
                    raw_response, reasoning_content = await self.llm_service.call_llm_with_reasoning(
                        fallback_prompt, 
                        model_pool="premium", 
                        stage=f"{stage_prefix}切面深度问答降级 - {facet}"
                    )
                    
                    # 提取正文部分进行质量检查
                    check_body = raw_response
                    if "</think>" in raw_response:
                        check_body = raw_response.split("</think>")[-1].strip()
                        if not reasoning_content and "<think>" in raw_response:
                            parts = raw_response.split("</think>")[0].split("<think>")
                            if len(parts) > 1:
                                reasoning_content = parts[1].strip()
                        
                    is_passed, reason = check_answer_quality(check_body, reasoning_content, simplify=simplify)
                    if is_passed:
                        break
                    logger.warning(f"Quality Guardrail FAILED on fallback attempt {fb_attempt}: {reason}. Retrying...")
                
                # 降级后依然无法通过质量检查，丢弃该切面
                if not is_passed:
                    logger.critical(f"Quality guard failed repeatedly for facet '{facet}'. DROPPING this facet.")
                    return facet, None
                
                if "<think>" in raw_response and f"facet =" not in raw_response:
                    response = raw_response.replace("<think>", f"<think><facet = {facet}>")
                elif not raw_response.startswith("<think"):
                    if reasoning_content:
                        cleaned_reasoning = reasoning_content.replace("<think>", "").replace("</think>", "").strip()
                        response = f"<think><facet = {facet}>\n{cleaned_reasoning}\n</think>\n" + raw_response
                    else:
                        mock_think = f"<think><facet = {facet}>\n{facet}切面推理过程\n</think>\n"
                        response = mock_think + raw_response
                else:
                    response = raw_response
                    
            return facet, response

    async def run_parallel_answers(self, query: str, facets: List[str], refs: List[Dict[str, str]], task_id_label: str = "") -> List[Dict[str, str]]:
        """
        并发运行所有切面的问答智能体。
        使用信号量控制最大并发数。
        
        :param query: 当前查询问题
        :param facets: 切面列表
        :param refs: 参考文献列表
        :param task_id_label: 任务ID标签，用于日志追踪
        :return: 包含各切面及其对应回答的字典列表，如: [{"planner": 切面, "answer": 回答}, ...]
        """
        from core.governance.facet_strategy import (
            FacetGovernanceFilter, DropDirtyFacetStrategy, RenameAndRepairStrategy, RedirectToSimpleStrategy
        )
        from core.rag.evidence_scope_router import EvidenceScopeRouter

        semaphore = asyncio.Semaphore(config.CONCURRENT_QA_LIMIT)
        facets, invalid_facets = filter_valid_facets(facets)
        if invalid_facets:
            logger.critical(f"Invalid facets rejected before parallel answers: {invalid_facets}")
            record_generation_audit({
                "stage": "parallel_answer_gate",
                "status": "invalid_rejected",
                "task_id": task_id_label,
                "query": query,
                "invalid_facets": invalid_facets,
            })
        if len(facets) < 2:
            logger.critical(f"Parallel answer generation aborted: fewer than 2 valid facets for query '{query}'.")
            return []

        # 1. 🚦 Q-Facet 兼容过滤器治理
        gov_filter = FacetGovernanceFilter(self.llm_service)
        gov_context = {"audit_log": []}
        governed_facets = []

        for facet in facets:
            decision = await gov_filter.evaluate_compatibility(query, facet)
            
            if decision.facet_action == "DROP" or decision.compatibility == "FORCED_SKIP":
                strategy = DropDirtyFacetStrategy()
                res = await strategy.apply(query, facet, gov_context)
            elif decision.facet_action == "RENAME":
                strategy = RenameAndRepairStrategy(decision.target_facet)
                res = await strategy.apply(query, facet, gov_context)
            elif decision.facet_action == "REDIRECT_SIMPLE" or decision.compatibility == "COMPATIBLE_SIMPLE":
                strategy = RedirectToSimpleStrategy(decision.target_facet or facet)
                res = await strategy.apply(query, facet, gov_context)
            else: # KEEP
                res = {"action": "keep", "facet": facet, "simplify": False}
                
            if res["action"] != "drop" and res["facet"]:
                governed_facets.append({
                    "facet": res["facet"],
                    "simplify": res.get("simplify", False)
                })

        # 记录前置治理审计日志
        if gov_context["audit_log"]:
            logger.info(f"Facet governance audit for query '{query}': {gov_context['audit_log']}")
            for log in gov_context["audit_log"]:
                record_generation_audit({
                    "stage": "facet_governance",
                    "status": "governed",
                    "task_id": task_id_label,
                    "query": query,
                    "log": log
                })

        if len(governed_facets) < 1:
            logger.warning(f"No valid facets remain after initial governance for query '{query}'. Retrying with explicit negative examples...")
            prompt = (
                f"你是一个资深的医学多视角数据集设计专家。\n"
                f"对于主问题：'{query}'\n"
                f"我们之前规划的以下分析视角由于与问题不兼容/强套偏题已被丢弃：{facets}\n"
                f"请重新为主问题规划 2-3 个合理、严谨且与问题强契合的规范分析视角/切面。\n"
                f"只返回 JSON 数组格式，例如：[\"视角1\", \"视角2\"]，不要输出任何其他多余字符。"
            )
            try:
                response = await self.llm_service.call_llm(prompt, model_pool="lightweight", stage=f"[{task_id_label}] 视角切面紧急重新规划" if task_id_label else "视角切面紧急重新规划")
                from pipeline import parse_json_safely
                new_facets = parse_json_safely(response, [])
                new_facets, _ = filter_valid_facets(new_facets)
                if len(new_facets) >= 1:
                    logger.info(f"Successfully replanned new facets: {new_facets} for query '{query}'")
                    # 对新的 facets 再次执行一次 evaluate_compatibility 治理
                    for facet in new_facets:
                        decision = await gov_filter.evaluate_compatibility(query, facet)
                        if decision.facet_action == "DROP" or decision.compatibility == "FORCED_SKIP":
                            strategy = DropDirtyFacetStrategy()
                            res = await strategy.apply(query, facet, gov_context)
                        elif decision.facet_action == "RENAME":
                            strategy = RenameAndRepairStrategy(decision.target_facet)
                            res = await strategy.apply(query, facet, gov_context)
                        elif decision.facet_action == "REDIRECT_SIMPLE" or decision.compatibility == "COMPATIBLE_SIMPLE":
                            strategy = RedirectToSimpleStrategy(decision.target_facet or facet)
                            res = await strategy.apply(query, facet, gov_context)
                        else: # KEEP
                            res = {"action": "keep", "facet": facet, "simplify": False}
                            
                        if res["action"] != "drop" and res["facet"]:
                            governed_facets.append({
                                "facet": res["facet"],
                                "simplify": res.get("simplify", False)
                            })
            except Exception as e:
                logger.error(f"Failed to replan facets for query '{query}': {e}")

        if len(governed_facets) < 1:
            logger.critical(f"Parallel answer generation aborted: no valid facets remain after governance for query '{query}'.")
            return []

        # 2. 🚦 证据域 RAG 检索分级路由过滤
        router = EvidenceScopeRouter()
        from core.governance.facet_strategy import classify_intent_by_rule
        intent = classify_intent_by_rule(query)
        routed_refs = router.route_references(query, intent, refs or [])
        
        # 隔离物理屏蔽与无用证据，只保留 CORE 和 BOUNDARY
        active_refs = routed_refs["CORE"] + routed_refs["BOUNDARY"]
        boundary_refs = routed_refs["BOUNDARY"]
        
        # 记录 RAG 路由审计日志
        blocked_count = len(routed_refs["BLOCKED"])
        unused_count = len(routed_refs["UNUSED"])
        if blocked_count > 0 or unused_count > 0:
            log_msg = f"EvidenceScopeRouter: filtered out {blocked_count} blocked and {unused_count} unused refs."
            logger.info(log_msg)
            record_generation_audit({
                "stage": "evidence_scope_routing",
                "status": "routed",
                "task_id": task_id_label,
                "query": query,
                "blocked_count": blocked_count,
                "unused_count": unused_count,
                "log": log_msg
            })

        # 创建并发任务列表，分发治理后属性及过滤后 refs
        tasks = [
            self.answer_single_facet(
                query, 
                f_info["facet"], 
                active_refs, 
                semaphore, 
                simplify=f_info["simplify"], 
                boundary_refs=boundary_refs,
                task_id_label=task_id_label
            ) 
            for f_info in governed_facets
        ]
        # 并发执行所有任务
        results = await asyncio.gather(*tasks)
        
        planners = []
        # 过滤掉被丢弃(返回None)的切面，组装结果
        for facet, answer in results:
            if answer is None:
                continue
            planners.append({
                "planner": facet,
                "answer": answer
            })
        return planners

    async def synthesize_answers(self, query: str, planners: List[Dict[str, str]], task_id_label: str = "") -> str:
        """
        将过滤后的各切面回答综合凝练为单一的连贯最终摘要回答。
        路由至轻量级大模型执行。
        
        :param query: 当前查询问题
        :param planners: 包含各切面回答的字典列表
        :param task_id_label: 任务ID标签，用于日志追踪
        :return: 综合凝练后的最终摘要字符串
        """
        answers_clean = []
        # 提取回答正文，去除思考块内容
        for p in planners:
            ans = p["answer"]
            if "usse" in ans:
                parts = ans.split("usse")
                ans_body = parts[-1].strip()
            else:
                ans_body = ans.strip()
            answers_clean.append(ans_body)
            
        # 渲染综合凝练提示词并调用轻量级大模型
        prompt = PromptRenderer.render(prompts.MULTI_ANSWER_SYNTHESIS_TEMPLATE, query=query, answers=answers_clean)
        stage_prefix = f"[{task_id_label}] " if task_id_label else ""
        summary = await self.llm_service.call_llm(prompt, model_pool="lightweight", stage=f"{stage_prefix}切面问答综合凝练")
        logger.info("Successfully synthesized final answer summary.")
        return summary

    async def generate_next_question(self, context_list: List[Dict[str, str]], history: List[Dict[str, Any]], previous_summary: str, task_id_label: str = "") -> str:
        """
        在多轮对话中生成下一个问题。
        路由至轻量级大模型执行。
        
        :param context_list: 上下文信息列表
        :param history: 历史对话记录
        :param previous_summary: 上一轮的摘要回答
        :param task_id_label: 任务ID标签，用于日志追踪
        :return: 生成的下一轮问题字符串
        """
        # 渲染下一问生成提示词
        prompt = PromptRenderer.render(
            prompts.NEXT_QUESTION_TEMPLATE,
            context_list=context_list,
            history=history,
            summary=previous_summary
        )
        stage_prefix = f"[{task_id_label}] " if task_id_label else ""
        # 调用轻量级大模型
        next_q = await self.llm_service.call_llm(prompt, model_pool="lightweight", stage=f"{stage_prefix}多轮对话下一问生成")
        # 清理多余的引号和空格
        next_q = next_q.strip().strip('"').strip("'")
        return next_q

    async def generate_single_round(
        self, 
        query: str, 
        refs: List[Dict[str, str]], 
        history: List[Dict[str, Any]] = None,
        task_id_label: str = ""
    ) -> Dict[str, Any]:
        """
        执行一次完整的单轮问答生成流程。
        包含：规划切面 -> 预处理切面 -> 并发生成回答 -> 冗余过滤 -> 综合凝练。
        
        :param query: 当前查询问题
        :param refs: 参考文献列表
        :param history: 历史对话记录，默认为None
        :param task_id_label: 任务ID标签，用于日志追踪
        :return: 单轮对话数据字典，包含 Q(问题), planners(切面回答), history(历史), summary(摘要)
        """
        logger.info(f"--- Starting Round QA for Q: '{query}' ---")
        
        # 1. 规划初始切面
        initial_facets = await self.plan_facets(query, task_id_label=task_id_label)
        if not initial_facets:
            raise SampleQuarantineException(f"Facet planning failed validation; sample quarantined for query: {query}")
            
        # 2. 预处理切面（缩减或扩充）
        final_facets = await self.preprocess_facets(query, initial_facets, task_id_label=task_id_label)
        if len(final_facets) < 2:
            raise SampleQuarantineException(f"Facet preprocessing produced fewer than 2 valid facets; sample quarantined for query: {query}")

        # 3. 并发生成各切面回答
        planners = await self.run_parallel_answers(query, final_facets, refs, task_id_label=task_id_label)
        if not planners:
            raise SampleQuarantineException(f"No valid facet answers generated; sample quarantined for query: {query}")
        
        # 4. 使用策略模式执行冗余过滤
        non_redundant_planners = await self.redundancy_filter.filter_redundancy(
            query, 
            planners, 
            task_id_label=task_id_label
        )
        if not non_redundant_planners:
            non_redundant_planners = planners
            
        # 5. 综合凝练最终摘要
        summary = await self.synthesize_answers(query, non_redundant_planners, task_id_label=task_id_label)
        
        # 组装单轮结果数据
        round_data = {
            "Q": query,
            "planners": non_redundant_planners,
            "history": list(history) if history else [],
            "summary": summary
        }
        return round_data

    async def generate_multi_round_dataset(self, intent: Dict[str, Any] = None, task_id_label: str = "") -> Dict[str, Any]:
        """
        编排基于 Graph-RAG 意图引导的稳健对话生成流水线。
        负责多轮对话的整体调度，包括意图注入、图谱获取、上下文准备及多轮迭代生成。
        
        :param intent: 意图字典，包含主题、种子实体和关系，用于引导生成。若为None则自动从图谱提取
        :param task_id_label: 任务ID标签，用于日志追踪
        :return: 最后一轮的对话数据字典，并附加上完整的参考文献(refs)
        """
        log_prefix = f"[{task_id_label}] " if task_id_label else ""
        num_rounds = int(config.NUM_ROUNDS) if hasattr(config, "NUM_ROUNDS") else 1
        logger.info(f"{log_prefix}=== Starting Multi-Round Generation: {num_rounds} Rounds ===")
        
        # 1. 获取随机知识图谱数据
        try:
            graph_data = await self.graph_service.fetch_random_knowledge_graph(count=1)
        except Exception as e:
            logger.critical(f"{log_prefix}Failed to fetch random KG: {e}")
            graph_data = {"entities": [], "relationships": []}
            
        # 保证图谱数据结构完整
        if "entities" not in graph_data or not graph_data["entities"]:
            graph_data["entities"] = []
        if "relationships" not in graph_data or not graph_data["relationships"]:
            graph_data["relationships"] = []
            
        # 2. 确定对话主题，并根据意图注入种子数据
        if intent is not None:
            # 存在外部意图，使用意图中的主题，并注入种子实体和关系到图谱数据中
            selected_theme = intent.get("theme", "临床医学研究")
            for seed in intent.get("seeds", []):
                if not any(e.get("name") == seed["name"] for e in graph_data["entities"]):
                    graph_data["entities"].append({
                        "id": random.randint(10000, 99999),
                        "name": seed["name"],
                        "type": seed["type"],
                        "description": seed["description"]
                    })
            for rel in intent.get("rels", []):
                if not any(r.get("sourceName") == rel["sourceName"] and r.get("targetName") == rel["targetName"] for r in graph_data["relationships"]):
                    graph_data["relationships"].append(rel)
        else:
            # 无外部意图，根据图谱现有的关系或实体自动生成主题
            if graph_data.get("relationships"):
                rel = random.choice(graph_data["relationships"])
                src = rel.get("sourceName", "")
                tgt = rel.get("targetName", "")
                relation = rel.get("relationship", "")
                selected_theme = f"{src}与{tgt}的{relation}临床诊疗规范与医学循证"
            elif graph_data.get("entities"):
                names = [e.get("name") for e in graph_data["entities"][:2]]
                selected_theme = "与".join(names) + "的临床应用研究与用药指南"
            else:
                selected_theme = "常见疾病的循证医学用药指南"
                
        logger.info(f"{log_prefix}--- Intention-Guided Graph-RAG Theme: '{selected_theme}' ---")
        
        # 3. 准备上下文和参考引用，生成初始问题
        context_list, refs = await self._prepare_context_and_refs(graph_data, query=selected_theme)
        q1 = await self.generate_initial_question(context_list, task_id_label=task_id_label)
        
        history = []
        current_q = q1
        
        # 4. 多轮迭代生成
        for r in range(1, num_rounds + 1):
            logger.info(f"{log_prefix}=== Running Round {r} / {num_rounds} ===")
            # 执行当前轮次的单轮生成
            round_result = await self.generate_single_round(current_q, refs, history, task_id_label=task_id_label)
            history.append(round_result)
            
            # 如果不是最后一轮，准备下一轮的问题
            if r < num_rounds:
                # 30%的概率扩展对话视野：额外获取新的图谱数据补充进上下文
                if random.random() < 0.3:
                    try:
                        logger.info(f"{log_prefix}Expanding dialog horizon...")
                        additional_graph = await self.graph_service.fetch_random_knowledge_graph(count=1)
                        new_context, new_refs = await self._prepare_context_and_refs(additional_graph, query=current_q)
                        context_list.extend(new_context)
                        refs.extend(new_refs)
                    except Exception as e:
                        logger.warning(f"Failed to fetch extra entities: {e}")
                        
                # 生成下一轮问题
                current_q = await self.generate_next_question(context_list, history, round_result["summary"], task_id_label=task_id_label)
                
        logger.info(f"{log_prefix}=== Multi-Round Dialog Generation Completed! ===")
        # 将参考文献附加到最后一轮的结果中并返回
        history[-1]["refs"] = refs
        return history[-1]
