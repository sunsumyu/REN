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

HARD_FACT_RETRIEVAL_PATTERNS = [
    r"推荐用法用量",
    r"用法用量是什么",
    r"主要不良反应",
    r"不良反应有哪些",
    r"什么类型.{0,8}临床试验",
    r"临床试验.{0,8}什么类型",
    r"规格是",
    r"批准文号",
    r"生产厂家",
    r"具体数值",
    r"具体剂量",
    r"总\w{0,3}组数",
    r"总\w{0,3}数量",
    r"每组\w{0,4}数",
    r"共\w{0,2}组",
]

SOFT_FACT_RETRIEVAL_PATTERNS = [
    r"是多少",
    r"有什么",
    r"有哪些",
    r"是什么",
    r"什么类型",
    r"如何分类",
    r"列举",
    r"有几",
    r"是第几",
    r"是哪\w{0,2}年",
]

CLINICAL_FRICTION_MARKERS = [
    "患者",
    "合并",
    "既往",
    "正在服用",
    "肝功能",
    "肾功能",
    "禁忌",
    "风险",
    "权衡",
    "机制",
    "因果",
    "鉴别",
    "调整",
    "特殊人群",
    "外推",
    "获益",
    "冲突",
    "边界",
    "监测",
]


def is_fact_retrieval_question(question: str) -> bool:
    """
    Detect one-hop lookup questions that are too shallow for Think CoT training.
    This gate is intentionally conservative: rejected samples are quarantined,
    not written to the dataset.
    """
    q = (question or "").strip()
    if not q:
        return True
    if any(re.search(pattern, q) for pattern in HARD_FACT_RETRIEVAL_PATTERNS):
        return True
    if not any(re.search(pattern, q) for pattern in SOFT_FACT_RETRIEVAL_PATTERNS):
        return False
    has_clinical_friction = any(marker in q for marker in CLINICAL_FRICTION_MARKERS)
    return not (has_clinical_friction and len(q) >= 45)


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
        
        # 纯数据容器型实体类型黑名单：这类实体只含实验参数，无机制信息，
        # 生成的问题几乎只能是纯事实提取题（几只动物/几组等），对 CoT 训练无价值。
        # 仍保留在 refs 供 CoT 回答时作为事实锚点引用。
        FACT_CONTAINER_ENTITY_TYPES = {"group", "result", "measurement", "statistic"}

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
                "entity_id": str(ent_id),
                "metadata": {"type": "entity", "entity_type": ent_type}
            }
            # 层次一过滤：纯数据容器型实体不加入 context_list（问题生成源），
            # 但始终加入 refs（事实锚点），保留其数据价值。
            if ent_type.lower() not in FACT_CONTAINER_ENTITY_TYPES:
                context_list.append(item)
            else:
                logger.debug(f"[实体过滤] '{name}'（type={ent_type}）被识别为纯数据容器型实体，跳过加入 context_list，仅保留在 refs。")
            refs.append({
                "context": f"概念定义: {name} (类型: {ent_type}) - {description}",
                "source": f"refs:《实体库:{name}》",
                "metadata": {"type": "entity", "entity_type": ent_type}
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
                "source": source_str,
                "metadata": {"type": "relationship", "relationship": relation}
            }
            context_list.append(item)
            refs.append({
                "context": f"知识关联: 【{src}】 --({relation})--> 【{tgt}】",
                "source": f"refs:《图谱关系:{src}-{tgt}》",
                "metadata": {"type": "relationship", "relationship": relation}
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
        # 第一道物理防线：语料字数前置拦截
        total_context_len = sum(len(c.get("context", "")) for c in context_list)
        if total_context_len < 100:
            logger.warning(f"[{task_id_label}] 第一道防线拦截：语料信息量极低 (仅 {total_context_len} 字)，直接阻断。")
            raise SampleQuarantineException(f"上下文信息量过低（{total_context_len} 字），不足以支撑复杂推理题的构建。")

        # 渲染问题生成提示词
        prompt = PromptRenderer.render(prompts.QUESTION_CREATOR_TEMPLATE, context_list=context_list)
        stage_prefix = f"[{task_id_label}] " if task_id_label else ""
        # 调用旗舰大模型（具备高指令服从度，支持源头熔断机制）
        response = await self.llm_service.call_llm(prompt, model_pool="premium", stage=f"{stage_prefix}初始问题生成")
        
        # 安全解析JSON响应
        from pipeline import parse_json_safely
        parsed_result = parse_json_safely(response, {})
        
        if isinstance(parsed_result, dict):
            questions = parsed_result.get("questions", [])
        elif isinstance(parsed_result, list):
            questions = parsed_result
        else:
            questions = []
            
        if not questions:
            # 提取图谱/路径的实体和来源诊断信息
            sources = sorted(list(set(c.get("source", "Unknown") for c in context_list)))
            entity_names = []
            for src in sources:
                # 尝试从临床路径格式中匹配病种
                match = re.search(r'临床路径-(.+?)(?:临床路径)?(?:（\d+年版）)?$', src)
                if match:
                    entity_names.append(match.group(1))
                else:
                    # 或者从一般的《xxx》格式中匹配
                    match2 = re.search(r'《(.+?)》', src)
                    if match2:
                        entity_names.append(match2.group(1))
            entity_names = sorted(list(set(entity_names))) if entity_names else ["未知病种"]

            error_msg = (
                f"Failed to generate questions from context.\n"
                f"  - Task Label: {task_id_label}\n"
                f"  - Associated Entities: {entity_names}\n"
                f"  - Sources: {sources}\n"
                f"  - Raw LLM Response: {repr(response)}\n"
                f"  - Parsed Result: {repr(parsed_result)}"
            )
            logger.error(f"{stage_prefix}{error_msg}")

            raise Exception(
                f"Failed to generate questions from context.\n"
                f"    - Associated Entities: {entity_names}\n"
                f"    - Context Sources: {sources[:3]}\n"
                f"    - Raw Response Snippet (First 150 chars): {repr(response)[:150]}"
            )

        # 层次三：正则网关——移除事实提取型问题候选，最多重试 2 次
        filtered = [q for q in questions if not is_fact_retrieval_question(q)]
        retry_count = 0
        while not filtered and retry_count < 2:
            retry_count += 1
            logger.warning(f"{stage_prefix}所有候选问题均为事实提取型，重新生成（第 {retry_count} 次重试）...")
            resp2 = await self.llm_service.call_llm(prompt, model_pool="premium", stage=f"{stage_prefix}初始问题生成(重试{retry_count})")
            parsed2 = parse_json_safely(resp2, {})
            if isinstance(parsed2, dict):
                questions2 = parsed2.get("questions", [])
            elif isinstance(parsed2, list):
                questions2 = parsed2
            else:
                questions2 = []
            filtered = [q for q in questions2 if not is_fact_retrieval_question(q)]
        if not filtered:
            logger.error(f"{stage_prefix}第三道防线拦截：连续 {retry_count+1} 次生成的候选问题均为单跳事实查询题，斩断退路。")
            raise SampleQuarantineException("连续生成的候选问题均为单跳事实查询题，系统拒绝入库。")

        filtered_out = len(questions) - len(filtered)
        if filtered_out > 0:
            logger.info(f"{stage_prefix}事实提取题过滤：移除 {filtered_out} 题，剩余 {len(filtered)} 题可用。")

        # 随机选择一个问题
        selected_q = random.choice(filtered)
        logger.info(f"Generated {len(questions)} questions (filtered to {len(filtered)}). Selected: '{selected_q}'")
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

    async def answer_single_facet(
        self,
        query: str,
        facet: str,
        refs: List[Dict[str, str]],
        semaphore: asyncio.Semaphore,
        simplify: bool = False,
        boundary_refs: List[Dict[str, str]] = None,
        evidence_contract: Dict[str, Any] = None,
        task_id_label: str = ""
    ) -> Tuple[str, str]:
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
        from core.evidence_contract import detect_forbidden_expansion, render_evidence_contract_prompt

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
            contract_prompt = render_evidence_contract_prompt(evidence_contract)
            user_prompt = f"{task_prompt}\n\n{context_prompt}{contract_prompt}"
            
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
                        structured_check_text = "\n".join([
                            result.answer_body,
                            getattr(result, "_reasoning_content", ""),
                            result.final_conclusion_summary,
                            "\n".join(e.summary for e in result.evidences),
                            "\n".join(step.logic for step in result.reasoning_chains),
                        ])
                        violations = detect_forbidden_expansion(structured_check_text, evidence_contract)
                        if violations:
                            is_passed = False
                            reason = f"evidence contract violation: {violations}"
                            record_generation_audit({
                                "stage": "answer_evidence_contract",
                                "status": "violation",
                                "task_id": task_id_label,
                                "query": query,
                                "facet": facet,
                                "attempt": q_attempt + 1,
                                "violations": violations,
                            })
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
                        violations = detect_forbidden_expansion(
                            f"{check_body}\n{reasoning_content}",
                            evidence_contract,
                        )
                        if violations:
                            is_passed = False
                            reason = f"evidence contract violation: {violations}"
                            record_generation_audit({
                                "stage": "answer_evidence_contract",
                                "status": "violation",
                                "task_id": task_id_label,
                                "query": query,
                                "facet": facet,
                                "attempt": fb_attempt + 1,
                                "fallback": True,
                                "violations": violations,
                            })
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

    async def _govern_single_facet(self, query: str, facet: str, gov_filter, gov_context: dict) -> dict:
        from core.governance.facet_strategy import DropDirtyFacetStrategy, RenameAndRepairStrategy, RedirectToSimpleStrategy
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
        return res

    async def _govern_facets_with_audit(self, query: str, facets: List[str], task_id_label: str) -> List[Dict[str, Any]]:
        from core.governance.facet_strategy import FacetGovernanceFilter
        gov_filter = FacetGovernanceFilter(self.llm_service)
        gov_context = {"audit_log": []}
        
        tasks = [self._govern_single_facet(query, facet, gov_filter, gov_context) for facet in facets]
        results = await asyncio.gather(*tasks)
        
        governed_facets = []
        for res in results:
            if res["action"] != "drop" and res["facet"]:
                governed_facets.append({
                    "facet": res["facet"],
                    "simplify": res.get("simplify", False)
                })

        if gov_context["audit_log"]:
            try:
                from utils.visual_printer import print_facet_governance_audit
                print_facet_governance_audit(query, gov_context["audit_log"], task_id_label)
            except Exception as e:
                logger.error(f"[VisualPrinter Error] Failed to print facet governance audit: {e}")
                logger.info(f"Facet governance audit for query '{query}': {gov_context['audit_log']}")
                
            for log in gov_context["audit_log"]:
                record_generation_audit({
                    "stage": "facet_governance",
                    "status": "governed",
                    "task_id": task_id_label,
                    "query": query,
                    "log": log
                })
        return governed_facets

    async def _emergency_replan_facets(self, query: str, old_facets: List[str], task_id_label: str) -> List[str]:
        import prompts
        from pipeline import parse_json_safely
        logger.warning(f"No valid facets remain after initial governance for query '{query}'. Retrying with explicit negative examples...")
        
        domain_name = getattr(config, "DOMAIN_NAME", "医学")
        prompt = prompts.render_prompt(
            prompts.EMERGENCY_REPLAN_TEMPLATE, 
            domain_name=domain_name, 
            query=query, 
            old_facets=old_facets
        )
        
        try:
            response = await self.llm_service.call_llm(prompt, model_pool="lightweight", stage=f"[{task_id_label}] 视角切面紧急重新规划" if task_id_label else "视角切面紧急重新规划")
            new_facets = parse_json_safely(response, [])
            new_facets, _ = filter_valid_facets(new_facets)
            if len(new_facets) >= 1:
                logger.info(f"Successfully replanned new facets: {new_facets} for query '{query}'")
                return new_facets
        except Exception as e:
            logger.error(f"Failed to replan facets for query '{query}': {e}")
        return []

    async def run_parallel_answers(
        self,
        query: str,
        facets: List[str],
        refs: List[Dict[str, str]],
        task_id_label: str = ""
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        并发运行所有切面的问答智能体。
        使用信号量控制最大并发数。
        """
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

        # 1. 🚦 Q-Facet 并发兼容过滤器治理
        governed_facets = await self._govern_facets_with_audit(query, facets, task_id_label)

        # 兜底重规划
        if len(governed_facets) < 1:
            new_facets = await self._emergency_replan_facets(query, facets, task_id_label)
            if new_facets:
                governed_facets = await self._govern_facets_with_audit(query, new_facets, task_id_label)

        if len(governed_facets) < 1:
            logger.critical(f"Parallel answer generation aborted: no valid facets remain after governance for query '{query}'.")
            return []

        # 2. 🚦 证据域 RAG 检索分级路由过滤
        router = EvidenceScopeRouter()
        from core.evidence_contract import build_evidence_contract
        from core.governance.facet_strategy import classify_intent_by_rule
        intent = classify_intent_by_rule(query)
        routed_refs = await router.route_references(query, intent, refs or [])
        
        # 隔离物理屏蔽与无用证据，只保留 CORE 和 BOUNDARY
        active_refs = routed_refs["CORE"] + routed_refs["BOUNDARY"]
        boundary_refs = routed_refs["BOUNDARY"]
        evidence_contract = build_evidence_contract(query, refs or [], routed_refs)
        
        # 记录 RAG 路由审计日志
        blocked_count = len(routed_refs["BLOCKED"])
        unused_count = len(routed_refs["UNUSED"])
        record_generation_audit({
            "stage": "evidence_contract",
            "status": evidence_contract.get("evidence_status", "unknown"),
            "task_id": task_id_label,
            "query": query,
            "allowed_fact_count": evidence_contract.get("allowed_fact_count", 0),
            "core_fact_count": evidence_contract.get("core_fact_count", 0),
            "boundary_fact_count": evidence_contract.get("boundary_fact_count", 0),
            "forbidden_expansions": evidence_contract.get("forbidden_expansions", []),
        })
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

        # 3. 创建并发任务列表，分发治理后属性及过滤后 refs
        tasks = [
            self.answer_single_facet(
                query, 
                f_info["facet"], 
                active_refs, 
                semaphore, 
                simplify=f_info["simplify"], 
                boundary_refs=boundary_refs,
                evidence_contract=evidence_contract,
                task_id_label=task_id_label
            ) 
            for f_info in governed_facets
        ]
        # 并发执行所有任务
        results = await asyncio.gather(*tasks)
        
        planners = []
        for facet, answer in results:
            if answer is None:
                continue
            planners.append({
                "planner": facet,
                "answer": answer
            })
        return planners, evidence_contract

    async def synthesize_answers(
        self,
        query: str,
        planners: List[Dict[str, str]],
        task_id_label: str = "",
        evidence_contract: Dict[str, Any] = None
    ) -> str:
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
        if evidence_contract:
            from core.evidence_contract import detect_forbidden_expansion, render_evidence_contract_prompt
            prompt += render_evidence_contract_prompt(evidence_contract)
        stage_prefix = f"[{task_id_label}] " if task_id_label else ""
        feedback = ""
        for attempt in range(3):
            summary = await self.llm_service.call_llm(
                prompt + feedback,
                model_pool="lightweight",
                stage=f"{stage_prefix}切面问答综合凝练 attempt-{attempt + 1}"
            )
            violations = detect_forbidden_expansion(summary, evidence_contract) if evidence_contract else []
            if not violations:
                logger.info("Successfully synthesized final answer summary.")
                return summary

            record_generation_audit({
                "stage": "summary_evidence_contract",
                "status": "violation",
                "task_id": task_id_label,
                "query": query,
                "attempt": attempt + 1,
                "violations": violations,
            })
            logger.warning(
                f"{stage_prefix}Summary evidence contract violation on attempt {attempt + 1}: {violations}"
            )
            feedback = (
                "\n\n【上一版输出违反证据契约】\n"
                f"违规项: {violations}\n"
                "请重新综合。只能保留允许事实；对于证据未提供的信息，只能说明证据不足，禁止补写具体药代、替代药或临床研究结论。\n"
            )

        raise SampleQuarantineException(f"Summary repeatedly violated evidence contract for query: {query}")

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
        feedback = ""
        for attempt in range(3):
            retry_prompt = prompt + feedback
            next_q = await self.llm_service.call_llm(
                retry_prompt,
                model_pool="lightweight",
                stage=f"{stage_prefix}多轮对话下一问生成 attempt-{attempt + 1}"
            )
            # 清理多余的引号和空格
            next_q = next_q.strip().strip('"').strip("'")
            if not is_fact_retrieval_question(next_q):
                return next_q

            logger.warning(
                f"{stage_prefix}下一轮问题命中事实提取型拦截（attempt {attempt + 1}）：{next_q}"
            )
            feedback = (
                "\n\n<previous_validation_failure>\n"
                f"上一次输出 `{next_q}` 属于单跳事实查询题，会被质量网关判定为推演复杂度不及格。"
                "请改写为带患者背景、合并症/禁忌/药代冲突或证据边界权衡的临床推理题；"
                "不要问“是什么/有哪些/用法用量/试验类型”等可直接摘录的问题。\n"
                "</previous_validation_failure>"
            )

        raise SampleQuarantineException("连续生成的下一轮问题均为单跳事实查询题，系统拒绝入库。")

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
        planners, evidence_contract = await self.run_parallel_answers(query, final_facets, refs, task_id_label=task_id_label)
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
        summary = await self.synthesize_answers(
            query,
            non_redundant_planners,
            task_id_label=task_id_label,
            evidence_contract=evidence_contract
        )
        
        # 组装单轮结果数据
        round_data = {
            "Q": query,
            "planners": non_redundant_planners,
            "history": list(history) if history else [],
            "summary": summary,
            "evidence_contract": evidence_contract
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
            logger.info(f"{log_prefix}AAA Fetched KG graph_data successfully.")
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
        
        # 3. 准备上下文和参考引用
        context_list, refs = await self._prepare_context_and_refs(graph_data, query=selected_theme)
        logger.info(f"{log_prefix}AAABPrepared context_list: {len(context_list)} items, refs: {len(refs)} items")

        # 生成初始问题
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
