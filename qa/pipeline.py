# -*- coding: utf-8 -*-
import asyncio
import json
import logging
from typing import List, Dict, Any, Tuple
from api_client import APIClient
from core.pipeline_workflow import PipelineWorkflow
from strategies.redundancy_filter.llm_filter import LLMRedundancyFilterStrategy

logger = logging.getLogger("MedicalQA.PipelineProxy")

# Re-export parse functions to keep compatibility
def extract_json_block(text: str) -> str:
    """
    从文本中提取 JSON 代码块或 JSON 对象/数组字符串。

    该函数首先尝试匹配 Markdown 格式的 JSON 代码块（```json ... ```）。
    如果未找到代码块，则查找文本中第一个 '{' 或 '[' 的位置，并匹配对应的结束符号 '}' 或 ']'，
    以提取最外层的 JSON 结构。

    Args:
        text (str): 包含潜在 JSON 数据的原始文本字符串。

    Returns:
        str: 提取出的纯 JSON 字符串。如果无法提取有效结构，则返回去除首尾空白后的原始文本。
    """
    from api_client import APIClient
    # Fallback import locally if needed
    import re
    text = text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    first_brace = text.find('{')
    first_bracket = text.find('[')
    start = -1
    end = -1
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start = first_brace
        end = text.rfind('}')
    elif first_bracket != -1:
        start = first_bracket
        end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text


def repair_truncated_json(text: str) -> str:
    """
    Repair common LLM JSON tail truncation, such as a missing final `}` after
    an otherwise complete object. This is intentionally narrow: it only balances
    brackets/braces outside quoted strings and removes trailing commas.
    """
    import re

    repaired = (text or "").strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    stack = []
    in_string = False
    escape = False
    pairs = {"{": "}", "[": "]"}
    closers = set(pairs.values())

    for ch in repaired:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in closers:
            if stack and ch == stack[-1]:
                stack.pop()
            else:
                return repaired

    if in_string:
        return repaired
    return repaired + "".join(reversed(stack))


def extract_questions_fallback(text: str) -> Any:
    """
    Salvage a valid `"questions": [...]` array when surrounding JSON is broken,
    commonly because a discarded `think` field contains unescaped quotes.
    """
    import re

    raw = text or ""
    match = re.search(r'"questions"\s*:\s*\[', raw)
    if not match:
        return None

    start = raw.find("[", match.start())
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                array_text = raw[start:idx + 1]
                try:
                    questions = json.loads(array_text)
                except json.JSONDecodeError:
                    return None
                if isinstance(questions, list):
                    return {"questions": questions}
                return None
    return None

def parse_json_safely(text: str, default_value: Any = None) -> Any:
    """
    安全地解析 JSON 字符串。

    首先使用 extract_json_block 清理文本，然后尝试将其解析为 Python 对象。
    如果解析失败，记录错误日志并返回默认值。

    Args:
        text (str): 待解析的 JSON 字符串。
        default_value (Any, optional): 解析失败时返回的默认值。默认为 None。

    Returns:
        Any: 解析后的 Python 对象（如 dict, list 等），或在发生 JSONDecodeError 时返回 default_value。
    """
    clean_text = extract_json_block(text)
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        repaired_text = repair_truncated_json(clean_text)
        if repaired_text != clean_text:
            try:
                return json.loads(repaired_text)
            except json.JSONDecodeError:
                pass
        questions_payload = extract_questions_fallback(text)
        if questions_payload is not None:
            return questions_payload
        logger.error(f"JSON parsing error: {e}. Raw text was:\n{text}")
        return default_value

class MedicalQAPipeline:
    """
    Surgically elegant backward-compatible Proxy Pipeline representing the 
    legacy MedicalQAPipeline interface, routing execution to modularized Workflow.
    """
    def __init__(self, api_client: APIClient):
        """
        初始化 MedicalQAPipeline 代理实例。

        构建底层的 PipelineWorkflow 实例，注入必要的服务依赖（LLM 服务、图谱服务）
        以及冗余过滤策略。

        Args:
            api_client (APIClient): API 客户端实例，提供 LLM 和 Graph 服务接口。
        """
        self.api_client = api_client
        if api_client is not None:
            self.workflow = PipelineWorkflow(
                llm_service=api_client.llm_service,
                graph_service=api_client.graph_service,
                redundancy_filter=LLMRedundancyFilterStrategy(api_client.llm_service)
            )

    async def _prepare_context_and_refs(self, graph_data: Dict[str, Any], query: str = "") -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        准备上下文信息和参考文献列表。

        基于图谱数据和用户查询，检索相关的背景知识和引用源。

        Args:
            graph_data (Dict[str, Any]): 图谱数据字典，包含节点和关系信息。
            query (str, optional): 用户查询字符串。默认为空字符串。

        Returns:
            Tuple[List[Dict[str, str]], List[Dict[str, str]]]: 
                一个元组，包含两个列表：
                1. 上下文列表 (context_list)，每个元素为包含文本信息的字典。
                2. 参考文献列表 (refs)，每个元素为包含引用元数据的字典。
        """
        return await self.workflow._prepare_context_and_refs(graph_data, query)

    async def generate_initial_question(self, context_list: List[Dict[str, str]], task_id_label: str = "") -> str:
        """
        根据提供的上下文生成初始问题或澄清性问题。

        Args:
            context_list (List[Dict[str, str]]): 上下文信息列表。
            task_id_label (str, optional): 任务标识标签，用于日志追踪。默认为空字符串。

        Returns:
            str: 生成的初始问题字符串。
        """
        return await self.workflow.generate_initial_question(context_list, task_id_label)

    async def plan_facets(self, query: str, task_id_label: str = "") -> List[str]:
        """
        规划查询的各个 facet（方面/子问题）。

        将复杂的用户查询分解为多个独立的、可并行处理的子任务或方面。

        Args:
            query (str): 用户原始查询字符串。
            task_id_label (str, optional): 任务标识标签，用于日志追踪。默认为空字符串。

        Returns:
            List[str]: 规划出的 facet 列表，每个元素为一个子问题或方面描述。
        """
        return await self.workflow.plan_facets(query, task_id_label)

    async def preprocess_facets(self, query: str, facets: List[str], task_id_label: str = "") -> List[str]:
        """
        预处理规划好的 facets。

        对生成的 facet 列表进行优化、去重或格式化，以确保后续处理的有效性。

        Args:
            query (str): 用户原始查询字符串。
            facets (List[str]): 原始 facet 列表。
            task_id_label (str, optional): 任务标识标签，用于日志追踪。默认为空字符串。

        Returns:
            List[str]: 预处理后的 facet 列表。
        """
        return await self.workflow.preprocess_facets(query, facets, task_id_label)

    async def answer_single_facet(self, query: str, facet: str, refs: List[Dict[str, str]], semaphore: asyncio.Semaphore, task_id_label: str = "") -> Tuple[str, str]:
        """
        针对单个 facet 生成答案。

        结合原始查询、特定 facet 和参考文献，调用 LLM 生成局部答案。
        使用信号量控制并发请求。

        Args:
            query (str): 用户原始查询字符串。
            facet (str): 当前处理的 facet（子问题）。
            refs (List[Dict[str, str]]): 相关的参考文献列表。
            semaphore (asyncio.Semaphore): 异步信号量，用于限制并发数量。
            task_id_label (str, optional): 任务标识标签，用于日志追踪。默认为空字符串。

        Returns:
            Tuple[str, str]: 一个元组，包含：
                1. 生成的答案内容。
                2. 推理过程或中间状态信息。
        """
        return await self.workflow.answer_single_facet(
            query,
            facet,
            refs,
            semaphore,
            task_id_label=task_id_label,
        )

    async def run_parallel_answers(self, query: str, facets: List[str], refs: List[Dict[str, str]], task_id_label: str = "") -> List[Dict[str, str]]:
        """
        并行执行所有 facet 的答案生成。

        并发处理多个 facet，收集每个 facet 的答案结果。

        Args:
            query (str): 用户原始查询字符串。
            facets (List[str]): 需要处理的 facet 列表。
            refs (List[Dict[str, str]]): 相关的参考文献列表。
            task_id_label (str, optional): 任务标识标签，用于日志追踪。默认为空字符串。

        Returns:
            List[Dict[str, str]]: 包含每个 facet 答案结果的字典列表。
        """
        planners, _ = await self.workflow.run_parallel_answers(query, facets, refs, task_id_label)
        return planners

    async def synthesize_answers(self, query: str, planners: List[Dict[str, str]], task_id_label: str = "") -> str:
        """
        综合所有 facet 的答案生成最终回复。

        将并行生成的各个 facet 答案整合成一个连贯、完整的最终回答。

        Args:
            query (str): 用户原始查询字符串。
            planners (List[Dict[str, str]]): 包含各 facet 答案及规划信息的列表。
            task_id_label (str, optional): 任务标识标签，用于日志追踪。默认为空字符串。

        Returns:
            str: 综合后的最终答案字符串。
        """
        return await self.workflow.synthesize_answers(query, planners, task_id_label)

    async def generate_next_question(self, context_list: List[Dict[str, str]], history: List[Dict[str, Any]], previous_summary: str, task_id_label: str = "") -> str:
        """
        基于对话历史和上下文生成下一个追问或澄清问题。

        用于多轮对话场景，根据之前的交互总结当前状态并决定下一步提问。

        Args:
            context_list (List[Dict[str, str]]): 当前上下文信息列表。
            history (List[Dict[str, Any]]): 历史对话记录列表。
            previous_summary (str): 上一轮对话的摘要。
            task_id_label (str, optional): 任务标识标签，用于日志追踪。默认为空字符串。

        Returns:
            str: 生成的下一个问题字符串。
        """
        return await self.workflow.generate_next_question(context_list, history, previous_summary, task_id_label)

    async def generate_single_round(
        self, 
        query: str, 
        refs: List[Dict[str, str]], 
        history: List[Dict[str, Any]] = None,
        task_id_label: str = ""
    ) -> Dict[str, Any]:
        """
        执行单轮问答生成流程。

        整合上下文准备、facet 规划、并行回答和答案综合步骤，完成一次完整的问答交互。

        Args:
            query (str): 用户查询字符串。
            refs (List[Dict[str, str]]): 参考文献列表。
            history (List[Dict[str, Any]], optional): 历史对话记录。默认为 None。
            task_id_label (str, optional): 任务标识标签，用于日志追踪。默认为空字符串。

        Returns:
            Dict[str, Any]: 包含最终答案及相关元数据的字典。
        """
        return await self.workflow.generate_single_round(query, refs, history, task_id_label)

    async def generate_multi_round_dataset(self, intent: Dict[str, Any] = None, task_id_label: str = "") -> Dict[str, Any]:
        """
        生成多轮对话数据集。

        模拟或实际执行多轮交互，生成用于训练或评估的多轮对话数据。

        Args:
            intent (Dict[str, Any], optional): 用户意图字典，用于引导对话生成。默认为 None。
            task_id_label (str, optional): 任务标识标签，用于日志追踪。默认为空字符串。

        Returns:
            Dict[str, Any]: 包含多轮对话数据集的字典。
        """
        return await self.workflow.generate_multi_round_dataset(intent, task_id_label)

    def _check_answer_quality(self, answer_body: str, reasoning_content: str = "") -> Tuple[bool, str]:
        """
        Runs quality guardrails to filter out refusals and prompt pollutions (backward compatible wrapper).
        
        检查答案质量，过滤拒绝回答和提示词污染。

        Args:
            answer_body (str): 生成的答案正文。
            reasoning_content (str, optional): 推理过程内容。默认为空字符串。

        Returns:
            Tuple[bool, str]: 一个元组，包含：
                1. 布尔值，表示答案是否通过质量检查。
                2. 字符串，表示检查结果详情或错误信息。
        """
        from strategies.quality_gate.answer_guard import check_answer_quality
        return check_answer_quality(answer_body, reasoning_content)
