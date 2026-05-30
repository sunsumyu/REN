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

def parse_json_safely(text: str, default_value: Any = None) -> Any:
    clean_text = extract_json_block(text)
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}. Raw text was:\n{text}")
        return default_value

class MedicalQAPipeline:
    """
    Surgically elegant backward-compatible Proxy Pipeline representing the 
    legacy MedicalQAPipeline interface, routing execution to modularized Workflow.
    """
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        if api_client is not None:
            self.workflow = PipelineWorkflow(
                llm_service=api_client.llm_service,
                graph_service=api_client.graph_service,
                redundancy_filter=LLMRedundancyFilterStrategy(api_client.llm_service)
            )

    async def _prepare_context_and_refs(self, graph_data: Dict[str, Any], query: str = "") -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        return await self.workflow._prepare_context_and_refs(graph_data, query)

    async def generate_initial_question(self, context_list: List[Dict[str, str]], task_id_label: str = "") -> str:
        return await self.workflow.generate_initial_question(context_list, task_id_label)

    async def plan_facets(self, query: str, task_id_label: str = "") -> List[str]:
        return await self.workflow.plan_facets(query, task_id_label)

    async def preprocess_facets(self, query: str, facets: List[str], task_id_label: str = "") -> List[str]:
        return await self.workflow.preprocess_facets(query, facets, task_id_label)

    async def answer_single_facet(self, query: str, facet: str, refs: List[Dict[str, str]], semaphore: asyncio.Semaphore, task_id_label: str = "") -> Tuple[str, str]:
        return await self.workflow.answer_single_facet(query, facet, refs, semaphore, task_id_label)

    async def run_parallel_answers(self, query: str, facets: List[str], refs: List[Dict[str, str]], task_id_label: str = "") -> List[Dict[str, str]]:
        return await self.workflow.run_parallel_answers(query, facets, refs, task_id_label)

    async def synthesize_answers(self, query: str, planners: List[Dict[str, str]], task_id_label: str = "") -> str:
        return await self.workflow.synthesize_answers(query, planners, task_id_label)

    async def generate_next_question(self, context_list: List[Dict[str, str]], history: List[Dict[str, Any]], previous_summary: str, task_id_label: str = "") -> str:
        return await self.workflow.generate_next_question(context_list, history, previous_summary, task_id_label)

    async def generate_single_round(
        self, 
        query: str, 
        refs: List[Dict[str, str]], 
        history: List[Dict[str, Any]] = None,
        task_id_label: str = ""
    ) -> Dict[str, Any]:
        return await self.workflow.generate_single_round(query, refs, history, task_id_label)

    async def generate_multi_round_dataset(self, intent: Dict[str, Any] = None, task_id_label: str = "") -> Dict[str, Any]:
        return await self.workflow.generate_multi_round_dataset(intent, task_id_label)

    def _check_answer_quality(self, answer_body: str, reasoning_content: str = "") -> Tuple[bool, str]:
        """
        Runs quality guardrails to filter out refusals and prompt pollutions (backward compatible wrapper).
        """
        from strategies.quality_gate.answer_guard import check_answer_quality
        return check_answer_quality(answer_body, reasoning_content)

