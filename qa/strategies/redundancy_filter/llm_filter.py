# -*- coding: utf-8 -*-
import logging
from typing import List, Dict
import prompts
from services.llm_service import ILLMService

logger = logging.getLogger("MedicalQA.RedundancyFilter")

from abc import ABC, abstractmethod

class IRedundancyFilterStrategy(ABC):
    @abstractmethod
    async def filter_redundancy(self, query: str, planners: List[Dict[str, str]], task_id_label: str = "") -> List[Dict[str, str]]:
        pass

class LLMRedundancyFilterStrategy(IRedundancyFilterStrategy):
    def __init__(self, llm_service: ILLMService):
        self.llm_service = llm_service

    async def filter_redundancy(self, query: str, planners: List[Dict[str, str]], task_id_label: str = "") -> List[Dict[str, str]]:
        """
        Call Facet Redundancy Detector to filter out redundant perspective answers using lightweight LLM.
        """
        prompt = prompts.render_prompt(prompts.FACET_REDUNDANCY_DETECTOR_TEMPLATE, query=query, planners=planners)
        stage_prefix = f"[{task_id_label}] " if task_id_label else ""
        
        # We leverage the lightweight model pool for high performance and low latency
        response = await self.llm_service.call_llm(prompt, model_pool="lightweight", stage=f"{stage_prefix}视角切面冗余度检测与去重过滤")
        
        # Safely parse JSON array
        import json
        try:
            from pipeline import parse_json_safely
            indices_to_remove = parse_json_safely(response, [])
        except ImportError:
            # Fallback inline parsing
            clean_text = response.strip()
            if clean_text.startswith("```"):
                clean_text = "\n".join(clean_text.splitlines()[1:])
            if clean_text.endswith("```"):
                clean_text = "\n".join(clean_text.splitlines()[:-1])
            clean_text = clean_text.strip()
            first_bracket = clean_text.find('[')
            end_bracket = clean_text.rfind(']')
            if first_bracket != -1 and end_bracket != -1:
                clean_text = clean_text[first_bracket:end_bracket+1]
            try:
                indices_to_remove = json.loads(clean_text)
            except Exception:
                indices_to_remove = []

        if not isinstance(indices_to_remove, list):
            logger.warning(f"Redundancy detector returned invalid format: {response}. Skipping filtering.")
            return planners
            
        logger.info(f"Redundancy detector indices to remove (0-indexed): {indices_to_remove}")
        
        filtered_planners = []
        for i, p in enumerate(planners):
            if i not in indices_to_remove:
                filtered_planners.append(p)
                
        logger.info(f"Filtered planners: {len(filtered_planners)} out of {len(planners)} remaining.")
        return filtered_planners
