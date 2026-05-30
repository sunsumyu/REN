# -*- coding: utf-8 -*-
import re
import logging
from typing import Set
from services.llm_service import ILLMService

logger = logging.getLogger("MedicalQA.HealingService")

from abc import ABC, abstractmethod

class IHealingService(ABC):
    @abstractmethod
    async def verify_and_repair_academic_entities(self, purified_text: str, q: str, facet: str, line_num: int = None) -> str:
        pass

class HealingService(IHealingService):
    def __init__(self, llm_service: ILLMService):
        self.llm_service = llm_service

    async def verify_and_repair_academic_entities(self, purified_text: str, q: str, facet: str, line_num: int = None) -> str:
        """
        Runs PubMed validation and repairs hallucinated academic abbreviations or drug-related entities.
        """
        entities: Set[str] = set(re.findall(r'\b[A-Z]+-\d+\b', purified_text))
        if not entities:
            return purified_text

        from retrieval.restricted_search import RestrictedSearchService
        search_service = RestrictedSearchService()
        repaired_text = purified_text

        HEURISTIC_REPAIR_MAP = {
            "OER-1": "OAT-1",
            "OER-3": "OAT-3",
            "OER": "OAT",
        }

        for entity in entities:
            if entity in HEURISTIC_REPAIR_MAP:
                target = HEURISTIC_REPAIR_MAP[entity]
                repaired_text = repaired_text.replace(entity, target)
                logger.warning(f"🔧 [Heuristic Repair] Mapped '{entity}' -> '{target}'")
                continue

            search_query = f'"{entity}" AND (kidney OR "renal" OR "liver" OR "pharma" OR "PBP")'
            try:
                refs = await search_service.search(query=search_query, entity_name=entity)
                pubmed_mentions = sum(1 for ref in refs if "ncbi.nlm.nih.gov" in ref.source)
                
                if pubmed_mentions == 0:
                    logger.warning(f"🚨 [PubMed Alert] Term '{entity}' has 0 NCBI mentions. Treating as hallucination.")
                    
                    repair_prompt = f"""你是一个拥有极深中医和西医底蕴的顶级医学专家。
在以下思维链中，模型产生了一个非医学/药理学常识或虚构的缩写/实体: "{entity}"。
请分析上下文并将其修改为标准、真实的医学词汇（如合并严重肾脏排泄功能受损时常发生的阴离子通道转运体错误，可修正为 "OAT-1" 或 "OAT-3"；如果是青霉素结合蛋白，可修正为 "PBP"；如果是多酚氧化酶或类似转运蛋白，可修正为 "OAT"等）。

原问题: {q}
切面视角: {facet}
原思考链: 
\"\"\"
{purified_text}
\"\"\"

请输出修正并提纯后的完整思考链，不要带有任何 <think> 标记，也不要有任何 Markdown 围栏或解释。"""
                    
                    stage_prefix = f"[{line_num}行] " if line_num else ""
                    corrected = await self.llm_service.call_llm(repair_prompt, model_pool="premium", stage=f"{stage_prefix}学术实体自愈 - {entity}")
                    repaired_text = corrected.replace("<think>", "").replace("</think>", "").strip()
                    logger.info(f"✨ [AI Healed] Successfully repaired hallucinated term '{entity}' in thought trace.")
                    break
                    
            except Exception as e:
                logger.error(f"⚠️ Error verifying entity '{entity}': {e}. Skipping.")
                
        return repaired_text
