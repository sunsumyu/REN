# -*- coding: utf-8 -*-
import logging
from typing import Dict, Any, Set
import config

logger = logging.getLogger("MedicalQA.GraphService")

from abc import ABC, abstractmethod

class IGraphService(ABC):
    @abstractmethod
    async def fetch_random_knowledge_graph(self, count: int = config.DEFAULT_ENTITY_COUNT, kb_id: int = config.DEFAULT_KNOWLEDGE_BASE_ID, hop_count: int = config.DEFAULT_HOP_COUNT) -> Dict[str, Any]:
        pass

class GraphService(IGraphService):
    def __init__(self, llm_service):
        # We inject the LLMService (which wraps the Async HTTP client)
        self.llm_service = llm_service
        self.seen_entity_ids: Set[str] = set()

    async def fetch_random_knowledge_graph(
        self, 
        count: int = config.DEFAULT_ENTITY_COUNT,
        kb_id: int = config.DEFAULT_KNOWLEDGE_BASE_ID,
        hop_count: int = config.DEFAULT_HOP_COUNT
    ) -> Dict[str, Any]:
        """
        Fetch a random medical subgraph. Automatically passes seen entity IDs to exclude them.
        """
        exclude_ids_str = ",".join(list(self.seen_entity_ids))
        
        params = {
            "count": count,
            "knowledgeBaseId": kb_id,
            "hopCount": hop_count
        }
        if exclude_ids_str:
            params["entityIds"] = exclude_ids_str
            
        logger.info(f"Fetching random KG data (excluding {len(self.seen_entity_ids)} entities)...")
        
        response_data = await self.llm_service._request_with_retry("GET", config.GRAPH_API_URL, params=params)
        
        if not response_data.get("success", False):
            raise Exception(f"Knowledge Graph API failed: {response_data.get('msg', 'Unknown error')}")
            
        graph_data = response_data.get("data", {})
        entities = graph_data.get("entities", [])
        
        new_entities_count = 0
        for entity in entities:
            entity_id = str(entity.get("id"))
            if entity_id not in self.seen_entity_ids:
                self.seen_entity_ids.add(entity_id)
                new_entities_count += 1
                
        logger.info(f"Successfully retrieved {len(entities)} entities (registered {new_entities_count} new entities).")
        return graph_data
