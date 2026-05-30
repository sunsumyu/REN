# -*- coding: utf-8 -*-
import asyncio
import httpx
import logging
from typing import List, Dict, Any, Tuple
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MedicalQA.APIClientProxy")

# Re-export visual printing from utils
from utils.visual_printer import print_token_usage
from services.llm_service import LLMService
from services.graph_service import GraphService

class APIClient:
    """
    Surgically elegant backward-compatible Proxy Client representing the 
    legacy APIClient interface, internally routing to modularized Services.
    """
    def __init__(self):
        # Instantiate modularized httpx client and Semaphore locally to preserve compatibility
        self.httpx_client = httpx.AsyncClient(timeout=60.0)
        self.global_semaphore = asyncio.Semaphore(config.GLOBAL_API_SEMAPHORE)
        
        # Instantiate services via Dependency Injection
        self.llm_service = LLMService(self.httpx_client, self.global_semaphore)
        self.graph_service = GraphService(self.llm_service)
        
        # Maintain compatibility state variables
        self.seen_entity_ids = self.graph_service.seen_entity_ids

    @property
    def supported_models(self) -> List[str]:
        return self.llm_service.supported_models

    @supported_models.setter
    def supported_models(self, val: List[str]):
        self.llm_service.supported_models = val

    async def close(self):
        await self.httpx_client.aclose()

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        return await self.llm_service._request_with_retry(method, url, **kwargs)

    async def fetch_random_knowledge_graph(
        self, 
        count: int = config.DEFAULT_ENTITY_COUNT,
        kb_id: int = config.DEFAULT_KNOWLEDGE_BASE_ID,
        hop_count: int = config.DEFAULT_HOP_COUNT
    ) -> Dict[str, Any]:
        return await self.graph_service.fetch_random_knowledge_graph(count, kb_id, hop_count)

    async def init_supported_models(self):
        await self.llm_service.init_supported_models()

    def _resolve_model(self, model_pool: str, is_structured: bool = False) -> str:
        return self.llm_service._resolve_model(model_pool, is_structured)

    async def call_llm(self, prompt: str, system_prompt: str = "", model_pool: str = "premium", stage: str = "") -> str:
        return await self.llm_service.call_llm(prompt, system_prompt, model_pool, stage)

    async def call_llm_with_reasoning(self, prompt: str, system_prompt: str = "", model_pool: str = "premium", stage: str = "") -> Tuple[str, str]:
        return await self.llm_service.call_llm_with_reasoning(prompt, system_prompt, model_pool, stage)

    async def call_llm_structured(self, messages: List[Dict[str, str]], response_model: type, model_pool: str = "premium", stage: str = "") -> Any:
        return await self.llm_service.call_llm_structured(messages, response_model, model_pool, stage)
