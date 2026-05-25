import asyncio
import httpx
import logging
import random
from typing import List, Dict, Any, Set
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self):
        self.seen_entity_ids: Set[str] = set()
        # Using httpx.AsyncClient for high performance asynchronous calls
        self.client = httpx.AsyncClient(timeout=60.0)

    async def close(self):
        await self.client.aclose()

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """
        Execute an HTTP request with exponential backoff and random jitter retry.
        """
        retries = 0
        backoff = config.RETRY_BACKOFF_FACTOR
        
        while True:
            try:
                response = await self.client.request(method, url, **kwargs)
                
                # Check for rate limit or server errors
                if response.status_code == 429:
                    logger.warning(f"Rate limited (429) on {url}. Retrying...")
                elif response.status_code >= 500:
                    logger.warning(f"Server error ({response.status_code}) on {url}. Retrying...")
                elif 400 <= response.status_code < 500:
                    error_msg = f"API Client Error ({response.status_code}) on {url}: {response.text}"
                    logger.critical(error_msg)
                    raise Exception(error_msg)  # 致命错误，直接抛出，中断重试循环
                else:
                    response.raise_for_status()
                    return response.json()
            except (httpx.HTTPError, httpx.NetworkError) as e:
                logger.error(f"HTTP/Network error on request {method} {url}: {e}")
            
            retries += 1
            if retries > config.MAX_RETRIES:
                logger.critical(f"Max retries ({config.MAX_RETRIES}) reached for {url}. Raising error.")
                raise Exception(f"Failed to fetch from {url} after {config.MAX_RETRIES} attempts.")
            
            # Exponential backoff with jitter
            sleep_time = (backoff ** retries) + random.uniform(0.1, 0.5)
            logger.info(f"Sleeping for {sleep_time:.2f} seconds before retry {retries}...")
            await asyncio.sleep(sleep_time)

    async def fetch_random_knowledge_graph(
        self, 
        count: int = config.DEFAULT_ENTITY_COUNT,
        kb_id: int = config.DEFAULT_KNOWLEDGE_BASE_ID,
        hop_count: int = config.DEFAULT_HOP_COUNT
    ) -> Dict[str, Any]:
        """
        Fetch a random medical subgraph. Automatically passes seen entity IDs to exclude them.
        """
        # Convert seen_entity_ids to comma-separated string
        exclude_ids_str = ",".join(list(self.seen_entity_ids))
        
        params = {
            "count": count,
            "knowledgeBaseId": kb_id,
            "hopCount": hop_count
        }
        if exclude_ids_str:
            params["entityIds"] = exclude_ids_str
            
        logger.info(f"Fetching random KG data (excluding {len(self.seen_entity_ids)} entities)...")
        
        # We use GET method
        response_data = await self._request_with_retry("GET", config.GRAPH_API_URL, params=params)
        
        # Verify response success
        if not response_data.get("success", False):
            raise Exception(f"Knowledge Graph API failed: {response_data.get('msg', 'Unknown error')}")
            
        graph_data = response_data.get("data", {})
        entities = graph_data.get("entities", [])
        
        # Add new entity IDs to seen registry to avoid duplicates next time
        new_entities_count = 0
        for entity in entities:
            entity_id = str(entity.get("id"))
            if entity_id not in self.seen_entity_ids:
                self.seen_entity_ids.add(entity_id)
                new_entities_count += 1
                
        logger.info(f"Successfully retrieved {len(entities)} entities (registered {new_entities_count} new entities).")
        return graph_data

    async def call_llm(self, prompt: str, system_prompt: str = "") -> str:
        """
        Call the Large Language Model completions API.
        """
        if not config.LLM_API_KEY:
            logger.warning("LLM_API_KEY is not set! Call might fail if the server requires authentication.")
            
        headers = {
            "Content-Type": "application/json"
        }
        
        # Handle Bearer token prefix if missing
        api_key = config.LLM_API_KEY.strip()
        if api_key:
            if api_key.startswith("Bearer "):
                headers["Authorization"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["Authorization"] = "Bearer dummy"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "temperature": 0.1,  # Low temperature for highly precise structured reasoning
        }
        
        logger.info(f"Calling LLM ({config.LLM_MODEL})...")
        response_data = await self._request_with_retry("POST", config.LLM_API_URL, headers=headers, json=data)
        
        # Parse standard chat completion response
        try:
            content = response_data["choices"][0]["message"]["content"]
            return content.strip()
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse LLM response format: {response_data}")
            raise Exception(f"Invalid LLM response format: {e}")
