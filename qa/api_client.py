import asyncio
import httpx
import logging
import random
from typing import List, Dict, Any, Set
from pydantic import BaseModel
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
                    res_json = response.json()
                    if isinstance(res_json, dict) and "error" in res_json and res_json["error"]:
                        err_detail = res_json["error"]
                        err_msg = err_detail.get("message", "Unknown API error")
                        err_code = err_detail.get("code", "")
                        logger.warning(f"API gateway returned error payload inside HTTP 200: {err_msg} (code: {err_code})")
                        raise httpx.HTTPStatusError(
                            message=f"API payload error: {err_msg} (code: {err_code})",
                            request=response.request,
                            response=response
                        )
                    return res_json
            except (httpx.HTTPStatusError, httpx.HTTPError, httpx.NetworkError) as e:
                logger.error(f"HTTP/API error on request {method} {url}: {e}")
            
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

    async def call_llm_structured(self, messages: List[Dict[str, str]], response_model: type) -> Any:
        """
        Calls the LLM using API-level Structured Outputs with JSON Schema strict constraints.
        Includes a 2-attempt Self-Healing loop to correct formats recursively in case of Pydantic validation failures.
        Returns an instance of the validated response_model.
        """
        if not config.LLM_API_KEY:
            logger.warning("LLM_API_KEY is not set! Call might fail.")
            
        headers = {
            "Content-Type": "application/json"
        }
        
        # Handle Bearer token
        api_key = config.LLM_API_KEY.strip()
        if api_key:
            if api_key.startswith("Bearer "):
                headers["Authorization"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["Authorization"] = "Bearer dummy"

        # Self-healing retry loop (limit to 2 retries)
        max_healing_attempts = 2
        current_messages = list(messages)
        
        for attempt in range(max_healing_attempts + 1):
            data = {
                "model": config.LLM_MODEL,
                "messages": current_messages,
                "temperature": 0.1,  # Keep low temperature for strict schema generation
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": response_model.model_json_schema()
                    }
                }
            }
            
            if attempt > 0:
                logger.warning(f"Self-Healing formatted retry attempt {attempt} / {max_healing_attempts} for model: {response_model.__name__}")
            else:
                logger.info(f"Calling LLM ({config.LLM_MODEL}) in Structured Output Mode for {response_model.__name__}...")
                
            response_data = await self._request_with_retry("POST", config.LLM_API_URL, headers=headers, json=data)
            
            try:
                content = response_data["choices"][0]["message"]["content"]
                # Validate and parse directly into the Pydantic model
                return response_model.model_validate_json(content)
            except Exception as e:
                logger.error(f"Pydantic Validation or parsing failed on attempt {attempt}: {e}")
                if attempt >= max_healing_attempts:
                    logger.critical(f"Self-Healing exhausted all {max_healing_attempts} retries for model: {response_model.__name__}")
                    raise e
                
                # Fetch incorrect content safely
                raw_incorrect = ""
                try:
                    raw_incorrect = response_data["choices"][0]["message"]["content"]
                except Exception:
                    raw_incorrect = str(response_data)
                
                # Append assistant incorrect answer and correction prompt as user
                current_messages.append({"role": "assistant", "content": raw_incorrect})
                
                # Construct detailed error description to feed back
                error_desc = str(e)
                feedback_prompt = (
                    f"Your previous response failed Pydantic validation with the following error:\n"
                    f"'{error_desc}'\n\n"
                    f"Please output a corrected, valid JSON object that strictly adheres to the requested JSON schema. "
                    f"Ensure all required fields are filled correctly and values match any defined Enums exactly. "
                    f"Do not add any markup wrapping, markdown fences, or conversational prefix/suffix text."
                )
                current_messages.append({"role": "user", "content": feedback_prompt})
