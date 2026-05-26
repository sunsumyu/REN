import asyncio
import httpx
import logging
import random
import json
from typing import List, Dict, Any, Set, Tuple
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
        self.supported_models: List[str] = []

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
                        
                        # 针对模型不存在的致命错误，立即向上抛出标准 Exception，打破重试循环以进行自愈降级
                        if err_code == 20201 or "模型不存在" in err_msg or "model_not_found" in err_msg:
                            raise Exception(f"API fatal error: {err_msg} (code: {err_code})")
                            
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

    async def init_supported_models(self):
        """
        Dynamically fetch the list of supported models from the gateway.
        This queries the /v1/models endpoint, which standard OpenAI-compatible gateways expose.
        """
        if self.supported_models:
            return
            
        headers = {
            "Content-Type": "application/json"
        }
        api_key = config.LLM_API_KEY.strip() if config.LLM_API_KEY else "dummy"
        if api_key:
            if api_key.startswith("Bearer "):
                headers["Authorization"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["Authorization"] = "Bearer dummy"
            
        # Dynamically build standard OpenAI models endpoint
        url = config.LLM_API_URL.replace("/chat/completions", "/models")
        try:
            response = await self.client.get(url, headers=headers, timeout=10.0)
            if response.status_code == 200:
                res_data = response.json()
                if isinstance(res_data, dict) and "data" in res_data:
                    self.supported_models = [item.get("id") for item in res_data["data"] if item.get("id")]
                    logger.info(f"Successfully loaded {len(self.supported_models)} supported models from gateway dynamically!")
            else:
                logger.warning(f"Failed to query supported models from {url}: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"Error querying supported models from {url}: {e}")

    def _resolve_model(self, model_pool: str, is_structured: bool = False) -> str:
        """
        Resolve the model pool name to the actual model identifier from config,
        with dynamic support verification and auto-matching fallback.
        Supports structured upgrade to premium pool for JSON schemas.
        """
        pool = model_pool.lower()
        
        # 针对结构化输出 (Structured Outputs)，如果轻量模型在网关上可能不支持 strict schema，
        # 我们自动升级路由到 premium 旗舰模型，以确保 100% 的格式契合与零超时失败。
        if is_structured and pool == "lightweight":
            logger.info("Structured call detected in lightweight pool. Routing to premium pool for schema compatibility.")
            pool = "premium"
            
        # 1. Map pool to configured model name
        if pool == "lightweight":
            configured = getattr(config, "MODEL_POOL_LIGHTWEIGHT", config.LLM_MODEL)
        elif pool == "judge":
            configured = getattr(config, "MODEL_POOL_JUDGE", config.LLM_MODEL)
        elif pool == "premium":
            configured = getattr(config, "MODEL_POOL_PREMIUM", config.LLM_MODEL)
        else:
            configured = config.LLM_MODEL
            
        # 2. If supported_models list is not loaded yet (or empty), return configured directly
        if not self.supported_models:
            return configured
            
        # 3. Check if the configured model is supported by the gateway
        if configured in self.supported_models:
            return configured
            
        # 4. If not supported, apply smart model auto-matching based on pool name
        logger.warning(f"Configured model '{configured}' for pool '{model_pool}' is not supported by this gateway.")
        
        if pool == "lightweight":
            # Search for available lightweight models in descending order of performance
            for candidate in ["deepseek-v4-flash", "qwen3.6-plus", "qwen-plus", "qwen-turbo", "glm-4-flash", "gpt-4o-mini"]:
                if candidate in self.supported_models:
                    logger.info(f"Auto-selected available lightweight model '{candidate}' as fallback.")
                    return candidate
                    
        # General default fallback to any supported main model or first available model
        if config.LLM_MODEL in self.supported_models:
            logger.info(f"Auto-selected main LLM_MODEL '{config.LLM_MODEL}' as fallback.")
            return config.LLM_MODEL
            
        # Absolute fallback to first model in list if everything else fails
        fallback = self.supported_models[0]
        logger.warning(f"Absolutely all pool fallbacks failed. Selecting first gateway model '{fallback}'.")
        return fallback

    async def call_llm(self, prompt: str, system_prompt: str = "", model_pool: str = "premium") -> str:
        """
        Call the Large Language Model completions API with smart model pool routing and dynamic model-not-found self-healing fallback.
        """
        await self.init_supported_models()
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

        resolved_model = self._resolve_model(model_pool)

        data = {
            "model": resolved_model,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE,
            "top_p": config.LLM_TOP_P,
            "frequency_penalty": config.LLM_FREQUENCY_PENALTY,
        }
        
        logger.info(f"Calling LLM ({resolved_model}) [Pool: {model_pool}]...")
        try:
            response_data = await self._request_with_retry("POST", config.LLM_API_URL, headers=headers, json=data)
        except Exception as e:
            if ("模型不存在" in str(e) or "20201" in str(e)) and resolved_model != config.LLM_MODEL:
                logger.warning(f"Model '{resolved_model}' from pool '{model_pool}' is not supported by the upstream gateway. Gracefully falling back to default LLM_MODEL '{config.LLM_MODEL}'...")
                resolved_model = config.LLM_MODEL
                data["model"] = resolved_model
                logger.info(f"Retrying Call with Fallback LLM ({resolved_model})...")
                response_data = await self._request_with_retry("POST", config.LLM_API_URL, headers=headers, json=data)
            else:
                raise e
        
        # Parse standard chat completion response
        try:
            content = response_data["choices"][0]["message"]["content"]
            return content.strip()
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse LLM response format: {response_data}")
            raise Exception(f"Invalid LLM response format: {e}")

    async def call_llm_with_reasoning(self, prompt: str, system_prompt: str = "", model_pool: str = "premium") -> Tuple[str, str]:
        """
        Call the Large Language Model completions API with smart model pool routing and dynamic model-not-found self-healing fallback.
        Returns a tuple of (content, reasoning_content).
        """
        await self.init_supported_models()
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

        resolved_model = self._resolve_model(model_pool)

        data = {
            "model": resolved_model,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE,
            "top_p": config.LLM_TOP_P,
            "frequency_penalty": config.LLM_FREQUENCY_PENALTY,
        }
        
        logger.info(f"Calling LLM with reasoning ({resolved_model}) [Pool: {model_pool}]...")
        try:
            response_data = await self._request_with_retry("POST", config.LLM_API_URL, headers=headers, json=data)
        except Exception as e:
            if ("模型不存在" in str(e) or "20201" in str(e)) and resolved_model != config.LLM_MODEL:
                logger.warning(f"Model '{resolved_model}' from pool '{model_pool}' is not supported by the upstream gateway. Gracefully falling back to default LLM_MODEL '{config.LLM_MODEL}'...")
                resolved_model = config.LLM_MODEL
                data["model"] = resolved_model
                logger.info(f"Retrying Call with Fallback LLM ({resolved_model})...")
                response_data = await self._request_with_retry("POST", config.LLM_API_URL, headers=headers, json=data)
            else:
                raise e
        
        # Parse standard chat completion response
        try:
            message = response_data["choices"][0]["message"]
            content = message["content"]
            reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
            return content.strip(), reasoning_content.strip()
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse LLM response format: {response_data}")
            raise Exception(f"Invalid LLM response format: {e}")

    async def call_llm_structured(self, messages: List[Dict[str, str]], response_model: type, model_pool: str = "premium") -> Any:
        """
        Calls the LLM using API-level Structured Outputs with JSON Schema strict constraints.
        Includes a 2-attempt Self-Healing loop to correct formats recursively in case of Pydantic validation failures.
        Supports model pool routing and dynamic model-not-found self-healing fallback.
        Returns an instance of the validated response_model.
        """
        await self.init_supported_models()
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
        
        resolved_model = self._resolve_model(model_pool, is_structured=True)
        
        # 探测模型类型以进行响应格式适配：
        # 如果是 OpenAI 的 GPT 旗舰模型，我们使用最严苛的 'json_schema' (strict=True) 以获得完美的概率锁止；
        # 对于其它非 GPT 旗舰模型（如 DeepSeek, Qwen 等），我们自动降级为通用标准的 'json_object' (JSON Mode)，
        # 以完美绕过内网网关代理对非 OpenAI 模型在 json_schema 上的刚性拦截 (Code: 10009 / invalid_request_error)。
        is_openai = resolved_model.lower().startswith("gpt")
        
        if is_openai:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": response_model.model_json_schema()
                }
            }
        else:
            # 采用类似 Hello-Agents 的最大兼容性方案：
            # 直接舍弃原生的 `response_format` 字段，以彻底绕过网关层面对 JSON Mode / Schema 的严苛检查。
            # 完全依赖 Prompt Engineering + 客户端的 Pydantic 校验与重试机制 (Self-Healing)。
            response_format = None
            schema_str = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
            current_messages.insert(0, {
                "role": "system", 
                "content": f"You are a rigorous data processing API. You must output ONLY a valid JSON object that strictly adheres to the following JSON schema. Do not include markdown fences (like ```json), conversational text, or explanations.\n\nSchema:\n{schema_str}"
            })
            
        for attempt in range(max_healing_attempts + 1):
            data = {
                "model": resolved_model,
                "messages": current_messages,
                "temperature": config.LLM_TEMPERATURE,
                "top_p": config.LLM_TOP_P,
                "frequency_penalty": config.LLM_FREQUENCY_PENALTY,
            }
            if response_format:
                data["response_format"] = response_format
            
            if attempt > 0:
                logger.warning(f"Self-Healing formatted retry attempt {attempt} / {max_healing_attempts} for model: {response_model.__name__} ({resolved_model})")
            else:
                logger.info(f"Calling LLM ({resolved_model}) [Pool: {model_pool}] in Structured Output Mode for {response_model.__name__}...")
                
            try:
                response_data = await self._request_with_retry("POST", config.LLM_API_URL, headers=headers, json=data)
            except Exception as e:
                if ("模型不存在" in str(e) or "20201" in str(e)) and resolved_model != config.LLM_MODEL:
                    logger.warning(f"Model '{resolved_model}' from pool '{model_pool}' is not supported by the upstream gateway. Gracefully falling back to default LLM_MODEL '{config.LLM_MODEL}'...")
                    resolved_model = config.LLM_MODEL
                    data["model"] = resolved_model
                    logger.info(f"Retrying Call in Structured Mode with Fallback LLM ({resolved_model})...")
                    response_data = await self._request_with_retry("POST", config.LLM_API_URL, headers=headers, json=data)
                else:
                    raise e
            
            try:
                message = response_data["choices"][0]["message"]
                content = message["content"]
                reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
                
                # Validate and parse directly into the Pydantic model
                obj = response_model.model_validate_json(content)
                # Dynamically attach reasoning_content to the object
                object.__setattr__(obj, "_reasoning_content", reasoning_content.strip())
                return obj
            except Exception as e:
                logger.error(f"Pydantic Validation or parsing failed on attempt {attempt}: {e}")
                if attempt >= max_healing_attempts:
                    logger.critical(f"Self-Healing exhausted all {max_healing_attempts} retries for model: {response_model.__name__} ({resolved_model})")
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

