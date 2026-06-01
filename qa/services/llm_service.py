# -*- coding: utf-8 -*-
import asyncio
import httpx
import logging
import random
import json
import time
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
import config
from utils.visual_printer import print_token_usage

logger = logging.getLogger("MedicalQA.LLMService")

class ILLMService(ABC):
    @abstractmethod
    async def call_llm(self, prompt: str, system_prompt: str = "", model_pool: str = "premium", stage: str = "", max_tokens: int = None) -> str:
        pass
        
    @abstractmethod
    async def call_llm_with_reasoning(self, prompt: str, system_prompt: str = "", model_pool: str = "premium", stage: str = "", max_tokens: int = None) -> Tuple[str, str]:
        pass
        
    @abstractmethod
    async def call_llm_structured(self, messages: List[Dict[str, str]], response_model: type, model_pool: str = "premium", stage: str = "") -> Any:
        pass

class LLMService(ILLMService):
    def __init__(self, http_client: httpx.AsyncClient, global_semaphore: asyncio.Semaphore):
        self.client = http_client
        self.global_semaphore = global_semaphore
        self.supported_models: List[str] = []

    async def init_supported_models(self):
        """
        Dynamically fetch the list of supported models from the gateway.
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
                    logger.info(f"Successfully loaded {len(self.supported_models)} supported models dynamically!")
            else:
                logger.warning(f"Failed to query supported models from {url}: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"Error querying supported models from {url}: {e}")

    def _resolve_model(self, model_pool: str, is_structured: bool = False) -> str:
        """
        Resolve the model pool name to the actual model identifier from config.
        """
        pool = model_pool.lower()
        
        # 针对结构化输出，若轻量模型可能不支持 strict schema，自动升级路由至 premium 旗舰模型
        if is_structured and pool == "lightweight":
            logger.info("Structured call detected in lightweight pool. Routing to premium pool for schema compatibility.")
            pool = "premium"
            
        # Map pool to configured model name
        if pool == "lightweight":
            model_candidate = config.MODEL_POOL_LIGHTWEIGHT
        else:
            model_candidate = config.MODEL_POOL_PREMIUM
            
        # Verify compatibility against dynamically loaded models, with fallback
        if self.supported_models and model_candidate not in self.supported_models:
            logger.warning(f"Configured model '{model_candidate}' is not present in supported models list.")
            if config.LLM_MODEL in self.supported_models:
                logger.info(f"Routing to verified default LLM_MODEL: '{config.LLM_MODEL}'")
                return config.LLM_MODEL
            elif self.supported_models:
                fallback_first = self.supported_models[0]
                logger.info(f"Routing to first available verified model: '{fallback_first}'")
                return fallback_first
                
        return model_candidate

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """
        Executes HTTP requests with exponential backoff on transient errors.
        Respects global concurrency semaphore rate limiting.
        """
        retries = 0
        backoff = 2.0
        
        while True:
            req_start_time = time.time()
            try:
                async with self.global_semaphore:
                    response = await self.client.request(method, url, timeout=120.0, **kwargs)
                    response.raise_for_status()
                    
                    res_json = response.json()
                    # Check for API-level business errors packaged in 200 responses
                    if isinstance(res_json, dict) and res_json.get("error"):
                        err = res_json.get("error", {})
                        err_msg = err.get("message", "Unknown API error")
                        err_code = err.get("code", "api_error")
                        logger.error(f"API-level error returned from {url}: {err_msg} (code: {err_code})")
                        raise httpx.HTTPStatusError(
                            message=f"API payload error: {err_msg} (code: {err_code})",
                            request=response.request,
                            response=response
                        )
                    return res_json
            except (httpx.HTTPStatusError, httpx.HTTPError, httpx.NetworkError) as e:
                elapsed = time.time() - req_start_time
                logger.error(f"HTTP/API error on request {method} {url} (elapsed: {elapsed:.2f}s): {e}")
                # 🚨 [504 Gateway Timeout 熔断保护]：
                # 如果是网关对大模型超时未返回强行进行了 HTTP 504 关闭，直接物理熔断抛出异常，不进行退避重试
                if isinstance(e, httpx.HTTPStatusError) and e.response is not None and e.response.status_code == 504:
                    logger.critical(f"🚨 [504 Timeout Intercept] Gateway timed out waiting for LLM after {elapsed:.2f}s and closed HTTP connection. Aborting retries to prevent queue congestion.")
                    raise e
            
            retries += 1
            if retries > config.MAX_RETRIES:
                logger.critical(f"Max retries ({config.MAX_RETRIES}) reached for {url}. Raising error.")
                raise Exception(f"Failed to fetch from {url} after {config.MAX_RETRIES} attempts.")
            
            sleep_time = (backoff ** retries) + random.uniform(0.1, 0.5)
            logger.info(f"Sleeping for {sleep_time:.2f} seconds before retry {retries}...")
            await asyncio.sleep(sleep_time)

    def _build_request_headers(self) -> Dict[str, str]:
        """集中构造具有授权签名的 HTTP 请求头"""
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
        return headers

    def _prepare_messages(self, prompt: str, system_prompt: str = "") -> List[Dict[str, str]]:
        """集中包装对话上下文消息队列"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def _execute_with_fallback(
        self, 
        headers: Dict[str, str], 
        data: Dict[str, Any], 
        resolved_model: str, 
        model_pool: str, 
        stage: str, 
        attempt: int = 0
    ) -> Tuple[Dict[str, Any], str]:
        """集中控制 HTTP 调用生命周期，实现计时、弹性降级路由及 Token 漂亮打印的极致 DRY 抽取"""
        start_time = time.time()
        try:
            response_data = await self._request_with_retry("POST", config.LLM_API_URL, headers=headers, json=data)
        except Exception as e:
            if ("模型不存在" in str(e) or "20201" in str(e)) and resolved_model != config.LLM_MODEL:
                logger.warning(f"Model '{resolved_model}' not supported. Falling back to default LLM_MODEL '{config.LLM_MODEL}'...")
                resolved_model = config.LLM_MODEL
                data["model"] = resolved_model
                logger.info(f"Retrying Call with Fallback LLM ({resolved_model})...")
                start_time = time.time()
                response_data = await self._request_with_retry("POST", config.LLM_API_URL, headers=headers, json=data)
            else:
                raise e
        
        duration = time.time() - start_time
        usage = response_data.get("usage", {})
        
        current_stage = stage
        if attempt > 0:
            current_stage = f"{stage} (自愈重试 {attempt})"
        print_token_usage(current_stage, resolved_model, duration, usage)
        return response_data, resolved_model

    async def call_llm(self, prompt: str, system_prompt: str = "", model_pool: str = "premium", stage: str = "", max_tokens: int = None) -> str:
        await self.init_supported_models()
        headers = self._build_request_headers()
        messages = self._prepare_messages(prompt, system_prompt)
        resolved_model = self._resolve_model(model_pool)

        data = {
            "model": resolved_model,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE,
            "top_p": config.LLM_TOP_P,
            "frequency_penalty": config.LLM_FREQUENCY_PENALTY,
        }
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        
        logger.info(f"Calling LLM ({resolved_model}) [Pool: {model_pool}]...")
        response_data, resolved_model = await self._execute_with_fallback(headers, data, resolved_model, model_pool, stage)
        
        try:
            content = response_data["choices"][0]["message"]["content"]
            return content.strip()
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse LLM response format: {response_data}")
            raise Exception(f"Invalid LLM response format: {e}")

    async def call_llm_with_reasoning(self, prompt: str, system_prompt: str = "", model_pool: str = "premium", stage: str = "", max_tokens: int = None) -> Tuple[str, str]:
        await self.init_supported_models()
        headers = self._build_request_headers()
        messages = self._prepare_messages(prompt, system_prompt)
        resolved_model = self._resolve_model(model_pool)

        data = {
            "model": resolved_model,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE,
            "top_p": config.LLM_TOP_P,
            "frequency_penalty": config.LLM_FREQUENCY_PENALTY,
        }
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        
        logger.info(f"Calling LLM with reasoning ({resolved_model}) [Pool: {model_pool}]...")
        response_data, resolved_model = await self._execute_with_fallback(headers, data, resolved_model, model_pool, stage)
        
        try:
            message = response_data["choices"][0]["message"]
            content = message["content"]
            reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
            return content.strip(), reasoning_content.strip()
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse LLM response format: {response_data}")
            raise Exception(f"Invalid LLM response format: {e}")

    async def call_llm_structured(self, messages: List[Dict[str, str]], response_model: type, model_pool: str = "premium", stage: str = "") -> Any:
        """
        Calls the LLM using API-level Structured Outputs with JSON Schema strict constraints.
        """
        await self.init_supported_models()
        headers = self._build_request_headers()
        
        max_healing_attempts = 2
        current_messages = list(messages)
        resolved_model = self._resolve_model(model_pool, is_structured=True)
        
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
                
            response_data, resolved_model = await self._execute_with_fallback(
                headers, data, resolved_model, model_pool, stage, attempt=attempt
            )
            
            try:
                message = response_data["choices"][0]["message"]
                content = message["content"]
                reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
                
                # Clean JSON block
                clean_content = content.strip()
                import re
                match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', clean_content, re.IGNORECASE)
                if match:
                    clean_content = match.group(1).strip()
                
                first_brace = clean_content.find('{')
                first_bracket = clean_content.find('[')
                start = -1
                end = -1
                if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
                    start = first_brace
                    end = clean_content.rfind('}')
                elif first_bracket != -1:
                    start = first_bracket
                    end = clean_content.rfind(']')
                
                if start != -1 and end != -1 and end > start:
                    clean_content = clean_content[start:end+1]

                obj = response_model.model_validate_json(clean_content)
                object.__setattr__(obj, "_reasoning_content", reasoning_content.strip())
                return obj
            except Exception as e:
                logger.error(f"Pydantic Validation or parsing failed on attempt {attempt}: {e}")
                if attempt >= max_healing_attempts:
                    logger.critical(f"Self-Healing exhausted all {max_healing_attempts} retries for model: {response_model.__name__} ({resolved_model})")
                    raise e
                
                raw_incorrect = ""
                try:
                    raw_incorrect = response_data["choices"][0]["message"]["content"]
                except Exception:
                    raw_incorrect = str(response_data)
                
                current_messages.append({"role": "assistant", "content": raw_incorrect})
                
                error_desc = str(e)
                feedback_prompt = (
                    f"Your previous response failed Pydantic validation with the following error:\n"
                    f"'{error_desc}'\n\n"
                    f"Please output a corrected, valid JSON object that strictly adheres to the requested JSON schema. "
                    f"Ensure all required fields are filled correctly and values match any defined Enums exactly. "
                    f"Do not add any markup wrapping, markdown fences, or conversational prefix/suffix text."
                )
                current_messages.append({"role": "user", "content": feedback_prompt})
