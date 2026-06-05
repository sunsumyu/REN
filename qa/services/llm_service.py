# -*- coding: utf-8 -*-
import asyncio
import httpx
import logging
import random
import json
import time
import os
import re
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
import config
from utils.visual_printer import print_token_usage

logger = logging.getLogger("MedicalQA.LLMService")

class ILLMService(ABC):
    @abstractmethod
    async def call_llm(self, prompt: str, system_prompt: str = "", model_pool: str = "premium", stage: str = "", max_tokens: int = None) -> str:
        """
        调用大语言模型生成文本回复。

        Args:
            prompt (str): 用户输入的主要提示词。
            system_prompt (str, optional): 系统级指令，用于设定模型角色或行为准则。默认为空字符串。
            model_pool (str, optional): 模型池标识，如 "premium" 或 "lightweight"，用于选择不同性能/成本的模型。默认为 "premium"。
            stage (str, optional): 当前业务阶段标识，用于日志记录和监控追踪。默认为空字符串。
            max_tokens (int, optional): 限制生成的最大 token 数量。默认为 None（使用默认配置）。

        Returns:
            str: 模型生成的文本内容。
        """
        pass
        
    @abstractmethod
    async def call_llm_with_reasoning(self, prompt: str, system_prompt: str = "", model_pool: str = "premium", stage: str = "", max_tokens: int = None) -> Tuple[str, str]:
        """
        调用大语言模型生成文本回复，并额外获取模型的推理过程内容。

        Args:
            prompt (str): 用户输入的主要提示词。
            system_prompt (str, optional): 系统级指令，用于设定模型角色或行为准则。默认为空字符串。
            model_pool (str, optional): 模型池标识，如 "premium" 或 "lightweight"。默认为 "premium"。
            stage (str, optional): 当前业务阶段标识，用于日志记录和监控追踪。默认为空字符串。
            max_tokens (int, optional): 限制生成的最大 token 数量。默认为 None。

        Returns:
            Tuple[str, str]: 一个元组，包含 (最终回答内容, 推理过程内容)。
        """
        pass
        
    @abstractmethod
    async def call_llm_structured(self, messages: List[Dict[str, str]], response_model: type, model_pool: str = "premium", stage: str = "") -> Any:
        """
        调用大语言模型进行结构化输出，确保返回结果符合指定的 Pydantic 模型结构。

        Args:
            messages (List[Dict[str, str]]): 对话消息列表，包含 role 和 content。
            response_model (type): 期望返回数据的 Pydantic BaseModel 类。
            model_pool (str, optional): 模型池标识。默认为 "premium"。
            stage (str, optional): 当前业务阶段标识。默认为空字符串。

        Returns:
            Any: 实例化的 response_model 对象。
        """
        pass

class LLMService(ILLMService):
    def __init__(self, http_client: httpx.AsyncClient, global_semaphore: asyncio.Semaphore):
        """
        初始化 LLM 服务实例。

        Args:
            http_client (httpx.AsyncClient): 异步 HTTP 客户端实例，用于发送请求。
            global_semaphore (asyncio.Semaphore): 全局并发控制信号量，用于限制同时进行的 API 调用数量。
        """
        self.client = http_client
        self.global_semaphore = global_semaphore
        self.supported_models: List[str] = []

    STRUCTURED_LEAK_PATTERNS = [
        r"rigorous\s+data\s+processing\s+api",
        r"strictly\s+adheres?\s+to\s+the\s+following\s+json\s+schema",
        r"valid\s+json\s+object\s+that\s+strictly\s+adheres?",
        r"do\s+not\s+include\s+markdown\s+fences",
        r"requested\s+json\s+schema",
        r"pydantic\s+validation",
        r"required\s+fields\s+are\s+filled\s+correctly",
        r'"\$defs"\s*:',
        r'"properties"\s*:',
        r'"type"\s*:\s*"object"',
    ]

    @classmethod
    def _find_structured_prompt_leak(cls, value: Any, path: str = "$") -> Tuple[str, str]:
        """
        递归检查数据结构中是否包含提示词泄露或 Schema 描述信息。

        Args:
            value (Any): 待检查的数据值（字符串、字典或列表）。
            path (str, optional): 当前数据在结构中的路径标识，用于错误定位。默认为 "$"。

        Returns:
            Tuple[str, str]: 如果发现泄露，返回 (泄露位置路径, 匹配的正则模式)；否则返回 ("", "")。
        """
        if isinstance(value, str):
            lowered = value.lower()
            for pattern in cls.STRUCTURED_LEAK_PATTERNS:
                if re.search(pattern, lowered, flags=re.IGNORECASE):
                    return path, pattern
            return "", ""
        if isinstance(value, dict):
            for key, child in value.items():
                leak_path, pattern = cls._find_structured_prompt_leak(child, f"{path}.{key}")
                if leak_path:
                    return leak_path, pattern
        if isinstance(value, list):
            for index, child in enumerate(value):
                leak_path, pattern = cls._find_structured_prompt_leak(child, f"{path}[{index}]")
                if leak_path:
                    return leak_path, pattern
        return "", ""

    @classmethod
    def _assert_no_structured_prompt_leak(cls, obj: Any, response_model: type) -> None:
        """
        验证结构化输出对象中是否包含不应出现的提示词或 Schema 泄露内容。

        Args:
            obj (Any): 待验证的对象，通常为 Pydantic 模型实例或字典。
            response_model (type): 预期的响应模型类，用于错误提示信息。

        Raises:
            ValueError: 如果检测到提示词泄露，抛出异常并指出泄露位置。
        """
        try:
            payload = obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj
        except Exception:
            payload = obj
        leak_path, pattern = cls._find_structured_prompt_leak(payload)
        if leak_path:
            raise ValueError(
                f"{response_model.__name__} contains structured prompt/schema leakage at {leak_path}: {pattern}"
            )

    async def init_supported_models(self, force: bool = False):
        """
        动态从网关获取支持的模型列表。
        支持离线缓存 'supported_models_cache.json' 以应对网络故障。

        Args:
            force (bool, optional): 是否强制刷新模型列表，忽略本地缓存。默认为 False。
        """
        cache_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "supported_models_cache.json")
        
        if self.supported_models and not force:
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
                try:
                    res_data = response.json()
                except json.JSONDecodeError as je:
                    logger.error(
                        f"❌ 获取支持模型时 JSON 解析失败 (JSONDecodeError): {url}\n"
                        f"  - 响应状态码: {response.status_code}\n"
                        f"  - 响应头部: {dict(response.headers)}\n"
                        f"  - 响应内容 (前1000字): {repr(response.text[:1000])}"
                    )
                    raise je
                if isinstance(res_data, dict) and "data" in res_data:
                    self.supported_models = [item.get("id") for item in res_data["data"] if item.get("id")]
                    logger.info(f"Successfully loaded {len(self.supported_models)} supported models dynamically from gateway!")
                    # Write to local cache
                    try:
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(self.supported_models, f, ensure_ascii=False, indent=2)
                        logger.info(f"Updated supported models cache file: {cache_file}")
                    except Exception as cache_err:
                        logger.warning(f"Failed to save supported models to cache: {cache_err}")
                    return
            else:
                logger.warning(f"Failed to query supported models from {url}: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"Error querying supported models from {url}: {e}")
            
        # Fallback to local cache if query failed or gateway was unreachable
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.supported_models = json.load(f)
                logger.info(f"Loaded supported models from local cache file: {cache_file} (Total: {len(self.supported_models)} models)")
            except Exception as cache_err:
                logger.error(f"Failed to load supported models from cache file: {cache_err}")

    async def _resolve_model(self, model_pool: str, is_structured: bool = False, force_refresh_on_missing: bool = True) -> str:
        """
        将模型池名称解析为实际使用的模型标识符。
        如果候选模型不在当前加载的列表中，则动态从网关强制刷新。

        Args:
            model_pool (str): 模型池名称（如 "lightweight", "judge", "premium"）。
            is_structured (bool, optional): 是否为结构化输出调用。如果是，可能会自动升级模型以保证 Schema 兼容性。默认为 False。
            force_refresh_on_missing (bool, optional): 当模型未找到时是否强制刷新列表。默认为 True。

        Returns:
            str: 解析后的实际模型 ID。
        """
        pool = model_pool.lower()
        
        # 针对结构化输出，若轻量模型可能不支持 strict schema，自动升级路由至 premium 旗舰模型
        if is_structured and pool == "lightweight":
            logger.info("Structured call detected in lightweight pool. Routing to premium pool for schema compatibility.")
            pool = "premium"
            
        # Map pool to configured model name
        if pool == "lightweight":
            model_candidate = config.MODEL_POOL_LIGHTWEIGHT
        elif pool == "judge":
            model_candidate = config.MODEL_POOL_JUDGE
        else:
            model_candidate = config.MODEL_POOL_PREMIUM
            
        def find_match():
            if not self.supported_models:
                return None
            for m in self.supported_models:
                if m == model_candidate:
                    return m
                # 检查是否存在带前缀的精确匹配 (支持 - 和 / 分隔符，例如 bsp-deepseek-v4-pro 或 deepseek/deepseek-v4-pro)
                if m.endswith(f"-{model_candidate}") or m.endswith(f"/{model_candidate}") or f"-{model_candidate}-" in m or f"/{model_candidate}/" in m:
                    logger.info(f"✨ [AI Model Match] Matched configured '{model_candidate}' to verified API model: '{m}'")
                    return m
            return None

        # 1. Try finding in currently loaded list
        matched_model = find_match()
        
        # 2. If not matched, force refresh from gateway to see if new models became available
        if not matched_model and force_refresh_on_missing:
            logger.warning(f"⚠️ Model candidate '{model_candidate}' not found in current supported models. Forcing dynamic re-fetch from gateway...")
            await self.init_supported_models(force=True)
            matched_model = find_match()
            
        if matched_model:
            return matched_model
            
        # 3. Fallback routing if still not found
        logger.warning(f"Configured model '{model_candidate}' is not present in supported models list: {self.supported_models}")
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
        执行带有指数退避重试机制的 HTTP 请求。
        遵守全局并发信号量的速率限制。

        Args:
            method (str): HTTP 请求方法（如 "GET", "POST"）。
            url (str): 请求 URL。
            **kwargs: 传递给 httpx 客户端的其他参数（如 headers, json 等）。

        Returns:
            Dict[str, Any]: 解析后的 JSON 响应数据。

        Raises:
            Exception: 当达到最大重试次数后仍然失败时抛出异常。
        """
        retries = 0
        backoff = 2.0
        
        while True:
            req_start_time = time.time()
            try:
                async with self.global_semaphore:
                    response = await self.client.request(method, url, timeout=120.0, **kwargs)
                    response.raise_for_status()
                    
                    try:
                        res_json = response.json()
                    except json.JSONDecodeError as je:
                        logger.error(
                            f"❌ 大模型接口请求 JSON 解析失败 (JSONDecodeError): {url}\n"
                            f"  - 响应状态码: {response.status_code}\n"
                            f"  - 响应头部: {dict(response.headers)}\n"
                            f"  - 响应内容 (前1000字): {repr(response.text[:1000])}"
                        )
                        raise je
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
                # 🚨 [504 Gateway Timeout 退避重试]：
                # 如果是网关对大模型超时未返回强行进行了 HTTP 504 关闭，改为允许退避重试以增加容错性
                if isinstance(e, httpx.HTTPStatusError) and e.response is not None and e.response.status_code == 504:
                    logger.warning(f"⚠️ [504 Timeout Intercept] Gateway timed out waiting for LLM after {elapsed:.2f}s. Will retry to recover.")
            
            retries += 1
            if retries > config.MAX_RETRIES:
                logger.critical(f"Max retries ({config.MAX_RETRIES}) reached for {url}. Raising error.")
                raise Exception(f"Failed to fetch from {url} after {config.MAX_RETRIES} attempts.")
            
            sleep_time = (backoff ** retries) + random.uniform(0.1, 0.5)
            logger.info(f"Sleeping for {sleep_time:.2f} seconds before retry {retries}...")
            await asyncio.sleep(sleep_time)

    def _build_request_headers(self) -> Dict[str, str]:
        """
        集中构造具有授权签名的 HTTP 请求头。

        Returns:
            Dict[str, str]: 包含 Content-Type 和 Authorization 的请求头字典。
        """
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
        """
        集中包装对话上下文消息队列。

        Args:
            prompt (str): 用户提示词。
            system_prompt (str, optional): 系统提示词。默认为空字符串。

        Returns:
            List[Dict[str, str]]: 格式化后的消息列表。
        """
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
        stage: str
    ) -> Tuple[Dict[str, Any], str, float]:
        """
        集中控制 HTTP 调用生命周期，实现计时和弹性降级路由。

        Args:
            headers (Dict[str, str]): HTTP 请求头。
            data (Dict[str, Any]): 请求体数据。
            resolved_model (str): 已解析的目标模型 ID。
            model_pool (str): 模型池标识。
            stage (str): 业务阶段标识。

        Returns:
            Tuple[Dict[str, Any], str, float]: 包含 (响应数据, 实际使用的模型ID, 耗时秒数) 的元组。

        Raises:
            Exception: 如果请求失败且无法通过降级恢复，则抛出原始异常。
        """
        total_start_time = time.time()
        try:
            response_data = await self._request_with_retry("POST", config.LLM_API_URL, headers=headers, json=data)
        except Exception as e:
            if ("模型不存在" in str(e) or "20201" in str(e)) and resolved_model != config.LLM_MODEL:
                logger.warning(f"Model '{resolved_model}' not supported. Falling back to default LLM_MODEL '{config.LLM_MODEL}'...")
                resolved_model = config.LLM_MODEL
                data["model"] = resolved_model
                logger.info(f"Retrying Call with Fallback LLM ({resolved_model})...")
                response_data = await self._request_with_retry("POST", config.LLM_API_URL, headers=headers, json=data)
            else:
                raise e
        
        duration = time.time() - total_start_time
        return response_data, resolved_model, duration

    async def call_llm(self, prompt: str, system_prompt: str = "", model_pool: str = "premium", stage: str = "", max_tokens: int = None) -> str:
        """
        调用大语言模型生成文本回复。

        Args:
            prompt (str): 用户输入的主要提示词。
            system_prompt (str, optional): 系统级指令。默认为空字符串。
            model_pool (str, optional): 模型池标识。默认为 "premium"。
            stage (str, optional): 业务阶段标识。默认为空字符串。
            max_tokens (int, optional): 最大生成 token 数。默认为 None。

        Returns:
            str: 模型生成的文本内容。
        """
        await self.init_supported_models()
        headers = self._build_request_headers()
        messages = self._prepare_messages(prompt, system_prompt)
        resolved_model = await self._resolve_model(model_pool)

        data = {
            "model": resolved_model,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE,
            "top_p": config.LLM_TOP_P,
            "frequency_penalty": config.LLM_FREQUENCY_PENALTY,
        }
        # 统计当前并发调用数
        if hasattr(self, "global_semaphore"):
            active_sem = self.global_semaphore._value if hasattr(self.global_semaphore, "_value") else "N/A"
            total_sem = config.GLOBAL_API_SEMAPHORE
            try:
                active_count = max(0, total_sem - int(active_sem))
            except Exception:
                active_count = "N/A"
            logger.info(f"Calling LLM ({resolved_model}) [Pool: {model_pool}] (Current Active/Limit: {active_count}/{total_sem})...")
        else:
            logger.info(f"Calling LLM ({resolved_model}) [Pool: {model_pool}]...")
        response_data, resolved_model, duration = await self._execute_with_fallback(headers, data, resolved_model, model_pool, stage)
        print_token_usage(stage, resolved_model, duration, response_data.get("usage", {}))
        
        try:
            content = response_data["choices"][0]["message"]["content"]
            return content.strip()
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Failed to parse LLM response format: {response_data}")
            raise Exception(f"Invalid LLM response format: {e}")

    async def call_llm_with_reasoning(self, prompt: str, system_prompt: str = "", model_pool: str = "premium", stage: str = "", max_tokens: int = None) -> Tuple[str, str]:
        """
        调用大语言模型生成文本回复，并额外获取模型的推理过程内容。

        Args:
            prompt (str): 用户输入的主要提示词。
            system_prompt (str, optional): 系统级指令。默认为空字符串。
            model_pool (str, optional): 模型池标识。默认为 "premium"。
            stage (str, optional): 业务阶段标识。默认为空字符串。
            max_tokens (int, optional): 最大生成 token 数。默认为 None。

        Returns:
            Tuple[str, str]: 一个元组，包含 (最终回答内容, 推理过程内容)。
        """
        await self.init_supported_models()
        headers = self._build_request_headers()
        messages = self._prepare_messages(prompt, system_prompt)
        resolved_model = await self._resolve_model(model_pool)

        data = {
            "model": resolved_model,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE,
            "top_p": config.LLM_TOP_P,
            "frequency_penalty": config.LLM_FREQUENCY_PENALTY,
        }
        # 统计当前并发调用数
        if hasattr(self, "global_semaphore"):
            active_sem = self.global_semaphore._value if hasattr(self.global_semaphore, "_value") else "N/A"
            total_sem = config.GLOBAL_API_SEMAPHORE
            try:
                active_count = max(0, total_sem - int(active_sem))
            except Exception:
                active_count = "N/A"
            logger.info(f"Calling LLM with reasoning ({resolved_model}) [Pool: {model_pool}] (Current Active/Limit: {active_count}/{total_sem})...")
        else:
            logger.info(f"Calling LLM with reasoning ({resolved_model}) [Pool: {model_pool}]...")
        response_data, resolved_model, duration = await self._execute_with_fallback(headers, data, resolved_model, model_pool, stage)
        print_token_usage(stage, resolved_model, duration, response_data.get("usage", {}))
        
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
        调用大语言模型进行结构化输出，确保返回结果符合指定的 Pydantic 模型结构。
        支持自愈重试机制，当解析失败时自动反馈错误给模型进行修正。

        Args:
            messages (List[Dict[str, str]]): 对话消息列表。
            response_model (type): 期望返回数据的 Pydantic BaseModel 类。
            model_pool (str, optional): 模型池标识。默认为 "premium"。
            stage (str, optional): 业务阶段标识。默认为空字符串。

        Returns:
            Any: 实例化的 response_model 对象，其中包含额外的 _reasoning_content 属性。

        Raises:
            Exception: 当所有自愈重试尝试均失败时抛出异常。
        """
        await self.init_supported_models()
        headers = self._build_request_headers()
        
        max_healing_attempts = 2
        current_messages = list(messages)
        resolved_model = await self._resolve_model(model_pool, is_structured=True)
        
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
                "content": (
                    f"你是结构化函数调用参数生成器。请为虚拟函数 {response_model.__name__} 生成 arguments。\n"
                    "只允许返回一个 JSON object，不要返回 Markdown、解释、对话前后缀或代码块。\n"
                    "下面的 JSON Schema 只是字段约束，绝不能把 schema、字段说明、system/user/assistant 提示文本复制进任何字段值。\n\n"
                    f"Function arguments JSON Schema:\n{schema_str}"
                )
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
                # 统计当前Premium池的并发调用数
                if hasattr(self, "global_semaphore"):
                    active_sem = self.global_semaphore._value if hasattr(self.global_semaphore, "_value") else "N/A"
                    # 因为 Semaphore 信号量是递减的，当前活跃数 = 总量 - 信号量当前可用值
                    total_sem = config.GLOBAL_API_SEMAPHORE
                    try:
                        active_count = max(0, total_sem - int(active_sem))
                    except Exception:
                        active_count = "N/A"
                    logger.info(f"Calling LLM ({resolved_model}) [Pool: {model_pool}] (Current Active/Limit: {active_count}/{total_sem}) in Structured Output Mode for {response_model.__name__}...")
                else:
                    logger.info(f"Calling LLM ({resolved_model}) [Pool: {model_pool}] in Structured Output Mode for {response_model.__name__}...")
                
            response_data, resolved_model, duration = await self._execute_with_fallback(
                headers, data, resolved_model, model_pool, stage
            )
            current_stage = stage
            if attempt > 0:
                current_stage = f"{stage} (自愈重试 {attempt})"
            print_token_usage(current_stage, resolved_model, duration, response_data.get("usage", {}))
            
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
                self._assert_no_structured_prompt_leak(obj, response_model)
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
                    "上一轮函数参数 JSON 未通过程序校验，错误如下：\n"
                    f"{error_desc}\n\n"
                    "请重新输出一个可被解析的 JSON object。"
                    "字段必须完整，枚举值必须精确匹配约束。"
                    "不要输出 Markdown、解释、对话前后缀，也不要把校验错误、schema 或提示文本复制进任何字段值。"
                )
                current_messages.append({"role": "user", "content": feedback_prompt})
