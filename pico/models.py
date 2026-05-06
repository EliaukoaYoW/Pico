
import json
import re
import time
from traceback import print_tb
import urllib.error
import urllib.request
from http.client import RemoteDisconnected


def _normalize_versioned_base_url(base_url):
    base = str(base_url).rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base

def _extract_openai_text(data):
    if data.get("output_text"):
        return data["output_text"]
    
    for item in data.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict):
                text = content.get("text")
                if text:
                    return text
    choices = data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        return text
    return ""

def _extract_openai_text_from_sse(body_text):
    last_response = None
    deltas = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type", "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
            continue
        if event_type == "response.output_text.done":
            text = event.get("text")
            if text and isinstance(text, str):
                return text
        part = event.get("part")
        if isinstance(part, dict):
            text = part.get("text")
            if text and isinstance(text, str):
                return text
        item = event.get("item")
        if isinstance(item, dict):
            text = _extract_openai_text({"output": [item]})
            if text:
                return text
        response = event.get("response")
        if isinstance(response, dict):
            last_response = response
            text = _extract_openai_text(response)
            if text:
                return text
        text = _extract_openai_text(event)
        if text:
            return text
    if deltas:
        return "".join(deltas)
    if isinstance(last_response, dict):
        return _extract_openai_text(last_response)
    return ""


def _extract_openai_response_from_sse(body_text):
    last_response = None
    deltas = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        response = event.get("response")
        if isinstance(response, dict):
            last_response = response
            if event.get("type") == "response.completed":
                text = _extract_openai_text(response)
                if text:
                    return text, response
        event_type = event.get("type", "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                deltas.append(delta)
        elif event_type == "response.output_text.done":
            text = event.get("text")
            if text and isinstance(text, str):
                return text, last_response or {}
        else:
            text = _extract_openai_text(event)
            if text:
                return text, event
    if deltas:
        return "".join(deltas), last_response or {}
    if isinstance(last_response, dict):
        return _extract_openai_text(last_response), last_response
    return "", {}


def _extract_usage_cache_details(data):
    # 把不同 OpenAI-compatible 返回里的 usage 字段整理成统一结构，让 runtime/trace/report 不需要关心 provider 细节。
    usage = data.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached_tokens = int(input_details.get("cached_tokens") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": cached_tokens,
        "cache_hit": cached_tokens > 0,
    }


class FakeModelClient:
    def __init__(self, outputs):
        self.outputs = outputs
        self.prompts = []
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}
    
    def complete(self, prompt, max_new_tokens, **kwargs):
        self.prompts.append(prompt)
        if not getattr(self, "last_completion_metadata", None):
            self.last_completion_metadata = {}
        if not self.outputs:
            raise RuntimeError("fake model ran out of outputs")
        return self.outputs.pop(0)


class OllamaModelClient:
    def __init__(self, model, host, temperature, top_p, timeout):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}
    
    def complete(self, prompt, max_new_tokens, **kwargs):
        # Ollama 当前不支持我们这里接入的 prompt cache 语义，所以 runtime 传下来的缓存参数会被忽略。
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": False,
            "think": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
        }
        request = urllib.request.Request(
            f"{self.host}/api/v1/completions",  # 不同点
            method = "POST",
            headers = {
                "Content-Type": "application/json"
            },
            data = json.dumps(payload).encode("utf-8")
        )
        try:
            with urllib.request.urlopen(request, timeout = self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                "Could not reach Ollama.\n"
                "Make sure `ollama serve` is running and the model is available.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}"
            ) from e
        
        if data.get("error"):
            raise RuntimeError(f"Ollama request failed with error: {data['error']}")
        return data.get("response", "")


class OpenAICompatibleModelClient:
    def __init__(self, model, base_url, api_key, temperature, timeout):
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.supports_prompt_cache = any(host in self.base_url for host in ("openai.com", "right.codes"))
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        """向 OpenAI-compatible `/responses` 接口发起一次模型调用。

        为什么存在：
        runtime 不应该知道 HTTP 细节、SSE 细节、usage 字段长什么样，
        更不应该自己去判断 prompt cache 参数要不要带。这个函数把这些后端
        细节都包起来，对上层暴露统一的 `complete()` 行为。

        输入 / 输出：
        - 输入：完整 prompt、最大输出 token，以及可选的 prompt cache 参数
        - 输出：模型最终文本；同时把 usage / cached_tokens 等元数据写进 `self.last_completion_metadata`

        在 agent 链路里的位置：它位于 `Pico.ask()` 的模型调用阶段，
        是稳定前缀缓存复用链路真正落到 provider API 的地方。
        """
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "input":[
                {
                    "role":"user",
                    "content":[
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ]
                }
            ],
            "max_output_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.supports_prompt_cache and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if self.supports_prompt_cache and prompt_cache_retention:
            payload["prompt_cache_retention"] = prompt_cache_retention

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            method = "POST",
            headers = headers,
            data = json.dumps(payload).encode("utf-8")
        )
        attempts = 3
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout = self.timeout) as response:
                    body_text = response.read().decode("utf-8")
                    headers = getattr(response, "headers", {})
                    content_type = headers.get("Content-Type", "")
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                if e.code >= 500 and attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"OpenAI-compatible failed with HTTP {e.code}: {body}") from e
            except (urllib.error.URLError, ConnectionRefusedError) as e:
                if attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    "Could not reach the OpenAI-compatible backend.\n"
                    f"Base URL: {self.base_url}\n"
                    f"Model: {self.model}"
                ) from e
        if content_type.startswith("text/event-stream") or body_text.lstrip().startswith("data:"):
            text, response_data = _extract_openai_response_from_sse(body_text)
            if response_data and isinstance(response_data, dict):
                self.last_completion_metadata = {
                    "prompt_cache_supported": self.supports_prompt_cache,
                    "prompt_cache_key": prompt_cache_key,
                    "prompt_cache_retention": prompt_cache_retention,
                    **_extract_usage_cache_details(response_data),
                }
            if text:
                return text
            raise RuntimeError("OpenAI-compatible error: could not extract text from event stream response")

        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"OpenAI-compatible response is not valid JSON: {e}") from e
        if data.get("error"):
            raise RuntimeError(f"OpenAI-compatible error: {data.get('error')}")
        self.last_completion_metadata = {
            "prompt_cache_supported": self.supports_prompt_cache,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": prompt_cache_retention,
            **_extract_usage_cache_details(data)
        }
        return _extract_openai_text(data)
    

class SiliconflowModelClient:
    def __init__(self, model, base_url, api_key, temperature, timeout):
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.supports_prompt_cache = any(host in self.base_url for host in ("openai.com", "right.codes","siliconflow.cn"))
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, prompt_cache_key=None, prompt_cache_retention=None):
        self.last_completion_metadata = {}
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            method = "POST",
            headers = headers,
            data = json.dumps(payload).encode("utf-8")
        )
        attempts = 3
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout = self.timeout) as response:
                    body_text = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                if e.code >= 500 and attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"Siliconflow request failed with HTTP {e.code}: {body}") from e
            except (urllib.error.URLError, ConnectionRefusedError) as e:
                if attempt < attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise RuntimeError(
                    "Could not reach the Siliconflow backend.\n"
                    f"Base URL: {self.base_url}\n"
                    f"Model: {self.model}"
                ) from e

        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                "Siliconflow error: backend returned non-JSON content that could not be parsed"
            ) from e
        if data.get("error"):
            raise RuntimeError(f"Siliconflow error: {data.get('error')}")
        
        self.last_completion_metadata = {
            "prompt_cache_supported": self.supports_prompt_cache,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": prompt_cache_retention,
            **_extract_usage_cache_details(data)
        }
        return _extract_openai_text(data)