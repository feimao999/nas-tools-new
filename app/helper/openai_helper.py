import ipaddress
import json
import re
import time
from urllib.parse import urlparse

import requests

try:
    import log
except ImportError:
    log = None

from app.utils import OpenAISessionCache
from app.utils.commons import singleton
from config import Config


class _AIRateLimitError(Exception):
    """AI 接口触发频率限制或配额不足"""


@singleton
class OpenAiHelper:
    _provider = "openai"
    _api_key = ""
    _api_url = ""
    _model = ""
    _web_search = False
    _tavily_enable = False
    _tavily_api_key = ""
    _proxies = None
    _timeout = 60

    _TAVILY_ENDPOINT = "https://api.tavily.com/search"
    _NATIVE_SEARCH_COOLDOWN_SECONDS = 600
    _tavily_cache = {}
    _native_search_cooldown_until = {}

    _GEMINI_SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    _PROVIDERS = ("openai", "claude", "gemini", "local")

    _DEFAULT_URLS = {
        "openai": "https://api.openai.com",
        "claude": "https://api.anthropic.com",
        "gemini": "https://generativelanguage.googleapis.com",
        "local": "http://127.0.0.1:11434/v1",
    }

    _DEFAULT_MODELS = {
        "openai": "gpt-4o-mini",
        "claude": "claude-sonnet-4-5",
        "gemini": "gemini-2.5-flash",
        "local": "qwen2.5:7b",
    }

    _OPENAI_CHAT_PREFIXES = ("gpt-", "chatgpt-", "o1", "o3", "o4", "gpt")
    _OPENAI_NON_CHAT_KEYWORDS = (
        "embedding", "whisper", "tts", "dall-e", "davinci", "babbage",
        "moderation", "realtime", "audio", "transcribe", "image",
    )
    _GEMINI_NON_CHAT_KEYWORDS = (
        "embedding", "aqa", "imagen", "tts", "image-generation", "veo",
    )

    def __init__(self):
        self.init_config()

    def init_config(self):
        """
        初始化 AI 配置
        """
        cfg = Config().get_config("openai") or {}
        provider = (cfg.get("provider") or "openai").strip().lower()
        if provider not in self._PROVIDERS:
            provider = "openai"
        self._provider = provider
        self._api_key = (cfg.get("api_key") or "").strip()
        self._api_url = (cfg.get("api_url") or "").strip().rstrip("/")
        default_model = self._DEFAULT_MODELS.get(self._provider, "gpt-4o-mini")
        self._model = (cfg.get("model") or "").strip() or default_model
        self._web_search = bool(cfg.get("web_search"))
        self._tavily_enable = bool(cfg.get("tavily_enable"))
        self._tavily_api_key = (cfg.get("tavily_api_key") or "").strip()
        self._proxies = Config().get_proxies()

    def get_state(self):
        """
        检查 AI 服务是否已启用（每次读取前同步最新配置）
        """
        self.init_config()
        if self._provider in ("claude", "gemini"):
            return bool(self._api_key)
        if self._provider == "local":
            return bool(self._api_url)
        return bool(self._api_key) or bool(self._api_url)

    @staticmethod
    def __save_session(session_id, message):
        """
        保存会话
        :param session_id: 会话ID
        :param message: 助手回复消息
        :return:
        """
        seasion = OpenAISessionCache.get(session_id)
        if seasion:
            seasion.append({
                "role": "assistant",
                "content": message
            })
            OpenAISessionCache.set(session_id, seasion)

    @staticmethod
    def __get_session(session_id, message):
        """
        获取会话
        :param session_id: 会话ID
        :param message: 当前用户输入
        :return: 会话上下文
        """
        seasion = OpenAISessionCache.get(session_id)
        if seasion:
            seasion.append({
                "role": "user",
                "content": message
            })
        else:
            seasion = [
                {
                    "role": "system",
                    "content": "请在接下来的对话中请使用中文回复，并且内容尽可能详细。"
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
            OpenAISessionCache.set(session_id, seasion)
        return seasion

    @staticmethod
    def __clear_session(session_id):
        """
        清除会话
        :param session_id: 会话ID
        :return:
        """
        if OpenAISessionCache.get(session_id):
            OpenAISessionCache.delete(session_id)

    @staticmethod
    def _is_local_or_private_host(url):
        """
        判断目标 URL 是否为本地或内网地址
        """
        try:
            hostname = (urlparse(url).hostname or "").strip().lower()
        except Exception:
            return False
        if not hostname:
            return False
        if hostname in ("localhost", "host.docker.internal") or hostname.endswith(".local"):
            return True
        try:
            ip = ipaddress.ip_address(hostname)
            return ip.is_private or ip.is_loopback or ip.is_link_local
        except ValueError:
            return False

    def _get_request_proxies(self, url, provider=None):
        """
        本地 AI 或内网地址自动绕过系统代理，外部服务沿用全局代理设置
        """
        active_provider = provider or self._provider
        if active_provider == "local" or self._is_local_or_private_host(url):
            return None
        return self._proxies or None

    @classmethod
    def _normalize_openai_base(cls, api_url):
        """
        规范化 OpenAI 兼容接口路径前缀
        """
        base = (api_url or "").strip().rstrip("/")
        if not base:
            return f"{cls._DEFAULT_URLS['openai']}/v1"
        if base.endswith("/chat/completions"):
            return base[:-len("/chat/completions")]
        if base.endswith("/models"):
            return base[:-len("/models")]
        if base.endswith("/v1") or base.endswith("/v1beta/openai") or base.endswith("/api/v1"):
            return base
        return f"{base}/v1"

    @classmethod
    def _build_openai_endpoint(cls, api_url):
        """
        规范化 OpenAI 兼容接口请求地址
        """
        return f"{cls._normalize_openai_base(api_url)}/chat/completions"

    @classmethod
    def _build_openai_models_endpoint(cls, api_url):
        """
        规范化 OpenAI 兼容接口模型列表地址
        """
        return f"{cls._normalize_openai_base(api_url)}/models"

    @classmethod
    def _build_claude_base(cls, api_url):
        """
        规范化 Anthropic Claude 基础路径
        """
        base = (api_url or "").strip().rstrip("/") or cls._DEFAULT_URLS["claude"]
        if base.endswith("/messages"):
            base = base[:-len("/messages")]
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    @staticmethod
    def _claude_headers(api_key):
        """
        构造 Anthropic Claude 请求头
        """
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

    @classmethod
    def _build_gemini_base(cls, api_url):
        """
        规范化 Google Gemini 基础路径
        """
        base = (api_url or "").strip().rstrip("/") or cls._DEFAULT_URLS["gemini"]
        if base.endswith("/v1beta") or base.endswith("/v1"):
            return base
        return f"{base}/v1beta"

    @classmethod
    def _build_gemini_endpoint(cls, api_url, model):
        """
        规范化 Google Gemini generateContent 请求地址
        """
        return f"{cls._build_gemini_base(api_url)}/models/{model}:generateContent"

    @staticmethod
    def _normalize_messages(message, prompt=None):
        """
        统一转换为标准的 OpenAI messages 列表
        """
        if isinstance(message, list):
            return message
        if prompt:
            return [
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ]
        return [{"role": "user", "content": message}]

    @staticmethod
    def _extract_json(text):
        """
        从 AI 返回文本中提取并解析 JSON（兼容联网搜索前言文本、Markdown 代码块与推理模型的 <think> 标签）
        """
        if not text:
            return {}
        cleaned = re.sub(r"<think>.*?</think>", "", str(text), flags=re.IGNORECASE | re.DOTALL).strip()
        fenced_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        for block in reversed(fenced_blocks):
            try:
                parsed = json.loads(block.strip())
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            decoder = json.JSONDecoder()
            brace_positions = [m.start() for m in re.finditer(r"\{", cleaned)]
            for pos in reversed(brace_positions):
                try:
                    parsed, _ = decoder.raw_decode(cleaned[pos:])
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    continue
            raise

    @staticmethod
    def _raise_for_http_error(resp):
        """
        检查 HTTP 状态码并抛出带可读信息的异常
        """
        if resp.status_code == 429:
            raise _AIRateLimitError(resp.text or "HTTP 429 Too Many Requests")
        if resp.status_code >= 400:
            detail = ""
            try:
                err_json = resp.json()
                if isinstance(err_json, dict):
                    err = err_json.get("error")
                    if isinstance(err, dict):
                        detail = err.get("message") or str(err)
                    elif err:
                        detail = str(err)
            except Exception:
                detail = resp.text[:300] if resp.text else ""
            raise requests.exceptions.HTTPError(
                f"HTTP {resp.status_code}: {detail or resp.reason}",
                response=resp,
            )

    def _call_openai_compatible(self, messages, user=None, json_mode=False, use_web_search=False, **kwargs):
        """
        调用 OpenAI / Ollama / LM Studio / DeepSeek / One-API 等兼容 Chat Completions 接口
        """
        endpoint = self._build_openai_endpoint(self._api_url)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": messages,
        }
        if self._provider == "openai" and user:
            payload["user"] = user
        is_search_model = "search" in (self._model or "").lower()
        if not is_search_model:
            for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
                if key in kwargs and kwargs[key] is not None:
                    payload[key] = kwargs[key]
        if use_web_search and self._provider == "openai" and not is_search_model:
            if not self._api_url or "api.openai.com" in self._api_url.lower():
                payload["web_search_options"] = {"search_context_size": "low"}

        resp = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            proxies=self._get_request_proxies(endpoint),
            timeout=self._timeout,
        )
        self._raise_for_http_error(resp)
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"AI 响应中缺少 choices 字段: {data}")
        content = (choices[0].get("message") or {}).get("content")
        return content or ""

    def _call_claude(self, messages, json_mode=False, use_web_search=False, **kwargs):
        """
        调用 Anthropic Claude Messages 原生接口
        """
        endpoint = f"{self._build_claude_base(self._api_url)}/messages"
        headers = self._claude_headers(self._api_key)

        system_parts = []
        converted = []
        for msg in messages:
            role = (msg.get("role") or "user").lower()
            text = msg.get("content") or ""
            if role == "system":
                if text:
                    system_parts.append(text)
                continue
            target_role = "assistant" if role == "assistant" else "user"
            if converted and converted[-1]["role"] == target_role:
                converted[-1]["content"] = f"{converted[-1]['content']}\n{text}"
            else:
                converted.append({"role": target_role, "content": text})

        payload = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": converted,
        }
        if system_parts:
            payload["system"] = "\n".join(system_parts)
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        if use_web_search:
            payload["tools"] = [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 2,
                }
            ]

        resp = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            proxies=self._get_request_proxies(endpoint, provider="claude"),
            timeout=self._timeout,
        )
        self._raise_for_http_error(resp)
        data = resp.json()
        blocks = data.get("content") or []
        texts = [
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        result = "\n".join(t for t in texts if t)
        if not result:
            raise ValueError(f"Claude 响应中缺少有效 content 字段: {data}")
        return result

    def _call_gemini(self, messages, json_mode=False, use_web_search=False, **kwargs):
        """
        调用 Google Gemini generateContent 原生接口
        """
        endpoint = self._build_gemini_endpoint(self._api_url, self._model)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }

        system_parts = []
        contents = []
        for msg in messages:
            role = (msg.get("role") or "user").lower()
            text = msg.get("content") or ""
            if role == "system":
                if text:
                    system_parts.append({"text": text})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
            else:
                contents.append({"role": "user", "parts": [{"text": text}]})

        payload = {
            "contents": contents,
            "safetySettings": self._GEMINI_SAFETY_SETTINGS,
        }
        if system_parts:
            payload["system_instruction"] = {"parts": system_parts}

        gen_config = {}
        if kwargs.get("temperature") is not None:
            gen_config["temperature"] = kwargs["temperature"]
        if kwargs.get("top_p") is not None:
            gen_config["topP"] = kwargs["top_p"]
        # Gemini 开启 google_search 工具时不能同时强制 responseMimeType=application/json（部分模型会返回 400）
        if json_mode and not use_web_search:
            gen_config["responseMimeType"] = "application/json"
        if gen_config:
            payload["generationConfig"] = gen_config
        if use_web_search:
            payload["tools"] = [{"google_search": {}}]

        resp = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            proxies=self._get_request_proxies(endpoint),
            timeout=self._timeout,
        )
        self._raise_for_http_error(resp)
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise ValueError(f"Gemini 响应中缺少 candidates 字段: {data}")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        texts = [
            part.get("text", "")
            for part in parts
            if isinstance(part, dict) and "text" in part and part.get("thought") is not True
        ]
        return "".join(texts)

    def list_models(self, provider=None, api_key=None, api_url=None):
        """
        从对应 AI 服务商拉取可用模型列表（优先使用传入参数，未传入则回退当前配置）
        :return: 排序去重后的模型 ID 列表
        """
        self.init_config()
        active_provider = (self._provider if provider is None else provider).strip().lower()
        if active_provider not in self._PROVIDERS:
            active_provider = "openai"
        active_key = (self._api_key if api_key is None else api_key).strip()
        active_url = (self._api_url if api_url is None else api_url).strip().rstrip("/")

        if active_provider == "claude":
            endpoint = f"{self._build_claude_base(active_url)}/models"
            resp = requests.get(
                endpoint,
                headers=self._claude_headers(active_key),
                params={"limit": 1000},
                proxies=self._get_request_proxies(endpoint, provider="claude"),
                timeout=20,
            )
            self._raise_for_http_error(resp)
            data = resp.json()
            ids = [item.get("id") for item in (data.get("data") or []) if item.get("id")]
            return sorted(set(ids))

        if active_provider == "gemini":
            endpoint = f"{self._build_gemini_base(active_url)}/models"
            headers = {"x-goog-api-key": active_key}
            ids = []
            page_token = None
            for _ in range(5):
                params = {"pageSize": 1000}
                if page_token:
                    params["pageToken"] = page_token
                resp = requests.get(
                    endpoint,
                    headers=headers,
                    params=params,
                    proxies=self._get_request_proxies(endpoint, provider="gemini"),
                    timeout=20,
                )
                self._raise_for_http_error(resp)
                data = resp.json()
                for item in data.get("models") or []:
                    methods = item.get("supportedGenerationMethods") or []
                    if "generateContent" not in methods:
                        continue
                    name = item.get("name") or ""
                    model_id = name[len("models/"):] if name.startswith("models/") else name
                    if not model_id:
                        continue
                    lower_id = model_id.lower()
                    if any(kw in lower_id for kw in self._GEMINI_NON_CHAT_KEYWORDS):
                        continue
                    ids.append(model_id)
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
            return sorted(set(ids))

        # openai / local
        endpoint = self._build_openai_models_endpoint(active_url)
        headers = {"Content-Type": "application/json"}
        if active_key:
            headers["Authorization"] = f"Bearer {active_key}"
        resp = requests.get(
            endpoint,
            headers=headers,
            proxies=self._get_request_proxies(endpoint, provider=active_provider),
            timeout=20,
        )
        self._raise_for_http_error(resp)
        data = resp.json()
        raw_ids = [item.get("id") for item in (data.get("data") or []) if item.get("id")]
        is_official_openai = active_provider == "openai" and (
            not active_url or "api.openai.com" in active_url.lower()
        )
        if not is_official_openai:
            return sorted(set(raw_ids))

        filtered = []
        for model_id in raw_ids:
            lower_id = model_id.lower()
            if not lower_id.startswith(self._OPENAI_CHAT_PREFIXES):
                continue
            if any(kw in lower_id for kw in self._OPENAI_NON_CHAT_KEYWORDS):
                continue
            filtered.append(model_id)
        return sorted(set(filtered))

    @classmethod
    def _build_tavily_query(cls, raw_input):
        """
        将复杂媒体文件名/种子名清洗为优先命中中文官方译名的 Tavily 检索词
        """
        lines = [ln.strip() for ln in str(raw_input or "").splitlines() if ln.strip()]
        if not lines:
            return ""

        def _clean_segment(seg):
            seg = re.sub(r"\.(mkv|mp4|avi|ts|m2ts|iso|rmvb|flv|webm|mov|wmv)$", "", seg, flags=re.IGNORECASE)
            # 先剥离方括号内的纯 CRC32 校验码与纯集数块（如 [09]、[11v2]、[A1B2C3D4]），保留含片名的方括号
            seg = re.sub(r"\[(?:[0-9A-Fa-f]{8}|\d{1,3}(?:v\d)?)\]", " ", seg, flags=re.IGNORECASE)
            seg = re.sub(r"[\[\]()【】（）_]+", " ", seg)
            seg = re.sub(r"(?<=\w)\.(?=\w)", " ", seg)
            noise_pattern = (
                r"\b(?:"
                r"4K|8K|2160p|1080p|1080i|720p|576p|480p|UHD|FHD|"
                r"BDRip|BDMV|BluRay|Blu-ray|WEB-DL|WEBRip|HDTV|DVDRip|REMUX|BDRemux|"
                r"HEVC|AVC|H\.?265|H\.?264|x265|x264|AV1|10bit|8bit|Hi10P|Hi444\w*|Ma10p|Ma444\w*|yuv\w*|"
                r"HDR10\+?|HDR|DoVi|DV|SDR|"
                r"FLAC(?:x\d+)?|AAC(?:x\d+)?|AC3|EAC3|DDP[\d.]*|DTS(?:-HD)?|TrueHD|Atmos|LPCM|MP3|"
                r"JPN|JAP|CHS|CHT|JPSC|JPTC|BIG5|GB|ENG|CHI|KOR|60fps|24fps|"
                r"DBD-Raws|VCB-Studio|LoliHouse|Lilith-Raws|Sakurato|ANi|Baha|B-Global|Nekomoe\s+kissaten|"
                r"S\d{1,2}E\d{1,3}|EP?\d{1,3}|v\d"
                r")\b"
            )
            seg = re.sub(noise_pattern, " ", seg, flags=re.IGNORECASE)
            seg = re.sub(r"\s+", " ", seg).strip(" -._/")
            seg = re.sub(r"(?:\s*-\s*\d{1,3}|\s+0\d{1,2})$", "", seg).strip(" -._/")
            return seg

        parts = [p.strip() for p in re.split(r"[\\/]+", lines[0]) if p.strip()]
        cleaned_parts = [_clean_segment(p) for p in parts]
        cleaned_parts = [p for p in cleaned_parts if p and not p.isdigit()]
        if len(cleaned_parts) >= 2 and (
            cleaned_parts[-1].lower() == cleaned_parts[-2].lower()
            or cleaned_parts[-1].lower() in cleaned_parts[-2].lower()
        ):
            cleaned = cleaned_parts[-2]
        elif cleaned_parts:
            cleaned = cleaned_parts[-1]
        else:
            cleaned = lines[0][:100].strip()

        # 若附带了副标题描述行，补充前 40 字符辅助定位中文译名
        if len(lines) > 1:
            sub = re.sub(r"^(?:Subtitle|Torrent description|副标题)[:：\s]*", "", lines[1], flags=re.IGNORECASE).strip()
            if sub and sub not in cleaned:
                cleaned = f"{cleaned} {sub[:40]}".strip()

        cleaned = cleaned[:110].strip()
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", cleaned))
        if not has_chinese:
            cleaned = f"{cleaned} 中文 译名 TMDB"
        elif "tmdb" not in cleaned.lower():
            cleaned = f"{cleaned} TMDB"
        return cleaned

    def _search_tavily(self, raw_input):
        """
        调用 Tavily Search API 获取网页标题与摘要（带内存缓存，每次 basic 检索仅消耗 1 credit）
        """
        query = self._build_tavily_query(raw_input)
        if not query or not self._tavily_api_key:
            return ""
        cached = self._tavily_cache.get(query)
        if cached is not None:
            if log:
                log.debug(f"【AI】Tavily 命中查询缓存：{query}")
            return cached

        if log:
            log.debug(f"【AI】正在调用 Tavily 搜索：{query}")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._tavily_api_key}",
        }
        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
            "include_answer": True,
        }
        resp = requests.post(
            self._TAVILY_ENDPOINT,
            headers=headers,
            json=payload,
            proxies=self._get_request_proxies(self._TAVILY_ENDPOINT, provider="tavily"),
            timeout=15,
        )
        self._raise_for_http_error(resp)
        data = resp.json() if resp.content else {}
        lines = []
        answer = (data.get("answer") or "").strip()
        if answer:
            lines.append(f"Summary: {answer[:400]}")
        for idx, item in enumerate(data.get("results") or [], start=1):
            if not isinstance(item, dict):
                continue
            title = (item.get("title") or "").strip()
            content = re.sub(r"\s+", " ", (item.get("content") or "").strip())[:280]
            url = (item.get("url") or "").strip()
            if title or content:
                lines.append(f"[{idx}] {title} ({url}): {content}")
        formatted = "\n".join(lines).strip()
        if len(self._tavily_cache) >= 256:
            self._tavily_cache.pop(next(iter(self._tavily_cache)), None)
        self._tavily_cache[query] = formatted
        return formatted

    @staticmethod
    def _inject_search_context(messages, search_context):
        """
        将外部搜索摘要注入到最后一条 user 消息中，供离线/免费模型直接使用
        """
        if not search_context:
            return messages
        enriched = [dict(m) for m in messages]
        suffix = (
            "\n\nWeb search reference results (use these real facts to determine the official Chinese title, "
            "original-language title, TMDB season number, and release year):\n"
            f"{search_context}"
        )
        for i in range(len(enriched) - 1, -1, -1):
            if (enriched[i].get("role") or "user").lower() == "user":
                enriched[i]["content"] = f"{enriched[i].get('content') or ''}{suffix}"
                return enriched
        enriched.append({"role": "user", "content": suffix.strip()})
        return enriched

    def _dispatch_model(self, messages, user="NAStool", json_mode=False, use_web_search=False, **kwargs):
        if self._provider == "claude":
            return self._call_claude(messages, json_mode=json_mode, use_web_search=use_web_search, **kwargs)
        if self._provider == "gemini":
            return self._call_gemini(messages, json_mode=json_mode, use_web_search=use_web_search, **kwargs)
        return self._call_openai_compatible(
            messages, user=user, json_mode=json_mode, use_web_search=use_web_search, **kwargs
        )

    def __get_model(
        self,
        message,
        prompt=None,
        user="NAStool",
        json_mode=False,
        use_web_search=False,
        use_tavily=False,
        search_query_source=None,
        **kwargs,
    ):
        """
        按优先级调用 AI 模型：
        1. 若开启模型原生联网且非本地模型：优先尝试模型原生联网；
        2. 若模型不支持联网、模型原生联网配额耗尽/报错，或仅开启了 Tavily：自动回退使用 Tavily 搜索网页摘要注入提示词；
        3. 若 Tavily 亦未启用或失败：自动降级为纯离线模型识别。
        """
        messages = self._normalize_messages(message, prompt=prompt)
        native_available = bool(use_web_search and self._provider != "local")
        tavily_available = bool(use_tavily and self._tavily_api_key)
        cooldown_key = (self._provider, self._model, (self._api_key or "")[:12])
        native_cooled = tavily_available and (
            time.time() < self._native_search_cooldown_until.get(cooldown_key, 0)
        )

        # 1. 模型原生联网优先
        if native_available and not native_cooled:
            try:
                return self._dispatch_model(
                    messages, user=user, json_mode=json_mode, use_web_search=True, **kwargs
                )
            except (_AIRateLimitError, requests.exceptions.HTTPError, requests.exceptions.RequestException) as e:
                if tavily_available:
                    self._native_search_cooldown_until[cooldown_key] = (
                        time.time() + self._NATIVE_SEARCH_COOLDOWN_SECONDS
                    )
                    msg = f"【AI】模型原生联网不可用或配额受限，自动回退至 Tavily 搜索：{str(e)}"
                else:
                    msg = f"【AI】联网搜索不可用或达到联网配额上限，自动降级为离线识别重试：{str(e)}"
                log.warn(msg) if log else print(msg)

        # 2. 回退或直接使用 Tavily 外部搜索增强
        if tavily_available:
            query_source = search_query_source if search_query_source is not None else (
                message if isinstance(message, str) else ""
            )
            try:
                tavily_context = self._search_tavily(query_source)
                if tavily_context:
                    enriched_messages = self._inject_search_context(messages, tavily_context)
                    return self._dispatch_model(
                        enriched_messages, user=user, json_mode=json_mode, use_web_search=False, **kwargs
                    )
            except Exception as e:
                msg = f"【AI】Tavily 搜索不可用或额度耗尽，自动降级为离线识别：{str(e)}"
                log.warn(msg) if log else print(msg)

        # 3. 纯离线模型调用
        return self._dispatch_model(
            messages, user=user, json_mode=json_mode, use_web_search=False, **kwargs
        )

    def get_media_name(self, filename):
        """
        第一阶段：从文件名或种子名中提取媒体名称、原名、多路 TMDB 搜索词与季集要素（支持模型联网优先 + Tavily 回退）
        :param filename: 文件名或种子名（可附带副标题）
        :return: Json dict | None
        """
        if not self.get_state():
            return None
        result = ""
        try:
            any_search_enabled = bool(
                (self._web_search and self._provider != "local")
                or (self._tavily_enable and self._tavily_api_key)
            )
            search_hint = (
                "If the release is a recent/unfamiliar movie, anime theatrical film (e.g. 'Movie 29'), or non-Chinese title, "
                "use web search or any provided web search reference results to find its official Chinese title, original-language title (e.g. Japanese), and release year on TMDB/Wikipedia.\n"
                if any_search_enabled
                else ""
            )
            _filename_prompt = (
                "You are an expert media filename & torrent parser for TMDB metadata matching.\n"
                f"{search_hint}"
                "Rules:\n"
                "1. Strip release/fansub groups ([DBD-Raws], [VCB-Studio], etc.), resolutions, codecs, audio tags, and non-official sequel numbering like 'Movie 29', 'M29', 'The Movie 29', '剧场版29' that TMDB does NOT include in official titles.\n"
                "2. Prefer the official Chinese title found in web search results (e.g. '直至魔女消逝') for 'title'; ONLY put a Chinese title in 'title' if confirmed in web search results or known with high confidence, otherwise put the clean original/English title from the filename (NEVER invent literal word-for-word translations).\n"
                "3. Put official original-language title (e.g. Japanese '名探偵コナン ハイウェイの堕天使') in 'original_title' if known, else ''.\n"
                "4. Put clean English title (without 'Movie XX' numbering, e.g. 'Detective Conan: Fallen Angel of the Highway') in 'en_title' if present/known, else ''.\n"
                "5. Put 2 to 4 TMDB search queries in 'search_keywords' ordered from specific to broad: MUST include standalone distinctive subtitles (e.g. 'Fallen Angel of the Highway') and localized franchise aliases (e.g. 'Case Closed: Fallen Angel of the Highway') when applicable.\n"
                "6. Set 'type' to 'movie' for films & theatrical anime movies ('Movie XX'/'The Movie'/'剧场版' is ALWAYS 'movie' with season=null, episode=null), or 'tv' for episodic TV/anime series.\n"
                "7. For multi-season anime where each season uses a different subtitle/arc name (e.g. 'Kaguya-sama: Ultra Romantic', 'Demon Slayer: Entertainment District Arc / 游郭篇'): put the parent TV series title in 'title'/'original_title'/'en_title' (put the arc subtitle in 'search_keywords'), and set 'season' to its actual TMDB season number (e.g. 2, 3, 4). Use web search if enabled to verify.\n"
                "8. For series specials ('OVA', 'OAD', 'SP', 'Special', '特典') belonging to a TV/anime series (not standalone theatrical films): set 'type' to 'tv', 'season' to 0 (TMDB Specials / S00), and use the same parent TV series title so they organize into the same show folder.\n"
                "9. Return ONLY a valid JSON object without explanation:\n"
                '{"title":string,"original_title":string,"en_title":string,"search_keywords":[string],"type":"movie"|"tv"|null,"version":string,"part":string,"year":string,"resolution":string,"season":number|null,"episode":number|null}'
            )
            result = self.__get_model(
                prompt=_filename_prompt,
                message=filename,
                json_mode=True,
                use_web_search=self._web_search,
                use_tavily=self._tavily_enable,
                search_query_source=filename,
            )
            return self._extract_json(result)
        except _AIRateLimitError as e:
            msg = f"【AI】接口达到频率或额度上限，跳过本次AI识别：{str(e)}"
            log.warn(msg) if log else print(msg)
            return {}
        except Exception as e:
            msg = f"【AI】解析媒体名称失败（{str(e)}）：{result}"
            log.warn(msg) if log else print(msg)
            return {}

    def select_tmdb_candidate(self, filename, candidates, parsed_info=None):
        """
        第二阶段：当字面全等未直接命中时，让 AI 从 TMDB 真实检索候选列表中选出匹配的 tmdb_id
        :param filename: 原始文件名/种子名
        :param candidates: TMDB 候选字典列表
        :param parsed_info: 第一阶段 AI 解析结果（可选参考）
        :return: {"tmdb_id": int|None, "type": "movie"|"tv"|None}
        """
        if not self.get_state() or not candidates:
            return {}
        result = ""
        try:
            _select_prompt = (
                "You match a media filename/torrent against real TMDB search results.\n"
                "Compare the filename and parsed hints against each candidate's title, original_title, year, type, and overview.\n"
                "Rules:\n"
                "1. Recognize cross-language equivalents (e.g. English subtitle 'Fallen Angel of the Highway' == Japanese 'ハイウェイの堕天使' == Chinese '高速公路的堕天使'; 'Detective Conan' == 'Case Closed' == '名侦探柯南').\n"
                "2. Distinguish sequels/installments carefully: do NOT pick a different movie number or TV series if the subtitle/year does not match.\n"
                "3. If NONE of the candidates matches this media file, return {\"tmdb_id\": null, \"type\": null}.\n"
                "4. Return ONLY a valid JSON object:\n"
                '{"tmdb_id":number|null,"type":"movie"|"tv"|null}'
            )
            hint_obj = {}
            if isinstance(parsed_info, dict):
                for k in ("title", "original_title", "en_title", "type", "year", "season", "episode"):
                    if parsed_info.get(k) not in (None, ""):
                        hint_obj[k] = parsed_info.get(k)
            user_msg = (
                f"Filename:\n{filename}\n\n"
                f"Parsed hints:\n{json.dumps(hint_obj, ensure_ascii=False)}\n\n"
                f"TMDB candidates:\n{json.dumps(candidates, ensure_ascii=False)}"
            )
            result = self.__get_model(
                prompt=_select_prompt,
                message=user_msg,
                json_mode=True,
                use_web_search=False,
                temperature=0,
            )
            return self._extract_json(result)
        except Exception as e:
            msg = f"【AI】候选列表二次匹配失败（{str(e)}）：{result}"
            log.warn(msg) if log else print(msg)
            return {}

    def get_answer(self, text, userid):
        """
        获取答案
        :param text: 输入文本
        :param userid: 用户ID
        :return:
        """
        if not self.get_state():
            return ""
        try:
            if not userid:
                return "用户信息错误"
            userid = str(userid)
            if text == "#清除":
                self.__clear_session(userid)
                return "会话已清除"
            # 获取历史上下文
            messages = self.__get_session(userid, text)
            result = self.__get_model(message=messages, user=userid)
            if result:
                self.__save_session(userid, result)
            return result
        except _AIRateLimitError as e:
            return f"请求被AI服务限流了，{str(e)}"
        except requests.exceptions.ConnectionError:
            return "AI服务网络连接失败！"
        except requests.exceptions.Timeout:
            return "没有接收到AI服务的返回消息！"
        except Exception as e:
            return f"请求AI服务出现错误：{str(e)}"

    def translate_to_zh(self, text):
        """
        翻译为中文
        :param text: 输入文本
        """
        if not self.get_state():
            return False, None
        system_prompt = "You are a translation engine that can only translate text and cannot interpret it."
        user_prompt = f"translate to zh-CN:\n\n{text}"
        result = ""
        try:
            result = self.__get_model(
                prompt=system_prompt,
                message=user_prompt,
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            ).strip()
            return True, result
        except Exception as e:
            print(f"{str(e)}：{result}")
            return False, str(e)

    def get_question_answer(self, question):
        """
        从给定问题和选项中获取正确答案
        :param question: 问题及选项
        :return: str | dict | None
        """
        if not self.get_state():
            return None
        result = ""
        try:
            _question_prompt = "下面我们来玩一个游戏，你是老师，我是学生，你需要回答我的问题，我会给你一个题目和几个选项，你的回复必须是给定选项中正确答案对应的序号，请直接回复数字"
            result = self.__get_model(prompt=_question_prompt, message=question)
            return result
        except Exception as e:
            print(f"{str(e)}：{result}")
            return {}
