import ipaddress
import json
import re
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
    _proxies = None
    _timeout = 60

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

        payload = {"contents": contents}
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

    def _dispatch_model(self, messages, user="NAStool", json_mode=False, use_web_search=False, **kwargs):
        if self._provider == "claude":
            return self._call_claude(messages, json_mode=json_mode, use_web_search=use_web_search, **kwargs)
        if self._provider == "gemini":
            return self._call_gemini(messages, json_mode=json_mode, use_web_search=use_web_search, **kwargs)
        return self._call_openai_compatible(
            messages, user=user, json_mode=json_mode, use_web_search=use_web_search, **kwargs
        )

    def __get_model(self, message, prompt=None, user="NAStool", json_mode=False, use_web_search=False, **kwargs):
        """
        按当前配置的服务商调用对应 AI 模型并返回文本内容；若开启联网但模型不支持或联网额度用尽，自动降级离线重试
        """
        messages = self._normalize_messages(message, prompt=prompt)
        enable_search = bool(use_web_search and self._provider != "local")
        if enable_search:
            try:
                return self._dispatch_model(
                    messages, user=user, json_mode=json_mode, use_web_search=True, **kwargs
                )
            except (_AIRateLimitError, requests.exceptions.HTTPError) as e:
                msg = f"【AI】联网搜索不可用或达到联网配额上限，自动降级为离线识别重试：{str(e)}"
                log.warn(msg) if log else print(msg)
        return self._dispatch_model(
            messages, user=user, json_mode=json_mode, use_web_search=False, **kwargs
        )

    def get_media_name(self, filename):
        """
        第一阶段：从文件名或种子名中提取媒体名称、原名、多路 TMDB 搜索词与季集要素（支持可选联网检索）
        :param filename: 文件名或种子名（可附带副标题）
        :return: Json dict | None
        """
        if not self.get_state():
            return None
        result = ""
        try:
            search_hint = (
                "If the release is a recent/unfamiliar movie, anime theatrical film (e.g. 'Movie 29'), or non-Chinese title, "
                "use web search to find its official Chinese title, original-language title (e.g. Japanese), and release year on TMDB/Wikipedia.\n"
                if self._web_search and self._provider != "local"
                else ""
            )
            _filename_prompt = (
                "You are an expert media filename & torrent parser for TMDB metadata matching.\n"
                f"{search_hint}"
                "Rules:\n"
                "1. Strip release/fansub groups ([DBD-Raws], [VCB-Studio], etc.), resolutions, codecs, audio tags, and non-official sequel numbering like 'Movie 29', 'M29', 'The Movie 29', '剧场版29' that TMDB does NOT include in official titles.\n"
                "2. Put official Chinese title in 'title' ONLY if known with high confidence; otherwise put the clean title from the filename (do NOT guess fake translations).\n"
                "3. Put official original-language title (e.g. Japanese '名探偵コナン ハイウェイの堕天使') in 'original_title' if known, else ''.\n"
                "4. Put clean English title (without 'Movie XX' numbering, e.g. 'Detective Conan: Fallen Angel of the Highway') in 'en_title' if present/known, else ''.\n"
                "5. Put 2 to 4 TMDB search queries in 'search_keywords' ordered from specific to broad: MUST include standalone distinctive subtitles (e.g. 'Fallen Angel of the Highway') and localized franchise aliases (e.g. 'Case Closed: Fallen Angel of the Highway') when applicable.\n"
                "6. Set 'type' to 'movie' for films & theatrical anime movies ('Movie XX'/'The Movie'/'剧场版' is ALWAYS 'movie' with season=null, episode=null), or 'tv' for episodic TV/anime series.\n"
                "7. Return ONLY a valid JSON object without explanation:\n"
                '{"title":string,"original_title":string,"en_title":string,"search_keywords":[string],"type":"movie"|"tv"|null,"version":string,"part":string,"year":string,"resolution":string,"season":number|null,"episode":number|null}'
            )
            result = self.__get_model(
                prompt=_filename_prompt,
                message=filename,
                json_mode=True,
                use_web_search=self._web_search,
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
