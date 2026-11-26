import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _ensure_stub_dependencies():
    try:
        import requests  # noqa: F401
    except ImportError:
        requests_mod = types.ModuleType("requests")
        exceptions_mod = types.ModuleType("requests.exceptions")

        class RequestException(Exception):
            pass

        class Timeout(RequestException):
            pass

        class ConnectionError(RequestException):
            pass

        class HTTPError(RequestException):
            def __init__(self, *args, response=None):
                super().__init__(*args)
                self.response = response

        exceptions_mod.RequestException = RequestException
        exceptions_mod.Timeout = Timeout
        exceptions_mod.ConnectionError = ConnectionError
        exceptions_mod.HTTPError = HTTPError
        requests_mod.exceptions = exceptions_mod
        requests_mod.post = MagicMock()
        requests_mod.get = MagicMock()
        sys.modules["requests"] = requests_mod
        sys.modules["requests.exceptions"] = exceptions_mod

    class _MemoryCache(dict):
        def get(self, key, default=None):
            return super().get(key, default)

        def set(self, key, value):
            self[key] = value

        def delete(self, key):
            self.pop(key, None)

    class _StubConfig:
        openai_cfg = {}
        laboratory_cfg = {}
        app_cfg = {"rmt_tmdbkey": "tmdb-key", "rmt_match_mode": "normal"}
        media_cfg = {"tmdb_language": "zh"}
        proxies = {"https": "http://127.0.0.1:7890"}

        def get_config(self, node=None):
            if node == "openai":
                return dict(self.openai_cfg)
            if node == "laboratory":
                return dict(self.laboratory_cfg)
            if node == "app":
                return dict(self.app_cfg)
            if node == "media":
                return dict(self.media_cfg)
            return {}

        def get_proxies(self):
            return self.proxies

        def get_tmdbapi_url(self):
            return "api.tmdb.org"

    if "log" not in sys.modules:
        log_mod = types.ModuleType("log")
        log_mod.info = lambda *a, **k: None
        log_mod.warn = lambda *a, **k: None
        log_mod.error = lambda *a, **k: None
        log_mod.debug = lambda *a, **k: None
        sys.modules["log"] = log_mod

    if "config" not in sys.modules:
        config_mod = types.ModuleType("config")
        config_mod.Config = _StubConfig
        config_mod.KEYWORD_BLACKLIST = []
        config_mod.KEYWORD_SEARCH_WEIGHT_3 = 3
        config_mod.KEYWORD_SEARCH_WEIGHT_2 = 2
        config_mod.KEYWORD_SEARCH_WEIGHT_1 = 1
        config_mod.KEYWORD_STR_SIMILARITY_THRESHOLD = 0.5
        config_mod.KEYWORD_DIFF_SCORE_THRESHOLD = 0.5
        sys.modules["config"] = config_mod

    if "app" not in sys.modules:
        app_pkg = types.ModuleType("app")
        app_pkg.__path__ = [str(ROOT_DIR / "app")]
        sys.modules["app"] = app_pkg

    if "app.helper" not in sys.modules:
        helper_pkg = types.ModuleType("app.helper")
        helper_pkg.__path__ = [str(ROOT_DIR / "app" / "helper")]
        sys.modules["app.helper"] = helper_pkg

    if "app.utils" not in sys.modules:
        utils_mod = types.ModuleType("app.utils")
        utils_mod.OpenAISessionCache = _MemoryCache()
        sys.modules["app.utils"] = utils_mod

    if "app.utils.commons" not in sys.modules:
        commons_mod = types.ModuleType("app.utils.commons")
        commons_mod.singleton = lambda cls: cls
        sys.modules["app.utils.commons"] = commons_mod


_ensure_stub_dependencies()

import requests  # noqa: E402
from app.helper.openai_helper import OpenAiHelper  # noqa: E402
from app.utils import OpenAISessionCache  # noqa: E402
from config import Config  # noqa: E402


def _make_response(status_code=200, json_data=None, text="", reason="OK"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.reason = reason
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("no json")
    return resp


class TestOpenAiHelperMultiAI(unittest.TestCase):
    def setUp(self):
        OpenAISessionCache.clear()
        Config.openai_cfg = {}
        Config.proxies = {"https": "http://127.0.0.1:7890"}
        self.helper = OpenAiHelper()

    def test_get_state_across_providers(self):
        Config.openai_cfg = {"provider": "openai", "api_key": "", "api_url": ""}
        self.assertFalse(self.helper.get_state())

        Config.openai_cfg = {"provider": "openai", "api_key": "sk-test", "api_url": ""}
        self.assertTrue(self.helper.get_state())

        Config.openai_cfg = {"provider": "claude", "api_key": "", "api_url": ""}
        self.assertFalse(self.helper.get_state())

        Config.openai_cfg = {"provider": "claude", "api_key": "sk-ant-test", "api_url": ""}
        self.assertTrue(self.helper.get_state())

        Config.openai_cfg = {"provider": "gemini", "api_key": "", "api_url": ""}
        self.assertFalse(self.helper.get_state())

        Config.openai_cfg = {"provider": "gemini", "api_key": "AIza-test", "api_url": ""}
        self.assertTrue(self.helper.get_state())

        Config.openai_cfg = {"provider": "local", "api_key": "", "api_url": ""}
        self.assertFalse(self.helper.get_state())

        Config.openai_cfg = {
            "provider": "local",
            "api_key": "",
            "api_url": "http://127.0.0.1:11434/v1",
        }
        self.assertTrue(self.helper.get_state())

    def test_endpoint_builders(self):
        self.assertEqual(
            OpenAiHelper._build_openai_endpoint(""),
            "https://api.openai.com/v1/chat/completions",
        )
        self.assertEqual(
            OpenAiHelper._build_openai_endpoint("https://api.deepseek.com"),
            "https://api.deepseek.com/v1/chat/completions",
        )
        self.assertEqual(
            OpenAiHelper._build_openai_endpoint("http://127.0.0.1:11434/v1"),
            "http://127.0.0.1:11434/v1/chat/completions",
        )
        self.assertEqual(
            OpenAiHelper._build_openai_endpoint("https://generativelanguage.googleapis.com/v1beta/openai"),
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        )
        self.assertEqual(
            OpenAiHelper._build_openai_endpoint("http://localhost:8000/api/v1/chat/completions"),
            "http://localhost:8000/api/v1/chat/completions",
        )

        self.assertEqual(
            OpenAiHelper._build_claude_base(""),
            "https://api.anthropic.com/v1",
        )
        self.assertEqual(
            OpenAiHelper._build_claude_base("https://claude-proxy.example.com/v1"),
            "https://claude-proxy.example.com/v1",
        )

        self.assertEqual(
            OpenAiHelper._build_gemini_endpoint("", "gemini-2.5-flash"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        )
        self.assertEqual(
            OpenAiHelper._build_gemini_endpoint("https://my-proxy.example.com/v1beta", "gemini-2.5-pro"),
            "https://my-proxy.example.com/v1beta/models/gemini-2.5-pro:generateContent",
        )

    def test_proxy_bypass_for_local_and_private_hosts(self):
        Config.openai_cfg = {
            "provider": "local",
            "api_url": "http://192.168.1.50:11434/v1",
        }
        self.helper.init_config()
        self.assertIsNone(self.helper._get_request_proxies("http://192.168.1.50:11434/v1/chat/completions"))

        Config.openai_cfg = {
            "provider": "openai",
            "api_url": "http://host.docker.internal:1234/v1",
        }
        self.helper.init_config()
        self.assertIsNone(self.helper._get_request_proxies("http://host.docker.internal:1234/v1/chat/completions"))

        Config.openai_cfg = {
            "provider": "gemini",
            "api_key": "AIza-test",
        }
        self.helper.init_config()
        self.assertEqual(
            self.helper._get_request_proxies("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"),
            {"https": "http://127.0.0.1:7890"},
        )

    def test_get_media_name_claude(self):
        Config.openai_cfg = {
            "provider": "claude",
            "api_key": "sk-ant-test",
            "model": "",
        }
        claude_reply = {
            "content": [
                {
                    "type": "text",
                    "text": '{"title":"Dune","version":"","part":"","year":"2021","resolution":"2160p","season":null,"episode":null}',
                }
            ]
        }
        with patch.object(requests, "post", return_value=_make_response(200, claude_reply)) as mock_post:
            res = self.helper.get_media_name("Dune.2021.2160p.WEB-DL.mkv")
            self.assertEqual(res["title"], "Dune")
            self.assertEqual(res["year"], "2021")
            self.assertIsNone(res["season"])

            call_url = mock_post.call_args[0][0]
            call_kwargs = mock_post.call_args[1]
            self.assertEqual(call_url, "https://api.anthropic.com/v1/messages")
            self.assertEqual(call_kwargs["headers"]["x-api-key"], "sk-ant-test")
            self.assertEqual(call_kwargs["headers"]["anthropic-version"], "2023-06-01")
            self.assertIn("system", call_kwargs["json"])
            self.assertEqual(call_kwargs["json"]["max_tokens"], 4096)
            self.assertTrue(all(m["role"] != "system" for m in call_kwargs["json"]["messages"]))

    def test_translate_to_zh_claude_omits_top_p(self):
        Config.openai_cfg = {
            "provider": "claude",
            "api_key": "sk-ant-test",
            "model": "claude-sonnet-4-5",
        }
        claude_reply = {"content": [{"type": "text", "text": "沙丘简介"}]}
        with patch.object(requests, "post", return_value=_make_response(200, claude_reply)) as mock_post:
            ok, text = self.helper.translate_to_zh("Dune overview")
            self.assertTrue(ok)
            self.assertEqual(text, "沙丘简介")
            payload = mock_post.call_args[1]["json"]
            self.assertEqual(payload["temperature"], 0)
            self.assertNotIn("top_p", payload)

    def test_get_media_name_gemini(self):
        Config.openai_cfg = {
            "provider": "gemini",
            "api_key": "AIza-test",
            "model": "",
        }
        gemini_reply = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"thought": True, "text": "thinking..."},
                            {
                                "text": '```json\n{"title":"Inception","version":"","part":"","year":"2010","resolution":"1080p","season":null,"episode":null}\n```'
                            },
                        ]
                    }
                }
            ]
        }
        with patch.object(requests, "post", return_value=_make_response(200, gemini_reply)) as mock_post:
            res = self.helper.get_media_name("Inception.2010.1080p.BluRay.mkv")
            self.assertEqual(res["title"], "Inception")
            self.assertEqual(res["year"], "2010")
            self.assertIsNone(res["season"])

            call_url = mock_post.call_args[0][0]
            call_kwargs = mock_post.call_args[1]
            self.assertIn("/v1beta/models/gemini-2.5-flash:generateContent", call_url)
            self.assertEqual(call_kwargs["headers"]["x-goog-api-key"], "AIza-test")
            self.assertEqual(call_kwargs["json"]["generationConfig"]["responseMimeType"], "application/json")
            self.assertIn("system_instruction", call_kwargs["json"])

    def test_get_media_name_local_with_think_tags(self):
        Config.openai_cfg = {
            "provider": "local",
            "api_key": "",
            "api_url": "http://127.0.0.1:11434/v1",
            "model": "qwen3:8b",
        }
        local_reply = {
            "choices": [
                {
                    "message": {
                        "content": '<think>分析文件名</think>\n{"title":"Breaking Bad","version":"","part":"","year":"2008","resolution":"2160p","season":1,"episode":5}'
                    }
                }
            ]
        }
        with patch.object(requests, "post", return_value=_make_response(200, local_reply)) as mock_post:
            res = self.helper.get_media_name("Breaking.Bad.S01E05.2160p.mkv")
            self.assertEqual(res["title"], "Breaking Bad")
            self.assertEqual(res["season"], 1)
            self.assertEqual(res["episode"], 5)

            call_url = mock_post.call_args[0][0]
            call_kwargs = mock_post.call_args[1]
            self.assertEqual(call_url, "http://127.0.0.1:11434/v1/chat/completions")
            self.assertNotIn("Authorization", call_kwargs["headers"])
            self.assertIsNone(call_kwargs["proxies"])
            self.assertEqual(call_kwargs["json"]["model"], "qwen3:8b")

    def test_list_models_openai_official_filtered(self):
        Config.openai_cfg = {"provider": "gemini", "api_key": "AIza-saved"}
        self.helper.init_config()
        openai_models = {
            "data": [
                {"id": "gpt-4o"},
                {"id": "text-embedding-3-small"},
                {"id": "whisper-1"},
                {"id": "o3-mini"},
                {"id": "dall-e-3"},
            ]
        }
        with patch.object(requests, "get", return_value=_make_response(200, openai_models)) as mock_get:
            models = self.helper.list_models(provider="openai", api_key="sk-test", api_url="")
            self.assertEqual(models, ["gpt-4o", "o3-mini"])
            self.assertEqual(mock_get.call_args[0][0], "https://api.openai.com/v1/models")
            # 确认传入临时参数不会覆盖实例本身保存的 provider
            self.assertEqual(self.helper._provider, "gemini")

    def test_list_models_local_unfiltered(self):
        local_models = {
            "data": [
                {"id": "qwen2.5:7b"},
                {"id": "deepseek-r1:14b"},
            ]
        }
        with patch.object(requests, "get", return_value=_make_response(200, local_models)) as mock_get:
            models = self.helper.list_models(provider="local", api_key="", api_url="http://127.0.0.1:11434/v1")
            self.assertEqual(models, ["deepseek-r1:14b", "qwen2.5:7b"])
            call_url = mock_get.call_args[0][0]
            call_kwargs = mock_get.call_args[1]
            self.assertEqual(call_url, "http://127.0.0.1:11434/v1/models")
            self.assertNotIn("Authorization", call_kwargs["headers"])
            self.assertIsNone(call_kwargs["proxies"])

    def test_list_models_claude(self):
        claude_models = {
            "data": [
                {"id": "claude-sonnet-4-5"},
                {"id": "claude-3-5-haiku-20241022"},
            ]
        }
        with patch.object(requests, "get", return_value=_make_response(200, claude_models)) as mock_get:
            models = self.helper.list_models(provider="claude", api_key="sk-ant-test", api_url="")
            self.assertEqual(models, ["claude-3-5-haiku-20241022", "claude-sonnet-4-5"])
            call_url = mock_get.call_args[0][0]
            call_kwargs = mock_get.call_args[1]
            self.assertEqual(call_url, "https://api.anthropic.com/v1/models")
            self.assertEqual(call_kwargs["headers"]["x-api-key"], "sk-ant-test")

    def test_list_models_gemini(self):
        gemini_models = {
            "models": [
                {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
            ]
        }
        with patch.object(requests, "get", return_value=_make_response(200, gemini_models)) as mock_get:
            models = self.helper.list_models(provider="gemini", api_key="AIza-test", api_url="")
            self.assertEqual(models, ["gemini-2.5-flash"])
            call_url = mock_get.call_args[0][0]
            call_kwargs = mock_get.call_args[1]
            self.assertEqual(call_url, "https://generativelanguage.googleapis.com/v1beta/models")
            self.assertEqual(call_kwargs["headers"]["x-goog-api-key"], "AIza-test")

    def test_get_answer_session_and_errors(self):
        Config.openai_cfg = {"provider": "openai", "api_key": "sk-test"}
        reply = {"choices": [{"message": {"content": "你好，我是AI助手"}}]}
        with patch.object(requests, "post", return_value=_make_response(200, reply)):
            ans = self.helper.get_answer("你好", "user1")
            self.assertEqual(ans, "你好，我是AI助手")
            history = OpenAISessionCache.get("user1")
            self.assertEqual(history[-1], {"role": "assistant", "content": "你好，我是AI助手"})

        self.assertEqual(self.helper.get_answer("#清除", "user1"), "会话已清除")
        self.assertIsNone(OpenAISessionCache.get("user1"))

        with patch.object(requests, "post", return_value=_make_response(429, text="rate limit")):
            self.assertIn("限流", self.helper.get_answer("hello", "user1"))

        with patch.object(requests, "post", side_effect=requests.exceptions.Timeout("timed out")):
            self.assertEqual(self.helper.get_answer("hello", "user1"), "没有接收到AI服务的返回消息！")

        with patch.object(requests, "post", side_effect=requests.exceptions.ConnectionError("conn err")):
            self.assertEqual(self.helper.get_answer("hello", "user1"), "AI服务网络连接失败！")

    def test_translate_and_question_answer(self):
        Config.openai_cfg = {"provider": "gemini", "api_key": "AIza-test"}
        trans_reply = {"candidates": [{"content": {"parts": [{"text": " 电影简介 "}]}}]}
        with patch.object(requests, "post", return_value=_make_response(200, trans_reply)):
            ok, text = self.helper.translate_to_zh("Movie overview")
            self.assertTrue(ok)
            self.assertEqual(text, "电影简介")

        qa_reply = {"candidates": [{"content": {"parts": [{"text": "2"}]}}]}
        with patch.object(requests, "post", return_value=_make_response(200, qa_reply)):
            ans = self.helper.get_question_answer("1+1=? 1:1 2:2")
            self.assertEqual(ans, "2")


def _load_media_class():
    from enum import Enum

    class MediaType(Enum):
        MOVIE = "电影"
        TV = "电视剧"
        ANIME = "动漫"

    class MatchMode(Enum):
        NORMAL = "normal"
        STRICT = "strict"

    class _DummyMetaInfo:
        def __init__(self, title, subtitle=None, mtype=None, filePath=None, **kwargs):
            self.org_string = title
            self.subtitle = subtitle
            self._name = "WrongRegexTitle" if "MessyGroup" in (title or "") else (title or "")
            self.year = None
            self.type = mtype or MediaType.TV
            self.begin_season = None
            self.end_season = None
            self.begin_episode = None
            self.end_episode = None
            self.tmdb_info = None

        def get_name(self):
            return self._name

        def get_season_string(self):
            return f"S{self.begin_season:02d}" if self.begin_season else "S01"

        def get_episode_string(self):
            return f"E{self.begin_episode:02d}" if self.begin_episode else ""

        def set_season(self, sea):
            if sea is not None and str(sea).isdigit():
                self.begin_season = int(sea)

        def set_episode(self, ep):
            if ep is not None and str(ep).isdigit():
                self.begin_episode = int(ep)

        def set_tmdb_info(self, info):
            self.tmdb_info = info

    class _DummyMetaHelper:
        def __init__(self):
            self.store = {}

        def get_meta_data_by_key(self, key):
            return self.store.get(key) or {}

        def update_meta_data(self, mapping):
            self.store.update(mapping)

    if "zhconv" not in sys.modules:
        zh = types.ModuleType("zhconv")
        zh.convert = lambda s, _: s
        sys.modules["zhconv"] = zh

    if "lxml" not in sys.modules:
        lxml_mod = types.ModuleType("lxml")
        lxml_mod.etree = MagicMock()
        sys.modules["lxml"] = lxml_mod

    sys.modules["app.helper"].MetaHelper = _DummyMetaHelper

    if "app.media" not in sys.modules:
        media_pkg = types.ModuleType("app.media")
        media_pkg.__path__ = [str(ROOT_DIR / "app" / "media")]
        sys.modules["app.media"] = media_pkg

    metainfo_mod = types.ModuleType("app.media.meta.metainfo")
    metainfo_mod.MetaInfo = _DummyMetaInfo
    sys.modules["app.media.meta"] = types.ModuleType("app.media.meta")
    sys.modules["app.media.meta.metainfo"] = metainfo_mod

    tmdb_mod = types.ModuleType("app.media.tmdbv3api")
    for cls_name in ("TMDb", "Search", "Movie", "TV", "Person", "Find", "Discover", "Trending", "Episode", "Genre"):
        setattr(tmdb_mod, cls_name, MagicMock)
    tmdb_mod.TMDbException = Exception
    sys.modules["app.media.tmdbv3api"] = tmdb_mod

    utils_mod = sys.modules["app.utils"]
    for attr in ("PathUtils", "EpisodeFormat", "RequestUtils", "NumberUtils", "StringUtils"):
        setattr(utils_mod, attr, MagicMock())
    utils_mod.StringUtils.xstr = lambda v: "" if v is None else str(v)
    utils_mod.cacheman = {"tmdb_supply": {}}

    types_mod = types.ModuleType("app.utils.types")
    types_mod.MediaType = MediaType
    types_mod.MatchMode = MatchMode
    sys.modules["app.utils.types"] = types_mod

    from app.media.media import Media
    return Media, MediaType


class TestMediaAIPriorityAndFallback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Media, cls.MediaType = _load_media_class()

    def setUp(self):
        self.Media._ai_mem_cache.clear()
        Config.laboratory_cfg = {"chatgpt_enable": True}
        Config.openai_cfg = {
            "provider": "gemini",
            "api_key": "AIza-test",
            "model": "gemini-2.5-flash",
            "ai_priority": False,
        }
        self.media = self.Media()

        def fake_search_tmdb(file_media_name, search_type, first_media_year=None, media_year=None, season_number=None):
            if file_media_name == "WrongRegexTitle":
                return {"id": 999, "name": "错误匹配的剧集", "media_type": self.MediaType.TV, "genres": [1]}
            if file_media_name == "幕府将军":
                return {"id": 126308, "name": "幕府将军", "media_type": self.MediaType.TV, "genres": [1]}
            return {}

        self.media._Media__search_tmdb = MagicMock(side_effect=fake_search_tmdb)
        self.media._Media__search_multi_tmdb = MagicMock(return_value={})
        self.media.get_tmdb_info = MagicMock(
            side_effect=lambda mtype, tmdbid, **kw: {"id": tmdbid, "name": "幕府将军" if tmdbid == 126308 else "错误匹配的剧集", "media_type": mtype, "genres": [1]}
        )

    def test_default_mode_uses_tmdb_first_even_when_wrong(self):
        Config.openai_cfg["ai_priority"] = False
        with patch.object(requests, "post") as mock_post:
            info = self.media.get_media_info("[MessyGroup] Shogun S02E03 2160p", subtitle="幕府将军 第二季 第3集")
            self.assertEqual(info.tmdb_info["id"], 999)
            mock_post.assert_not_called()

    def test_ai_priority_runs_ai_first_and_fixes_title_and_season(self):
        Config.openai_cfg["ai_priority"] = True
        gemini_reply = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"title":"幕府将军","en_title":"Shogun","type":"tv","version":"","part":"","year":"2024","resolution":"2160p","season":2,"episode":3}'
                            }
                        ]
                    }
                }
            ]
        }
        with patch.object(requests, "post", return_value=_make_response(200, gemini_reply)) as mock_post:
            info = self.media.get_media_info("[MessyGroup] Shogun S02E03 2160p", subtitle="幕府将军 第二季 第3集")
            self.assertEqual(info.tmdb_info["id"], 126308)
            self.assertEqual(info.begin_season, 2)
            self.assertEqual(info.begin_episode, 3)
            mock_post.assert_called_once()

    def test_ai_priority_falls_back_to_tmdb_on_rate_limit_and_recovers_later(self):
        Config.openai_cfg["ai_priority"] = True
        # 第一次：AI 触发 429 配额上限 -> 自动回退常规 TMDB（不会把失败结果锁死在缓存中）
        with patch.object(requests, "post", return_value=_make_response(429, text="quota exceeded")):
            fallback_info = self.media.get_media_info("[MessyGroup] Shogun S02E03 2160p", cache=False)
            self.assertEqual(fallback_info.tmdb_info["id"], 999)

        # 第二次：配额恢复 -> 再次优先走 AI 并拿到正确结果
        gemini_reply = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"title":"幕府将军","en_title":"Shogun","type":"tv","version":"","part":"","year":"2024","resolution":"2160p","season":2,"episode":3}'
                            }
                        ]
                    }
                }
            ]
        }
        with patch.object(requests, "post", return_value=_make_response(200, gemini_reply)):
            recovered_info = self.media.get_media_info("[MessyGroup] Shogun S02E03 2160p", cache=False)
            self.assertEqual(recovered_info.tmdb_info["id"], 126308)
            self.assertEqual(recovered_info.begin_season, 2)

    def test_gemini_web_search_grounding_and_auto_downgrade(self):
        Config.openai_cfg = {
            "provider": "gemini",
            "api_key": "AIza-test",
            "model": "gemini-2.5-flash",
            "web_search": True,
        }
        helper = OpenAiHelper()
        grounded_reply = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Found via Google Search:\n"},
                            {
                                "text": '```json\n{"title":"名侦探柯南：高速公路的堕天使","original_title":"名探偵コナン ハイウェイの堕天使","en_title":"Detective Conan: Fallen Angel of the Highway","search_keywords":["Fallen Angel of the Highway"],"type":"movie","version":"","part":"","year":"2026","resolution":"2160p","season":null,"episode":null}\n```'
                            },
                        ]
                    }
                }
            ]
        }
        with patch.object(requests, "post", return_value=_make_response(200, grounded_reply)) as mock_post:
            res = helper.get_media_name(
                "[4K][DBD-Raws][Detective Conan Movie 29 Fallen Angel of the Highway][2160P][BDRip].mkv"
            )
            self.assertEqual(res["title"], "名侦探柯南：高速公路的堕天使")
            self.assertEqual(res["year"], "2026")
            payload = mock_post.call_args[1]["json"]
            self.assertEqual(payload["tools"], [{"google_search": {}}])
            # 开启 google_search 时不能带 responseMimeType=application/json
            self.assertNotIn("responseMimeType", payload.get("generationConfig", {}))

        # 若联网配额用尽或模型不支持 google_search 返回 400，应自动无感降级为离线 JSON 模式重试
        offline_reply = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"title":"Detective Conan: Fallen Angel of the Highway","original_title":"","en_title":"Detective Conan: Fallen Angel of the Highway","search_keywords":["Fallen Angel of the Highway"],"type":"movie","version":"","part":"","year":"","resolution":"2160p","season":null,"episode":null}'
                            }
                        ]
                    }
                }
            ]
        }
        with patch.object(
            requests,
            "post",
            side_effect=[
                _make_response(400, json_data={"error": {"message": "Search grounding not supported"}}),
                _make_response(200, offline_reply),
            ],
        ) as mock_post:
            res2 = helper.get_media_name(
                "[4K][DBD-Raws][Detective Conan Movie 29 Fallen Angel of the Highway][2160P][BDRip].mkv"
            )
            self.assertEqual(res2["search_keywords"], ["Fallen Angel of the Highway"])
            self.assertEqual(mock_post.call_count, 2)
            second_payload = mock_post.call_args_list[1][1]["json"]
            self.assertNotIn("tools", second_payload)
            self.assertEqual(second_payload["generationConfig"]["responseMimeType"], "application/json")

    def test_claude_web_search_with_preamble_text(self):
        Config.openai_cfg = {
            "provider": "claude",
            "api_key": "sk-ant-test",
            "model": "claude-sonnet-4-5",
            "web_search": True,
        }
        helper = OpenAiHelper()
        claude_reply = {
            "content": [
                {"type": "text", "text": "Let me search for Detective Conan Movie 29..."},
                {"type": "server_tool_use", "name": "web_search"},
                {
                    "type": "text",
                    "text": '{"title":"名侦探柯南：高速公路的堕天使","original_title":"名探偵コナン ハイウェイの堕天使","en_title":"Detective Conan: Fallen Angel of the Highway","search_keywords":["Fallen Angel of the Highway"],"type":"movie","version":"","part":"","year":"2026","resolution":"2160p","season":null,"episode":null}',
                },
            ]
        }
        with patch.object(requests, "post", return_value=_make_response(200, claude_reply)) as mock_post:
            res = helper.get_media_name("Detective Conan Movie 29 Fallen Angel of the Highway.mkv")
            self.assertEqual(res["title"], "名侦探柯南：高速公路的堕天使")
            payload = mock_post.call_args[1]["json"]
            self.assertEqual(payload["tools"][0]["type"], "web_search_20250305")

    def test_two_stage_conan_movie_29_candidate_selection(self):
        """
        模拟柯南 M29 场景：即使第一阶段精确相等未命中（例如离线模式且 TMDB 译名为 Case Closed / 日文原名），
        第二阶段通过副标题 'Fallen Angel of the Highway' 从 TMDB 召回候选列表并由 AI 精准拍板命中
        """
        Config.openai_cfg = {
            "provider": "gemini",
            "api_key": "AIza-test",
            "model": "gemini-2.5-flash",
            "ai_priority": True,
            "web_search": False,
        }
        # 常规精确全等返回空（模拟 Detective Conan != Case Closed 导致的字面不相等）
        self.media._Media__search_tmdb = MagicMock(return_value={})
        self.media._Media__search_multi_tmdb = MagicMock(return_value={})

        # 模拟 TMDB 原始搜索结果：包含 M28 干扰项与 M29 真条目
        def fake_movies_search(params):
            q = params.get("query", "")
            if "Fallen Angel of the Highway" in q:
                return [
                    {
                        "id": 280001,
                        "title": "名侦探柯南：独眼的残像",
                        "original_title": "名探偵コナン 隻眼の残像",
                        "release_date": "2025-04-18",
                        "overview": "第28部剧场版",
                    },
                    {
                        "id": 290029,
                        "title": "名侦探柯南：高速公路的堕天使",
                        "original_title": "名探偵コナン ハイウェイの堕天使",
                        "release_date": "2026-04-10",
                        "overview": "第29部剧场版，神奈川县警萩原千速与失控黑色摩托车路西法的事件",
                    },
                ]
            return []

        self.media.search = MagicMock()
        self.media.search.movies = MagicMock(side_effect=fake_movies_search)
        self.media.search.multi = MagicMock(return_value=[])
        self.media.get_tmdb_info = MagicMock(
            side_effect=lambda mtype, tmdbid, **kw: {
                "id": tmdbid,
                "title": "名侦探柯南：高速公路的堕天使",
                "original_title": "名探偵コナン ハイウェイの堕天使",
                "media_type": self.MediaType.MOVIE,
                "genres": [16],
            }
        )

        stage1_reply = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"title":"Detective Conan: Fallen Angel of the Highway","original_title":"","en_title":"Detective Conan: Fallen Angel of the Highway","search_keywords":["Detective Conan Fallen Angel of the Highway","Fallen Angel of the Highway"],"type":"movie","version":"BDRip","part":"","year":"","resolution":"2160p","season":null,"episode":null}'
                            }
                        ]
                    }
                }
            ]
        }
        stage2_reply = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '{"tmdb_id":290029,"type":"movie"}'}
                        ]
                    }
                }
            ]
        }

        fname = "[4K][DBD-Raws][Detective Conan Movie 29 Fallen Angel of the Highway][2160P][BDRip][HEVC-10bit][JPN][FLACx3].mkv"
        with patch.object(
            requests,
            "post",
            side_effect=[_make_response(200, stage1_reply), _make_response(200, stage2_reply)],
        ) as mock_post:
            info = self.media.get_media_info(fname, cache=False)
            self.assertEqual(info.tmdb_info["id"], 290029)
            self.assertEqual(info.tmdb_info["title"], "名侦探柯南：高速公路的堕天使")
            self.assertEqual(info.type, self.MediaType.MOVIE)
            self.assertIsNone(info.begin_season)
            self.assertIsNone(info.begin_episode)
            self.assertEqual(mock_post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
