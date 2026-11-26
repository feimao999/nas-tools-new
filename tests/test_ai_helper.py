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
        proxies = {"https": "http://127.0.0.1:7890"}

        def get_config(self, node=None):
            if node == "openai":
                return dict(self.openai_cfg)
            return {}

        def get_proxies(self):
            return self.proxies

    if "config" not in sys.modules:
        config_mod = types.ModuleType("config")
        config_mod.Config = _StubConfig
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


if __name__ == "__main__":
    unittest.main()
