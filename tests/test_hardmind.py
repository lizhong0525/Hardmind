# -*- coding: utf-8 -*-
"""Hardmind（app.py）修复回归测试。

运行：python -m pytest tests/test_hardmind.py -q
"""
import importlib.util
import json
import os
import sys
import threading
import time
import unittest
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)


def _load_app():
    spec = importlib.util.spec_from_file_location("hardmind_app",
                                                  os.path.join(BASE, "app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


APP = _load_app()


class TestUclaScoring(unittest.TestCase):
    def test_valid_ratings(self):
        total, level = APP.score_ucla([3] * 20)
        self.assertEqual(total, 51)
        self.assertEqual(level, "High")

    def test_out_of_range_returns_none(self):
        # 越界分值（0 或 5）视为无效施测，而不是错误计分
        self.assertEqual(APP.score_ucla([0] * 20), (None, None))
        self.assertEqual(APP.score_ucla([5] + [3] * 19), (None, None))
        self.assertEqual(APP.score_ucla(["3"] * 20), (None, None))

    def test_wrong_length(self):
        self.assertEqual(APP.score_ucla([3] * 19), (None, None))
        self.assertEqual(APP.score_ucla(None), (None, None))


class TestParseUcla(unittest.TestCase):
    def test_clean_array(self):
        raw = "[2, 3, 1, 2, 3, 2, 1, 2, 3, 2, 3, 1, 2, 3, 2, 1, 3, 2, 3, 1]"
        self.assertEqual(len(APP.parse_ucla_response(raw)), 20)

    def test_prose(self):
        raw = "好的，以下是我的评分 [2, 3, 1, 2, 3, 2, 1, 2, 3, 2, 3, 1, 2, 3, 2, 1, 3, 2, 3, 1] 谢谢"
        self.assertEqual(len(APP.parse_ucla_response(raw)), 20)

    def test_garbage(self):
        self.assertIsNone(APP.parse_ucla_response("我不知道"))


class TestDetection(unittest.TestCase):
    def test_cn_loneliness_clues(self):
        # 原实现仅英文词表，中文对话无法触发；现已加入中文线索
        self.assertTrue(APP.detect_loneliness_tendency("最近总是一个人吃饭，也没人说话"))
        self.assertTrue(APP.detect_loneliness_tendency("我感觉好孤独"))
        self.assertFalse(APP.detect_loneliness_tendency("今天天气不错，一起去打球吧"))

    def test_similarity(self):
        self.assertAlmostEqual(APP.compute_similarity("a b c", "a b c"), 1.0)
        self.assertAlmostEqual(APP.compute_similarity("a b", "c d"), 0.0)
        self.assertAlmostEqual(APP.compute_similarity("", ""), 0.0)

    def test_similarity_cjk(self):
        # 中文按标点分词：两段不同中文不应被当作"单个词相同"
        s1 = "今天一个人吃饭，有点无聊"
        s2 = "今天一个人吃饭，有点无聊"
        s3 = "明天和朋友打球，非常开心"
        self.assertAlmostEqual(APP.compute_similarity(s1, s2), 1.0)
        self.assertLess(APP.compute_similarity(s1, s3), 1.0)

    def test_end_detection_word_boundary(self):
        # "stopped" 不应触发 stop 结束检测
        self.assertFalse(APP.detect_conversation_end("I stopped by the library today and borrowed a novel"))
        self.assertTrue(APP.detect_conversation_end("Okay stop"))


class TestPrompts(unittest.TestCase):
    def test_ucla_prompt_is_neutral(self):
        """修复回归：评估提示词不得再预置结论（priming）。"""
        char = {"name": "T", "background": "bg", "personality": "p"}
        prompt = APP.build_ucla_assessment_prompt(char, "Please rate how you have been feeling recently, based on your current real state.")
        for bad in ["generally okay", "heartwarming", "warm, genuine", "feel lighter",
                    "BEFORE this conversation", "AFTER this heartwarming"]:
            self.assertNotIn(bad, prompt)

    def test_think_prompt_uses_total_rounds(self):
        prompt = APP.build_think_prompt("")
        self.assertIn(f"Do {APP.total_rounds} rounds", prompt)
        self.assertNotIn("Do 30 rounds" if APP.total_rounds != 30 else "Do 999 rounds", prompt)

    def test_build_chat_prompt_missing_fields(self):
        # 缺少可选字段时不得抛异常
        char = {"name": "X", "model": "qwen2.5:7b"}
        prompt = APP.build_chat_prompt(char)
        self.assertIn("X", prompt)


class TestSoulLoading(unittest.TestCase):
    def test_load_soul_a(self):
        soul = APP.load_soul_file("soul_a.md")
        self.assertEqual(soul["name"], "Chen Yu")
        self.assertIn("personality", soul)
        self.assertIn("background", soul)

    def test_load_missing_file(self):
        self.assertIsNone(APP.load_soul_file("no_such_file.md"))


class TestTemplates(unittest.TestCase):
    def test_render_replaces_keys(self):
        out = APP.render_template("index.html").decode("utf-8")
        self.assertIn("AI与心理学", out)
        # index.html 不应残留未替换的模板占位符
        self.assertNotIn("{{", out)

    def test_chat_render_keys(self):
        out = APP.render_template("chat.html", char_a_name="陈屿", char_b_name="林晗",
                                  char_a_initial="陈", char_b_initial="林",
                                  char_a_style="s1", char_b_style="s2",
                                  selected_role="a").decode("utf-8")
        self.assertIn("陈屿", out)
        self.assertIn("林晗", out)
        # 修复回归：不应再出现服务端 JS 三元表达式占位符
        self.assertNotIn("{{ selected_role", out)


class TestLiveServer(unittest.TestCase):
    """真实启动 HTTP 服务做集成验证（GET 类接口，不依赖 Ollama）。"""

    @classmethod
    def setUpClass(cls):
        from http.server import HTTPServer
        APP.load_characters()  # 生产环境在 __main__ 中调用；测试环境手动初始化
        cls.server = HTTPServer(("127.0.0.1", 0), APP.Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _get(self, path):
        url = f"http://127.0.0.1:{self.port}{path}"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, b""

    def test_index(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("AI与心理学".encode("utf-8"), body)

    def test_static_css(self):
        status, body = self._get("/static/style.css")
        self.assertEqual(status, 200)
        self.assertGreater(len(body), 100)

    def test_path_traversal_blocked(self):
        # 路径穿越防护：/static/../app.py 必须被拒绝
        status, _ = self._get("/static/../app.py")
        self.assertEqual(status, 403)

    def test_soul_api(self):
        status, body = self._get("/api/soul")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertIn("character_a", data)

    def test_chat_page_renders(self):
        status, body = self._get("/chat")
        self.assertEqual(status, 200)
        text = body.decode("utf-8")
        self.assertIn("Chen Yu", text)
        self.assertNotIn("{{", text)

    def test_unknown_route(self):
        status, _ = self._get("/no/such/path")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
