# -*- coding: utf-8 -*-
"""E5 文化等价性模块单元测试。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from psybench.scales import UCLA3, PHQ9, GAD7
from psybench.souls import get_soul
from psybench.culture import run_e5, analyze_e5, _welch_with_ci
from psybench.assessment import build_survey_messages
from psybench.providers import MockProvider


class TestBilingualScales(unittest.TestCase):
    def test_items_for(self):
        self.assertEqual(len(UCLA3.items_for("en")), 20)
        self.assertEqual(len(PHQ9.items_for("en")), 9)
        self.assertEqual(len(GAD7.items_for("en")), 7)
        self.assertEqual(UCLA3.items_for("zh"), UCLA3.items)

    def test_options_alignment_en(self):
        self.assertEqual(UCLA3.options_for("en")[0], "Never")
        self.assertEqual(PHQ9.options_for("en")[0], "Not at all")


class TestBilingualSouls(unittest.TestCase):
    def test_en_prompt(self):
        p = get_soul("chen_yu").build_system_prompt("en")
        self.assertIn("Chen Yu", p)
        self.assertIn("sophomore", p)
        self.assertNotIn("孤独", p)

    def test_zh_prompt_unchanged(self):
        p = get_soul("chen_yu").build_system_prompt("zh")
        self.assertIn("陈屿", p)


class TestEnSurveyBlind(unittest.TestCase):
    def test_en_survey_no_construct_words(self):
        soul = get_soul("chen_yu")
        for scale in [UCLA3, PHQ9, GAD7]:
            msgs = build_survey_messages(soul, scale, lang="en")
            sys_text = " ".join(m["content"] for m in msgs if m["role"] == "system")
            self.assertNotIn(scale.name, sys_text)
            self.assertNotIn("loneliness", sys_text)
            self.assertNotIn("depression", sys_text)
            user_text = " ".join(m["content"] for m in msgs if m["role"] == "user")
            self.assertIn("JSON", user_text)


class TestWelchWithCi(unittest.TestCase):
    def test_normal(self):
        a = [1, 2, 3, 4, 5]
        b = [6, 7, 8, 9, 10]
        st = _welch_with_ci(a, b)
        self.assertLess(st["p"], 0.01)
        self.assertAlmostEqual(st["diff"], -5.0)
        self.assertEqual(len(st["ci95"]), 2)

    def test_constant_guard(self):
        st = _welch_with_ci([3, 3, 3], [3, 3, 3])
        self.assertIsNone(st["t"])
        self.assertEqual(st["diff"], 0.0)
        self.assertEqual(st["ci95"], [0.0, 0.0])


class TestAnalyzeE5(unittest.TestCase):
    def _synthetic(self):
        """构造合成数据：中文语境总分高于英文；条目 2（非反向题）有明显 DIF。

        注意：条目 1 是反向题，不能用它构造方向断言（反向计分会翻转方向）。
        条目均加少量噪声，避免零方差退化。
        """
        rng = np.random.default_rng(42)
        e5 = {}
        for sid in ["chen_yu", "lin_han", "mo_ran", "zhou_yang"]:
            e5[sid] = {}
            for skey in ["ucla3"]:
                zh_base = [2] * 20
                zh_base[1] = 3  # 条目 2：中文更高（非反向题，计分方向一致）
                en_base = [2] * 20
                en_base[1] = 1  # 条目 2：英文更低
                zh_rows, en_rows = [], []
                for i in range(5):
                    zrow = np.clip(np.round(zh_base + rng.normal(0, 0.45, 20)), 1, 4).astype(int)
                    erow = np.clip(np.round(en_base + rng.normal(0, 0.45, 20)), 1, 4).astype(int)
                    zh_rows.append({"items": [int(x) for x in zrow], "total": int(UCLA3.score(zrow))})
                    en_rows.append({"items": [int(x) for x in erow], "total": int(UCLA3.score(erow))})
                e5[sid][skey] = {"zh": zh_rows, "en": en_rows}
        return e5

    def test_dif_flag(self):
        a = analyze_e5(self._synthetic())
        dif = a["dif"]["ucla3"]
        self.assertEqual(dif[0]["item_idx"], 2)  # 条目 2 差异最大
        self.assertTrue(dif[0]["flag"])

    def test_summary_direction(self):
        a = analyze_e5(self._synthetic())
        s = a["summary"]["ucla3"]
        self.assertGreater(s["mean_diff_zh_minus_en"], 0)
        self.assertIsNotNone(s["ci95"])


class TestRunE5Mock(unittest.TestCase):
    def test_mock_flow(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = MockProvider(seed=3)
            d = run_e5(p, ["chen_yu", "lin_han"], ["ucla3"], 2, tmp)
            self.assertIn("chen_yu", d)
            self.assertEqual(len(d["chen_yu"]["ucla3"]["zh"]), 2)
            self.assertEqual(len(d["chen_yu"]["ucla3"]["en"]), 2)
            a = analyze_e5(d)
            self.assertIn("summary", a)


if __name__ == "__main__":
    unittest.main(verbosity=2)
