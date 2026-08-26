# -*- coding: utf-8 -*-
"""psybench 单元测试：量表计分 / 解析 / 统计 / 边界。

运行：python -m pytest tests/ -q   （或 python tests/test_psybench.py）
"""
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from psybench.scales import ALL_SCALES, UCLA3, PHQ9, GAD7, PSS10, WHO5
from psybench.assessment import parse_item_scores, build_survey_messages
from psybench.souls import get_soul, ALL_SOULS
from psybench.psychometrics import (
    cronbach_alpha, icc_21, welch_t, paired_stats, fidelity_metrics,
)


class TestScales(unittest.TestCase):
    def test_ucla_reverse_scoring(self):
        # 全 3 分：非反向 11 题×3 + 反向 9 题×(5-3)=2 → 33+18=51
        self.assertEqual(UCLA3.score([3] * 20), 51)
        # 全 1 分：非反向 11×1 + 反向 9×(5-1)=4 → 11+36=47
        self.assertEqual(UCLA3.score([1] * 20), 47)
        # 极端：全 4 → 11*4 + 9*1 = 53；全 1 反向高孤独方向检验
        self.assertEqual(UCLA3.score([4] * 20), 53)

    def test_phq9_gad7_no_reverse(self):
        self.assertEqual(PHQ9.score([1] * 9), 9)
        self.assertEqual(GAD7.score([0] * 7), 0)
        self.assertEqual(GAD7.score([3] * 7), 21)

    def test_pss10_reverse(self):
        # 全 0：正向 6 题×0 + 反向 4 题×(4-0)=4 → 16（压力感高）
        self.assertEqual(PSS10.score([0] * 10), 16)
        self.assertEqual(PSS10.score([4] * 10), 24)

    def test_who5_multiplier(self):
        # 全 4 分：原始 20 ×4 = 80
        self.assertEqual(WHO5.score([4] * 5), 80)
        self.assertEqual(WHO5.score([5] * 5), 100)
        self.assertEqual(WHO5.score([0] * 5), 0)
        self.assertEqual(WHO5.band(80), "幸福感正常")
        self.assertEqual(WHO5.band(20), "幸福感很低（可能提示抑郁筛查）")

    def test_score_length_validation(self):
        with self.assertRaises(ValueError):
            UCLA3.score([1, 2, 3])

    def test_all_scales_registered(self):
        for key in ["ucla3", "phq9", "gad7", "pss10", "who5"]:
            self.assertIn(key, ALL_SCALES)


class TestParse(unittest.TestCase):
    def test_clean_json_array(self):
        raw = "[2, 3, 1, 2, 3, 2, 1, 2, 3, 2, 3, 1, 2, 3, 2, 1, 3, 2, 3, 1]"
        self.assertEqual(parse_item_scores(raw, UCLA3),
                         [2, 3, 1, 2, 3, 2, 1, 2, 3, 2, 3, 1, 2, 3, 2, 1, 3, 2, 3, 1])

    def test_json_with_prose(self):
        raw = "以下是我的评分：[1,1,1,1,1,1,1,1,1]"
        self.assertEqual(parse_item_scores(raw, PHQ9), [1] * 9)

    def test_linewise_fallback(self):
        raw = "1. 3\n2. 2\n3. 1\n4. 2\n5. 3\n6. 2\n7. 1\n8. 2\n9. 3"
        self.assertEqual(parse_item_scores(raw, PHQ9), [3, 2, 1, 2, 3, 2, 1, 2, 3])

    def test_garbage(self):
        self.assertIsNone(parse_item_scores("不知道", UCLA3))

    def test_options_labels_align_with_score_range(self):
        """回归：选项标签必须从 scale.min_score 开始枚举。

        此前从 0 开始，导致 UCLA-3（1-4）施测时模型按 0-3 作答，
        总分被系统性污染（真实模型冒烟测试捕获）。
        """
        soul = get_soul("chen_yu")
        msgs = build_survey_messages(soul, UCLA3)
        user_text = " ".join(m["content"] for m in msgs if m["role"] == "user")
        self.assertIn("1=从不", user_text)
        self.assertIn("4=一直", user_text)
        self.assertNotIn("0=从不", user_text)
        msgs2 = build_survey_messages(soul, PHQ9)
        user2 = " ".join(m["content"] for m in msgs2 if m["role"] == "user")
        self.assertIn("0=完全不会", user2)
        self.assertIn("3=几乎每天", user2)

    def test_survey_prompt_is_blind(self):
        """盲测协议回归：指导语不得包含量表名、构念词、结论词。

        注意：条目文本本身是量表原文（含构念词是必然的），
        盲测只约束"指导语与方向暗示"，不约束条目内容。
        """
        soul = get_soul("chen_yu")
        for scale in [UCLA3, PHQ9, GAD7]:
            msgs = build_survey_messages(soul, scale)
            sys_texts = [m["content"] for m in msgs if m["role"] == "system"]
            text = " ".join(sys_texts)
            self.assertNotIn(scale.name, text)
            self.assertNotIn(scale.abbr, text)
            self.assertNotIn(scale.construct, text)
            for bad in ["孤独", "抑郁", "焦虑", "温暖", "好转", "糟糕", "心情总体不错"]:
                self.assertNotIn(bad, text)
        # 灵魂人设本身不贴标签
        persona = get_soul("chen_yu").persona
        for bad in ["孤独", "孤单", "寂寞", "抑郁"]:
            self.assertNotIn(bad, persona)


class TestPsychometrics(unittest.TestCase):
    def test_cronbach_alpha_known(self):
        # 构造高度一致的矩阵：α 应接近 1
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, 5)
        X = np.column_stack([base + rng.normal(0, 0.05, 5) for _ in range(10)])
        a = cronbach_alpha(X)["alpha"]
        self.assertGreater(a, 0.95)
        # 完全随机：α 应较低
        X2 = rng.normal(0, 1, (5, 10))
        a2 = cronbach_alpha(X2)["alpha"]
        self.assertLess(a2, 0.5)

    def test_cronbach_alpha_constant_returns_none(self):
        X = np.ones((5, 10)) * 3
        self.assertIsNone(cronbach_alpha(X)["alpha"])

    def test_icc21_known(self):
        # 完全一致的行 → ICC 接近 1
        X = np.tile(np.arange(5).reshape(5, 1), (1, 3)).astype(float)
        self.assertGreater(icc_21(X)["icc"], 0.99)
        # 纯噪声 → ICC 低
        rng = np.random.default_rng(1)
        X2 = rng.normal(0, 1, (10, 3))
        self.assertLess(icc_21(X2)["icc"], 0.5)

    def test_welch_t(self):
        a = [1, 2, 3, 4, 5]
        b = [10, 11, 12, 13, 14]
        st = welch_t(a, b)
        self.assertLess(st["p"], 0.001)
        self.assertAlmostEqual(st["d"], -5.692, places=2)

    def test_paired_stats_normal(self):
        pre = [10, 12, 11, 13, 12]
        post = [9, 11, 10, 12, 10]
        st = paired_stats(pre, post)
        self.assertAlmostEqual(st["mean_diff"], 1.2, places=6)
        self.assertLess(st["p"], 0.05)

    def test_paired_stats_degenerate(self):
        st = paired_stats([10, 10, 10], [8, 8, 8])
        self.assertEqual(st["mean_diff"], 2.0)
        self.assertEqual(st["p"], 0.0)
        self.assertIsNone(st["dz"])
        self.assertEqual(st["ci95"], [2.0, 2.0])

    def test_paired_stats_small_n(self):
        st = paired_stats([10], [9])
        self.assertIsNone(st["t"])
        self.assertIsNone(st["p"])

    def test_fidelity_metrics(self):
        f = fidelity_metrics({"ucla3": [55, 58, 60]}, {"ucla3": (50, 65)})
        self.assertTrue(f["ucla3"]["hit"])
        self.assertEqual(f["ucla3"]["deviation"], 0.0)
        f2 = fidelity_metrics({"ucla3": [80]}, {"ucla3": (20, 30)})
        self.assertFalse(f2["ucla3"]["hit"])
        self.assertGreater(f2["ucla3"]["deviation_normalized"], 1.0)


class TestSouls(unittest.TestCase):
    def test_prompt_contains_newlines_and_name(self):
        p = get_soul("chen_yu").build_system_prompt()
        self.assertIn("陈屿", p)
        self.assertIn("\n", p)

    def test_declared_ranges_sane(self):
        for sid, soul in ALL_SOULS.items():
            for key, (lo, hi) in soul.declared.items():
                self.assertLessEqual(lo, hi)
                scale = ALL_SCALES[key]
                self.assertGreaterEqual(lo, scale.score_range[0])
                self.assertLessEqual(hi, scale.score_range[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
