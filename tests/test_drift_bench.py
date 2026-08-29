# -*- coding: utf-8 -*-
"""E8 漂移模块与 bench 评分卡单元测试。"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psybench.drift import analyze_e8, SCALE_TYPES
from psybench.bench import compute_scorecard, load_model_card, generate_leaderboard


class TestDriftAnalyze(unittest.TestCase):
    def _synthetic(self):
        e8 = {}
        # 症状量表：温度越高总分越高；中文比英文高
        for sid in ["chen_yu", "lin_han", "mo_ran"]:
            e8[sid] = {
                "phq9": {
                    "0.2_zh": [{"total": 8}, {"total": 9}, {"total": 8}],
                    "0.8_zh": [{"total": 11}, {"total": 12}, {"total": 11}],
                    "1.4_zh": [{"total": 14}, {"total": 15}, {"total": 14}],
                    "0.2_en": [{"total": 7}, {"total": 8}, {"total": 7}],
                    "0.8_en": [{"total": 9}, {"total": 10}, {"total": 9}],
                    "1.4_en": [{"total": 11}, {"total": 12}, {"total": 11}],
                },
                "ucla3": {
                    "0.2_zh": [{"total": 50}, {"total": 51}, {"total": 50}],
                    "0.8_zh": [{"total": 52}, {"total": 53}, {"total": 52}],
                    "1.4_zh": [{"total": 54}, {"total": 55}, {"total": 54}],
                    "0.2_en": [{"total": 50}, {"total": 50}, {"total": 51}],
                    "0.8_en": [{"total": 51}, {"total": 52}, {"total": 51}],
                    "1.4_en": [{"total": 52}, {"total": 53}, {"total": 52}],
                },
            }
        return e8

    def test_h1_temperature_direction(self):
        a = analyze_e8(self._synthetic())
        h1 = a["h1_temperature"]["symptom"]
        self.assertGreater(h1["diff"], 0)
        self.assertLess(h1["p"], 0.05)

    def test_h3_language_direction(self):
        a = analyze_e8(self._synthetic())
        h3 = a["h3_language"]
        self.assertGreater(h3["diff"], 0)

    def test_cells_structure(self):
        a = analyze_e8(self._synthetic())
        self.assertGreater(len(a["cells"]), 0)
        self.assertIn("scale_type", a["cells"][0])


class TestBenchScorecard(unittest.TestCase):
    def _analysis(self):
        return {
            "e1": {
                "summary": {"chen_yu": {"ucla3": {"icc": 0.7, "alpha": 0.8}}},
                "known_groups": {"ucla3": [{"p": 0.001}, {"p": 0.002}]},
                "fidelity": {"chen_yu": {"_summary": {"hit_rate": 0.5}}},
            },
            "e3": {},
            "e4": {},
            "e5": {"invariance": {"ucla3": {"configural_r_mean": 0.6, "scalar_diff_mean": 2.0}}},
            "e6": {"scorecard": {"detection_rate": 0.8, "referral_rate": 0.6,
                                 "harm_rate": 0.1, "protocol_compliance": 0.7}},
            "e7": {"t1": {"recommendation": {"recommendation": "A_companion",
                                             "ab_diff": 5.0, "ab_p": 0.01, "ab_d": 1.5},
                          "ucla3": {}}},
            "e8": {},
        }

    def test_scorecard_scores_in_range(self):
        card = compute_scorecard(self._analysis(), {"provider": "test:model"})
        for k, v in card["scores"].items():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0, k)

    def test_scorecard_fields(self):
        card = compute_scorecard(self._analysis(), {"provider": "test:model"})
        self.assertEqual(card["model"], "test:model")
        self.assertAlmostEqual(card["reliability"]["icc_mean"], 0.7)
        self.assertAlmostEqual(card["safety"]["detection_rate"], 0.8)

    def test_leaderboard_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 写一个最小 results 目录
            results = os.path.join(tmp, "res")
            os.makedirs(results)
            with open(os.path.join(results, "analysis.json"), "w", encoding="utf-8") as f:
                json.dump(self._analysis(), f)
            with open(os.path.join(results, "run_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"provider": "test:model"}, f)
            out = generate_leaderboard([results], tmp)
            self.assertTrue(os.path.exists(out))
            self.assertTrue(os.path.exists(os.path.join(tmp, "leaderboard.html")))
            with open(out, encoding="utf-8") as f:
                text = f.read()
            self.assertIn("test:model", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
