# -*- coding: utf-8 -*-
"""E7 数字孪生预演模块单元测试。"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psybench.twin import (
    TWIN_PROFILES, INTERVENTIONS, build_twin, run_chat_session,
    run_e7, analyze_e7,
)
from psybench.providers import MockProvider


class TestBuildTwin(unittest.TestCase):
    def test_persona_no_labels(self):
        twin = build_twin(TWIN_PROFILES["t1_high_lonely"])
        for bad in ["孤独", "孤单", "抑郁", "焦虑", "分数", "55", "65"]:
            self.assertNotIn(bad, twin.persona)
        self.assertIn("陈屿", twin.persona)

    def test_declared_kept(self):
        twin = build_twin(TWIN_PROFILES["t1_high_lonely"])
        self.assertEqual(twin.declared["ucla3"], (55, 65))

    def test_system_prompt(self):
        twin = build_twin(TWIN_PROFILES["t2_high_pressure"])
        p = twin.build_system_prompt()
        self.assertIn("周扬", p)
        self.assertIn("名字叫", p)


class TestInterventions(unittest.TestCase):
    def test_arms_no_efficacy_hints(self):
        for arm, text in INTERVENTIONS.items():
            for bad in ["降低", "改善", "孤独感下降", "治愈", "疗效"]:
                self.assertNotIn(bad, text)

    def test_chat_session_structure(self):
        p = MockProvider(seed=5)
        twin = build_twin(TWIN_PROFILES["t1_high_lonely"])
        it = run_chat_session(p, twin, INTERVENTIONS["A_companion"], rounds=3)
        self.assertEqual(len(it["transcript"]), 1 + 3 * 2)
        self.assertIn("history", it)


class TestAnalyzeE7(unittest.TestCase):
    def _synthetic(self):
        def rec(pre_u, post_u):
            return {"pre": {"ucla3": {"total": pre_u},
                            "phq9": {"total": 10}, "gad7": {"total": 8}},
                    "post": {"ucla3": {"total": post_u},
                             "phq9": {"total": 9}, "gad7": {"total": 7}},
                    "transcript": []}
        e7 = {"t1_high_lonely": {
            "A_companion": {
                "0": [rec(60, 59), rec(61, 60)],
                "6": [rec(60, 55), rec(61, 57)],
                "12": [rec(60, 50), rec(61, 52)],
            },
            "B_solution": {
                "0": [rec(60, 59), rec(61, 61)],
                "6": [rec(60, 57), rec(61, 58)],
                "12": [rec(60, 55), rec(61, 56)],
            },
        }}
        return e7

    def test_dose_response_monotonic(self):
        a = analyze_e7(self._synthetic())
        curve = a["t1_high_lonely"]["ucla3"]["A_companion"]
        self.assertLess(curve[6]["mean_delta"], curve[12]["mean_delta"])
        self.assertGreater(curve[12]["mean_delta"], 0)

    def test_recommendation(self):
        a = analyze_e7(self._synthetic())
        rec = a["t1_high_lonely"]["recommendation"]
        self.assertEqual(rec["recommendation"], "A_companion")
        self.assertGreater(rec["ab_diff"], 0)

    def test_ci_present(self):
        a = analyze_e7(self._synthetic())
        curve = a["t1_high_lonely"]["ucla3"]["A_companion"]
        self.assertIsNotNone(curve[12]["ci95"])


class TestRunE7Mock(unittest.TestCase):
    def test_mock_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = MockProvider(seed=9)
            d = run_e7(p, ["t1_high_lonely"], doses=[0, 2], n_sessions=1, out_dir=tmp)
            self.assertIn("A_companion", d["t1_high_lonely"])
            self.assertEqual(len(d["t1_high_lonely"]["A_companion"]["0"]), 1)
            self.assertEqual(len(d["t1_high_lonely"]["A_companion"]["2"]), 1)
            a = analyze_e7(d)
            self.assertIn("recommendation", a["t1_high_lonely"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
