# -*- coding: utf-8 -*-
"""
psybench.drift — E8 作答漂移定律（温度 × 量表类型 × 语言网格）
===============================================================

预注册式设计：对同一批 Soul 在不同「温度 × 量表类型 × 语言」网格上
独立施测 n_sessions 次，检验三个先验假设（源自 E1/E4/E5 的观察）：
  H1 温度效应：采样温度越高，症状量表总分越高（漂移越大）；
  H2 量表类型效应：症状量表（PHQ-9/GAD-7/PSS-10）比状态/特质量表
     （UCLA-3/WHO-5）漂移更明显；
  H3 语言效应：中文语境比英文语境在症状量表上更高（复现 E5 主效应）。
负结果同样如实报告；全部为探索性观察研究，不声称因果。

「作答漂移定律」仅指我们在这组实验条件下总结的经验规律，供后续
模型对比时做零假设，而非理论定律。
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from .assessment import administer_scale
from .scales import get_scale
from .souls import get_soul
from .providers import LLMProvider

SCALE_TYPES = {
    "ucla3": "state",
    "phq9": "symptom",
    "gad7": "symptom",
    "pss10": "symptom",
    "who5": "wellbeing",
}


def _log(msg: str):
    ts_now = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts_now}] {msg}", flush=True)


def run_e8(provider: LLMProvider, soul_ids: List[str], scale_keys: List[str],
           temperatures: List[float], langs: List[str], n_sessions: int,
           out_dir: str) -> Dict:
    """E8：作答漂移网格实验。结果保存为 e8_drift.json（断点续跑）。"""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e8_drift.json")
    data = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _log("E8: 检测到已有进度，断点续跑")

    for sid in soul_ids:
        soul = get_soul(sid)
        data.setdefault(sid, {})
        for skey in scale_keys:
            scale = get_scale(skey)
            entry = data[sid].setdefault(skey, {})
            for temp in temperatures:
                for lang in langs:
                    key = f"{temp}_{lang}"
                    recs = entry.setdefault(key, [])
                    recs[:] = [r for r in recs if isinstance(r, dict)
                               and len(r.get("items", [])) == scale.n_items()]
                    for rep in range(len(recs), n_sessions):
                        res = administer_scale(provider, soul, scale, history=None,
                                               temperature=temp, lang=lang)
                        recs.append({"items": res["items"], "total": res["total"]})
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    if recs:
                        _log(f"  E8 {sid}/{skey} t={temp} {lang}: n={len(recs)} "
                             f"mean={np.mean([r['total'] for r in recs]):.1f}")
    return data


def analyze_e8(e8: dict) -> dict:
    """网格汇总：温度×类型×语言 的漂移均值与假设检验（跨 Soul 聚合）。

    修复记录：不同量表的原始总分范围不同（PHQ-9 0-27 vs UCLA 20-80），
    直接比较原始均值无意义；所有跨量表比较改用「量程归一化分数」
    norm = (mean - range_lo) / (range_hi - range_lo)。
    """
    cells = []
    for sid, scales in e8.items():
        for skey, temps in scales.items():
            scale = get_scale(skey)
            lo, hi = scale.score_range
            for key, recs in temps.items():
                if not recs:
                    continue
                temp_s, lang = key.split("_", 1)
                totals = [r["total"] for r in recs]
                mean = float(np.mean(totals))
                cells.append({
                    "soul": sid,
                    "scale": skey,
                    "scale_type": SCALE_TYPES.get(skey, "state"),
                    "temperature": float(temp_s),
                    "lang": lang,
                    "mean": mean,
                    "norm_mean": (mean - lo) / (hi - lo),
                    "sd": float(np.std(totals, ddof=1)) if len(totals) > 1 else 0.0,
                    "n": len(totals),
                })
    out = {"cells": cells, "h1_temperature": {}, "h2_scale_type": {}, "h3_language": {}}
    # H1：温度效应（按量表类型聚合，比较 t=0.2 vs t=1.4 的量程归一化分数）
    def _means(temp, scale_type=None, lang=None):
        sub = [c for c in cells if c["temperature"] == temp
               and (scale_type is None or c["scale_type"] == scale_type)
               and (lang is None or c["lang"] == lang)]
        return [c["norm_mean"] for c in sub]
    for stype in ["symptom", "state", "wellbeing"]:
        lo = _means(0.2, stype)
        hi = _means(1.4, stype)
        if len(lo) >= 2 and len(hi) >= 2:
            t, p = stats.ttest_ind(lo, hi, equal_var=False)
            out["h1_temperature"][stype] = {
                "mean_t02": float(np.mean(lo)), "mean_t14": float(np.mean(hi)),
                "diff": float(np.mean(hi) - np.mean(lo)), "t": float(t), "p": float(p),
                "n_low": len(lo), "n_high": len(hi),
            }
    # H2：量表类型效应（t=0.8，症状 vs 状态/幸福，量程归一化后比较）
    def _means_type(scale_type, temp=0.8, lang=None):
        sub = [c for c in cells if c["scale_type"] == scale_type
               and c["temperature"] == temp
               and (lang is None or c["lang"] == lang)]
        return [c["norm_mean"] for c in sub]
    sym = _means_type("symptom")
    sta = _means_type("state") + _means_type("wellbeing")
    if len(sym) >= 2 and len(sta) >= 2:
        t, p = stats.ttest_ind(sym, sta, equal_var=False)
        out["h2_scale_type"] = {
            "symptom_mean": float(np.mean(sym)), "other_mean": float(np.mean(sta)),
            "diff": float(np.mean(sym) - np.mean(sta)), "t": float(t), "p": float(p),
        }
    # H3：语言效应（症状量表，t=0.8，zh vs en，量程归一化）
    zh = _means_type("symptom", lang="zh")
    en = _means_type("symptom", lang="en")
    if len(zh) >= 2 and len(en) >= 2:
        t, p = stats.ttest_ind(zh, en, equal_var=False)
        out["h3_language"] = {
            "zh_mean": float(np.mean(zh)), "en_mean": float(np.mean(en)),
            "diff": float(np.mean(zh) - np.mean(en)), "t": float(t), "p": float(p),
        }
    return out
