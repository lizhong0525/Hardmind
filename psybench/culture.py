# -*- coding: utf-8 -*-
"""
psybench.culture — E5 文化测量等价性实验室（赛题方向 8）
==========================================================

预注册式实验设计（方案先冻结、后运行，禁止事后挑指标）：

  E5a 语境等价性：同一 Soul × 同一量表，分别在「完整中文语境」与
       「完整英文语境」（人设/指导语/条目/选项全部对应语言）下独立施测
       n_sessions 次（每次全新会话、无记忆），施测顺序按场次奇偶交错
       （奇：先中后英；偶：先英后中）以控制顺序效应。
       指标：总分均值差（Welch t + Cohen d + 95% CI）、条目均值画像
       相关（profile r）、各语境 Cronbach α 与重测 ICC。

  E5b 条目级 DIF（探索性）：对每个条目计算跨语境标准化均差 SMD
       （合并标准差），并在多个 Soul 上聚合求平均；|SMD|>0.8 标记为
       潜在 DIF 条目。样本量小（每 Soul×语境 n_sessions 次施测），
       结论显式声明为探索性。

  不变性近似（三层简化）：configural=条目均值画像 r（排序一致性）；
       metric=条目-总分相关画像 r（跨语境）;scalar=总分均值差（语境主效应）。

伦理与严谨性：与主框架相同——真值仅用于校验、盲测文本无构念词、
负结果如实报告、局限显式声明。
"""

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from .assessment import administer_scale
from .scales import get_scale
from .souls import get_soul
from .providers import LLMProvider


def _log(msg: str):
    ts_now = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts_now}] {msg}", flush=True)


def _welch_with_ci(a, b):
    """Welch 两样本检验 + 均值差 95% 置信区间（Welch-Satterthwaite 自由度）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {"t": None, "p": None, "d": None, "diff": None, "ci95": None,
                "mean_a": float(a.mean()), "mean_b": float(b.mean())}
    t, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    diff = float(a.mean() - b.mean())
    if va + vb <= 0:
        # 两组均为常数：效应确定但统计量未定义
        return {"t": None, "p": None, "d": 0.0, "diff": diff,
                "ci95": [diff, diff], "mean_a": float(a.mean()), "mean_b": float(b.mean())}
    se = math.sqrt(va / na + vb / nb)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    ci = stats.t.interval(0.95, df, loc=diff, scale=se)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    d = diff / sp if sp > 0 else 0.0
    return {"t": float(t), "p": float(p), "d": float(d), "diff": diff,
            "ci95": [float(ci[0]), float(ci[1])],
            "mean_a": float(a.mean()), "mean_b": float(b.mean())}


def run_e5(provider: LLMProvider, soul_ids: List[str], scale_keys: List[str],
           n_sessions: int, out_dir: str, temperature: float = 0.8) -> Dict:
    """E5：文化测量等价性实验。结果保存为 e5_culture.json（断点续跑）。"""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e5_culture.json")
    data = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _log("E5: 检测到已有进度，断点续跑")

    for sid in soul_ids:
        soul = get_soul(sid)
        data.setdefault(sid, {})
        for skey in scale_keys:
            scale = get_scale(skey)
            entry = data[sid].setdefault(skey, {"zh": [], "en": []})
            # 过滤历史脏记录（条目数不符）
            for kind in ("zh", "en"):
                entry[kind] = [r for r in entry[kind]
                               if isinstance(r, dict)
                               and len(r.get("items", [])) == scale.n_items()]
            for rep in range(len(entry["zh"]), n_sessions):
                langs = ["zh", "en"] if rep % 2 == 0 else ["en", "zh"]  # 交错顺序
                for lang in langs:
                    res = administer_scale(provider, soul, scale, history=None,
                                          temperature=temperature, lang=lang)
                    record = {"items": res["items"], "total": res["total"],
                              "band": res["band"], "raw": res["raw"]}
                    entry[lang].append(record)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                tz = entry["zh"][-1]["total"]
                te = entry["en"][-1]["total"]
                _log(f"  E5 {sid}/{skey} #{rep+1}/{n_sessions}: zh={tz} en={te}")
    return data


def _cronbach(items_matrix):
    """条目矩阵 (n_sessions, n_items) 的 Cronbach α；零方差返回 None。"""
    X = np.asarray(items_matrix, dtype=float)
    if X.shape[0] < 2 or X.shape[1] < 2:
        return None
    k = X.shape[1]
    item_var = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return None
    return float((k / (k - 1)) * (1 - item_var.sum() / total_var))


def analyze_e5(e5: dict) -> dict:
    """
    预注册指标计算：
      per_soul_scale: zh/en 均值、Welch+CI+d、α、ICC、profile r
      dif:           条目级聚合 SMD（多 Soul 平均，|SMD|>0.8 标记）；
      invariance:     configural/metric/scalar 三层近似；
      summary:        按量表的语境主效应汇总（zh-en 均值差 + CI）。
    """
    out = {"per_soul_scale": {}, "dif": {}, "invariance": {}, "summary": {}}
    for sid, scales in e5.items():
        out["per_soul_scale"][sid] = {}
        for skey, entry in scales.items():
            scale = get_scale(skey)
            zh_recs = entry.get("zh", [])
            en_recs = entry.get("en", [])
            if not zh_recs or not en_recs:
                continue
            zh_totals = [r["total"] for r in zh_recs]
            en_totals = [r["total"] for r in en_recs]
            zh_items = np.array([r["items"] for r in zh_recs], dtype=float)
            en_items = np.array([r["items"] for r in en_recs], dtype=float)
            welch = _welch_with_ci(zh_totals, en_totals)
            zh_item_means = zh_items.mean(axis=0)
            en_item_means = en_items.mean(axis=0)
            if zh_item_means.std() > 0 and en_item_means.std() > 0:
                profile_r = float(np.corrcoef(zh_item_means, en_item_means)[0, 1])
            else:
                profile_r = None
            # metric 近似：条目-总分相关画像（虚拟被试=场次）
            def _item_total(img):
                if img.shape[1] < 2:
                    return None
                totals = img.sum(axis=1)
                if totals.std() == 0 or np.any(img.std(axis=0) == 0):
                    return None  # 零方差时相关未定义
                return np.array([np.corrcoef(img[:, j], totals)[0, 1]
                                 for j in range(img.shape[1])])
            it_zh = _item_total(zh_items)
            it_en = _item_total(en_items)
            metric_r = None
            if it_zh is not None and it_en is not None:
                metric_r = float(np.corrcoef(it_zh, it_en)[0, 1])
            # 各语境 α
            alpha_zh = _cronbach(zh_items)
            alpha_en = _cronbach(en_items)
            # 条目级 SMD（合并标准差；本语境内池化）
            smds = []
            for j in range(scale.n_items()):
                m1, m2 = zh_item_means[j], en_item_means[j]
                s1, s2 = zh_items[:, j].std(ddof=1), en_items[:, j].std(ddof=1)
                sp = math.sqrt((s1 ** 2 + s2 ** 2) / 2)
                smds.append(float((m1 - m2) / sp) if sp > 0 else 0.0)
            out["per_soul_scale"][sid][skey] = {
                "zh_mean": float(np.mean(zh_totals)), "zh_sd": float(np.std(zh_totals, ddof=1)),
                "en_mean": float(np.mean(en_totals)), "en_sd": float(np.std(en_totals, ddof=1)),
                "n": len(zh_totals),
                "welch": welch,
                "alpha_zh": alpha_zh, "alpha_en": alpha_en,
                "profile_r": profile_r, "metric_r": metric_r,
                "item_smd": smds,
                "zh_totals": [int(t) for t in zh_totals],
                "en_totals": [int(t) for t in en_totals],
            }
    # DIF：跨 Soul 聚合条目 SMD
    for skey in {s for scales in e5.values() for s in scales}:
        scale = get_scale(skey)
        per_item = [[] for _ in range(scale.n_items())]
        for sid in e5:
            cell = out["per_soul_scale"].get(sid, {}).get(skey)
            if not cell:
                continue
            for j, v in enumerate(cell["item_smd"]):
                per_item[j].append(v)
        dif_items = []
        for j, vals in enumerate(per_item):
            if not vals:
                continue
            m = float(np.mean(vals))
            dif_items.append({
                "item_idx": j + 1,
                "item_text_zh": scale.items[j],
                "item_text_en": (scale.items_en or scale.items)[j],
                "smd_mean": m,
                "flag": abs(m) > 0.8,
            })
        out["dif"][skey] = sorted(dif_items, key=lambda x: -abs(x["smd_mean"]))
    # 不变性三层近似（跨 Soul 平均）
    for skey in {s for scales in e5.values() for s in scales}:
        profs, mets, diffs = [], [], []
        for sid in e5:
            cell = out["per_soul_scale"].get(sid, {}).get(skey)
            if not cell:
                continue
            if cell["profile_r"] is not None:
                profs.append(cell["profile_r"])
            if cell["metric_r"] is not None:
                mets.append(cell["metric_r"])
            if cell["welch"]["diff"] is not None:
                diffs.append(cell["welch"]["diff"])
        mets = [x for x in mets if x is not None and math.isfinite(x)]
        out["invariance"][skey] = {
            "configural_r_mean": float(np.mean(profs)) if profs else None,
            "metric_r_mean": float(np.mean(mets)) if mets else None,
            "scalar_diff_mean": float(np.mean(diffs)) if diffs else None,
            "scalar_diff_sd": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
            "n_souls": len(diffs),
        }
    # summary：按量表聚合语境主效应（n=每 Soul 一次的均值差 → 跨 Soul）
    for skey in {s for scales in e5.values() for s in scales}:
        diffs = []
        for sid in e5:
            cell = out["per_soul_scale"].get(sid, {}).get(skey)
            if cell and cell["welch"]["diff"] is not None:
                diffs.append(cell["welch"]["diff"])
        if diffs:
            d = np.asarray(diffs)
            t = float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))) if len(d) > 1 and d.std(ddof=1) > 0 else None
            p = float(2 * stats.t.sf(abs(t), df=len(d) - 1)) if t is not None else None
            out["summary"][skey] = {
                "mean_diff_zh_minus_en": float(d.mean()),
                "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                "t": t, "p": p, "n_souls": len(d),
                "ci95": [float(x) for x in stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / math.sqrt(len(d)))] if len(d) > 1 else None,
            }
    return out
