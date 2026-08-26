# -*- coding: utf-8 -*-
"""
psybench.psychometrics — 心理计量学统计工具箱
===============================================

将经典心理计量学指标迁移到「AI 心理仿真对象」的评价上：
  - Cronbach α        内部一致性（对同一 Soul 多次独立施测，
                      视每次施测为一名「虚拟被试」）
  - ICC(2,1)          重测信度（Shrout & Fleiss 1979，
                      绝对一致性，单次测量）
  - Cohen d / dz      效应量（组间 / 配对）
  - 配对 t 检验        干预前后差异显著性（含 95% 置信区间）
  - 已知组效度检验     高/低分组间的 Welch t 检验与效应量

所有指标均输出到结构化字典，供报告模块直接引用。
"""

import math
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats


def cronbach_alpha(matrix) -> Dict:
    """
    Cronbach α：matrix 形状为 (n_subjects, n_items)。
    这里 n_subjects = 同一 Soul 的独立施测次数（虚拟被试），
    n_items = 量表条目数。返回 α 与条目数。
    """
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        return {"alpha": None, "n_subjects": X.shape[0], "n_items": X.shape[1]}
    k = X.shape[1]
    item_var = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        # 总分零方差时 α 数学上未定义（0/0），返回 None 而非虚报 1.0
        return {"alpha": None, "n_subjects": X.shape[0], "n_items": k}
    alpha = (k / (k - 1)) * (1 - item_var.sum() / total_var)
    return {"alpha": float(alpha), "n_subjects": X.shape[0], "n_items": k}


def icc_21(matrix) -> Dict:
    """
    ICC(2,1)：双向随机效应、绝对一致性、单次测量（McGraw & Wong 1996）。
    matrix 形状为 (n_subjects, n_raters)；此处 n_raters = 重复施测次数。
    用于衡量同一 Soul 在不同场次施测中的分数稳定性（重测信度）。
    """
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        return {"icc": None, "n_subjects": X.shape[0], "n_raters": X.shape[1]}
    n, k = X.shape
    # 经典 ANOVA 分解（Shrout & Fleiss 1979 的 MSB/MSW/MSE）
    grand = X.mean()
    msb = k * np.sum((X.mean(axis=1) - grand) ** 2) / (n - 1)
    msw = n * np.sum((X.mean(axis=0) - grand) ** 2) / (k - 1)
    sse = np.sum((X - X.mean(axis=1, keepdims=True)
                 - X.mean(axis=0, keepdims=True) + grand) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    denom = msb + (k - 1) * mse + k * (msw - mse) / n
    if denom <= 0 or not math.isfinite(denom):
        return {"icc": None, "n_subjects": n, "n_raters": k,
                "msb": float(msb), "msw": float(msw), "mse": float(mse)}
    icc = (msb - mse) / denom
    return {"icc": float(icc), "n_subjects": n, "n_raters": k,
            "msb": float(msb), "msw": float(msw), "mse": float(mse)}


def cohen_d_indep(a, b) -> float:
    """独立样本 Cohen d（合并标准差）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                   / (na + nb - 2))
    if sp == 0:
        return float("inf") if a.mean() != b.mean() else 0.0
    return float((a.mean() - b.mean()) / sp)


def paired_stats(pre, post) -> Dict:
    """
    配对样本统计：t、p（双尾）、Cohen dz、均值差 95% 置信区间。
    pre/post 为同一批虚拟被试前后测的总分列表。
    """
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    if len(pre) != len(post) or len(pre) < 2:
        return {"n": len(pre), "t": None, "p": None, "dz": None,
                "mean_diff": None, "ci95": None}
    diff = pre - post  # 定义：改善 = 分数下降（如 UCLA 孤独分下降）
    n = len(diff)
    dbar = diff.mean()
    sd = diff.std(ddof=1)
    if sd > 0:
        t = dbar / (sd / math.sqrt(n))
        p = 2 * stats.t.sf(abs(t), df=n - 1)
        dz = dbar / sd
        se = sd / math.sqrt(n)
        ci = stats.t.interval(0.95, df=n - 1, loc=dbar, scale=se)
        ci = [float(ci[0]), float(ci[1])]
    else:
        # 前后测差值为常数（零方差）：效应确定但效应量不可定义
        t = float("inf") if dbar != 0 else 0.0
        p = 0.0 if dbar != 0 else 1.0
        dz = None
        ci = [float(dbar), float(dbar)]
    dz_out = float(dz) if dz is not None else None
    return {"n": n, "t": float(t), "p": float(p), "dz": dz_out,
            "mean_diff": float(dbar), "ci95": ci,
            "pre_mean": float(pre.mean()), "post_mean": float(post.mean()),
            "pre_sd": float(pre.std(ddof=1)), "post_sd": float(post.std(ddof=1))}


def welch_t(a, b) -> Dict:
    """Welch 两样本 t 检验（方差不齐稳健）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {"t": None, "p": None, "d": None}
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"t": float(t), "p": float(p),
            "d": cohen_d_indep(a, b),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "sd_a": float(a.std(ddof=1)), "sd_b": float(b.std(ddof=1))}


def fidelity_metrics(measured: Dict[str, List[float]], declared: Dict[str, tuple]):
    """
    仿真保真度：比较实测分数与声明真值区间。
    measured: {scale_key: [各场次总分, ...]}
    declared: {scale_key: (low, high)}
    返回每量表：mean、hit(落在区间内)、偏差（区间外时到区间的距离）、
    以及总体 hit 率与平均偏差（百分位当量）。
    """
    out = {}
    hits = 0
    total = 0
    devs = []
    for key, vals in measured.items():
        if key not in declared or not vals:
            continue
        low, high = declared[key]
        m = float(np.mean(vals))
        in_band = low <= m <= high
        dev = 0.0 if in_band else (m - high if m > high else low - m)
        span = max(high - low, 1.0)
        out[key] = {
            "mean": m, "declared": [low, high], "hit": in_band,
            "deviation": dev, "deviation_normalized": dev / span,
            "values": [float(v) for v in vals],
        }
        hits += int(in_band)
        total += 1
        devs.append(dev / span)
    out["_summary"] = {
        "hit_rate": hits / total if total else None,
        "mean_normalized_deviation": float(np.mean(devs)) if devs else None,
    }
    return out


def spearman_rank(measured_ranks: Dict[str, float], declared_centers: Dict[str, float]):
    """
    等级相关：实测均值排序 vs 声明区间中心排序（用于跨 Soul 一致性）。
    """
    keys = [k for k in declared_centers if k in measured_ranks]
    if len(keys) < 3:
        return {"rho": None, "p": None, "n": len(keys)}
    x = [declared_centers[k] for k in keys]
    y = [measured_ranks[k] for k in keys]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {"rho": None, "p": None, "n": len(keys)}
    try:
        rho, p = stats.spearmanr(x, y)
        return {"rho": float(rho), "p": float(p), "n": len(keys)}
    except Exception:
        return {"rho": None, "p": None, "n": len(keys)}
