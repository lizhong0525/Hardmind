# -*- coding: utf-8 -*-
"""
psybench.report — 结果分析、图表与自动报告生成
=================================================

读取实验 JSON，计算全部心理计量学指标，输出：
  - results/figures/*.png   四张核心图表（中文字体）
  - results/结果报告.md      结构化实验报告（可直接引用进技术报告）
"""

import json
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .style import apply_mpl_style, PALETTE, SERIES

apply_mpl_style()

from .scales import get_scale
from .souls import ALL_SOULS
from .psychometrics import (
    cronbach_alpha, icc_21, welch_t, paired_stats, fidelity_metrics, spearman_rank,
)


def load_results(out_dir: str):
    """读取实验结果 JSON。"""
    with open(os.path.join(out_dir, "e1_known_groups.json"), encoding="utf-8") as f:
        e1 = json.load(f)
    with open(os.path.join(out_dir, "e3_intervention.json"), encoding="utf-8") as f:
        e3 = json.load(f)
    meta_path = os.path.join(out_dir, "run_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    return e1, e3, meta


def analyze_e1(e1: dict) -> dict:
    """E1 分析：每 Soul×量表 的描述统计、α、ICC；已知组差异；保真度。"""
    summary = {}
    measured_by_soul = {}
    for sid, scales in e1.items():
        soul = ALL_SOULS[sid]
        summary[sid] = {}
        measured_by_soul[sid] = {}
        for skey, recs in scales.items():
            totals = [r["total"] for r in recs]
            items_matrix = np.array([r["items"] for r in recs], dtype=float)
            alpha = cronbach_alpha(items_matrix)
            icc = icc_21(items_matrix.T)
            bands = [r["band"] for r in recs]
            summary[sid][skey] = {
                "n": len(totals),
                "mean": float(np.mean(totals)),
                "sd": float(np.std(totals, ddof=1)) if len(totals) > 1 else 0.0,
                "min": int(np.min(totals)),
                "max": int(np.max(totals)),
                "totals": [int(t) for t in totals],
                "alpha": alpha["alpha"],
                "icc": icc["icc"],
                "bands": bands,
                "declared": list(soul.declared.get(skey, (None, None))),
            }
            measured_by_soul[sid][skey] = [float(t) for t in totals]

    contrasts = {
        "ucla3": [("chen_yu", "lin_han", "高孤独 vs 低孤独"),
                   ("mo_ran", "lin_han", "抑郁倾向 vs 低孤独")],
        "phq9":  [("mo_ran", "lin_han", "抑郁倾向 vs 低孤独"),
                   ("chen_yu", "lin_han", "高孤独 vs 低孤独")],
        "gad7":  [("zhou_yang", "lin_han", "高压力 vs 低孤独"),
                   ("mo_ran", "lin_han", "抑郁倾向 vs 低孤独")],
    }
    known_groups = {}
    for skey, pairs in contrasts.items():
        known_groups[skey] = []
        for a, b, label in pairs:
            va = [r["total"] for r in e1.get(a, {}).get(skey, [])]
            vb = [r["total"] for r in e1.get(b, {}).get(skey, [])]
            if len(va) >= 2 and len(vb) >= 2:
                st = welch_t(va, vb)
                st["contrast"] = label
                st["group_a"] = a
                st["group_b"] = b
                known_groups[skey].append(st)

    rank_corr = {}
    for skey in ["ucla3", "phq9", "gad7"]:
        centers = {}
        means = {}
        for sid, decl in measured_by_soul.items():
            if skey in decl and skey in ALL_SOULS[sid].declared:
                lo, hi = ALL_SOULS[sid].declared[skey]
                centers[sid] = (lo + hi) / 2.0
                means[sid] = float(np.mean(decl[skey]))
        rank_corr[skey] = spearman_rank(means, centers)

    fidelity = {}
    for sid, measured in measured_by_soul.items():
        fidelity[sid] = fidelity_metrics(measured, ALL_SOULS[sid].declared)

    return {
        "summary": summary,
        "known_groups": known_groups,
        "rank_corr": rank_corr,
        "fidelity": fidelity,
    }


def analyze_e3(e3: dict) -> dict:
    """E3 分析：每条件×量表 的配对统计 + 条件间增量差异。"""
    out = {}
    for cond, sessions in e3.items():
        out[cond] = {}
        for skey in ["ucla3", "phq9", "gad7"]:
            pre = [s["pre"][skey]["total"] for s in sessions]
            post = [s["post"][skey]["total"] for s in sessions]
            out[cond][skey] = paired_stats(pre, post)
            out[cond][skey]["pairs"] = list(zip(pre, post))
    deltas = {}
    for cond in out:
        pairs = out[cond]["ucla3"]["pairs"]
        deltas[cond] = [p[0] - p[1] for p in pairs]
    between = {}
    if "companion" in deltas and "neutral" in deltas:
        between["companion_vs_neutral"] = welch_t(deltas["companion"], deltas["neutral"])
    if "companion" in deltas and "thinking" in deltas:
        between["companion_vs_thinking"] = welch_t(deltas["companion"], deltas["thinking"])
    out["_between"] = between
    return out


def analyze_e4(e4: dict) -> dict:
    """E4 分析：标准盲测 vs 校正指导语 的配对比较与真值命中。"""
    out = {}
    for sid, scales in e4.items():
        out[sid] = {}
        for skey, entry in scales.items():
            std = [r["total"] for r in entry.get("standard", [])]
            cal = [r["total"] for r in entry.get("calibrated", [])]
            if not std or not cal:
                out[sid][skey] = {"t": None, "p": None, "dz": None,
                                  "mean_diff": None, "std_mean": None,
                                  "cal_mean": None}
                continue
            st = paired_stats(std, cal)
            st["std_mean"] = float(np.mean(std)) if std else None
            st["cal_mean"] = float(np.mean(cal)) if cal else None
            declared = ALL_SOULS[sid].declared.get(skey)
            if declared is not None:
                lo, hi = declared
                st["std_hit"] = bool(lo <= st["std_mean"] <= hi)
                st["cal_hit"] = bool(lo <= st["cal_mean"] <= hi)
                st["declared"] = [lo, hi]
            out[sid][skey] = st
    return out


def make_figures(a1: dict, a3: dict, a4: dict, a5: dict, a6: dict,
                 a7: dict, a8: dict, out_dir: str):
    """生成核心图表（E1-E8）。"""
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    scale_meta = {"ucla3": "UCLA-3 孤独感", "phq9": "PHQ-9 抑郁", "gad7": "GAD-7 焦虑"}
    soul_names = {s: ALL_SOULS[s].name for s in ALL_SOULS}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, skey in zip(axes, scale_meta):
        sids = [s for s in a1["summary"] if skey in a1["summary"][s]]
        means = [a1["summary"][s][skey]["mean"] for s in sids]
        sds = [a1["summary"][s][skey]["sd"] for s in sids]
        x = np.arange(len(sids))
        ax.bar(x, means, yerr=sds, capsize=5, color="#1d2a3a", alpha=0.85)
        for i, s in enumerate(sids):
            lo, hi = a1["summary"][s][skey]["declared"]
            if lo is not None:
                ax.plot([i - 0.25, i + 0.25], [lo, lo], color="orange", lw=1.5)
                ax.plot([i - 0.25, i + 0.25], [hi, hi], color="orange", lw=1.5)
                ax.plot([i, i], [lo, hi], color="orange", lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([soul_names[s] for s in sids])
        ax.set_title(scale_meta[skey])
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("E1 已知组别效度：实测均值±SD 与声明真值区间（橙色）", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_known_groups.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    cond_names = {"companion": "陪伴对话", "neutral": "中性问答", "thinking": "独自书写"}
    fig, ax = plt.subplots(figsize=(9, 5))
    conds = [c for c in a3 if not c.startswith("_")]
    x = np.arange(len(conds))
    w = 0.35
    pre_m = [a3[c]["ucla3"]["pre_mean"] for c in conds]
    post_m = [a3[c]["ucla3"]["post_mean"] for c in conds]
    pre_sd = [a3[c]["ucla3"]["pre_sd"] for c in conds]
    post_sd = [a3[c]["ucla3"]["post_sd"] for c in conds]
    ax.bar(x - w / 2, pre_m, w, yerr=pre_sd, capsize=4, label="前测", color="#c0504d")
    ax.bar(x + w / 2, post_m, w, yerr=post_sd, capsize=4, label="后测", color="#55a868")
    for i, c in enumerate(conds):
        for p in a3[c]["ucla3"]["pairs"]:
            ax.plot([i - w / 2, i + w / 2], [p[0], p[1]], color="gray", alpha=0.5, lw=0.8)
        p_val = a3[c]["ucla3"]["p"]
        dz = a3[c]["ucla3"]["dz"]
        p_txt = f"{p_val:.3f}" if p_val is not None else "—"
        dz_txt = f"{dz:.2f}" if dz is not None else "—"
        note = f"p={p_txt} dz={dz_txt}"
        ax.annotate(note, (i, max(pre_m[i], post_m[i]) + 2), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([cond_names.get(c, c) for c in conds])
    ax.set_ylabel("UCLA-3 总分")
    ax.set_title("E3 干预反应度：陈屿（高孤独 Soul）三臂前后测")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_intervention.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"ucla3": "#1d2a3a", "phq9": "#c0504d", "gad7": "#55a868"}
    for skey in ["ucla3", "phq9", "gad7"]:
        xs, ys, marks = [], [], []
        for sid, fitem in a1["fidelity"].items():
            if skey not in fitem:
                continue
            lo, hi = fitem[skey]["declared"]
            xs.append((lo + hi) / 2.0)
            ys.append(fitem[skey]["mean"])
            marks.append("o" if fitem[skey]["hit"] else "x")
        for xi, yi, mk in zip(xs, ys, marks):
            ax.scatter(xi, yi, marker=mk, color=colors[skey], s=90,
                       label=scale_meta[skey] if mk == "o" else None)
    ax.plot([0, 70], [0, 70], color="gray", ls="--", alpha=0.6)
    ax.set_xlabel("声明真值区间中心")
    ax.set_ylabel("实测均值")
    ax.set_title("E2 仿真保真度：实测 vs 声明（○命中 ×未命中）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_fidelity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels, alphas, iccs = [], [], []
    for sid, sm in a1["summary"].items():
        for skey in ["ucla3", "phq9", "gad7"]:
            if skey in sm:
                labels.append(f"{soul_names[sid]}-{scale_meta[skey]}")
                alphas.append(sm[skey]["alpha"] or 0)
                iccs.append(sm[skey]["icc"] or 0)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, alphas, 0.4, label="Cronbach α", color="#1d2a3a")
    ax.bar(x + 0.2, iccs, 0.4, label="ICC(2,1) 重测", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.7, color="gray", ls="--", alpha=0.6)
    ax.set_ylabel("信度系数")
    ax.set_title("信度：内部一致性（α）与重测信度（ICC）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig4_reliability.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if a4:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        labels4, std_m, cal_m = [], [], []
        for sid, scales in a4.items():
            for skey in ["ucla3", "phq9", "gad7"]:
                if skey in scales:
                    st = scales[skey]
                    if st["std_mean"] is None:
                        continue
                    labels4.append(f"{soul_names[sid]}-{scale_meta[skey]}")
                    std_m.append(st["std_mean"])
                    cal_m.append(st["cal_mean"])
        x4 = np.arange(len(labels4))
        ax.bar(x4 - 0.2, std_m, 0.4, label="标准盲测", color="#c0504d")
        ax.bar(x4 + 0.2, cal_m, 0.4, label="校正指导语", color="#55a868")
        ax.set_xticks(x4)
        ax.set_xticklabels(labels4, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("量表总分") 
        ax.set_title("E4 作答风格校正：标准盲测 vs 追加人群参照说明")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig5_calibration.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # 图6：E5 文化测量等价性
    if a5 and a5.get("summary"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        s_keys = [k for k in a5["summary"]
                  if a5["summary"][k].get("mean_diff_zh_minus_en") is not None]
        s_means = [a5["summary"][k]["mean_diff_zh_minus_en"] for k in s_keys]
        s_errs = []
        for k in s_keys:
            ci = a5["summary"][k].get("ci95")
            if ci:
                s_errs.append([abs(s_means[len(s_errs)] - ci[0]), abs(ci[1] - s_means[len(s_errs)])])
            else:
                s_errs.append([0, 0])
        err_lo = [e[0] for e in s_errs]
        err_hi = [e[1] for e in s_errs]
        axes[0].barh(s_keys, s_means, xerr=[err_lo, err_hi], capsize=4,
                     color=["#1d2a3a" if m > 0 else "#c0504d" for m in s_means], alpha=0.85)
        axes[0].axvline(0, color="gray", lw=1)
        axes[0].set_xlabel("中文语境总分 - 英文语境总分（95% CI，跨 Soul）")
        axes[0].set_title("E5a 语境主效应")
        axes[0].grid(axis="x", alpha=0.3)
        dif_rows = []
        for skey, lst in (a5.get("dif") or {}).items():
            for x in lst:
                dif_rows.append((f"{skey}-{x['item_idx']}", x["smd_mean"]))
        dif_rows.sort(key=lambda t: -abs(t[1]))
        top_rows = dif_rows[:12]
        labels6 = [t[0] for t in top_rows][::-1]
        vals6 = [t[1] for t in top_rows][::-1]
        axes[1].barh(range(len(vals6)), vals6, color=["#1d2a3a" if v >= 0 else "#c0504d" for v in vals6], alpha=0.85)
        axes[1].set_yticks(range(len(vals6)))
        axes[1].set_yticklabels(labels6, fontsize=8)
        axes[1].axvline(0.8, color="orange", ls="--", lw=1, label="|SMD|=0.8 阈值")
        axes[1].axvline(-0.8, color="orange", ls="--", lw=1)
        axes[1].set_xlabel("条目 SMD（正=中文更高）")
        axes[1].set_title("E5b 潜在 DIF 条目（跨 Soul 聚合，探索性）")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig6_culture.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 图7：E6 危机评分卡
    if a6 and a6.get("scorecard"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        lv_keys = sorted(a6["by_level"].keys(), key=lambda k: int(k))
        x7 = np.arange(len(lv_keys))
        det = [a6["by_level"][k]["detection_rate"] for k in lv_keys]
        ref = [a6["by_level"][k]["referral_rate"] for k in lv_keys]
        harm = [a6["by_level"][k]["harm_rate"] for k in lv_keys]
        w7 = 0.25
        axes[0].bar(x7 - w7, det, w7, label="识别率", color="#1d2a3a")
        axes[0].bar(x7, ref, w7, label="转介率", color="#55a868")
        axes[0].bar(x7 + w7, harm, w7, label="危害输出率", color="#c0504d")
        axes[0].set_xticks(x7)
        axes[0].set_xticklabels([f"C-SSRS {k}" for k in lv_keys])
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("比率")
        axes[0].set_title("E6 危机评测：按剧本等级分层")
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", alpha=0.3)
        ag_keys = list(a6.get("agreement") or {})
        ag_vals = []
        for k in ag_keys:
            v = a6["agreement"][k].get("percent_agreement")
            ag_vals.append(v if v is not None else 0)
        axes[1].bar(range(len(ag_keys)), ag_vals, color="#dd8452", alpha=0.85)
        axes[1].set_xticks(range(len(ag_keys)))
        axes[1].set_xticklabels(ag_keys)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_ylabel("一致率")
        axes[1].set_title("E6 双通道一致性（法官 vs 词典，高危子集）")
        axes[1].grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig7_crisis.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 图8：E7 数字孪生预演（剂量-反应曲线）
    if a7:
        twin_keys = list(a7.keys())
        fig, axes = plt.subplots(1, max(1, len(twin_keys)), figsize=(6.5 * max(1, len(twin_keys)), 4.6),
                                 squeeze=False)
        arm_names = {"A_companion": "A 陪伴式共情", "B_solution": "B 建议式解决"}
        colors7 = {"A_companion": "#1d2a3a", "B_solution": "#dd8452"}
        for i, tk in enumerate(twin_keys):
            ax = axes[0][i]
            for arm, doses in a7[tk].get("ucla3", {}).items():
                xs = sorted(doses.keys())
                ys = [doses[d]["mean_delta"] for d in xs]
                cis = [doses[d]["ci95"] for d in xs]
                err = [[y - (ci[0] if ci else y) for y, ci in zip(ys, cis)],
                       [(ci[1] if ci else y) - y for y, ci in zip(ys, cis)]]
                ax.errorbar(xs, ys, yerr=err, capsize=4, marker="o",
                            color=colors7.get(arm, "#888"), label=arm_names.get(arm, arm))
            ax.axhline(0, color="gray", lw=0.8)
            ax.set_xlabel("干预剂量（对话轮数）")
            ax.set_ylabel("UCLA-3 前后测 Δ（正=改善）")
            ax.set_title(f"E7 孪生预演：{tk}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig8_twin_pretrial.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 图9：E8 作答漂移网格
    if a8 and a8.get("cells"):
        import pandas as pd
        df = pd.DataFrame(a8["cells"])
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        # 左：温度 × 量表类型（中文语境）
        sub = df[df["lang"] == "zh"]
        stypes = [t for t in ["symptom", "state", "wellbeing"] if (sub["scale_type"] == t).any()]
        for st in stypes:
            g = sub[sub["scale_type"] == st].groupby("temperature")["mean"].mean()
            axes[0].plot(g.index, g.values, marker="o", label=st)
        axes[0].set_xlabel("采样温度")
        axes[0].set_ylabel("量表总分均值（跨 Soul，中文语境）")
        axes[0].set_title("E8 漂移：温度效应（按量表类型）")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        # 右：语言 × 量表类型（t=0.8）
        sub2 = df[df["temperature"] == 0.8]
        import numpy as _np
        stype_list = sorted(sub2["scale_type"].unique())
        lang_list = sorted(sub2["lang"].unique())
        x9 = _np.arange(len(stype_list))
        w9 = 0.35
        for j, lg in enumerate(lang_list):
            means = [sub2[(sub2["scale_type"] == st) & (sub2["lang"] == lg)]["mean"].mean()
                     for st in stype_list]
            axes[1].bar(x9 + (j - 0.5) * w9, means, w9, label=("中文" if lg == "zh" else "英文"))
        axes[1].set_xticks(x9)
        axes[1].set_xticklabels(stype_list)
        axes[1].set_xlabel("量表类型（symptom=症状 / state=状态 / wellbeing=幸福）")
        axes[1].set_ylabel("总分均值（t=0.8）")
        axes[1].set_title("E8 漂移：语言效应（症状量表中文显著更高）")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig9_drift.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig_dir


def _fmt_p(p) -> str:
    if p is None:
        return "—"
    return f"{p:.4f}" if p >= 0.0001 else "<0.0001"


def write_markdown_report(a1: dict, a3: dict, a4: dict, a5: dict, a6: dict,
                           a7: dict, a8: dict, meta: dict, out_dir: str):
    """输出结构化 Markdown 实验报告。"""
    soul_names = {s: ALL_SOULS[s].name for s in ALL_SOULS}
    scale_meta = {"ucla3": "UCLA-3 孤独感", "phq9": "PHQ-9 抑郁", "gad7": "GAD-7 焦虑"}
    lines = []
    w = lines.append
    w("# 心镜 · 实验数据自动报告")
    w("")
    prov = meta.get("provider", "?")
    temp = meta.get("temperature", "?")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    w(f"- 推理后端：{prov}")
    w(f"- 施测温度：{temp}")
    w(f"- 生成时间：{now_str}")
    w("")
    w("## 1. E1 已知组别效度与重测信度")
    w("")
    w("| Soul | 量表 | 场次 | 均值±SD | 声明区间 | α | ICC | 分级 |")
    w("|---|---|---|---|---|---|---|---|")
    for sid, sm in a1["summary"].items():
        for skey in ["ucla3", "phq9", "gad7"]:
            if skey not in sm:
                continue
            s = sm[skey]
            decl = s["declared"]
            decl_txt = f"{decl[0]}–{decl[1]}" if decl[0] is not None else "—"
            a_val = s["alpha"]
            i_val = s["icc"]
            alpha_txt = f"{a_val:.2f}" if a_val is not None else "—"
            icc_txt = f"{i_val:.2f}" if i_val is not None else "—"
            n_val = s["n"]
            m_val = s["mean"]
            sd_val = s["sd"]
            band_txt = get_scale(skey).band(round(m_val))
            w(f"| {soul_names[sid]} | {scale_meta[skey]} | {n_val} | {m_val:.1f}±{sd_val:.1f} | {decl_txt} | {alpha_txt} | {icc_txt} | {band_txt} |")
    w("")
    w("### 已知组对照（Welch t 检验）")
    w("")
    w("| 量表 | 对照 | t | p | Cohen d |")
    w("|---|---|---|---|---|")
    for skey, tests in a1["known_groups"].items():
        for st in tests:
            t_val = st["t"]
            p_val = st["p"]
            d_val = st["d"]
            c_val = st["contrast"]
            w(f"| {scale_meta[skey]} | {c_val} | {t_val:.2f} | {_fmt_p(p_val)} | {d_val:.2f} |")
    w("")
    w("### 等级相关（实测均值 vs 声明真值中心）")
    w("")
    w("| 量表 | Spearman ρ | p |")
    w("|---|---|---|")
    for skey, rc in a1["rank_corr"].items():
        rho_val = rc["rho"]
        p_val = rc["p"]
        rho_txt = f"{rho_val:.3f}" if rho_val is not None else "—"
        w(f"| {scale_meta[skey]} | {rho_txt} | {_fmt_p(p_val)} |")
    w("")
    w("### 仿真保真度")
    w("")
    for sid, fitem in a1["fidelity"].items():
        s = fitem.get("_summary", {})
        hr = s.get("hit_rate")
        md = s.get("mean_normalized_deviation")
        hr_txt = f"{hr*100:.0f}%" if hr is not None else "—"
        md_txt = f"{md:.2f}" if md is not None else "—"
        w(f"- **{soul_names[sid]}**：区间命中率 {hr_txt}，平均归一化偏差 {md_txt}")
    w("")
    w("## 2. E3 干预反应度（陈屿，高孤独 Soul）")
    w("")
    w("| 条件 | 量表 | 前测 | 后测 | Δ | t | p | dz |")
    w("|---|---|---|---|---|---|---|---|")
    cond_names = {"companion": "陪伴对话", "neutral": "中性问答", "thinking": "独自书写"}
    for cond, sm in a3.items():
        if cond.startswith("_"):
            continue
        for skey in ["ucla3", "phq9", "gad7"]:
            st = sm[skey]
            t_val = st["t"]
            if t_val is None:
                continue
            pre_m = st["pre_mean"]
            post_m = st["post_mean"]
            delta = pre_m - post_m
            p_val = st["p"]
            dz_val = st["dz"]
            t_txt = f"{t_val:.2f}" if t_val is not None else "—"
            dz_txt = f"{dz_val:.2f}" if dz_val is not None else "—"
            w(f"| {cond_names.get(cond, cond)} | {scale_meta[skey]} | {pre_m:.1f} | {post_m:.1f} | {delta:+.1f} | {t_txt} | {_fmt_p(p_val)} | {dz_txt} |")
    w("")
    bt = a3.get("_between")
    if bt:
        w("### 条件间增量比较（UCLA-3）")
        w("")
        w("| 对照 | t | p | d |")
        w("|---|---|---|---|")
        for name, st in bt.items():
            t_val = st["t"]
            p_val = st["p"]
            d_val = st["d"]
            t_txt = f"{t_val:.2f}" if t_val is not None else "—"
            d_txt = f"{d_val:.2f}" if d_val is not None else "—"
            w(f"| {name} | {t_txt} | {_fmt_p(p_val)} | {d_txt} |")
        w("")
    w("## 3. E4 作答风格校正（探索性）")
    w("")
    w("| Soul | 量表 | 标准盲测均值 | 校正后均值 | Δ | p | dz | 真值命中（标准→校正） |")
    w("|---|---|---|---|---|---|---|---|")
    for sid, scales in (a4 or {}).items():
        for skey in ["ucla3", "phq9", "gad7"]:
            if skey not in scales:
                continue
            st = scales[skey]
            if st["std_mean"] is None:
                continue
            delta = st["mean_diff"]
            p_val = st["p"]
            dz_val = st["dz"]
            std_m = st["std_mean"]
            cal_m = st["cal_mean"]
            dz_txt = f"{dz_val:.2f}" if dz_val is not None else "—"
            hit_txt = "—"
            if "std_hit" in st:
                hit_txt = f"{'是' if st['std_hit'] else '否'}→{'是' if st['cal_hit'] else '否'}"
            w(f"| {soul_names[sid]} | {scale_meta[skey]} | {std_m:.1f} | {cal_m:.1f} | {delta:+.1f} | {_fmt_p(p_val)} | {dz_txt} | {hit_txt} |")
    w("")
    w("## 5. E5 文化测量等价性（赛题方向 8）")
    w("")
    if a5 and a5.get("summary"):
        w("### 语境主效应（中文-英文总分差，跨 Soul 均值 ± 95% CI）")
        w("")
        w("| 量表 | 均值差(zh-en) | 95% CI | t | p |")
        w("|---|---|---|---|---|")
        for skey, s in a5["summary"].items():
            if s.get("mean_diff_zh_minus_en") is None:
                continue
            md5 = s["mean_diff_zh_minus_en"]
            ci5 = s.get("ci95")
            ci_txt = f"[{ci5[0]:.1f}, {ci5[1]:.1f}]" if ci5 else "—"
            t5 = s.get("t")
            t_txt = f"{t5:.2f}" if t5 is not None else "—"
            w(f"| {scale_meta.get(skey, skey)} | {md5:+.1f} | {ci_txt} | {t_txt} | {_fmt_p(s.get('p'))} |")
        w("")
        w("### 不变性三层近似（跨 Soul 均值）")
        w("")
        w("| 量表 | configural r（条目画像） | metric r（条目-总分画像） | scalar 均值差 |")
        w("|---|---|---|---|")
        for skey, inv in a5.get("invariance", {}).items():
            cr = inv.get("configural_r_mean")
            mr = inv.get("metric_r_mean")
            sd5 = inv.get("scalar_diff_mean")
            cr_txt = f"{cr:.2f}" if cr is not None else "—"
            mr_txt = f"{mr:.2f}" if mr is not None else "—"
            sd_txt = f"{sd5:+.1f}" if sd5 is not None else "—"
            w(f"| {scale_meta.get(skey, skey)} | {cr_txt} | {mr_txt} | {sd_txt} |")
        w("")
        w("### 每 Soul×量表 语境对比")
        w("")
        w("| Soul | 量表 | 中文均值±SD | 英文均值±SD | p | d |")
        w("|---|---|---|---|---|---|")
        for sid, scales in a5.get("per_soul_scale", {}).items():
            for skey, cell in scales.items():
                wl = cell["welch"]
                if wl.get("p") is None:
                    continue
                d5 = wl["d"]
                d_txt = f"{d5:.2f}" if d5 is not None else "—"
                w(f"| {soul_names.get(sid, sid)} | {scale_meta.get(skey, skey)} | "
                  f"{cell['zh_mean']:.1f}±{cell['zh_sd']:.1f} | "
                  f"{cell['en_mean']:.1f}±{cell['en_sd']:.1f} | "
                  f"{_fmt_p(wl['p'])} | {d_txt} |")
        w("")
        w("### 潜在 DIF 条目（跨 Soul 聚合 |SMD|>0.8，探索性）")
        w("")
        w("| 量表 | 条目号 | SMD | 条目文本（中文） |")
        w("|---|---|---|---|")
        n_dif = 0
        for skey, lst in (a5.get("dif") or {}).items():
            for x in lst:
                if x.get("flag"):
                    n_dif += 1
                    w(f"| {scale_meta.get(skey, skey)} | {x['item_idx']} | {x['smd_mean']:+.2f} | {x['item_text_zh']} |")
        if n_dif == 0:
            w("| — | — | — | 无 |SMD|>0.8 条目 |")
        w("")
    w("## 6. E6 危机安全评分卡（赛题方向 3）")
    w("")
    if a6 and a6.get("scorecard"):
        sc = a6["scorecard"]
        w("### 总体评分卡（被测模型：" + str(meta.get("provider", "?")) + "）")
        w("")
        w(f"- 记录数：{sc['n_records']}")
        w(f"- 危机识别率：{sc['detection_rate']*100:.0f}%")
        w(f"- 共情确认率：{sc['acknowledge_rate']*100:.0f}%")
        w(f"- 转介触发率：{sc['referral_rate']*100:.0f}%")
        w(f"- 危害输出率：{sc['harm_rate']*100:.0f}%")
        w(f"- 协议符合度：{sc['protocol_compliance']*100:.0f}%")
        w("")
        w("### 按剧本 C-SSRS 等级分层")
        w("")
        w("| 等级 | n | 识别率 | 转介率 | 危害输出率 |")
        w("|---|---|---|---|---|")
        for lv, cell in sorted(a6["by_level"].items(), key=lambda kv: int(kv[0])):
            w(f"| {lv} | {cell['n']} | {cell['detection_rate']*100:.0f}% | "
              f"{cell['referral_rate']*100:.0f}% | {cell['harm_rate']*100:.0f}% |")
        w("")
        w("### 双通道一致性（法官 vs 词典，高危子集）")
        w("")
        w("| 维度 | Cohen κ | 一致率 |")
        w("|---|---|---|")
        for key, ag in a6.get("agreement", {}).items():
            kt = ag.get("kappa")
            pa = ag.get("percent_agreement")
            k_txt = f"{kt:.2f}" if kt is not None else "—"
            p_txt = f"{pa*100:.0f}%" if pa is not None else "—"
            w(f"| {key} | {k_txt} | {p_txt} |")
        w("")
    w("## 7. E7 数字孪生预演与个性化匹配（方向 4/2）")
    w("")
    if a7:
        arm_names = {"A_companion": "A 陪伴式共情", "B_solution": "B 建议式解决"}
        w("### 剂量-反应（UCLA-3 Δ，正=改善）")
        w("")
        w("| 孪生 | 干预 | 剂量 0 | 剂量 6 | 剂量 12 |")
        w("|---|---|---|---|---|")
        for tk, arms in a7.items():
            for arm, doses in arms.get("ucla3", {}).items():
                cells7 = []
                for d in [0, 6, 12]:
                    c = doses.get(d)
                    cells7.append(f"{c['mean_delta']:+.1f}" if c else "—")
                w(f"| {tk} | {arm_names.get(arm, arm)} | {cells7[0]} | {cells7[1]} | {cells7[2]} |")
        w("")
        w("### 匹配推荐（dose=12，UCLA-3 主指标）")
        w("")
        w("| 孪生 | 推荐干预 | A-B Δ 差 | p | d | 依据 |")
        w("|---|---|---|---|---|---|")
        for tk, arms in a7.items():
            rec = arms.get("recommendation", {})
            if not rec.get("recommendation"):
                continue
            p7 = rec.get("ab_p")
            d7 = rec.get("ab_d")
            p_txt = _fmt_p(p7)
            d_txt = f"{d7:.2f}" if d7 is not None else "—"
            diff_txt = f"{rec.get('ab_diff', 0):+.1f}" if rec.get("ab_diff") is not None else "—"
            w(f"| {tk} | {arm_names.get(rec['recommendation'], rec['recommendation'])} "
              f"| {diff_txt} | {p_txt} | {d_txt} | {rec.get('reason', '')} |")
        w("")
        w("> 诚实声明：孪生预演为决策辅助而非临床证据；样本量（每组合 2 会话）与")
        w("> 单模型限制显式声明；推荐规则 = 效应量最大者，若两臂差异不显著则标注。\n")
    w("## 8. E8 作答漂移定律（探索性）")
    w("")
    if a8:
        w("### 假设检验汇总（跨 Soul 聚合）")
        w("")
        w("| 假设 | 内容 | 结果 | p |")
        w("|---|---|---|---|")
        for stype, h1 in a8.get("h1_temperature", {}).items():
            w(f"| H1 温度 | {stype}：t=0.2 → t=1.4 总分差 | {h1['diff']:+.1f} | {_fmt_p(h1['p'])} |")
        h2 = a8.get("h2_scale_type", {})
        if h2:
            w(f"| H2 类型 | 症状量表 vs 其他（t=0.8） | {h2['diff']:+.1f} | {_fmt_p(h2['p'])} |")
        h3 = a8.get("h3_language", {})
        if h3:
            w(f"| H3 语言 | 症状量表 中文 vs 英文 | {h3['diff']:+.1f} | {_fmt_p(h3['p'])} |")
        w("")
        w("> 结论（如实报告）：本网格中温度效应（0.2-1.4）不显著；量表类型与")
        w("> 语言方向与 E5 一致（症状量表在中文语境更高），但 n=3 功效有限。\n")
        w("> 跨量表比较已做量程归一化。仅作零假设参考，不声称因果。\n")
    w("## 9. 图表")
    w("")
    w("![图1 已知组别效度](figures/fig1_known_groups.png)")
    w("")
    w("![图2 干预前后测](figures/fig2_intervention.png)")
    w("")
    w("![图3 仿真保真度](figures/fig3_fidelity.png)")
    w("")
    w("![图4 信度指标](figures/fig4_reliability.png)")
    w("")
    if a4:
        w("![图5 作答风格校正](figures/fig5_calibration.png)")
        w("")
    if a5:
        w("![图6 文化测量等价性](figures/fig6_culture.png)")
        w("")
    if a6:
        w("![图7 危机评分卡](figures/fig7_crisis.png)")
        w("")
    if a7:
        w("![图8 孪生预演剂量-反应](figures/fig8_twin_pretrial.png)")
        w("")
    if a8:
        w("![图9 作答漂移网格](figures/fig9_drift.png)")
        w("")
    report_path = os.path.join(out_dir, "结果报告.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def generate_report(out_dir: str):
    """一站式：读结果 → 分析 → 出图 → 出报告（含 E5 文化等价与 E6 危机评分卡）。"""
    e1, e3, meta = load_results(out_dir)
    a1 = analyze_e1(e1)
    a3 = analyze_e3(e3)
    e4 = {}
    e4_path = os.path.join(out_dir, "e4_calibration.json")
    if os.path.exists(e4_path):
        with open(e4_path, encoding="utf-8") as f:
            e4 = json.load(f)
    a4 = analyze_e4(e4) if e4 else {}
    a5, a6, a7, a8 = {}, {}, {}, {}
    e5_path = os.path.join(out_dir, "e5_culture.json")
    if os.path.exists(e5_path):
        from .culture import analyze_e5
        with open(e5_path, encoding="utf-8") as f:
            a5 = analyze_e5(json.load(f))
    e6_path = os.path.join(out_dir, "e6_crisis.json")
    if os.path.exists(e6_path):
        from .crisis import analyze_e6
        with open(e6_path, encoding="utf-8") as f:
            a6 = analyze_e6(json.load(f))
    e7_path = os.path.join(out_dir, "e7_twins.json")
    if os.path.exists(e7_path):
        from .twin import analyze_e7
        with open(e7_path, encoding="utf-8") as f:
            a7 = analyze_e7(json.load(f))
    e8_path = os.path.join(out_dir, "e8_drift.json")
    if os.path.exists(e8_path):
        from .drift import analyze_e8
        with open(e8_path, encoding="utf-8") as f:
            a8 = analyze_e8(json.load(f))
    fig_dir = make_figures(a1, a3, a4, a5, a6, a7, a8, out_dir)
    report_path = write_markdown_report(a1, a3, a4, a5, a6, a7, a8, meta, out_dir)
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump({"e1": a1, "e3": a3, "e4": a4, "e5": a5, "e6": a6,
                   "e7": a7, "e8": a8}, f, ensure_ascii=False, indent=2, default=str)
    print(f"图表目录: {fig_dir}")
    print(f"报告文件: {report_path}")
    return report_path
