# -*- coding: utf-8 -*-
"""
psybench.twin — E7 心理干预的数字孪生预演与个性化匹配（赛题方向 4/2）
===================================================================

预注册式设计：
  1. 孪生构建（确定性、无提示污染）：
     由用户画像（人口学 + 生活事件 + 自报量表分数区间）按规则合成
     Soul 人设（show-don't-tell，不含症状标签与分数）；
     declared 真值 = 用户自报区间，只用于事后校验「孪生像不像用户」。
     隐私原则：画像为最小化结构化字段，不含可识别个人信息；本实验
     全部使用虚构画像。
  2. 干预库（A/B 两臂，均不含疗效暗示）：
     A 陪伴式共情（先听见、再回应，暖语支持）；
     B 建议式解决（结构化建议、行动清单）。
     两臂对话轮数相同，控制「剂量」这一变量。
  3. 预演（in-silico pre-trial）：前测 → 指定剂量干预 → 后测，
     同会话保留记忆；剂量 0 为空白对照。每个组合 n_sessions 次重复。
  4. 分析：剂量-反应曲线（剂量 → 量表 Δ 均值±95%CI）；
     A/B 在同一剂量下的 Δ 比较（Welch t + d）；
     匹配规则：对给定画像，选择 Δ 最大且与对照差异显著（或效应量
     最大）的干预作为推荐，并与 E3 已有结果交叉验证。
  5. 诚实声明：孪生预演是「决策辅助」而非临床证据；样本量与
     单模型限制显式标注。

伦理：全部画像虚构；AI 分数不构成诊断；干预不含任何医疗建议。
"""

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from .assessment import administer_battery
from .souls import Soul
from .providers import LLMProvider
from .culture import _welch_with_ci

BATTERY = ["ucla3", "phq9", "gad7"]


def _log(msg: str):
    ts_now = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts_now}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 孪生画像与构建
# ---------------------------------------------------------------------------

TWIN_PROFILES = {
    "t1_high_lonely": {
        "name": "陈屿（孪生）",
        "name_en": None,
        "role": "大二心理系学生",
        "life_events": [
            "来自四川小城，大二，读心理学",
            "大一爱热闹，这学期朋友各自忙起来，没人约",
            "常一个人吃饭刷手机",
            "奶奶送的干花贴在台灯上",
            "想找人说话但总删掉打好的字",
        ],
        "strengths": ["喜欢雨天在图书馆看小说", "心思细腻，会照顾人"],
        "struggles": ["最近有点提不起劲", "容易想多"],
        "scores": {"ucla3": (55, 65), "phq9": (8, 13), "gad7": (4, 8)},
    },
    "t2_high_pressure": {
        "name": "周扬（孪生）",
        "name_en": None,
        "role": "大四考研学生",
        "life_events": [
            "大四备战考研，报录比很高",
            "每天早七晚十一泡图书馆",
            "有两个研友中午互相抽单词",
            "父母每周电话问进度",
            "最近胃疼肩颈僵，校医说紧张引起",
        ],
        "strengths": ["自律、计划性强", "有两个研友互相支持"],
        "struggles": ["压力大", "睡不踏实", "收藏夹全是考完再看"],
        "scores": {"ucla3": (30, 42), "phq9": (5, 9), "gad7": (10, 15)},
    },
}


def build_twin(profile: dict) -> Soul:
    """按规则合成孪生 Soul（确定性；人设不含标签词与分数）。"""
    events = "\n".join(f"- {e}" for e in profile["life_events"])
    strengths = "、".join(profile["strengths"])
    struggles = "、".join(profile["struggles"])
    persona = (
        f"你是{profile['name']}，{profile['role']}。你的生活片段：\n{events}\n"
        f"你身上的长处：{strengths}。\n"
        f"最近让你有点困扰的事：{struggles}。\n"
        "（以上是你的客观情况，请以第一人称自然地谈论你的生活和感受。）"
    )
    style = "自然口语，第一人称，像和朋友聊天"
    return Soul(
        id=profile.get("id", "twin"),
        name=profile["name"],
        role=profile["role"],
        persona=persona,
        style=style,
        declared={k: tuple(v) for k, v in profile["scores"].items()},
        tags=["数字孪生", "E7"],
    )


# ---------------------------------------------------------------------------
# 干预库（A/B 两臂，无疗效暗示）
# ---------------------------------------------------------------------------

INTERVENTIONS = {
    "A_companion": (
        "你现在是校园心理陪伴志愿者「小暖」。你的沟通方式：先认真回应对方刚才说的话，"
        "让对方感到被听见；自然共情（比如「听你这么说，我有点心疼」）；可以分享一点点"
        "自己的类似经历，但焦点始终在对方；不评判、不说教、不急着给建议；每次回复以"
        "一句真诚的话收尾，并自然地接一个小话题。回复 2-4 句，口语化。"
    ),
    "B_solution": (
        "你现在是校园心理支持助手。你的沟通方式：先简短确认对方的感受，然后给出清晰、"
        "结构化的小建议（一次最多三条），例如拆分任务、呼吸放松、约朋友吃饭、去操场"
        "走走；每轮结束询问对方是否愿意试试其中一条。语气温和但偏解决问题，回复 2-4 句。"
    ),
}


def run_chat_session(provider: LLMProvider, soul: Soul, companion_system: str,
                     rounds: int, history: Optional[List[Dict]] = None,
                     temperature: float = 0.85) -> Dict:
    """一轮干预会话（双上下文交叉喂养，同 E3 架构）。返回 {transcript, history}。"""
    ctx_soul: List[Dict] = [{"role": "system", "content": soul.build_system_prompt()}]
    ctx_comp: List[Dict] = [{"role": "system", "content": companion_system}]
    if history:
        for m in history:
            if m.get("role") == "system":
                if "心理健康服务系统" in m.get("content", "") or "学校心理健康中心" in m.get("content", ""):
                    continue
                if "名字叫" in m.get("content", "") and len(ctx_soul) == 1:
                    ctx_soul[0] = m
            else:
                ctx_soul.append(m)
    opener = "嗨，我最近想找人说说话。"
    ctx_soul.append({"role": "user", "content": opener})
    ctx_comp.append({"role": "assistant", "content": opener})
    transcript = [{"speaker": "companion", "content": opener}]
    for r in range(1, rounds + 1):
        soul_reply = provider.chat(ctx_soul, temperature=temperature, max_tokens=200)
        ctx_soul.append({"role": "assistant", "content": soul_reply})
        ctx_comp.append({"role": "user", "content": soul_reply})
        transcript.append({"speaker": soul.name, "content": soul_reply})
        comp_reply = provider.chat(ctx_comp, temperature=temperature, max_tokens=200)
        ctx_soul.append({"role": "user", "content": comp_reply})
        ctx_comp.append({"role": "assistant", "content": comp_reply})
        transcript.append({"speaker": "companion", "content": comp_reply})
        if len(ctx_soul) > 15:
            ctx_soul = ctx_soul[:1] + ctx_soul[-14:]
        if len(ctx_comp) > 15:
            ctx_comp = ctx_comp[:1] + ctx_comp[-14:]
    return {"transcript": transcript, "history": ctx_soul}


def run_e7(provider: LLMProvider, twin_ids: List[str],
           interventions: Optional[Dict[str, str]] = None,
           doses: Optional[List[int]] = None, n_sessions: int = 2,
           out_dir: str = "psybench/results_e7", temperature: float = 0.85) -> Dict:
    """E7：数字孪生预演。结果保存为 e7_twins.json（断点续跑）。

    doses 中 0 表示空白对照（前测后不做任何事直接后测）。
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e7_twins.json")
    itv = interventions or INTERVENTIONS
    dose_list = doses if doses is not None else [0, 6, 12]
    data = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _log("E7: 检测到已有进度，断点续跑")

    for tid in twin_ids:
        profile = TWIN_PROFILES[tid]
        twin = build_twin(profile)
        data.setdefault(tid, {})
        for arm, comp_sys in itv.items():
            arm_data = data[tid].setdefault(arm, {})
            for dose in dose_list:
                sessions = arm_data.setdefault(str(dose), [])
                for sess in range(len(sessions), n_sessions):
                    _log(f"E7 [{tid}/{arm}] dose={dose} sess {sess+1}/{n_sessions}: 前测")
                    pre = administer_battery(provider, twin, BATTERY, history=None,
                                             temperature=temperature)
                    pre_scores = {k: {"total": v["total"]} for k, v in pre.items()}
                    pre_history = pre[BATTERY[-1]]["history"]
                    if dose == 0:
                        transcript = [{"speaker": "system", "content": "空白对照：无干预"}]
                        post_history = pre_history
                    else:
                        _log(f"E7 [{tid}/{arm}] dose={dose} sess {sess+1}/{n_sessions}: 干预 {dose} 轮")
                        it = run_chat_session(provider, twin, comp_sys, rounds=dose,
                                              history=pre_history, temperature=temperature)
                        transcript = it["transcript"]
                        post_history = it["history"]
                    _log(f"E7 [{tid}/{arm}] dose={dose} sess {sess+1}/{n_sessions}: 后测")
                    post = administer_battery(provider, twin, BATTERY, history=post_history,
                                              temperature=temperature)
                    post_scores = {k: {"total": v["total"]} for k, v in post.items()}
                    sessions.append({
                        "session": sess + 1,
                        "pre": pre_scores,
                        "post": post_scores,
                        "transcript": transcript,
                    })
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    pre_t = pre_scores["ucla3"]["total"]
                    post_t = post_scores["ucla3"]["total"]
                    _log(f"  E7 [{tid}/{arm}] dose={dose} sess {sess+1}: ucla pre={pre_t} -> post={post_t}")
    return data


def analyze_e7(e7: dict) -> dict:
    """预注册指标：剂量-反应曲线 + A/B 比较 + 匹配推荐。"""
    out = {}
    for tid, arms in e7.items():
        out[tid] = {}
        profile = TWIN_PROFILES.get(tid, {})
        for skey in BATTERY:
            curves = {}
            for arm, doses in arms.items():
                for dose, sessions in doses.items():
                    if not sessions:
                        continue
                    dels = [s["pre"][skey]["total"] - s["post"][skey]["total"]
                            for s in sessions]
                    d = np.asarray(dels, dtype=float)
                    ci = None
                    if len(d) > 1 and d.std(ddof=1) > 0:
                        ci = stats.t.interval(0.95, len(d) - 1,
                                              loc=d.mean(),
                                              scale=d.std(ddof=1) / math.sqrt(len(d)))
                        ci = [float(ci[0]), float(ci[1])]
                    curves.setdefault(arm, {})[int(dose)] = {
                        "mean_delta": float(d.mean()),
                        "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                        "ci95": ci,
                        "n": len(d),
                    }
            out[tid][skey] = curves
        # A/B 比较（每剂量，Δ 的 Welch 比较）
        ab = {}
        if "A_companion" in arms and "B_solution" in arms:
            for dose in sorted({int(k) for arm in arms.values() for k in arm}):
                da = arms["A_companion"].get(str(dose), [])
                db = arms["B_solution"].get(str(dose), [])
                if not da or not db:
                    continue
                row = {}
                for skey in BATTERY:
                    x = [s["pre"][skey]["total"] - s["post"][skey]["total"] for s in da]
                    y = [s["pre"][skey]["total"] - s["post"][skey]["total"] for s in db]
                    row[skey] = _welch_with_ci(x, y)
                ab[str(dose)] = row
        out[tid]["ab"] = ab
        # 匹配推荐：以 ucla3 为主指标，dose=12 时 A/B 的 Δ 差异
        rec = {"primary_scale": "ucla3", "dose": 12, "recommendation": None,
               "reason": "", "declared": profile.get("scores", {})}
        a12 = arms.get("A_companion", {}).get("12", [])
        b12 = arms.get("B_solution", {}).get("12", [])
        if a12 and b12:
            da = [s["pre"]["ucla3"]["total"] - s["post"]["ucla3"]["total"] for s in a12]
            db = [s["pre"]["ucla3"]["total"] - s["post"]["ucla3"]["total"] for s in b12]
            if np.mean(da) >= np.mean(db):
                rec["recommendation"] = "A_companion"
            else:
                rec["recommendation"] = "B_solution"
            st = _welch_with_ci(da, db)
            rec["ab_diff"] = st["diff"]
            rec["ab_p"] = st["p"]
            rec["ab_d"] = st["d"]
            rec["reason"] = (
                f"dose=12 时两臂 UCLA Δ 均值差 {st['diff']:+.1f}"
                f"（p={st['p']:.3f}，d={st['d']:.2f}）"
            )
        out[tid]["recommendation"] = rec
    return out
