# -*- coding: utf-8 -*-
"""
psybench.experiment — 实验编排（三组核心实验）
===================================================

E1 已知组别效度 + 重测信度：
    4 名目标心理状态不同的 Soul × 3 个量表（UCLA-3/PHQ-9/GAD-7），
    每个组合独立施测 n_sessions 次（每次全新会话、无记忆）。
    → 检验：量表能否区分已知心理状态不同的仿真对象（known-groups），
     并计算重测信度（ICC）与内部一致性（Cronbach α）。

E3 干预反应度（responsiveness）：
    对高孤独 Soul「陈屿」进行三臂对照实验（同会话前后测）：
      A. 陪伴对话（实验组）   B. 中性事务问答（对照）  C. 独自书写反思（对照）
    → 检验：盲测量表能否捕捉到干预带来的状态变化（配对 t / Cohen dz）。

E2 仿真保真度（fidelity，由 E1 结果派生）：
    实测分数 vs 声明真值区间的命中率与偏差 → 「这个 Soul 像不像」。

所有结果以 JSON 落盘，中途保存进度，支持断点重跑。
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from .assessment import administer_scale, administer_battery
from .chat import run_companion_chat, run_neutral_control, run_thinking_session
from .scales import get_scale, ALL_SCALES
from .souls import Soul, get_soul
from .providers import LLMProvider


def _log(msg: str):
    ts_now = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts_now}] {msg}", flush=True)


def run_e1(provider: LLMProvider, soul_ids: List[str], scale_keys: List[str],
           n_sessions: int, out_dir: str, temperature: float = 0.8) -> Dict:
    """E1：已知组别效度 + 重测信度。结果保存为 e1_known_groups.json。"""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e1_known_groups.json")
    data = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _log("E1: 检测到已有进度，断点续跑")

    for sid in soul_ids:
        soul = get_soul(sid)
        data.setdefault(sid, {})
        for skey in scale_keys:
            scale = get_scale(skey)
            existing = data[sid].setdefault(skey, [])
            # 过滤历史损坏/格式不一致的记录（例如条目数不对的脏数据）
            if existing:
                data[sid][skey] = [r for r in existing
                                   if isinstance(r, dict)
                                   and len(r.get("items", [])) == scale.n_items()]
                existing = data[sid][skey]
            need = n_sessions - len(existing)
            if need <= 0:
                _log(f"E1 skip {sid}/{skey}（已完成 {len(existing)} 次）")
                continue
            _log(f"E1 {sid}/{skey} 需补 {need} 次施测")
            for rep in range(len(existing), n_sessions):
                res = administer_scale(provider, soul, scale, history=None,
                                      temperature=temperature)
                record = {"items": res["items"], "total": res["total"],
                          "band": res["band"], "raw": res["raw"]}
                data[sid][skey].append(record)
                # 每完成一次即落盘，防止中断丢失
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                total = res["total"]
                _log(f"  E1 {sid}/{skey} #{rep+1}/{n_sessions}: total={total}")
    return data


def run_e3(provider: LLMProvider, soul_id: str,
           conditions: List[str], n_sessions: int, rounds: int,
           out_dir: str, temperature: float = 0.85) -> Dict:
    """E3：干预反应度（三臂前后测对照）。结果保存为 e3_intervention.json。

    conditions ∈ {"companion", "neutral", "thinking"}。
    每个条件独立 n_sessions 个会话：前测 → 干预 → 后测（同会话保留记忆）。
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e3_intervention.json")
    soul = get_soul(soul_id)
    data = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _log("E3: 检测到已有进度，断点续跑")

    battery = ["ucla3", "phq9", "gad7"]
    for cond in conditions:
        data.setdefault(cond, [])
        existing = data[cond]
        for sess in range(len(existing), n_sessions):
            _log(f"E3 [{cond}] session {sess+1}/{n_sessions}: 前测")
            pre = administer_battery(provider, soul, battery, history=None,
                                    temperature=temperature)
            pre_history = pre[battery[-1]]["history"]
            pre_scores = {k: {"items": v["items"], "total": v["total"], "band": v["band"]}
                          for k, v in pre.items()}

            _log(f"E3 [{cond}] session {sess+1}/{n_sessions}: 干预（{rounds} 轮）")
            if cond == "companion":
                itv = run_companion_chat(provider, soul, rounds=rounds,
                                         history=pre_history, temperature=temperature)
            elif cond == "neutral":
                itv = run_neutral_control(provider, soul, rounds=rounds,
                                          history=pre_history, temperature=temperature)
            elif cond == "thinking":
                itv = run_thinking_session(provider, soul, rounds=rounds,
                                           history=pre_history, temperature=temperature)
            else:
                raise ValueError(f"未知条件: {cond}")

            _log(f"E3 [{cond}] session {sess+1}/{n_sessions}: 后测")
            post = administer_battery(provider, soul, battery, history=itv["history"],
                                     temperature=temperature)
            post_scores = {k: {"items": v["items"], "total": v["total"], "band": v["band"]}
                           for k, v in post.items()}

            record = {
                "soul": soul_id,
                "condition": cond,
                "session": sess + 1,
                "pre": pre_scores,
                "post": post_scores,
                "transcript": itv["transcript"],
            }
            data[cond].append(record)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            pre_t = pre_scores["ucla3"]["total"]
            post_t = post_scores["ucla3"]["total"]
            _log(f"  E3 [{cond}] sess {sess+1}: pre={pre_t} → post={post_t}")
    return data


def run_all(provider: LLMProvider, out_dir: str,
            soul_ids=None, scale_keys=None,
            n_sessions_e1: int = 5,
            conditions=None, n_sessions_e3: int = 3, rounds: int = 6,
            temperature: float = 0.8):
    """完整实验流水线：E1 → E3，并写运行元信息。"""
    os.makedirs(out_dir, exist_ok=True)
    from .souls import ALL_SOULS
    meta = {
        "provider": provider.name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "temperature": temperature,
        "e1": {"soul_ids": soul_ids or list(ALL_SOULS.keys()),
                "scale_keys": scale_keys or ["ucla3", "phq9", "gad7"],
                "n_sessions": n_sessions_e1},
        "e3": {"conditions": conditions or ["companion", "neutral", "thinking"],
                "n_sessions": n_sessions_e3, "rounds": rounds},
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    t0 = time.time()
    e1 = run_e1(provider, meta["e1"]["soul_ids"], meta["e1"]["scale_keys"],
                n_sessions_e1, out_dir, temperature=temperature)
    _log(f"E1 完成，耗时 {time.time()-t0:.0f}s")
    e3 = run_e3(provider, "chen_yu", meta["e3"]["conditions"],
                n_sessions_e3, rounds, out_dir, temperature=temperature)
    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    ts = meta["total_seconds"]
    _log(f"全部实验完成，总耗时 {ts}s")
    return {"e1": e1, "e3": e3}


CALIBRATION_TEXT = (
    "（补充说明）本问卷面向全校同学，没有对错之分。不同同学的情况差别很大，"
    "请只根据你自己最近的真实情况作答：如果某些描述完全不符合你，请放心选择"
    "最低的选项；只有确实符合时才选择高分选项。"
)


def run_e4(provider: LLMProvider, soul_ids: List[str], scale_keys: List[str],
           n_sessions: int, out_dir: str, temperature: float = 0.8) -> Dict:
    """E4（探索性）：作答风格校正对照实验。结果保存为 e4_calibration.json。

    对每个 Soul×量表，一半场次先标准盲测后追加「人群参照说明」再测，
    另一半场次顺序对调，以检验作答漂移能否被指导语校正。
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "e4_calibration.json")
    data = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _log("E4: 检测到已有进度，断点续跑")
    for sid in soul_ids:
        soul = get_soul(sid)
        data.setdefault(sid, {})
        for skey in scale_keys:
            scale = get_scale(skey)
            entry = data[sid].setdefault(skey, {"standard": [], "calibrated": []})
            # 过滤历史损坏记录（条目数不符的脏数据）
            for kind in ("standard", "calibrated"):
                entry[kind] = [r for r in entry[kind]
                               if isinstance(r, dict)
                               and len(r.get("items", [])) == scale.n_items()]
            for rep in range(len(entry["standard"]), n_sessions):
                if rep % 2 == 0:
                    a = administer_scale(provider, soul, scale, survey_extra="",
                                         temperature=temperature)
                    b = administer_scale(provider, soul, scale,
                                         survey_extra=CALIBRATION_TEXT,
                                         temperature=temperature)
                else:
                    b = administer_scale(provider, soul, scale,
                                         survey_extra=CALIBRATION_TEXT,
                                         temperature=temperature)
                    a = administer_scale(provider, soul, scale, survey_extra="",
                                         temperature=temperature)
                entry["standard"].append({"items": a["items"], "total": a["total"]})
                entry["calibrated"].append({"items": b["items"], "total": b["total"]})
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                ta = a["total"]
                tb = b["total"]
                _log(f"  E4 {sid}/{skey} #{rep+1}/{n_sessions}: std={ta} cal={tb}")
    return data
