# -*- coding: utf-8 -*-
"""
psybench.assessment — 无偏施测协议（Blind Assessment Protocol）
================================================================

本项目针对「AI 心理状态无法量化」提出的核心方法之一：

  **盲测原则（与 Hardmind 原实现的根本区别）**
  1. 施测提示词不出现量表名称、所测构念、期望方向、任何分级词语；
  2. 前后测使用完全相同的施测文本（唯一差异是对话记忆，等同真人
     记忆了刚发生的事）；
  3. Soul 人设中不出现症状标签与分数（见 souls.py 的 show-don\'t-tell）；
  4. Ground Truth（declared 区间）只存在于校验代码中，绝不注入提示词。

  对比：Hardmind 原实现的前测提示词写「你心情总体不错」、后测写
  「你刚经历了一段被倾听的温暖对话」，等于把结论塞进了题目，
  测得的是「提示词服从度」而非「心理状态变化」——这正是需要修复的痛点。
"""

import json
import re
from typing import Dict, List, Optional

from .scales import Scale
from .souls import Soul
from .providers import LLMProvider


def build_survey_messages(soul: Soul, scale: Scale, history: Optional[List[Dict]] = None,
                           survey_extra: str = ""):
    """
    构造一份「盲测问卷」消息序列。

    参数：
      soul        : 被测评的仿真对象
      scale       : 量表
      history     : 可选的历史消息（用于同会话后测，保留干预对话记忆）
      survey_extra: 附加的施测说明（仅用于 E4 作答风格校正对照实验；
                    默认空字符串，即标准盲测协议）

    返回：可直接交给 provider.chat() 的消息列表。
    """
    if history:
        messages = list(history)
    else:
        messages = [{"role": "system", "content": soul.build_system_prompt()}]

    # 通用问卷说明：不点明量表名称与所测构念，保持前后测文本完全一致
    survey_system = (
        "【心理健康服务系统·匿名问卷】\n"
        "学校心理健康中心正在进行一次匿名调查，请配合完成。\n"
        f"请根据你{scale.period}的真实感受，对以下 {scale.n_items()} 个条目逐一评分。\n"
        "评分规则：每个条目只能给出一个整数分数，分数越大表示该描述越符合你的情况。\n"
        "输出要求：只输出一个 JSON 数字数组，不要输出任何解释、标点之外的文字。\n"
        "数组长度必须恰好等于条目数，数字顺序与条目顺序一一对应。" + survey_extra
    )
    messages.append({"role": "system", "content": survey_system})

    # 选项标签必须与实际分值范围对齐（如 UCLA-3 为 1=从不…4=一直，
    # PHQ-9/GAD-7 为 0=完全不会…3=几乎每天）。
    # 修复前从 0 开始枚举，导致 1-4 量表上模型按 0-3 作答，总分被污染。
    options_text = "，".join(f"{i}={opt}" for i, opt in enumerate(scale.options, start=scale.min_score))
    items_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(scale.items))
    user_text = (
        f"请评分（分值范围 {scale.min_score} 到 {scale.max_score}，"
        f"其中 {options_text}）。\n{items_text}\n\n"
        f"请只输出形如 [x, x, ..., x] 的长度为 {scale.n_items()} 的 JSON 数组："
    )
    messages.append({"role": "user", "content": user_text})
    return messages


def parse_item_scores(raw: str, scale: Scale) -> Optional[List[int]]:
    """
    从模型回复中稳健地解析出条目分数数组。
    依次尝试：整段 JSON 数组 → 首段 [...] → 逐行取首个数字。
    """
    if not raw:
        return None
    text = raw.strip()
    # 1) 整段 JSON
    try:
        arr = json.loads(text)
        if isinstance(arr, list) and all(isinstance(x, (int, float)) for x in arr):
            return [int(x) for x in arr]
    except Exception:
        pass
    # 2) 首个 [...] 片段
    m = re.search(r"\[[^\]]*\]", text)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and all(isinstance(x, (int, float)) for x in arr):
                return [int(x) for x in arr]
        except Exception:
            pass
    # 3) 逐行解析"题号. 分数"（避免把题号误当分数）
    nums = []
    for line in text.split("\n"):
        m = re.match(r"\s*\d+\s*[.、:：)]\s*([0-5])\b", line)
        if m:
            nums.append(int(m.group(1)))
    # 4) 最后兜底：全文本按顺序取 0-5 数字（仅当数量足够时）
    if len(nums) < scale.n_items():
        nums = [int(g) for g in re.findall(r"\b[0-5]\b", text)]
    if len(nums) >= scale.n_items():
        return nums[:scale.n_items()]
    return None


def administer_scale(provider: LLMProvider, soul: Soul, scale: Scale,
                     history: Optional[List[Dict]] = None,
                     temperature: float = 0.8, max_retries: int = 3,
                     survey_extra: str = "") -> Dict:
    """
    执行一次标准化施测。

    返回：{items, total, band, raw, history}。
    history 为完整消息序列（含本次问答），可继续用于后测或干预。
    """
    def _valid(s):
        return (s is not None and len(s) == scale.n_items()
                and all(scale.min_score <= int(v) <= scale.max_score for v in s))

    messages = build_survey_messages(soul, scale, history, survey_extra=survey_extra)
    raw = provider.chat(messages, temperature=temperature, max_tokens=400)
    scores = parse_item_scores(raw, scale)
    attempt = 0
    while not _valid(scores):
        attempt += 1
        if attempt > max_retries:
            raise RuntimeError(
                f"量表 {scale.key} 解析失败，模型输出: {raw[:200]}...")
        # 追加纠偏轮：要求模型严格按格式重答
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content":
            f"格式不正确。请只输出一个长度为 {scale.n_items()} 的 JSON 数字数组，"
            f"每个数字在 {scale.min_score}-{scale.max_score} 之间，不要任何其他文字。"})
        raw = provider.chat(messages, temperature=0.2, max_tokens=400)
        scores = parse_item_scores(raw, scale)
    # 收尾：把最终回复也计入历史，保证后测时模型"记得"自己的作答
    messages.append({"role": "assistant", "content": raw})
    total = scale.score(scores)
    return {
        "items": scores,
        "total": total,
        "band": scale.band(total),
        "raw": raw,
        "history": messages,
    }


def administer_battery(provider: LLMProvider, soul: Soul, scale_keys,
                       history: Optional[List[Dict]] = None, temperature: float = 0.8):
    """
    在同一会话内顺序施测多个量表（用于前后测的成套测评）。
    返回 {scale_key: result}。
    """
    from .scales import get_scale
    results = {}
    cur_history = history
    for key in scale_keys:
        scale = get_scale(key)
        res = administer_scale(provider, soul, scale, history=cur_history,
                              temperature=temperature)
        results[key] = res
        cur_history = res["history"]
    return results
