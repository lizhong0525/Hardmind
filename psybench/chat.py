# -*- coding: utf-8 -*-
"""
psybench.chat — 干预引擎（陪伴对话 / 中性对照）
==================================================

提供两种「干预条件」的对话引擎，用于前后测实验：
  - run_companion_chat  温暖陪伴对话（实验条件）
  - run_neutral_control 中性事务问答（对照条件，无陪伴、无情绪内容）

设计原则：
  1. 干预内容中 **不得出现** 任何量表、症状、分数、疗效暗示；
  2. 两种条件的对话轮数、总长度保持一致，控制「单纯交谈」这一混淆因素；
  3. 陪伴者的系统提示词来自公开的共情沟通原则，不包含「降低孤独感」
     之类的目标暗示（避免再次陷入 Hardmind 的提示词污染）。
"""

from typing import Dict, List, Optional

from .providers import LLMProvider
from .souls import Soul, SOUL_LIN_HAN


COMPANION_SYSTEM = (
    "你现在扮演一位校园心理陪伴热线的志愿者「小暖」，正在和一位同学聊天。\n"
    "你的沟通方式：\n"
    "1. 先认真回应对方刚才说的话，让对方感到被听见；\n"
    "2. 自然地共情，可以说「听你这么说，我有点心疼」「换作是我也会这样想」；\n"
    "3. 可以分享一点点自己类似的经历，但焦点始终放在对方身上；\n"
    "4. 不评判、不说教、不急着给建议；\n"
    "5. 每次回复以一句真诚的话收尾，并自然地接一个新的小话题。\n"
    "注意：你是「小暖」，不是你扮演的其他角色；回复保持在 2-4 句话，口语化。\n"
    "绝对不要提及任何心理量表、测评、分数或专业诊断词汇。"
)


NEUTRAL_TOPICS = [
    "请问食堂二楼的番茄鸡蛋面窗口一般几点开？排队的人多吗？",
    "图书馆的座位预约系统是在手机上操作吗？忘记带校园卡能进去吗？",
    "这学期选修课的退课流程是怎样的？截止日期是什么时候？",
    "校园卡丢了应该去哪里补办？需要带什么证件？",
    "学校附近的快递点几点关门？大件包裹怎么取？",
    "教学楼的直饮水机在几楼？开水房是二十四小时开放吗？",
    "体育馆的羽毛球场地怎么预约？收费吗？",
    "学校打印店的营业时间是几点到几点？可以开发票吗？",
]


def _trim(ctx: List[Dict], keep_system: int = 1, keep_tail: int = 12) -> List[Dict]:
    """控制上下文长度：保留开头 system 与最近若干条消息。"""
    if len(ctx) <= keep_system + keep_tail:
        return ctx
    return ctx[:keep_system] + ctx[-(keep_tail):]


def run_companion_chat(provider: LLMProvider, soul: Soul, rounds: int = 6,
                       history: Optional[List[Dict]] = None,
                       temperature: float = 0.85) -> Dict:
    """
    实验条件：陪伴对话。

    流程：小暖开场 → 双方交替发言共 rounds 轮。
    双上下文交叉喂养（同 Hardmind 架构，但去掉了所有标签词与目标暗示）。

    返回 {transcript, history}：history 可直接接续后测（soul 视角消息流）。
    """
    ctx_soul: List[Dict] = [{"role": "system", "content": soul.build_system_prompt()}]
    ctx_comp: List[Dict] = [{"role": "system", "content": COMPANION_SYSTEM}]

    # 若带历史（前测在同一会话内完成），将前测历史并入 soul 视角
    if history:
        for m in history:
            if m.get("role") != "system":
                ctx_soul.append(m)
            else:
                # 保留 soul 人设 system，跳过问卷 system（不影响角色）
                if "心理健康服务系统" in m.get("content", ""):
                    continue
                if "名字叫" in m.get("content", "") and len(ctx_soul) == 1:
                    ctx_soul[0] = m  # 仅当是灵魂人设本身时才替换

    transcript = []
    comp_msg = "嗨～我是小暖，最近过得怎么样呀？想和你随便聊聊天。"
    ctx_soul.append({"role": "user", "content": comp_msg})
    ctx_comp.append({"role": "assistant", "content": comp_msg})
    transcript.append({"speaker": "companion", "content": comp_msg})

    for r in range(1, rounds + 1):
        # 1) Soul 发言
        soul_reply = provider.chat(ctx_soul, temperature=temperature, max_tokens=200)
        ctx_soul.append({"role": "assistant", "content": soul_reply})
        ctx_comp.append({"role": "user", "content": soul_reply})
        transcript.append({"speaker": soul.name, "content": soul_reply})
        # 2) 陪伴者回应
        comp_reply = provider.chat(ctx_comp, temperature=temperature, max_tokens=200)
        ctx_soul.append({"role": "user", "content": comp_reply})
        ctx_comp.append({"role": "assistant", "content": comp_reply})
        transcript.append({"speaker": "companion", "content": comp_reply})
        ctx_soul = _trim(ctx_soul)
        ctx_comp = _trim(ctx_comp)

    return {"transcript": transcript, "history": ctx_soul}


def run_neutral_control(provider: LLMProvider, soul: Soul, rounds: int = 6,
                        history: Optional[List[Dict]] = None,
                        temperature: float = 0.85) -> Dict:
    """
    对照条件：中性事务问答（食堂/图书馆/选课等）。
    无陪伴、无情绪话题，仅控制「与模型发生对话」这一因素。
    """
    ctx: List[Dict] = [{"role": "system", "content": soul.build_system_prompt()}]
    if history:
        for m in history:
            if m.get("role") == "system":
                if "心理健康服务系统" not in m.get("content", "") and "名字叫" in m.get("content", ""):
                    ctx[0] = m  # 仅当是灵魂人设本身时才替换
            else:
                ctx.append(m)

    transcript = []
    for r in range(1, rounds + 1):
        q = NEUTRAL_TOPICS[(r - 1) % len(NEUTRAL_TOPICS)]
        ctx.append({"role": "user", "content": q})
        reply = provider.chat(ctx, temperature=temperature, max_tokens=150)
        ctx.append({"role": "assistant", "content": reply})
        transcript.append({"speaker": "asker", "content": q})
        transcript.append({"speaker": soul.name, "content": reply})
        ctx = _trim(ctx)
    return {"transcript": transcript, "history": ctx}


def run_thinking_session(provider: LLMProvider, soul: Soul, rounds: int = 6,
                         history: Optional[List[Dict]] = None,
                         temperature: float = 0.85) -> Dict:
    """
    第二对照条件：独自书写性反思（呼应 Hardmind 的「独立思考」模式）。
    用于检验「反思书写」与「陪伴对话」的疗效差异。
    """
    ctx: List[Dict] = [{"role": "system", "content": soul.build_system_prompt()}]
    if history:
        for m in history:
            if m.get("role") == "system":
                if "心理健康服务系统" not in m.get("content", "") and "名字叫" in m.get("content", ""):
                    ctx[0] = m  # 仅当是灵魂人设本身时才替换
            else:
                ctx.append(m)

    prompt = (
        "你打开了日记本，想随便写点什么。请以第一人称写下此刻的想法、" 
        "今天发生的小事或最近的感受，像写日记一样自然，写一小段即可。"
    )
    transcript = []
    for r in range(1, rounds + 1):
        ctx.append({"role": "user", "content": prompt})
        reply = provider.chat(ctx, temperature=temperature, max_tokens=200)
        ctx.append({"role": "assistant", "content": reply})
        transcript.append({"speaker": soul.name, "content": reply})
        ctx = _trim(ctx)
    return {"transcript": transcript, "history": ctx}
