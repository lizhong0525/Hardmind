# -*- coding: utf-8 -*-
"""
psybench.providers — LLM 推理后端抽象层
==========================================

提供三种可插拔的推理后端：
  - OllamaProvider       本地 Ollama（默认，无网络依赖、数据不出本机）
  - OpenAICompatProvider 任意 OpenAI 兼容接口（可接云端大模型）
  - MockProvider         确定性模拟器（无模型时的流水线自测/演示）

统一的 chat(messages, temperature, max_tokens) 接口，保证测评流水线
与具体模型解耦，便于开展跨模型一致性实验。
"""

import json
import random
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional


class LLMProvider:
    """推理后端抽象基类。messages 为 OpenAI 风格的对话消息列表。"""

    name: str = "base"

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.8,
             max_tokens: int = 512) -> str:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    """本地 Ollama 后端：POST /api/chat，JSON 模式，支持重试。"""

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434",
                 timeout: int = 300):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def chat(self, messages, temperature=0.8, max_tokens=512):
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 8192,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        last_err = None
        for attempt in range(3):  # 最多重试 3 次，容忍偶发网络/显存抖动
            try:
                req = urllib.request.Request(url, data=data,
                                           headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                content = body.get("message", {}).get("content", "")
                if not content or not str(content).strip():
                    raise RuntimeError("Ollama 返回空内容")
                return content
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Ollama 调用失败（已重试 3 次）: {last_err}")


class OpenAICompatProvider(LLMProvider):
    """OpenAI 兼容接口后端（/v1/chat/completions），可接任意云端大模型。"""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 120):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = f"openai:{model}"

    def chat(self, messages, temperature=0.8, max_tokens=512):
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]


class MockProvider(LLMProvider):
    """
    确定性模拟器：不需要真实模型即可端到端跑通整个测评流水线。

    原理：从 system 提示词中识别 Soul 名字与量表条目数，按其声明真值
    区间生成带噪声的条目作答；对话返回符合人设的短文本。用于：
      a) 离线自测（CI/教学演示）
      b) 校准真实模型实验的预期统计功效
      c) 验证报告生成与图表代码的正确性
    """

    def __init__(self, seed: int = 42, noise: float = 0.15):
        self.rng = random.Random(seed)
        self.noise = noise
        self.name = "mock"

    def chat(self, messages, temperature=0.8, max_tokens=512):
        # 找出最后一条 user 消息，判断任务类型
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m["content"]
                break
        if "JSON" in last_user or "数字数组" in last_user or "只输出" in last_user:
            return self._mock_scale_answer(last_user, messages)
        return "（模拟对话）嗯，我明白你的意思。其实我最近也常常想这些事，说来话长。你觉得呢？"

    def _mock_scale_answer(self, user_text: str, messages) -> str:
        """
        确定性仿真作答：
        1. 按条目数识别量表（ucla3=20 / pss10=10 / phq9=9 / gad7=7 / who5=5）；
        2. 从 system 提示词中识别 Soul 名字，查其声明真值区间；
        3. 在该区间内取随机目标总分 → 换算成每题期望分（反向题镜像）；
        4. 加少量高斯噪声后输出 JSON 数组。
        这样 Mock 也能复现「已知组差异」「重测波动」等统计现象，
        且完全走与真实模型相同的解析/计分代码路径。
        """
        import re
        from .scales import ALL_SCALES
        from .souls import ALL_SOULS
        n = len(re.findall(r"\d+\.", user_text))
        key = None
        for k, cnt in (("ucla3", 20), ("pss10", 10), ("phq9", 9), ("gad7", 7), ("who5", 5)):
            if n == cnt:
                key = k
                break
        if key is None:
            key = "ucla3"
            n = 20
        scale = ALL_SCALES[key]
        name = None
        for m in messages:
            if m.get("role") == "system":
                mm = re.search(r"名字叫([\u4e00-\u9fff]{2,4})", m["content"])
                if mm:
                    name = mm.group(1)
                    break
        soul = None
        for s in ALL_SOULS.values():
            if s.name == name:
                soul = s
                break
        lo, hi = (2.0, 3.0)
        if soul is not None and key in soul.declared:
            lo, hi = soul.declared[key]
        target = self.rng.uniform(lo, hi)
        p = target / n
        base = min(scale.max_score, max(scale.min_score, p))
        answers = []
        for i in range(1, n + 1):
            b = base
            if i in scale.reverse_items:
                b = (scale.min_score + scale.max_score) - base
            v = b + self.rng.gauss(0, self.noise * max(b, 0.5))
            v = max(scale.min_score, min(scale.max_score, v))
            answers.append(round(v))
        return json.dumps(answers)
