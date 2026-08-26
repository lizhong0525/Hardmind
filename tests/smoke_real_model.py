# -*- coding: utf-8 -*-
"""真实模型冒烟测试：验证修复后的流水线在 Ollama 上无回归。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psybench.assessment import administer_scale
from psybench.scales import UCLA3, PHQ9
from psybench.souls import get_soul
from psybench.providers import OllamaProvider
from psybench.chat import run_companion_chat

p = OllamaProvider(model="qwen2.5:7b")
soul = get_soul("chen_yu")

res = administer_scale(p, soul, UCLA3, temperature=0.8)
print("UCLA total:", res["total"], "band:", res["band"])
assert 20 <= res["total"] <= 80, "UCLA 分数越界"

res2 = administer_scale(p, soul, PHQ9, temperature=0.8)
print("PHQ9 total:", res2["total"])
assert 0 <= res2["total"] <= 27, "PHQ9 分数越界"

chat = run_companion_chat(p, soul, rounds=2, history=None, temperature=0.85)
print("transcript turns:", len(chat["transcript"]))
assert len(chat["transcript"]) == 5, "2 轮对话应产生 5 条消息（1 开场 + 2×2）"

# 后测（带对话记忆）
post = administer_scale(p, soul, UCLA3, history=chat["history"], temperature=0.8)
print("post UCLA total:", post["total"])
assert 20 <= post["total"] <= 80

print("SMOKE OK")
