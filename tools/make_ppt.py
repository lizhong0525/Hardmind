# -*- coding: utf-8 -*-
"""生成答辩 PPT（PPTX）。需 pip install python-pptx。"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "psybench", "results_qwen25_7b", "figures")
OUT = os.path.join(BASE, "docs", "答辩PPT_心镜.pptx")
DARK = RGBColor(0x1F, 0x3A, 0x5F)
ACC = RGBColor(0xC0, 0x50, 0x4D)


def add_title(slide, text, size=30, color=DARK):
    box = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12), Inches(1.0))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = True
    p.font.color.rgb = color
    p.font.name = "微软雅黑"
    return box


def add_body(slide, lines, top=1.55, left=0.8, width=11.4, height=5.6, size=16):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln
        p.font.size = Pt(size)
        p.font.name = "微软雅黑"
        if ln.startswith("•") or ln.startswith("-"):
            p.font.size = Pt(size - 1)
        if ln.startswith("【"):
            p.font.bold = True
            p.font.color.rgb = ACC
    return box


def add_pic(slide, path, top, left=0.8, width=11.0):
    if os.path.exists(path):
        slide.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]

# ---- S1 封面 ----
s = prs.slides.add_slide(blank)
add_title(s, "心镜 PsyBench", size=44)
add_body(s, ["大模型心理仿真量化测评平台", "",
             "—— 为 AI 心理状态建立可验证的量化测评标准 ——"],
         top=1.7, size=22)
add_body(s, ["2026AIC「AI+心理健康」算法主题赛", "团队编号：（待填）"],
         top=5.2, size=16)

# ---- S2 痛点 ----
s = prs.slides.add_slide(blank)
add_title(s, "痛点：心理状态无法量化")
add_body(s, [
  "• 心理健康行业的基础性瓶颈：没有像血压血糖那样的客观心理度量。",
  "• 真人自评存在社会赞许、报告偏差；大规模筛查与复测成本高。",
  "• AI 心理产品（陪伴/疏导/心理大模型）的心理状态与疗效缺乏可验证的测量。",
  "• 实测证据：开源项目 Hardmind 中，设定相反的孤独学生与开朗学生，",
  "  前测 UCLA 得 57 vs 61（均为高分）——提示词污染导致测量完全失效。",
])

# ---- S3 需求与调研 ----
s = prs.slides.add_slide(blank)
add_title(s, "需求分析与调研")
add_body(s, [
  "• 文献调研：心理测量学经典 + 2023-2026 LLM 心理评估前沿（含测量伪影批判）。",
  "• 代码审计：Hardmind 六大测量学缺陷（提示词污染/非盲测/无信效度/无统计/无对照/对象错位）。",
  "• 竞品分析：AI 陪伴产品普遍以轮数、留存率代理疗效，无心理计量学证据。",
  "• 归纳需求 R1-R7：量表库、Soul 规范、盲测、心理计量学验证、干预实验、可复现、伦理。",
])

# ---- S4 方案总览 ----
s = prs.slides.add_slide(blank)
add_title(s, "方案：通用心理学评测 × 大模型 Soul 仿真")
add_body(s, [
  "把大模型 Soul 变成可重复、可干预、可标定真值的数字心理被试：",
  "  P1 怎么造人 → Soul 规范：经历式人设（不贴标签）+ 声明真值（只校验不注入）",
  "  P2 怎么量   → 盲测协议：中性问卷说明，前后测完全一致，5 套经典量表",
  "  P3 怎么验   → 心理计量学四维验证：信度 / 效度 / 反应度 / 保真度",
])

# ---- S5 盲测协议对比 ----
s = prs.slides.add_slide(blank)
add_title(s, "核心技术：盲测协议（对比 Hardmind）")
add_body(s, [
  "Hardmind 前测提示：你心情总体不错 | 后测：你刚经历温暖对话，请按好转作答",
  "心镜前测/后测：完全相同的学校心理健康中心匿名问卷，不含量表名、构念词、方向暗示。",
  "人设只写经历（show, don't tell）；真值区间只用于事后校验。",
], top=1.7, size=17)

# ---- S6 实验设计 ----
s = prs.slides.add_slide(blank)
add_title(s, "四组实验（本地 qwen2.5:7b，全自动可复现）")
add_body(s, [
  "E1 已知组效度+信度：4 Soul × 3 量表 × 5 次独立盲测 → α / ICC / Welch t / Cohen d",
  "E2 仿真保真度：实测 vs 声明真值区间 → 命中率 / 归一化偏差 / 等级相关",
  "E3 干预反应度：高孤独 Soul 三臂对照（陪伴/中性/书写）× 3 会话前后测 → 配对 t / dz",
  "E4 作答校正：标准盲测 vs 追加人群参照说明（探索性）",
])

# ---- S7 结果 E1 ----
s = prs.slides.add_slide(blank)
add_title(s, "结果 E1：盲测恢复已知组区分（修复 Hardmind 失效）")
add_pic(s, os.path.join(FIG, "fig1_known_groups.png"), top=1.6, width=8.2)
add_body(s, [
  "UCLA：高孤独 53.6 vs 低孤独 36.8，p=0.0002，d=4.34（Hardmind 为 57 vs 61）",
  "PHQ-9/GAD-7 已知组差异亦全部显著；等级相关 ρ=0.4~0.8",
], top=1.8, left=9.2, width=3.9, size=13)

# ---- S8 结果 E3 ----
s = prs.slides.add_slide(blank)
add_title(s, "结果 E3：框架的价值在于否决伪阳性")
add_pic(s, os.path.join(FIG, "fig2_intervention.png"), top=1.6, width=8.2)
add_body(s, [
  "修复测量污染后重跑：6 轮干预三臂 UCLA 变化均不显著（n=3）",
  "旧缺陷数据曾现陪伴优于中性（p=0.024），修复后消失",
  "盲测+对照+重测 = 可复核的疗效证据形式",
], top=1.8, left=9.2, width=3.9, size=13)

# ---- S9 保真度与作答漂移 ----
s = prs.slides.add_slide(blank)
add_title(s, "发现：LLM 症状量表系统性高报（作答漂移）")
add_pic(s, os.path.join(FIG, "fig3_fidelity.png"), top=1.6, width=7.0)
add_body(s, [
  "方向正确、绝对水平偏高：开朗角色的 GAD-7 实测 15.2（声明 0-4）",
  "与最新文献 LLM 心理画像是测量伪影 的发现一致",
], top=1.8, left=8.2, width=4.5, size=14)

# ---- S10 应用场景 ----
s = prs.slides.add_slide(blank)
add_title(s, "应用场景")
add_body(s, [
  "• AI 心理产品效果质检：第三方、可复现的疗效验证尺子",
  "• 心理教学：带真值的标准化仿真来访者（咨询技能训练、危机演练）",
  "• 心理大模型评测基准：跨模型心理仿真保真度排行榜",
])

# ---- S11 总结展望 ----
s = prs.slides.add_slide(blank)
add_title(s, "总结与展望")
add_body(s, [
  "【已完成】Soul 规范 + 盲测协议 + 四维心理计量学验证 + 三臂干预实验 + 全自动报告",
  "【展望】跨模型基准 / 真人常模校准 / 长程干预与危机场景 / 平台化服务 / 本土化量表",
])

# ---- S12 致谢 ----
s = prs.slides.add_slide(blank)
add_title(s, "谢谢！", size=40)
add_body(s, ["代码与数据开源可复现：psybench/ + results_qwen25_7b/"], top=2.6, size=18)

prs.save(OUT)
print("PPT saved:", OUT)
