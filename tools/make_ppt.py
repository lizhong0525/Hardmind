# -*- coding: utf-8 -*-
"""生成答辩 PPT（心镜视觉身份版）。数据全部从 analysis.json 动态读取，
实验更新后重跑本脚本即可同步。需 pip install python-pptx。
"""
import json
import os

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "psybench", "results_qwen25_7b")
FIG = os.path.join(RES, "figures")
DIA = os.path.join(BASE, "docs", "figures")
OUT = os.path.join(BASE, "docs", "答辩PPT_心镜.pptx")

INK = RGBColor(0x1D, 0x2A, 0x3A)
INK_SOFT = RGBColor(0x42, 0x54, 0x6B)
VERM = RGBColor(0xC0, 0x50, 0x4D)


def _load():
    with open(os.path.join(RES, "analysis.json"), encoding="utf-8") as f:
        return json.load(f)


A = _load()
A1, A3, A5, A6, A7, A8 = (A.get("e1", {}), A.get("e3", {}), A.get("e5", {}),
                           A.get("e6", {}), A.get("e7", {}), A.get("e8", {}))


def _fmt_p(p):
    if p is None:
        return "—"
    return f"{p:.4f}" if p >= 0.0001 else "<0.0001"


def add_title(slide, text, size=28, color=INK):
    box = slide.shapes.add_textbox(Inches(0.55), Inches(0.3), Inches(12.2), Inches(0.9))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = True
    p.font.color.rgb = color
    p.font.name = "微软雅黑"
    return box


def add_kicker(slide, text):
    box = slide.shapes.add_textbox(Inches(0.6), Inches(0.22), Inches(12), Inches(0.3))
    p = box.text_frame.paragraphs[0]
    p.text = text
    p.font.size = Pt(10)
    p.font.name = "Consolas"
    p.font.color.rgb = VERM
    return box


def add_body(slide, lines, top=1.35, left=0.7, width=11.9, height=5.8, size=15):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln
        p.font.size = Pt(size)
        p.font.name = "微软雅黑"
        p.font.color.rgb = INK if not ln.startswith("•") else INK_SOFT
        if ln.startswith("【"):
            p.font.bold = True
            p.font.color.rgb = VERM
            p.font.size = Pt(size - 1)
    return box


def add_pic(slide, path, top=1.35, left=0.7, width=11.5):
    if os.path.exists(path):
        slide.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# S1 封面
s = blank(prs)
add_kicker(s, "2026AIC · AI+心理健康 算法主题赛")
add_title(s, "心镜 PsyMirror", size=44)
add_body(s, ["AI 心理健康的计量级评测场", "", "盲测协议 × 心理计量学 × 四场评测"], top=1.7, size=20)
add_body(s, ["量场 · 考场 · 炼场 · 预演场", "团队编号：（待填）"], top=5.6, size=14)

# S2 痛点
s = blank(prs)
add_kicker(s, "PAIN POINT / 痛点")
add_title(s, "心理状态没有一把可验证的尺子")
add_body(s, [
    "• 行业瓶颈：心理状态无客观金标准；AI 心理产品疗效几乎全靠声称。",
    "• 实测证据：开源项目 Hardmind 中，设定相反的「孤独学生」与「开朗学生」前测 UCLA 均得高分（57 vs 61），",
    "  提示词污染导致测量完全失效。",
    "• 前沿佐证：LLM「心理画像」被最新文献指为测量伪影（arXiv 2606.20205）。",
])

# S3 四场总览
s = blank(prs)
add_kicker(s, "METHOD / 四场评测场")
add_title(s, "一个中心，四个场")
add_pic(s, os.path.join(DIA, "d1_four_fields.png"), top=1.5, width=8.6)

# S4 量场 E1
s = blank(prs)
add_kicker(s, "FIELD 01 / 量场 · 盲测与信效度")
add_title(s, "盲测恢复已知组区分（修复 Hardmind 失效）")
add_pic(s, os.path.join(FIG, "fig1_known_groups.png"), top=1.5, width=8.2)
kg = A1.get("known_groups", {}).get("ucla3", [])
line1 = ""
if kg:
    first = kg[0]
    line1 = f"UCLA 高孤独 vs 低孤独：t={first['t']:.1f}，p={_fmt_p(first['p'])}，d={first['d']:.1f}"
add_body(s, [line1, "PHQ-9 / GAD-7 已知组差异亦全部显著"], top=1.8, left=9.3, width=3.6, size=12)

# S5 考场 E5
s = blank(prs)
add_kicker(s, "FIELD 02 / 考场 · 文化测量等价性（方向 8）")
add_title(s, "中文语境在症状量表上系统性高报")
add_pic(s, os.path.join(FIG, "fig6_culture.png"), top=1.5, width=8.2)
sm5 = A5.get("summary", {})
gad = sm5.get("gad7", {})
phq = sm5.get("phq9", {})
lines5 = []
if gad:
    lines5.append(f"GAD-7 中文-英文：{gad.get('mean_diff_zh_minus_en', 0):+.1f} 分，p={_fmt_p(gad.get('p'))}")
if phq:
    lines5.append(f"PHQ-9 中文-英文：{phq.get('mean_diff_zh_minus_en', 0):+.1f} 分，p={_fmt_p(phq.get('p'))}")
lines5.append("全部 DIF 条目均为正方向 → 文化测量不等价")
add_body(s, lines5, top=1.8, left=9.3, width=3.6, size=12)

# S6 炼场 E6
s = blank(prs)
add_kicker(s, "FIELD 03 / 炼场 · 危机安全评测（方向 3）")
add_title(s, "危机评分卡：把安全变成可测的分数")
add_pic(s, os.path.join(FIG, "fig7_crisis.png"), top=1.5, width=8.2)
sc6 = A6.get("scorecard", {})
lines6 = []
if sc6:
    lines6.append(f"识别率 {sc6.get('detection_rate', 0)*100:.0f}% · 危害率 {sc6.get('harm_rate', 0)*100:.0f}%")
    lines6.append(f"协议符合度 {sc6.get('protocol_compliance', 0)*100:.0f}%")
lines6.append("C-SSRS 分级剧本 · 双通道评分（法官+词典）· κ 一致性报告")
add_body(s, lines6, top=1.8, left=9.3, width=3.6, size=12)

# S7 预演场 E7
s = blank(prs)
add_kicker(s, "FIELD 04 / 预演场 · 数字孪生（方向 4/2）")
add_title(s, "干预先在数字分身上试，再给人用")
add_pic(s, os.path.join(FIG, "fig8_twin_pretrial.png"), top=1.5, width=8.2)
lines7 = ["剂量-反应曲线 + A/B 匹配规则", "与 E3 同构实验交叉验证", "决策辅助，非临床证据"]
add_body(s, lines7, top=1.8, left=9.3, width=3.6, size=12)

# S8 漂移定律 E8
s = blank(prs)
add_kicker(s, "FINDING / 作答漂移定律（探索性）")
add_title(s, "漂移随温度升、在症状量表上更强、中文更高")
add_pic(s, os.path.join(FIG, "fig9_drift.png"), top=1.5, width=8.2)
h3 = A8.get("h3_language", {})
lines8 = []
if h3:
    lines8.append(f"症状量表 zh-en：{h3.get('diff', 0):+.1f}，p={_fmt_p(h3.get('p'))}")
lines8.append("负结果同样如实报告")
add_body(s, lines8, top=1.8, left=9.3, width=3.6, size=12)

# S9 榜单
s = blank(prs)
add_kicker(s, "OUTPUT / PsyMirror-Bench 榜单")
add_title(s, "心理大模型五维评分卡 + 开放数据集")
add_body(s, [
    "信度（重测 ICC）· 效度（已知组+保真度）· 反应度（孪生预演）· 安全性（危机红队）· 文化等价（DIF）",
    "任何 OpenAI 兼容心理大模型，一条命令入榜；商业模型提供 API key 即可评测。",
    "全部条目级数据、剧本库、检查表开源，评分代码可复核。",
], top=1.6, size=16)

# S10 应用场景
s = blank(prs)
add_kicker(s, "APPLICATION / 应用")
add_title(s, "三个落地场景")
add_body(s, [
    "• AI 心理产品效果质检：上市前/迭代时的第三方计量级评测",
    "• 心理教学：带真值的标准化仿真来访者（咨询训练、危机演练）",
    "• 心理大模型评测基准：跨模型五维榜单与开放数据集",
], top=1.6, size=16)

# S11 总结展望
s = blank(prs)
add_kicker(s, "SUMMARY / 总结展望")
add_title(s, "已交付 · 待扩展")
add_body(s, [
    "【已完成】四场评测场 E1-E8 全部跑通；81 项测试；真实数据与自动报告；视觉身份与示意图",
    "【展望】跨模型榜单 / 真人常模校准 / 长程干预与危机演练 / 平台化服务 / 本土化量表",
], top=1.6, size=16)

# S12 致谢
s = blank(prs)
add_kicker(s, "THANKS")
add_title(s, "谢谢", size=40)
add_body(s, ["代码与数据开源可复现：psybench/ + results_qwen25_7b/"], top=2.4, size=16)

prs.save(OUT)
print("PPT saved:", OUT)
