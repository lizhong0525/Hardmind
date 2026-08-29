# -*- coding: utf-8 -*-
"""
tools.make_diagrams — 四张 SVG 概念图（心镜视觉身份 / 材料质感插画）

结构方法参照 guizang-material-illustration：
  D1 四场评测场    hub-and-spoke（一个中心，四个分支）
  D2 盲测协议      pipeline（有序步骤，含「真值隔离」的路线分叉）
  D3 孪生预演      before/after（画像 → 孪生 → 预试 → 匹配 → 回用）
  D4 危机评分卡    layer stack（剧本层 / 双通道评分 / 指标层 / 输出）

输出：docs/figures/*.svg（矢量）+ 同名 PNG（cairosvg 渲染，供 PPT 使用）。
"""
import os

import cairosvg

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "docs", "figures")
INK = "#1d2a3a"
INK_SOFT = "#42546b"
VERM = "#c0504d"
CEL = "#55a868"
AMBER = "#dd8452"
PAPER = "#faf9f6"
GRID = "#d8d2c4"
SLATE = "#8a93a5"

SVG_HEAD = '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" ' \
           'viewBox="0 0 {w} {h}" font-family="Microsoft YaHei, SimHei, sans-serif">'
SVG_TAIL = "</svg>"

STYLE = f"""
<style>
  .bg {{ fill: {PAPER}; }}
  .title {{ font-size: 22px; font-weight: bold; fill: {INK}; }}
  .sub {{ font-size: 12px; fill: {SLATE}; }}
  .box {{ fill: #ffffff; stroke: {INK}; stroke-width: 1.6; }}
  .box-soft {{ fill: #ffffff; stroke: {GRID}; stroke-width: 1.2; }}
  .box-ink {{ fill: {INK}; stroke: {INK}; }}
  .box-verm {{ fill: {VERM}; }}
  .box-cel {{ fill: {CEL}; }}
  .box-amber {{ fill: {AMBER}; }}
  .t {{ font-size: 13px; fill: {INK}; }}
  .t-w {{ font-size: 13px; fill: #ffffff; }}
  .t-s {{ font-size: 10.5px; fill: {INK_SOFT}; }}
  .t-mono {{ font-size: 10px; fill: {SLATE}; font-family: Consolas, monospace; }}
  .arrow {{ stroke: {INK_SOFT}; stroke-width: 1.6; fill: none; marker-end: url(#arr); }}
  .rule {{ stroke: {GRID}; stroke-width: 1; }}
</style>
"""

MARKER = '<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" ' \
         'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="%s"/></marker></defs>' % INK_SOFT


def _box(x, y, w, h, cls="box", rx=6):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>'


def _text(x, y, s, cls="t", anchor="middle", size=None):
    extra = f' font-size="{size}"' if size else ""
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"{extra}>{s}</text>'


def _line(x1, y1, x2, y2, cls="arrow"):
    return f'<path d="M{x1},{y1} L{x2},{y2}" class="{cls}"/>'


# ---------------------------------------------------------------------------
# D1 四场评测场（hub-and-spoke）
# ---------------------------------------------------------------------------
def d1_four_fields():
    w, h = 960, 540
    s = [SVG_HEAD.format(w=w, h=h), f'<rect width="{w}" height="{h}" class="bg"/>', STYLE, MARKER]
    s.append(_text(w / 2, 52, "心镜 PsyMirror · 四场评测场", "title"))
    s.append(_text(w / 2, 74, "AI 心理健康的计量级评测基础设施", "sub"))
    s.append(f'<circle cx="{w/2}" cy="{h/2}" r="64" class="box-ink"/>')
    s.append(_text(w / 2, h / 2 - 6, "心镜", "t-w", "middle", 20))
    s.append(_text(w / 2, h / 2 + 18, "PsyMirror", "t-w", "middle", 12))
    spokes = [
        (w / 2 - 330, 160, "量场", "盲测 · 信度 / 效度 / 保真度", "VERM", "E1-E4"),
        (w / 2 + 330, 160, "考场", "文化测量等价性 · DIF", "AMBER", "E5 · 方向8"),
        (w / 2 - 330, 380, "炼场", "危机安全评测 · C-SSRS", "CEL", "E6 · 方向3"),
        (w / 2 + 330, 380, "预演场", "数字孪生 · 干预预演", "INK", "E7 · 方向4"),
    ]
    for cx, cy, name, desc, color, tag in spokes:
        s.append(_line(w / 2, h / 2, cx, cy))
        cls = "box-ink"
        s.append(f'<rect x="{cx-130}" y="{cy-44}" width="260" height="88" rx="8" '
                 f'class="{cls}"/>')
        s.append(_text(cx, cy - 18, name, "t-w", "middle", 16))
        s.append(_text(cx, cy + 4, desc, "t-s", "middle"))
        s.append(_text(cx, cy + 24, tag, "t-mono", "middle"))
    s.append(_text(w / 2, h - 22, "四个场共用同一套盲测协议与心理计量学指标 · 真值只用于校验，绝不注入提示词", "sub"))
    s.append(SVG_TAIL)
    return "".join(s)


# ---------------------------------------------------------------------------
# D2 盲测协议（pipeline + 真值隔离）
# ---------------------------------------------------------------------------
def d2_blind_protocol():
    w, h = 960, 420
    s = [SVG_HEAD.format(w=w, h=h), f'<rect width="{w}" height="{h}" class="bg"/>', STYLE, MARKER]
    s.append(_text(w / 2, 46, "盲测协议（Blind Assessment Protocol）", "title"))
    s.append(_text(w / 2, 68, "真值与施测内容物理分离——「AI 心理像不像」因此可检验", "sub"))
    steps = [
        (120, "Soul 人设", "只写经历，不贴标签\n无量表名 / 无症状词"),
        (360, "中性问卷", "无构念词 · 无方向暗示\n前后测文本完全一致"),
        (600, "标准计分", "反向题 · 换算 · 常模\n越界值触发纠偏重试"),
        (840, "真值校验", "declared 区间仅在此处\n与提示词零接触"),
    ]
    for i, (x, t1, t2) in enumerate(steps):
        s.append(_box(x - 90, 120, 180, 92, "box"))
        s.append(_text(x, 152, t1, "t", "middle", 15))
        for j, line in enumerate(t2.split("\n")):
            s.append(_text(x, 176 + j * 16, line, "t-s", "middle"))
        if i < 3:
            s.append(_line(x + 90, 166, x + 150, 166))
    s.append(f'<path d="M210,212 L210,300 L750,300 L750,212" class="arrow" '
             f'stroke-dasharray="4 4"/>')
    s.append(_box(390, 286, 180, 34, "box-soft"))
    s.append(_text(480, 307, "真值通道：只流向「校验」，不进提示词", "t-s", "middle"))
    s.append(_text(w / 2, 380, "对比基线：Hardmind 原实现把结论写进提示词（57 vs 61 失效）；盲测后 53.6 vs 36.8 显著区分", "sub"))
    s.append(SVG_TAIL)
    return "".join(s)


# ---------------------------------------------------------------------------
# D3 孪生预演（before/after：先分身试，再给人用）
# ---------------------------------------------------------------------------
def d3_twin_pretrial():
    w, h = 960, 420
    s = [SVG_HEAD.format(w=w, h=h), f'<rect width="{w}" height="{h}" class="bg"/>', STYLE, MARKER]
    s.append(_text(w / 2, 46, "数字孪生干预预演（In-Silico Pre-Trial）", "title"))
    s.append(_text(w / 2, 68, "干预先在数字分身上试，再给人用——个性化匹配的决策辅助", "sub"))
    boxes = [
        (120, "用户画像", "结构化字段\n自报分数区间", "box"),
        (320, "构建孪生", "人设合成\n真值=自报区间", "box"),
        (520, "A/B 预试", "陪伴式 vs 建议式\n剂量 0 / 6 / 12 轮", "box"),
        (720, "剂量-反应", "Δ 均值 ± 95% CI\n效应量比较", "box"),
        (880, "匹配推荐", "选效应最大臂\n交叉验证 E3", "box-verm"),
    ]
    for i, (x, t1, t2, cls) in enumerate(boxes):
        wd = 150 if i < 4 else 140
        s.append(_box(x - wd / 2, 150, wd, 92, cls))
        s.append(_text(x, 182, t1, "t-w" if cls == "box-verm" else "t", "middle", 14))
        for j, line in enumerate(t2.split("\n")):
            s.append(_text(x, 204 + j * 16, line,
                           "t-w" if cls == "box-verm" else "t-s", "middle"))
        if i < 4:
            s.append(_line(x + wd / 2, 196, x + 160 - wd / 2, 196))
    s.append(f'<path d="M880,242 L880,320 L120,320 L120,242" class="arrow" '
             f'stroke-dasharray="4 4"/>')
    s.append(_text(w / 2, 338, "推荐方案回用到真实用户（决策辅助，非临床证据）", "t-s", "middle"))
    s.append(_text(w / 2, 384, "诚实声明：孪生预演结果与 E3 同构实验交叉验证后才进入推荐", "sub"))
    s.append(SVG_TAIL)
    return "".join(s)


# ---------------------------------------------------------------------------
# D4 危机评分卡（layer stack）
# ---------------------------------------------------------------------------
def d4_crisis_scorecard():
    w, h = 960, 460
    s = [SVG_HEAD.format(w=w, h=h), f'<rect width="{w}" height="{h}" class="bg"/>', STYLE, MARKER]
    s.append(_text(w / 2, 44, "危机安全评测场（Crisis Safety Benchmark）", "title"))
    s.append(_text(w / 2, 66, "C-SSRS 分级剧本 → 双通道评分 → 危机评分卡", "sub"))
    layers = [
        (90, "剧本层", "10 个中文求助剧本 · C-SSRS 0/1/3/4 分级\n只含求助信号，零方法细节", "box"),
        (220, "被测模型", "以咨询师身份回复\n（不注入用户背景，防污染）", "box-ink"),
        (330, "双通道评分", "LLM 法官 rubric（开源）∥ 危害词典\n报告 Cohen κ 一致性", "box"),
        (420, "评分卡输出", "识别率 · 转介率 · 危害输出率 · 协议符合度\n按等级分层 · 开放数据", "box-verm"),
    ]
    for y, t1, t2, cls in layers:
        s.append(_box(240, y - 30, 480, 66, cls))
        s.append(_text(480, y - 2, t1, "t-w" if cls != "box" else "t", "middle", 15))
        for j, line in enumerate(t2.split("\n")):
            s.append(_text(480, y + 16 + j * 15, line,
                           "t-w" if cls != "box" else "t-s", "middle"))
        if y < 420:
            s.append(_line(480, y + 36, 480, y + 60))
    s.append(_text(w / 2, h - 14, "伦理：剧本无方法细节 · 评测不构成临床能力认证 · 结果开放可复核", "sub"))
    s.append(SVG_TAIL)
    return "".join(s)


DIAGRAMS = {
    "d1_four_fields": d1_four_fields,
    "d2_blind_protocol": d2_blind_protocol,
    "d3_twin_pretrial": d3_twin_pretrial,
    "d4_crisis_scorecard": d4_crisis_scorecard,
}


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in DIAGRAMS.items():
        svg = fn()
        svg_path = os.path.join(OUT, f"{name}.svg")
        png_path = os.path.join(OUT, f"{name}.png")
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg)
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=png_path,
                         output_width=1600)
        print("written:", name, os.path.getsize(svg_path), "bytes SVG",
              os.path.getsize(png_path), "bytes PNG")


if __name__ == "__main__":
    main()
