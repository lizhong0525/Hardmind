# -*- coding: utf-8 -*-
"""
psybench.style — 「心镜」视觉身份与设计 token（设计美学方针的落地）

设计母题：心理计量仪器 × 临床档案纸。
  - 墨水蓝 Ink        #1d2a3a  标题/轴线（档案印墨）
  - 刻度红 Vermilion  #c0504d  警示/危险/重点（仪表红针）
  - 青瓷绿 Celadon    #55a868  改善/通过（量表向好方向）
  - 砂金 Amber        #dd8452  对照/次要序列
  - 纸张底 Paper      #faf9f6  图底（档案纸）
  - 数据等宽字体 Consolas（数字天生为仪器读数）
所有图表、示意图、榜单页共用此 token，保证「一眼可辨的视觉身份」。
"""

PALETTE = {
    "ink": "#1d2a3a",
    "ink_soft": "#42546b",
    "vermillion": "#c0504d",
    "celadon": "#55a868",
    "amber": "#dd8452",
    "paper": "#faf9f6",
    "paper_dark": "#efebe2",
    "grid": "#d8d2c4",
    "slate": "#8a93a5",
}

# 序列色（多系列图表按此顺序取色）
SERIES = ["#1d2a3a", "#c0504d", "#55a868", "#dd8452", "#42546b", "#8a93a5"]

# Matplotlib 全局样式（图表统一设计语言）
MPL_STYLE = {
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": PALETTE["ink"],
    "axes.linewidth": 0.9,
    "axes.labelcolor": PALETTE["ink"],
    "text.color": PALETTE["ink"],
    "xtick.color": PALETTE["ink_soft"],
    "ytick.color": PALETTE["ink_soft"],
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "legend.frameon": False,
    "axes.titlelocation": "left",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}

# HTML 榜单页样式（frontend-design：hero 即论点、字阶明确、仪器质感）
SITE_CSS = """
:root {
  --ink: #1d2a3a; --ink-soft: #42546b; --verm: #c0504d;
  --cel: #55a868; --amber: #dd8452; --paper: #faf9f6;
  --grid: #d8d2c4; --slate: #8a93a5;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink);
       font-family: "Microsoft YaHei", "PingFang SC", sans-serif; }
header { border-bottom: 1px solid var(--grid); padding: 56px 32px 40px; }
.eyebrow { font-family: Consolas, monospace; letter-spacing: 0.18em;
           color: var(--verm); font-size: 0.72em; text-transform: uppercase; }
h1 { font-size: 2.6em; margin: 10px 0 6px; letter-spacing: 0.02em; }
.lede { color: var(--ink-soft); max-width: 46em; line-height: 1.8; }
.hero { display: flex; gap: 48px; align-items: flex-end; flex-wrap: wrap;
        margin-top: 28px; }
.hero .rule { font-family: Consolas, monospace; color: var(--slate);
              font-size: 0.78em; letter-spacing: 0.1em; }
.hero b { font-family: Consolas, monospace; font-size: 1.5em; color: var(--ink); }
main { max-width: 1080px; margin: 0 auto; padding: 40px 32px 96px; }
table { width: 100%; border-collapse: collapse; margin: 8px 0 32px;
        font-family: Consolas, "Microsoft YaHei", monospace; }
th { background: var(--ink); color: var(--paper); text-align: center;
     padding: 12px 10px; font-weight: 600; letter-spacing: 0.06em; }
td { border-bottom: 1px solid var(--grid); padding: 13px 10px;
     text-align: center; }
tr:hover td { background: #f2efe7; }
td.dim-0 { color: var(--slate); }
.note { color: var(--slate); font-size: 0.85em; line-height: 1.7;
        border-left: 3px solid var(--verm); padding-left: 14px; }
footer { border-top: 1px solid var(--grid); color: var(--slate);
         padding: 24px 32px 40px; font-size: 0.85em; }
@media (max-width: 640px) { header { padding: 32px 18px 24px; }
  main { padding: 24px 18px 64px; } h1 { font-size: 1.8em; } }
"""


def apply_mpl_style():
    """把统一图表样式应用到当前 matplotlib 会话。"""
    import matplotlib.pyplot as plt
    plt.rcParams.update(MPL_STYLE)
