# -*- coding: utf-8 -*-
"""把关键 Markdown 文档转换为带排版样式的 HTML（浏览器打印即可出 PDF）。"""
import os
import sys

import markdown

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSS = """
body { font-family: "Microsoft YaHei", "SimSun", sans-serif; max-width: 900px;
       margin: 24px auto; padding: 0 20px; color: #222; line-height: 1.75; }
h1 { font-size: 1.5em; border-bottom: 3px solid #1f3a5f; padding-bottom: 8px; color: #1f3a5f; }
h2 { font-size: 1.25em; border-bottom: 1px solid #ccc; padding-bottom: 4px; color: #1f3a5f; }
h3 { font-size: 1.08em; color: #2c4a70; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.92em; }
th, td { border: 1px solid #999; padding: 6px 8px; text-align: left; }
th { background: #eef2f7; }
blockquote { border-left: 4px solid #1f3a5f; background: #f4f7fb;
             margin: 10px 0; padding: 8px 14px; color: #333; }
code { background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid #ddd; padding: 10px; overflow-x: auto; }
img { max-width: 100%; }
@media print { body { max-width: 100%; } }
"""


def convert(src_rel, dst_rel, title):
    src = os.path.join(BASE, src_rel)
    dst = os.path.join(BASE, dst_rel)
    with open(src, encoding="utf-8") as f:
        md = f.read()
    html_body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head>
<body>{html_body}</body></html>"""
    with open(dst, "w", encoding="utf-8") as f:
        f.write(page)
    print("HTML written:", dst)


if __name__ == "__main__":
    convert("report/技术报告_心镜.md", "report/技术报告_心镜.html", "心镜 PsyBench 技术报告")
    convert("docs/01_初步调研.md", "docs/01_初步调研.html", "心镜 初步调研报告")
    convert("psybench/results_qwen25_7b/结果报告.md",
            "psybench/results_qwen25_7b/结果报告.html", "心镜 实验数据自动报告")
    print("DONE")
