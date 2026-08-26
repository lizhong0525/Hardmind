# 心镜 PsyBench —— 参赛作品复现指南

> 2026AIC「AI+心理健康」算法主题赛 · 作品名称：心镜——大模型心理仿真量化测评平台

## 一、交付物地图

| 材料 | 路径 | 说明 |
|---|---|---|
| 初步调研报告 | docs/01_初步调研.md | 行业痛点、技术路线综述、Hardmind 可行性评估 |
| 技术报告 | report/技术报告_心镜.md | 按官方《技术报告参考大纲》撰写 |
| 作品简介（≤300字） | docs/competition/作品简介.txt | 按"问题-方案-效果-应用"撰写 |
| 官方赛题附件 | docs/competition/*.pdf / *.txt | 规则、提交要求、报告大纲 |
| 测评框架源码 | psybench/ | 8 模块，约 1500 行，全中文注释 |
| 实验原始数据 | psybench/results_qwen25_7b/ | E1/E3/E4 全部条目级作答与对话记录 |
| 自动实验报告 | psybench/results_qwen25_7b/结果报告.md | 表格 + 统计检验结果 |
| 图表 | psybench/results_qwen25_7b/figures/*.png | 5 张核心图表 |
| 离线回归 | （一键复现） | Mock 后端自测，无需模型，10 秒出报告 |
| 原 Hardmind 项目 | app.py, soul_*.md, conversations/ | 作为评估对象与缺陷证据保留 |

## 二、环境与依赖

- Python 3.11（numpy / scipy / matplotlib / pandas）
- Ollama 0.32+，模型 qwen2.5:7b（4.7GB，Q4_K_M）
- Windows/Linux 均可；GPU 8GB 显存即可（无 GPU 时请用 Mock 模式）

安装：
    ollama pull qwen2.5:7b
    pip install numpy scipy matplotlib

## 三、复现实验（约 40 分钟）

    cd Hardmind-main
    python -m psybench.run --provider ollama --model qwen2.5:7b         --out psybench/results_qwen25_7b --e1-sessions 5 --e3-sessions 3 --rounds 6

流水线自动完成：E1 已知组效度+信度 → E3 三臂干预 → 报告生成。
追加 E4 作答校正对照：

    python -m psybench.run --provider ollama --model qwen2.5:7b         --out psybench/results_qwen25_7b --only-e4 --e1-sessions 4

离线自测（无需模型）：

    python -m psybench.run --provider mock --out psybench/results_mock

生成可打印为 PDF 的 HTML 版报告（浏览器打印即可）：

    python tools/md_to_html.py

只重新生成报告/图表：

    python -m psybench.run --report-only --out psybench/results_qwen25_7b

所有实验支持断点续跑：中断后原命令重跑即可从上次进度继续。

## 四、实验设计速览

| 实验 | 问题 | 设计 | 指标 |
|---|---|---|---|
| E1 | 量表能否测出AI的已知心理差异 | 4 Soul × 3 量表 × 5 次独立盲测 | 均值±SD、α、ICC、Welch t、Cohen d |
| E2 | 模拟像不像 | 实测 vs 声明真值区间 | 命中率、归一化偏差、Spearman ρ |
| E3 | 干预有没有效 | 高孤独Soul × 三臂（陪伴/中性/书写）× 3 会话前后测 | 配对 t、dz、95% CI |
| E4 | 作答漂移能否校正 | 标准盲测 vs 追加人群参照说明 | 配对比较、真值命中变化 |

## 五、核心创新点（一句话版）

1. Soul 规范：人设"只写经历不贴标签"，声明真值与提示词物理分离——"AI 心理像不像"首次可检验；
2. 盲测协议：消除 Hardmind 类产品的提示词污染（实测证明其 57 vs 61 的失效）；
3. 心理计量学四维验证：信度（α/ICC）、效度（已知组）、反应度（干预）、保真度（真值命中）；
4. 全流程本地化、断点续跑、结果可复核，数据不出本机。

## 六、伦理声明

本作品所有"被试"均为虚构 AI 角色，不包含任何真实个人信息；
测量结果仅描述 AI 仿真对象，不构成对任何真人的临床诊断。
