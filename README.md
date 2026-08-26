# AI与心理学：虚拟陪伴与人类孤独问题

## 项目简介

本项目旨在研究AI虚拟陪伴对人类孤独感的影响。通过构建两个AI角色进行对话，模拟真实的人际互动，探讨虚拟陪伴是否能够有效降低孤独感。

## 核心功能

### 1. 双AI对话模式
- 两个AI角色（Chen Yu和Lin Xiaohan）进行100轮对话
- 用户可以作为其中一方发起对话
- 支持实时查看对话进度

### 2. 独立思考模式
- 单个AI进行100轮深度思考
- 模拟孤独个体的内心独白
- 每轮思考后生成洞察（Insight）

### 3. UCLA孤独量表评估
- 对话前后自动进行孤独感评估
- 基于UCLA Loneliness Scale Version 3
- 支持查看评估结果对比

### 4. 实时Soul更新
- 每轮对话后AI的性格特征实时演变
- 保持原始配置文件不变
- 基于对话内容动态更新AI角色

### 5. 重复检测与话题切换
- 实时检测对话重复度（阈值50%）
- 自动更换话题避免对话陷入循环
- 支持多种干预机制

## 技术架构

```
┌─────────────────────────────────────────────────────┐
│                    前端界面                          │
│   index.html | chat.html | think.html              │
└───────────────────┬─────────────────────────────────┘
                    │ HTTP
                    ▼
┌─────────────────────────────────────────────────────┐
│                    Flask后端                        │
│   app.py - 核心逻辑、API路由                        │
└───────────────────┬─────────────────────────────────┘
                    │ API
                    ▼
┌─────────────────────────────────────────────────────┐
│                   Ollama平台                        │
│   qwen2.5:7b - 大语言模型                          │
└─────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- Python 3.8+
- Ollama（已安装qwen2.5:7b模型）

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/lizhong0525/Hardmind.git
cd Hardmind
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **启动Ollama服务**
```bash
ollama serve
```

4. **启动应用**
```bash
python app.py
```

5. **访问应用**
```
http://localhost:5050
```

## 文件结构

```
Hardmind/
├── app.py                    # 主应用程序
├── soul_a.md                 # 角色A配置（Chen Yu - 孤独学生）
├── soul_b.md                 # 角色B配置（Lin Xiaohan - 阳光学生）
├── soul.md                   # 备用角色配置
├── templates/
│   ├── index.html            # 首页（模式选择）
│   ├── chat.html             # 双AI对话界面
│   └── think.html            # 独立思考界面
├── static/
│   └── style.css             # 样式文件
├── conversations/            # 聊天记录导出目录
└── README.md                 # 项目说明文档
```

## 使用说明

### 模式选择

1. **AI自我思考**：单个AI进行100轮深度思考
2. **双AI对话**：两个AI进行100轮对话，用户可选择角色发起对话

### 操作步骤

1. 打开首页选择模式
2. 输入初始对话内容（仅双AI模式）
3. 点击开始按钮
4. 等待对话完成（约5-10分钟）
5. 查看UCLA评估结果对比

## 核心算法

### 重复检测算法
```python
def compute_similarity(text1, text2):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / max(len(union), 1)
```

### 孤独感检测
- 检测直接孤独关键词（孤单、孤独、lonely等）
- 检测间接孤独线索（eating alone、no one、wish someone等）
- 支持多级阈值检测

## 实验设计

本项目用于验证以下假设：
- **假设1**：双AI对话能够有效降低AI角色的孤独感
- **假设2**：双AI对话模式比独立思考模式更能降低孤独感
- **假设3**：实时更新的Soul能够增强对话的真实性

## 数据采集

每次对话结束后自动导出：
- 完整对话记录（Markdown格式）
- UCLA前后测结果
- 对话统计数据

## 许可证

MIT License

## 作者

Li Zhong - lizhong0525@github.com