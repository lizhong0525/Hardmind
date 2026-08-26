# -*- coding: utf-8 -*-
"""
psybench.run — 命令行入口
===========================

用法示例：
  1) 离线自测（无需模型）：
     python -m psybench.run --provider mock --out results_mock
  2) 真实实验（本地 Ollama，qwen2.5:7b）：
     python -m psybench.run --provider ollama --model qwen2.5:7b --out results
  3) 只生成报告（实验已完成）：
     python -m psybench.run --report-only --out results
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="心镜 · AI 心理量化测评实验流水线")
    parser.add_argument("--provider", default="ollama",
                        choices=["ollama", "openai", "mock"])
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--base-url", default=None,
                        help="Ollama/OpenAI 兼容接口地址")
    parser.add_argument("--api-key", default="", help="OpenAI 兼容接口密钥")
    parser.add_argument("--out", default="psybench/results",
                        help="结果输出目录")
    parser.add_argument("--e1-sessions", type=int, default=5,
                        help="E1 每个 Soul×量表 的独立施测次数")
    parser.add_argument("--e3-sessions", type=int, default=3,
                        help="E3 每个条件的会话数")
    parser.add_argument("--rounds", type=int, default=6,
                        help="E3 干预对话轮数")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--only-e1", action="store_true")
    parser.add_argument("--only-e3", action="store_true")
    parser.add_argument("--only-e4", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    from .providers import OllamaProvider, OpenAICompatProvider, MockProvider
    if args.provider == "ollama":
        base = args.base_url or "http://localhost:11434"
        provider = OllamaProvider(model=args.model, base_url=base)
    elif args.provider == "openai":
        if not args.api_key or not args.base_url:
            print("--provider openai 需要 --api-key 与 --base-url")
            return
        provider = OpenAICompatProvider(args.api_key, args.base_url, args.model)
    else:
        provider = MockProvider()

    print(f"推理后端: {provider.name}")
    print(f"输出目录: {args.out}")

    if args.report_only:
        from .report import generate_report
        generate_report(args.out)
        return

    from .experiment import run_e1, run_e3, run_e4, run_all
    from .souls import ALL_SOULS
    if args.only_e1:
        run_e1(provider, list(ALL_SOULS.keys()), ["ucla3", "phq9", "gad7"],
               args.e1_sessions, args.out, temperature=args.temperature)
    elif args.only_e4:
        run_e4(provider, list(ALL_SOULS.keys()), ["ucla3", "phq9", "gad7"],
               args.e1_sessions, args.out, temperature=args.temperature)
    elif args.only_e3:
        run_e3(provider, "chen_yu", ["companion", "neutral", "thinking"],
               args.e3_sessions, args.rounds, args.out,
               temperature=args.temperature)
    else:
        run_all(provider, args.out,
                n_sessions_e1=args.e1_sessions,
                n_sessions_e3=args.e3_sessions,
                rounds=args.rounds,
                temperature=args.temperature)
        from .report import generate_report
        generate_report(args.out)


if __name__ == "__main__":
    main()
