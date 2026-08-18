"""CLI for M14: native Silverwing inference with KV-cache generation.

Usage:
    python scripts/generate.py --checkpoint experiments/checkpoints/best.pt \
        --prompt "Hello, how are you?" \
        --max-new-tokens 64 --temperature 0.8 --top-p 0.9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate completions with Silverwing Decoder V2 (KV-cache inference)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Inference config YAML",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Override checkpoint path"
    )
    parser.add_argument(
        "--model-config", type=str, default=None, help="Override model config YAML"
    )
    parser.add_argument(
        "--tokenizer-dir", type=str, default=None, help="Override tokenizer directory"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (cpu/cuda)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt string. If omitted, reads from stdin (one prompt per line).",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, help="Override max new tokens"
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Override temperature"
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k")
    parser.add_argument(
        "--top-p", type=float, default=None, help="Override top-p (nucleus)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Override repetition penalty",
    )
    parser.add_argument(
        "--prompt-template", type=str, default=None, help="Override prompt template"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (prompt and completion fields)",
    )
    args = parser.parse_args()

    from foundation.inference import Generator, InferenceConfig

    cfg = InferenceConfig.from_yaml(args.config)

    # Apply overrides from CLI
    overrides = {
        "checkpoint_path": args.checkpoint,
        "model_config_path": args.model_config,
        "tokenizer_dir": args.tokenizer_dir,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "prompt_template": args.prompt_template,
    }
    cfg = InferenceConfig(
        **{**cfg.to_dict(), **{k: v for k, v in overrides.items() if v is not None}}
    )

    gen = Generator.from_config(cfg)

    if args.prompt is not None:
        prompts = [args.prompt]
    else:
        print(
            "Reading prompts from stdin (one per line, Ctrl-D to finish):",
            file=sys.stderr,
        )
        prompts = [line.strip() for line in sys.stdin if line.strip()]
        if not prompts:
            print("No prompts provided.", file=sys.stderr)
            sys.exit(1)

    results = gen.generate(
        prompts,
        max_new_tokens=args.max_new_tokens or cfg.max_new_tokens,
        temperature=args.temperature
        if args.temperature is not None
        else cfg.temperature,
        top_k=args.top_k if args.top_k is not None else cfg.top_k,
        top_p=args.top_p if args.top_p is not None else cfg.top_p,
    )

    if args.json:
        for prompt, result in zip(prompts, results):
            print(
                json.dumps(
                    {
                        "prompt": prompt,
                        "completion": result.text,
                        "token_ids": result.token_ids,
                    }
                )
            )
    else:
        for prompt, result in zip(prompts, results):
            print(f"▶ {prompt}")
            print(f"  → {result.text}")
            print()


if __name__ == "__main__":
    main()
