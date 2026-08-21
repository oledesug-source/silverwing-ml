"""Install a trained checkpoint as the production model.

Copies a checkpoint + tokenizer into ``models/<name>/``, points
``configs/inference.yaml`` at them, and runs a generation smoke test so a bad
install fails here instead of at serving time.

Typical flow after Colab training finishes:
    1. Download from Drive: checkpoints/sft/final.pt and tokenizer-v2/
    2. python scripts/import_checkpoint.py --checkpoint ~/Downloads/final.pt \
           --tokenizer-dir ~/Downloads/tokenizer-v2 --name production
    3. python scripts/serve_platform.py   # platform now serves the new model

Examples:
    python scripts/import_checkpoint.py --checkpoint experiments/checkpoints/best.pt
    python scripts/import_checkpoint.py --checkpoint final.pt --tokenizer-dir tok/ --name production --no-smoke-test
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402


def _load_tokenizer_dir(src: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(f"tokenizer dir not found: {src}")
    required = ["vocab.json"]
    missing = [f for f in required if not (src / f).exists()]
    if missing:
        raise FileNotFoundError(f"tokenizer dir {src} is missing: {missing}")
    if not ((src / "merges.txt").exists() or (src / "merges.json").exists()):
        raise FileNotFoundError(f"tokenizer dir {src} is missing merges.txt/merges.json")
    return src


def main() -> int:
    parser = argparse.ArgumentParser(description="Install a checkpoint as the production model")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--tokenizer-dir", default="experiments/tokenizer", help="TokenizerV2 directory")
    parser.add_argument("--name", default="production", help="Install name (models/<name>/)")
    parser.add_argument("--config", default="configs/inference.yaml", help="Inference config to update")
    parser.add_argument("--device", default=None, help="Override device in the config (cpu/cuda)")
    parser.add_argument("--no-smoke-test", action="store_true", help="Skip generation smoke test")
    parser.add_argument("--prompt", default="The capital of France is", help="Smoke test prompt")
    args = parser.parse_args()

    ckpt_src = Path(args.checkpoint)
    if not ckpt_src.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_src}")

    tok_src = _load_tokenizer_dir(Path(args.tokenizer_dir))

    install_dir = Path("models") / args.name
    install_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dst = install_dir / "model.pt"
    tok_dst = install_dir / "tokenizer"

    print(f"Installing checkpoint -> {ckpt_dst}")
    shutil.copy2(ckpt_src, ckpt_dst)
    print(f"Installing tokenizer  -> {tok_dst}")
    if tok_dst.exists():
        shutil.rmtree(tok_dst)
    shutil.copytree(tok_src, tok_dst)

    config_path = Path(args.config)
    raw: dict = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    inf = raw.get("inference") or {}
    inf["checkpoint_path"] = str(ckpt_dst).replace("\\", "/")
    inf["tokenizer_dir"] = str(tok_dst).replace("\\", "/")
    if args.device:
        inf["device"] = args.device
    raw["inference"] = inf
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    print(f"Updated {config_path}:")
    print(f"  checkpoint_path: {inf['checkpoint_path']}")
    print(f"  tokenizer_dir:   {inf['tokenizer_dir']}")

    if args.no_smoke_test:
        print("Smoke test skipped (--no-smoke-test)")
        return 0

    print("\nSmoke test: loading model ...")
    from foundation.inference import Generator, InferenceConfig

    cfg = InferenceConfig.from_yaml(config_path)
    t0 = time.monotonic()
    gen = Generator.from_config(cfg)
    load_s = time.monotonic() - t0

    t0 = time.monotonic()
    result = gen.generate(args.prompt, max_new_tokens=32, temperature=0.7, top_p=0.9)
    gen_s = time.monotonic() - t0
    n_tokens = len(result.token_ids)

    print(f"  load: {load_s:.1f}s | generated {n_tokens} tokens in {gen_s:.1f}s "
          f"({n_tokens / max(gen_s, 1e-6):.1f} tok/s) on {cfg.device}")
    print(f"  prompt: {args.prompt!r}")
    print(f"  output: {result.text[:200]!r}")

    size_mb = ckpt_dst.stat().st_size / 1e6
    print(f"\nInstalled '{args.name}' ({size_mb:.0f} MB). Start serving with:")
    print("  python scripts/serve_platform.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
