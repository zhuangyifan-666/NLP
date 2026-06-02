#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from nlp_llm.config import deep_update, load_config, parse_overrides
from nlp_llm.data_pretrain import write_token_bin
from nlp_llm.tokenizer import load_tokenizer


def read_texts_file(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def load_texts_from_dataset(config: dict, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for TinyStories. Install manually with: python -m pip install -r requirements.txt"
        ) from exc
    name = config.get("dataset", "roneneldan/TinyStories")
    text_field = config.get("text_field", "text")
    max_samples = int(config.get(f"{split}_max_samples", config.get("max_samples", 100000)))
    try:
        ds = load_dataset(name, split=split)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {name}:{split}. If data is not cached, run the prepare command yourself with network access."
        ) from exc
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        text = str(row.get(text_field, "")).strip()
        if text:
            yield text


def tokenize_stream(tokenizer, texts, add_eos: bool = True) -> list[int]:
    tokens: list[int] = []
    for text in texts:
        tokens.extend(tokenizer.encode(text, add_bos=True, add_eos=add_eos))
    return tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tokenizer.yaml")
    ap.add_argument("--texts_file", default=None)
    ap.add_argument("--val_texts_file", default=None)
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()
    cfg = deep_update(load_config(args.config), parse_overrides(args.set))
    tokenizer = load_tokenizer(cfg.get("output_path", "outputs/tokenizer/tokenizer.json"))
    dtype = "uint16" if tokenizer.vocab_size < 65536 else "uint32"
    out_dir = Path(cfg.get("tinystories_out_dir", "data/tinystories"))
    train_texts = read_texts_file(args.texts_file) if args.texts_file else load_texts_from_dataset(cfg, "train")
    val_file = args.val_texts_file or args.texts_file
    val_texts = read_texts_file(val_file) if val_file else load_texts_from_dataset(cfg, "validation")
    train_tokens = tokenize_stream(tokenizer, train_texts)
    val_tokens = tokenize_stream(tokenizer, val_texts)
    write_token_bin(train_tokens, out_dir / "train.bin", dtype)
    write_token_bin(val_tokens, out_dir / "val.bin", dtype)
    meta = {
        "dtype": dtype,
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "tokenizer_path": cfg.get("output_path", "outputs/tokenizer/tokenizer.json"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_dir / 'train.bin'} ({len(train_tokens)} tokens)")
    print(f"wrote {out_dir / 'val.bin'} ({len(val_tokens)} tokens)")


if __name__ == "__main__":
    main()

