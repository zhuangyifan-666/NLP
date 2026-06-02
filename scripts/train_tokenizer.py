#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from nlp_llm.config import deep_update, load_config, parse_overrides


def read_texts_file(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def dataset_texts(config: dict):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required to load TinyStories. Install manually with: python -m pip install -r requirements.txt"
        ) from exc
    name = config.get("dataset", "roneneldan/TinyStories")
    split = config.get("split", "train")
    text_field = config.get("text_field", "text")
    max_samples = int(config.get("max_samples", 20000))
    try:
        ds = load_dataset(name, split=split)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset {name!r}. This usually means network/HF cache is unavailable. "
            "Run this command yourself after network is available: "
            "python scripts/train_tokenizer.py --config configs/tokenizer.yaml"
        ) from exc
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        text = str(row.get(text_field, "")).strip()
        if text:
            yield text


def train(config: dict, texts_file: str | None = None) -> None:
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.processors import TemplateProcessing
        from tokenizers.trainers import BpeTrainer
    except ImportError as exc:
        raise RuntimeError(
            "tokenizers is required. Install manually with: python -m pip install -r requirements.txt"
        ) from exc

    special_tokens = config.get("special_tokens", ["<pad>", "<bos>", "<eos>", "<unk>"])
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=int(config.get("vocab_size", 16000)),
        min_frequency=int(config.get("min_frequency", 2)),
        special_tokens=special_tokens,
    )
    iterator = read_texts_file(texts_file) if texts_file else dataset_texts(config)
    tokenizer.train_from_iterator(iterator, trainer=trainer, length=int(config.get("max_samples", 20000)))
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        special_tokens=[("<bos>", tokenizer.token_to_id("<bos>")), ("<eos>", tokenizer.token_to_id("<eos>"))],
    )
    out_path = Path(config.get("output_path", "outputs/tokenizer/tokenizer.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    meta = {
        "tokenizer_path": str(out_path),
        "vocab_size": tokenizer.get_vocab_size(),
        "dataset": config.get("dataset", "local_texts" if texts_file else "roneneldan/TinyStories"),
        "max_samples": config.get("max_samples"),
        "special_tokens": special_tokens,
    }
    meta_path = Path(config.get("meta_path", out_path.with_suffix(".meta.json")))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"saved tokenizer: {out_path}")
    print(f"saved meta: {meta_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tokenizer.yaml")
    ap.add_argument("--texts_file", default=None, help="Local newline text file for no-network smoke runs.")
    ap.add_argument("--set", nargs="*", default=None, help="Override config values, e.g. vocab_size=8000")
    args = ap.parse_args()
    config = deep_update(load_config(args.config), parse_overrides(args.set))
    train(config, texts_file=args.texts_file)


if __name__ == "__main__":
    main()
