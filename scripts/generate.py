#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

import torch

from nlp_llm.generation import generate_text
from nlp_llm.model import load_checkpoint, load_gpt_from_checkpoint
from nlp_llm.tokenizer import load_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    tokenizer_path = args.tokenizer or ckpt.get("tokenizer_path")
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_gpt_from_checkpoint(args.checkpoint, map_location=device).to(device)
    text = generate_text(model, tokenizer, args.prompt, device, args.max_new_tokens)
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": args.prompt, "text": text}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

