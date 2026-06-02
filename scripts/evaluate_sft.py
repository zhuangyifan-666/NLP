#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

import torch

from nlp_llm.data_sft import alpaca_prompt
from nlp_llm.generation import generate_text
from nlp_llm.model import load_checkpoint, load_gpt_from_checkpoint
from nlp_llm.tokenizer import load_tokenizer


DEFAULT_PROMPTS = [
    ("qa", "What is gravity?", ""),
    ("explain", "Explain why the sky is blue in simple words.", ""),
    ("writing", "Write a short story about friendship.", ""),
    ("summary", "Summarize the text.", "Cats sleep a lot. They like warm places."),
    ("reasoning", "If A is taller than B and B is taller than C, who is tallest?", ""),
] * 10


def load_prompts(path: str | None) -> list[tuple[str, str, str]]:
    if path is None or not Path(path).exists():
        return DEFAULT_PROMPTS[:50]
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append((obj.get("category", "custom"), obj["prompt"], obj.get("input", "")))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompts", default="reports/sft_eval_prompts.jsonl")
    ap.add_argument("--out", default="reports/manual_eval_sft.csv")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    tokenizer = load_tokenizer(args.tokenizer or ckpt.get("tokenizer_path"))
    model = load_gpt_from_checkpoint(args.checkpoint, map_location=device).to(device)
    rows = []
    for idx, (category, instruction, input_text) in enumerate(load_prompts(args.prompts), 1):
        prompt = alpaca_prompt(instruction, input_text)
        full_text = generate_text(model, tokenizer, prompt, device, max_new_tokens=args.max_new_tokens)
        response = full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text.strip()
        rows.append(
            {
                "id": idx,
                "category": category,
                "prompt": instruction,
                "input": input_text,
                "model_response": response,
                "score": "",
                "notes": "",
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote manual evaluation sheet: {out}")


if __name__ == "__main__":
    main()
