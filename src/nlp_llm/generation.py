from __future__ import annotations

import json
from pathlib import Path

import torch


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 50,
    top_p: float | None = None,
) -> str:
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=getattr(tokenizer, "eos_id", None),
    )[0].tolist()
    return tokenizer.decode(out)


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

