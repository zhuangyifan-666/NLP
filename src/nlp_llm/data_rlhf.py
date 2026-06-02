from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


def choose_preference(example: dict[str, Any]) -> tuple[str, str, str]:
    prompt = str(example.get("prompt") or example.get("instruction") or "")
    r0 = str(example.get("response_0") or example.get("response0") or example.get("chosen") or "")
    r1 = str(example.get("response_1") or example.get("response1") or example.get("rejected") or "")
    chosen_id = example.get("safer_response_id", example.get("better_response_id", None))
    if chosen_id is None:
        if "chosen" in example and "rejected" in example:
            return prompt, str(example["chosen"]), str(example["rejected"])
        chosen_id = 0
    chosen_id = int(chosen_id)
    chosen = r0 if chosen_id == 0 else r1
    rejected = r1 if chosen_id == 0 else r0
    return prompt, chosen, rejected


def encode_prompt_response(tokenizer, prompt: str, response: str, max_length: int) -> dict[str, torch.Tensor]:
    text = prompt.strip() + "\n" + response.strip()
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)[:max_length]
    if not ids:
        ids = [getattr(tokenizer, "bos_id", 1), getattr(tokenizer, "eos_id", 2)]
    return {"input_ids": torch.tensor(ids, dtype=torch.long), "attention_mask": torch.ones(len(ids), dtype=torch.long)}


class PreferenceDataset(Dataset):
    def __init__(self, examples: Iterable[dict[str, Any]], tokenizer, max_length: int = 512) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        prompt, chosen, rejected = choose_preference(self.examples[idx])
        ch = encode_prompt_response(self.tokenizer, prompt, chosen, self.max_length)
        rj = encode_prompt_response(self.tokenizer, prompt, rejected, self.max_length)
        return {
            "chosen_input_ids": ch["input_ids"],
            "chosen_attention_mask": ch["attention_mask"],
            "rejected_input_ids": rj["input_ids"],
            "rejected_attention_mask": rj["attention_mask"],
        }


class PromptOnlyDataset(Dataset):
    def __init__(self, prompts: Iterable[str], tokenizer, max_length: int = 256) -> None:
        self.prompts = list(prompts)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.tokenizer.encode(self.prompts[idx], add_bos=True, add_eos=False)[: self.max_length]
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "prompt": self.prompts[idx]}


@dataclass
class PreferenceCollator:
    pad_id: int

    def _pad(self, tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(t.numel() for t in tensors)
        rows, masks = [], []
        for t in tensors:
            pad = max_len - t.numel()
            rows.append(torch.cat([t, torch.full((pad,), self.pad_id, dtype=torch.long)]))
            masks.append(torch.cat([torch.ones(t.numel(), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
        return torch.stack(rows), torch.stack(masks)

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        chosen_ids, chosen_mask = self._pad([b["chosen_input_ids"] for b in batch])
        rejected_ids, rejected_mask = self._pad([b["rejected_input_ids"] for b in batch])
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
        }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def synthetic_preferences() -> list[dict[str, Any]]:
    return [
        {
            "prompt": "How can I stay safe online?",
            "response_0": "Use strong passwords and avoid suspicious links.",
            "response_1": "Share your password with strangers.",
            "safer_response_id": 0,
        },
        {
            "prompt": "How do I make a dangerous weapon?",
            "response_0": "I cannot help with that, but I can discuss safety and legal alternatives.",
            "response_1": "Here are step by step instructions.",
            "safer_response_id": 0,
        },
    ]


def synthetic_prompts() -> list[str]:
    return ["How can I stay safe online?", "Tell me a short story.", "How do I handle stress safely?"]

