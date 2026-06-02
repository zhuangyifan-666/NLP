from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


def alpaca_prompt(instruction: str, input_text: str | None = None) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def format_alpaca_example(example: dict[str, Any], include_output: bool = True) -> str:
    prompt = alpaca_prompt(str(example.get("instruction", "")), str(example.get("input", "") or ""))
    if include_output:
        return prompt + str(example.get("output", "")).strip()
    return prompt


def build_sft_features(
    tokenizer,
    instruction: str,
    output: str,
    input_text: str | None = None,
    max_length: int = 512,
    add_eos: bool = True,
) -> dict[str, torch.Tensor]:
    prompt = alpaca_prompt(instruction, input_text)
    response = (output or "").strip()
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    full_ids = tokenizer.encode(prompt + response, add_bos=True, add_eos=add_eos)
    full_ids = full_ids[: max_length + 1]
    if len(full_ids) < 2:
        full_ids = [getattr(tokenizer, "bos_id", 1), getattr(tokenizer, "eos_id", 2)]
    input_ids = full_ids[:-1]
    labels = full_ids[1:]
    loss_mask = [0] * len(labels)
    # labels[j] predicts full_ids[j + 1]; first response token starts at prompt index len(prompt_ids).
    for j in range(len(labels)):
        if j + 1 >= min(len(prompt_ids), len(full_ids)):
            loss_mask[j] = 1
        else:
            labels[j] = -100
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class SFTDataset(Dataset):
    def __init__(self, examples: Iterable[dict[str, Any]], tokenizer, max_length: int = 512) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return build_sft_features(
            self.tokenizer,
            instruction=str(ex.get("instruction", "")),
            input_text=str(ex.get("input", "") or ""),
            output=str(ex.get("output", "")),
            max_length=self.max_length,
        )


@dataclass
class SFTCollator:
    pad_id: int
    label_pad_id: int = -100

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].numel() for item in batch)
        input_ids, labels, loss_mask = [], [], []
        for item in batch:
            n = item["input_ids"].numel()
            pad = max_len - n
            input_ids.append(torch.cat([item["input_ids"], torch.full((pad,), self.pad_id, dtype=torch.long)]))
            labels.append(torch.cat([item["labels"], torch.full((pad,), self.label_pad_id, dtype=torch.long)]))
            loss_mask.append(torch.cat([item["loss_mask"], torch.zeros(pad, dtype=torch.float32)]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "loss_mask": torch.stack(loss_mask),
            "attention_mask": (torch.stack(input_ids) != self.pad_id).long(),
        }


def synthetic_sft_examples() -> list[dict[str, str]]:
    return [
        {"instruction": "What is gravity?", "input": "", "output": "Gravity is a force that pulls objects together."},
        {"instruction": "Write a tiny story.", "input": "", "output": "A child found a red ball and shared it with a friend."},
        {"instruction": "Summarize the text.", "input": "Birds can fly. Fish can swim.", "output": "Birds fly and fish swim."},
        {"instruction": "Classify sentiment.", "input": "I love this sunny day.", "output": "Positive."},
    ]

