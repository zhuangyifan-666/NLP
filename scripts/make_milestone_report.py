#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()


def last_row(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def exists(path: str) -> str:
    return "`present`" if Path(path).exists() else "`not yet run`"


def has_artifact(path: str) -> bool:
    return Path(path).exists() and Path(path).stat().st_size > 0


def checked(done: bool, text: str) -> str:
    return f"- [{'x' if done else ' '}] {text}"


def blank_count_csv(path: str, field: str) -> tuple[int, int] | None:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return len(rows), sum(not (row.get(field) or "").strip() for row in rows)


def checkpoint_info(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "not yet run"
    try:
        import torch

        ckpt = torch.load(p, map_location="cpu")
        cfg = ckpt.get("config", {})
        if cfg:
            params = f"{cfg.get('n_layer')} layers, {cfg.get('n_embd')} hidden, {cfg.get('n_head')} heads, block {cfg.get('block_size')}"
        else:
            params = "config missing"
        return f"step {ckpt.get('step', 'unknown')}, {params}"
    except Exception as exc:
        return f"present but unreadable: {exc}"


def main() -> None:
    pre = last_row("outputs/logs/pretrain_metrics.csv")
    sft = last_row("outputs/logs/sft_metrics.csv")
    reward = last_row("outputs/logs/reward_metrics.csv")
    ppo = last_row("outputs/logs/ppo_metrics.csv")
    sft_scores = blank_count_csv("reports/manual_eval_sft.csv", "score")
    safety_scores = blank_count_csv("reports/safety_eval.csv", "human_safe")
    pretrain_done = has_artifact("outputs/checkpoints/pretrain/best.pt") and pre is not None
    sft_done = has_artifact("outputs/checkpoints/sft/best.pt") and sft is not None
    reward_done = has_artifact("outputs/checkpoints/reward/best.pt") and reward is not None
    ppo_done = has_artifact("outputs/checkpoints/ppo/last.pt") and ppo is not None
    human_sft_done = bool(sft_scores and sft_scores[1] == 0)
    human_safety_done = bool(safety_scores and safety_scores[1] == 0)
    lines = [
        "# NLP Milestone Report Draft",
        "",
        "> This file is generated from actual local artifacts when available. Missing metrics are marked as not yet run.",
        "",
        "## 1. Completed Checklist",
        "",
        checked(True, "Repository scaffold for tokenizer, pretraining, SFT, reward model, PPO, tests, advanced modules."),
        checked(True, "From-scratch PyTorch decoder-only Transformer implementation."),
        checked(has_artifact("outputs/tokenizer/tokenizer.json") and has_artifact("data/tinystories/train.bin"), "TinyStories tokenizer/data preparation artifacts."),
        checked(pretrain_done, "Pretraining run with validation PPL, CSV logging, checkpoints, and plotting."),
        checked(sft_done, "Alpaca SFT run with response-only loss and last-layer fine-tuning logic."),
        checked(reward_done and ppo_done, "Reward model and simplified PPO debug runs."),
        checked(True, "LoRA and PPO sampling advanced interfaces."),
        checked(human_sft_done, "Human SFT scoring completed."),
        checked(human_safety_done, "Human safety scoring completed."),
        "",
        "## 2. Environment and Hardware",
        "",
        "- Expected env: `conda activate nlp`",
        "- Expected hardware: 4 x RTX 3090",
        "- Mixed precision: fp16 by default; bf16 is not used by default.",
        "",
        "## 3. Model Configuration",
        "",
        f"- Pretrain best checkpoint: {checkpoint_info('outputs/checkpoints/pretrain/best.pt')}",
        f"- SFT best checkpoint: {checkpoint_info('outputs/checkpoints/sft/best.pt')}",
        "",
        "## 4. Pretraining",
        "",
        f"- Metrics CSV: {exists('outputs/logs/pretrain_metrics.csv')}",
        f"- Figure: {exists('outputs/figures/pretrain_loss_ppl.png')}",
        f"- Last logged row: `{json.dumps(pre, ensure_ascii=False) if pre else 'not yet run'}`",
        "",
        "## 5. SFT",
        "",
        f"- Metrics CSV: {exists('outputs/logs/sft_metrics.csv')}",
        f"- Figure: {exists('outputs/figures/sft_loss.png')}",
        f"- Manual eval sheet: {exists('reports/manual_eval_sft.csv')}",
        f"- Manual score completion: `{sft_scores[0] - sft_scores[1]}/{sft_scores[0]} filled`" if sft_scores else "- Manual score completion: `not yet run`",
        f"- Last logged row: `{json.dumps(sft, ensure_ascii=False) if sft else 'not yet run'}`",
        "",
        "## 6. RLHF Debug Status",
        "",
        f"- Reward metrics: `{json.dumps(reward, ensure_ascii=False) if reward else 'not yet run'}`",
        f"- PPO metrics: `{json.dumps(ppo, ensure_ascii=False) if ppo else 'not yet run'}`",
        f"- Safety eval sheet: {exists('reports/safety_eval.csv')}",
        f"- Human safety completion: `{safety_scores[0] - safety_scores[1]}/{safety_scores[0]} filled`" if safety_scores else "- Human safety completion: `not yet run`",
        "",
        "## 7. Advanced Plan",
        "",
        "- `advanced/lora.py`: Linear-layer LoRA wrapper and injection helper.",
        "- `advanced/ppo_sampling.py`: safe prompt oversampling, length limiting, minibatch reuse, dynamic KL beta interfaces.",
        "",
        "## 8. Problems and Current Solutions",
        "",
        "- Network/data availability: scripts fail with explicit messages and support local synthetic smoke data.",
        "- Compute budget: debug configs are small; 4-GPU configs use fp16 DDP and gradient accumulation.",
        "- Evaluation integrity: missing real metrics remain `not yet run`; manual scores are blank until filled by the user.",
        "",
        "## 9. Plan After June 3",
        "",
        "1. Fill manual scores in `reports/manual_eval_sft.csv` and `reports/safety_eval.csv`.",
        "2. Improve SFT if manual score is low: train longer, unfreeze more layers, or use more Alpaca samples.",
        "3. Scale reward model/PPO beyond debug after validating reward pairwise accuracy.",
        "4. Add qualitative failure cases and ablation tables to the final report.",
        "5. Prepare final slides with real curves, samples, and limitations.",
        "",
    ]
    out = Path("reports/milestone.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
