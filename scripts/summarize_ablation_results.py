#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path

from _bootstrap import add_src_to_path
from summarize_manual_eval import summarize_safety, summarize_sft

ROOT = add_src_to_path()


def read_last_row(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def artifact(path: str | Path) -> str:
    p = Path(path)
    return "present" if p.exists() and p.stat().st_size > 0 else "missing"


def first_existing(*paths: str) -> str:
    for path in paths:
        if Path(path).exists():
            return path
    return paths[-1]


def log_entry(name: str, log_path: str, checkpoint: str, note: str) -> dict:
    return {
        "name": name,
        "log_path": log_path,
        "checkpoint": checkpoint,
        "checkpoint_status": artifact(checkpoint),
        "last_row": read_last_row(log_path),
        "note": note,
    }


def make_markdown(payload: dict) -> str:
    lines = [
        "# Ablation Summary",
        "",
        "This file is generated from local logs and manual-evaluation CSV files.",
        "",
        "## Training Logs",
        "",
        "| experiment | checkpoint | last row | note |",
        "|---|---|---|---|",
    ]
    for item in payload["training_logs"]:
        last = json.dumps(item["last_row"], ensure_ascii=False) if item["last_row"] else "not yet run"
        lines.append(f"| {item['name']} | {item['checkpoint_status']} | `{last}` | {item['note']} |")

    lines.extend(
        [
            "",
            "## Manual SFT Evaluation",
            "",
            "| experiment | filled | mean score | category means |",
            "|---|---:|---:|---|",
        ]
    )
    for item in payload["manual_sft"]:
        if not item.get("exists"):
            lines.append(f"| {item['name']} | 0 | not yet run |  |")
            continue
        category = json.dumps(item.get("by_category"), ensure_ascii=False)
        lines.append(
            f"| {item['name']} | {item.get('filled', 0)}/{item.get('total', 0)} | "
            f"{item.get('mean_score')} | `{category}` |"
        )

    lines.extend(
        [
            "",
            "## Safety Evaluation",
            "",
            "| experiment | human filled | human safe rate | heuristic safe rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in payload["safety"]:
        if not item.get("exists"):
            lines.append(f"| {item['name']} | 0 | not yet run | not yet run |")
            continue
        lines.append(
            f"| {item['name']} | {item.get('human_filled', 0)}/{item.get('total', 0)} | "
            f"{item.get('human_safe_rate_percent')} | {item.get('heuristic_safe_rate_percent')} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Model-scale ablation compares the base pretraining run against `pretrain_small` and its downstream SFT run.",
            "- SFT data-size ablation compares `sft_data_2500` against `sft_final` with the same final SFT hyperparameters except sample count.",
            "- LoRA ablation compares adapter-only SFT against the final partially-unfrozen SFT run.",
            "- RLHF safety comparison should use manually filled `human_safe` labels, not only heuristic labels.",
            "- Low manual/safety scores are kept as negative results and should not be rewritten as successful alignment.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    training_logs = [
        log_entry(
            "pretrain_base",
            "outputs/logs/pretrain_metrics.csv",
            "outputs/checkpoints/pretrain/best.pt",
            "Base model from the main run.",
        ),
        log_entry(
            "pretrain_small",
            "outputs/logs/pretrain_small_metrics.csv",
            "outputs/checkpoints/pretrain_small/best.pt",
            "Smaller model for model-scale ablation.",
        ),
        log_entry(
            "sft_final_10000",
            "outputs/logs/sft_final_metrics.csv",
            "outputs/checkpoints/sft_final/best.pt",
            "Final SFT: 10000 Alpaca samples, last 4 layers trainable.",
        ),
        log_entry(
            "sft_data_2500",
            "outputs/logs/sft_data_2500_metrics.csv",
            "outputs/checkpoints/sft_data_2500/best.pt",
            "Data-size ablation: 2500 Alpaca samples.",
        ),
        log_entry(
            "sft_lora",
            "outputs/logs/sft_lora_metrics.csv",
            "outputs/checkpoints/sft_lora/best.pt",
            "LoRA adapter-only SFT ablation.",
        ),
        log_entry(
            "sft_small_pretrain",
            "outputs/logs/sft_small_pretrain_metrics.csv",
            "outputs/checkpoints/sft_small_pretrain/best.pt",
            "SFT from the smaller pretrained checkpoint.",
        ),
        log_entry(
            "ppo_final",
            "outputs/logs/ppo_final_metrics.csv",
            "outputs/checkpoints/ppo_final/last.pt",
            "Final PPO/RLHF run.",
        ),
    ]

    manual_sft = []
    for name, path in [
        ("sft_final_10000", "reports/manual_eval_sft_final.csv"),
        ("sft_data_2500", "reports/manual_eval_sft_data_2500.csv"),
        ("sft_lora", "reports/manual_eval_sft_lora.csv"),
        ("sft_small_pretrain", "reports/manual_eval_sft_small_pretrain.csv"),
    ]:
        item = summarize_sft(path)
        item["name"] = name
        manual_sft.append(item)

    safety = []
    for name, path in [
        ("sft_final", "reports/safety_eval_sft_final.csv"),
        ("ppo_final", first_existing("reports/safety_eval_ppo_final_ablation.csv", "reports/safety_eval_final.csv")),
    ]:
        item = summarize_safety(path)
        item["name"] = name
        safety.append(item)

    payload = {
        "training_logs": training_logs,
        "manual_sft": manual_sft,
        "safety": safety,
    }
    json_out = Path("reports/ablation_summary.json")
    md_out = Path("reports/ablation_summary.md")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_out.write_text(make_markdown(payload), encoding="utf-8")
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")


if __name__ == "__main__":
    main()
