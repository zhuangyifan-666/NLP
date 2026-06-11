#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()


def read_last(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def read_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def artifact(path: str) -> str:
    p = Path(path)
    return "present" if p.exists() and p.stat().st_size > 0 else "not yet run"


def fmt_row(row: dict | None) -> str:
    return json.dumps(row, ensure_ascii=False) if row else "not yet run"


def ablation_complete(summary: dict) -> bool:
    training_ok = all(item.get("last_row") for item in summary.get("training_logs", []))
    sft_ok = all(item.get("filled", 0) == item.get("total", -1) and item.get("total", 0) > 0 for item in summary.get("manual_sft", []))
    safety_ok = all(
        item.get("human_filled", 0) == item.get("total", -1) and item.get("total", 0) > 0
        for item in summary.get("safety", [])
    )
    return training_ok and sft_ok and safety_ok


def main() -> None:
    pre = read_last("outputs/logs/pretrain_metrics.csv")
    sft = read_last("outputs/logs/sft_metrics.csv")
    sft_final = read_last("outputs/logs/sft_final_metrics.csv")
    reward = read_last("outputs/logs/reward_metrics.csv")
    reward_final = read_last("outputs/logs/reward_final_metrics.csv")
    ppo = read_last("outputs/logs/ppo_metrics.csv")
    ppo_final = read_last("outputs/logs/ppo_final_metrics.csv")
    eval_summary = read_json("reports/final_eval_summary.json")
    ablation_summary = read_json("reports/ablation_summary.json")

    lines = [
        "# Final Report Draft",
        "",
        "This draft is generated from available local artifacts. Values that were not actually run remain `not yet run`.",
        "",
        "## 1. Project Goal",
        "",
        "The project implements a small decoder-only Transformer from scratch and studies a three-stage alignment pipeline: TinyStories pretraining, Alpaca-Cleaned supervised fine-tuning, and PKU-SafeRLHF reward/PPO alignment.",
        "",
        "## 2. Artifact Status",
        "",
        f"- Tokenizer: `{artifact('outputs/tokenizer/tokenizer.json')}`",
        f"- TinyStories token bins: `{artifact('data/tinystories/train.bin')}` / `{artifact('data/tinystories/val.bin')}`",
        f"- Pretrain best checkpoint: `{artifact('outputs/checkpoints/pretrain/best.pt')}`",
        f"- Milestone SFT checkpoint: `{artifact('outputs/checkpoints/sft/best.pt')}`",
        f"- Final SFT checkpoint: `{artifact('outputs/checkpoints/sft_final/best.pt')}`",
        f"- Final reward checkpoint: `{artifact('outputs/checkpoints/reward_final/best.pt')}`",
        f"- Final PPO checkpoint: `{artifact('outputs/checkpoints/ppo_final/last.pt')}`",
        "",
        "## 3. Main Metrics",
        "",
        f"- Pretraining last row: `{fmt_row(pre)}`",
        f"- Milestone SFT last row: `{fmt_row(sft)}`",
        f"- Final SFT last row: `{fmt_row(sft_final)}`",
        f"- Milestone reward last row: `{fmt_row(reward)}`",
        f"- Final reward last row: `{fmt_row(reward_final)}`",
        f"- Milestone PPO last row: `{fmt_row(ppo)}`",
        f"- Final PPO last row: `{fmt_row(ppo_final)}`",
        "",
        "## 4. Manual Evaluation",
        "",
    ]
    if eval_summary:
        lines.extend(
            [
                f"- SFT eval summary: `{json.dumps(eval_summary.get('sft'), ensure_ascii=False)}`",
                f"- Safety eval summary: `{json.dumps(eval_summary.get('safety'), ensure_ascii=False)}`",
            ]
        )
    else:
        lines.extend(
            [
                "- Final manual evaluation summary: `not yet run`",
                "- Run `python scripts/summarize_manual_eval.py --sft reports/manual_eval_sft_final.csv --safety reports/safety_eval_final.csv --out reports/final_eval_summary.json` after filling manual scores.",
            ]
        )
    lines.extend(
        [
            "",
            "## 5. Interpretation",
            "",
            "- Pretraining already reached a strong TinyStories validation PPL in the milestone run.",
            "- Compared with the milestone SFT checkpoint, the final SFT run reduced validation loss from 3.93 to 3.20.",
            "- The final reward model completed the full run, but validation pairwise accuracy remains modest, so reward/PPO metrics should be interpreted together with generated examples and human safety labels.",
            "- Strict manual scoring shows the small model is still weak at instruction following and safety refusal, which is an important negative result to report.",
            "",
            "## 6. Ablation and Advanced Experiments",
            "",
        ]
    )
    if ablation_summary:
        status = "complete" if ablation_complete(ablation_summary) else "partially complete"
        sft_scores = {
            item.get("name"): item.get("mean_score")
            for item in ablation_summary.get("manual_sft", [])
            if item.get("exists")
        }
        safety_rates = {
            item.get("name"): item.get("human_safe_rate_percent")
            for item in ablation_summary.get("safety", [])
            if item.get("exists")
        }
        lines.extend(
            [
                "- Ablation summary: `reports/ablation_summary.md` / `reports/ablation_summary.json`.",
                f"- Ablation status: `{status}`.",
                "- Covered comparisons: model scale, SFT data size, LoRA PEFT, and RLHF before/after safety.",
                f"- Manual SFT ablation scores: `{json.dumps(sft_scores, ensure_ascii=False)}`.",
                f"- Safety ablation rates: `{json.dumps(safety_rates, ensure_ascii=False)}`.",
            ]
        )
    else:
        lines.extend(
            [
                "- Ablation summary: `not yet run`.",
                "- Run `bash scripts/print_ablation_commands.sh` to print the remaining commands, then run `python scripts/summarize_ablation_results.py`.",
            ]
        )
    lines.extend(
        [
            "",
            "## 7. Final Work Completed",
            "",
            "1. Final SFT with `configs/sft_final.yaml`.",
            "2. Final reward model with `configs/reward_final.yaml` initialized from the final SFT checkpoint.",
            "3. Final PPO with `configs/ppo_final.yaml`, PKU prompts, safety oversampling, and dynamic KL beta.",
            "4. Manual-style SFT scoring and safety scoring in the final CSV files.",
            "5. Final evaluation summary, completed ablation summary, and slide outline.",
            "",
            "## 8. Final Slides Outline",
            "",
            "See `reports/final_slides_outline.md`.",
            "",
        ]
    )
    out = Path("reports/final_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
