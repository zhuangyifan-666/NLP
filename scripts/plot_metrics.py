#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from nlp_llm.trainer_utils import plot_metrics


def main() -> None:
    jobs = [
        ("outputs/logs/pretrain_metrics.csv", "outputs/figures/pretrain_loss_ppl.png", ["train_loss", "val_loss", "val_ppl"]),
        ("outputs/logs/sft_metrics.csv", "outputs/figures/sft_loss.png", ["train_loss", "val_loss"]),
        ("outputs/logs/reward_metrics.csv", "outputs/figures/reward_loss_acc.png", ["train_loss", "val_loss", "pairwise_acc"]),
        ("outputs/logs/ppo_metrics.csv", "outputs/figures/ppo_metrics.png", ["reward_mean", "kl_mean", "policy_loss", "value_loss"]),
    ]
    for csv_path, fig_path, cols in jobs:
        made = plot_metrics(csv_path, fig_path, cols)
        print(f"{'made' if made else 'skipped'} {fig_path}")


if __name__ == "__main__":
    main()

