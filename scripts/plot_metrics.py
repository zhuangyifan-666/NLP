#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from nlp_llm.trainer_utils import plot_metrics


def main() -> None:
    jobs = [
        ("outputs/logs/pretrain_metrics.csv", "outputs/figures/pretrain_loss_ppl.png", ["train_loss", "val_loss", "val_ppl"]),
        ("outputs/logs/pretrain_small_metrics.csv", "outputs/figures/pretrain_small_loss_ppl.png", ["train_loss", "val_loss", "val_ppl"]),
        ("outputs/logs/sft_metrics.csv", "outputs/figures/sft_loss.png", ["train_loss", "val_loss"]),
        ("outputs/logs/sft_final_metrics.csv", "outputs/figures/sft_final_loss.png", ["train_loss", "val_loss"]),
        ("outputs/logs/sft_data_2500_metrics.csv", "outputs/figures/sft_data_2500_loss.png", ["train_loss", "val_loss"]),
        ("outputs/logs/sft_lora_metrics.csv", "outputs/figures/sft_lora_loss.png", ["train_loss", "val_loss"]),
        ("outputs/logs/sft_small_pretrain_metrics.csv", "outputs/figures/sft_small_pretrain_loss.png", ["train_loss", "val_loss"]),
        ("outputs/logs/reward_metrics.csv", "outputs/figures/reward_loss_acc.png", ["train_loss", "val_loss", "pairwise_acc"]),
        ("outputs/logs/reward_final_metrics.csv", "outputs/figures/reward_final_loss_acc.png", ["train_loss", "val_loss", "pairwise_acc"]),
        ("outputs/logs/ppo_metrics.csv", "outputs/figures/ppo_metrics.png", ["reward_mean", "kl_mean", "policy_loss", "value_loss"]),
        ("outputs/logs/ppo_final_metrics.csv", "outputs/figures/ppo_final_metrics.png", ["reward_mean", "kl_mean", "policy_loss", "value_loss"]),
    ]
    for csv_path, fig_path, cols in jobs:
        made = plot_metrics(csv_path, fig_path, cols)
        print(f"{'made' if made else 'skipped'} {fig_path}")


if __name__ == "__main__":
    main()
