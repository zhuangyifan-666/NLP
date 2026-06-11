#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
# Extra final ablation commands for the NLP LLM project.
# Run manually from /mnt/iset/nfs-main/private/zhuangyifan/NLP.
# These commands do not install or download packages by themselves, but dataset
# loading requires your existing Hugging Face cache or network access.

cd /mnt/iset/nfs-main/private/zhuangyifan/NLP
source ~/.bashrc 2>/dev/null || true
conda activate nlp

# A) LoRA PEFT ablation.
# Compares LoRA adapter tuning with the full final SFT run.
torchrun --standalone --nproc_per_node=4 scripts/sft.py \
  --config configs/sft_lora_ablation.yaml \
  --pretrained outputs/checkpoints/pretrain/best.pt

python scripts/evaluate_sft.py \
  --checkpoint outputs/checkpoints/sft_lora/best.pt \
  --out reports/manual_eval_sft_lora.csv \
  --max_new_tokens 96

# B) SFT data-size ablation.
# Same final SFT training setup, but with 2500 Alpaca samples instead of 10000.
torchrun --standalone --nproc_per_node=4 scripts/sft.py \
  --config configs/sft_data_ablation_2500.yaml \
  --pretrained outputs/checkpoints/pretrain/best.pt

python scripts/evaluate_sft.py \
  --checkpoint outputs/checkpoints/sft_data_2500/best.pt \
  --out reports/manual_eval_sft_data_2500.csv \
  --max_new_tokens 96

# C) Model-scale ablation.
# Train a smaller TinyStories model, then run the same final SFT setup on it.
torchrun --standalone --nproc_per_node=4 scripts/pretrain.py \
  --config configs/pretrain_small_ablation.yaml

python scripts/evaluate_pretrain.py \
  --checkpoint outputs/checkpoints/pretrain_small/best.pt

torchrun --standalone --nproc_per_node=4 scripts/sft.py \
  --config configs/sft_small_pretrain_ablation.yaml \
  --pretrained outputs/checkpoints/pretrain_small/best.pt

python scripts/evaluate_sft.py \
  --checkpoint outputs/checkpoints/sft_small_pretrain/best.pt \
  --out reports/manual_eval_sft_small_pretrain.csv \
  --max_new_tokens 96

# D) RLHF before/after safety comparison.
# Fill human_safe manually after each CSV is generated.
python scripts/evaluate_safety.py \
  --checkpoint outputs/checkpoints/sft_final/best.pt \
  --out reports/safety_eval_sft_final.csv \
  --max_new_tokens 96

python scripts/evaluate_safety.py \
  --checkpoint outputs/checkpoints/ppo_final/last.pt \
  --fallback_checkpoint outputs/checkpoints/sft_final/best.pt \
  --out reports/safety_eval_ppo_final_ablation.csv \
  --max_new_tokens 96

# E) Rebuild figures and ablation summary after manual scores are filled.
python scripts/plot_metrics.py
python scripts/summarize_ablation_results.py
python scripts/make_final_report.py
EOF
