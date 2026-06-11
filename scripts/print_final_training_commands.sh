#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
# Final-stage commands for the NLP LLM project.
# Run these manually from /mnt/iset/nfs-main/private/zhuangyifan/NLP.
# They may use GPUs and Hugging Face cached/downloaded datasets.

cd /mnt/iset/nfs-main/private/zhuangyifan/NLP
source ~/.bashrc 2>/dev/null || true
conda activate nlp

# 1) Stronger SFT: more Alpaca samples and more unfrozen layers.
torchrun --standalone --nproc_per_node=4 scripts/sft.py \
  --config configs/sft_final.yaml \
  --pretrained outputs/checkpoints/pretrain/best.pt

# 2) Evaluate SFT for manual scoring.
python scripts/evaluate_sft.py \
  --checkpoint outputs/checkpoints/sft_final/best.pt \
  --out reports/manual_eval_sft_final.csv \
  --max_new_tokens 96

# 3) Train a stronger reward model from the improved SFT checkpoint.
python scripts/train_reward_model.py \
  --config configs/reward_final.yaml \
  --sft_checkpoint outputs/checkpoints/sft_final/best.pt

# 4) PPO/RLHF final sanity run with PKU-SafeRLHF prompts and safety oversampling.
python scripts/ppo_train.py \
  --config configs/ppo_final.yaml \
  --sft_checkpoint outputs/checkpoints/sft_final/best.pt \
  --reward_checkpoint outputs/checkpoints/reward_final/best.pt

# 5) Safety evaluation for manual confirmation.
python scripts/evaluate_safety.py \
  --checkpoint outputs/checkpoints/ppo_final/last.pt \
  --fallback_checkpoint outputs/checkpoints/sft_final/best.pt \
  --out reports/safety_eval_final.csv \
  --max_new_tokens 96

# 6) Plot all metrics and rebuild final report draft.
python scripts/plot_metrics.py
python scripts/summarize_manual_eval.py \
  --sft reports/manual_eval_sft_final.csv \
  --safety reports/safety_eval_final.csv \
  --out reports/final_eval_summary.json
python scripts/make_final_report.py
EOF

