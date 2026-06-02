#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc 2>/dev/null || true
conda activate nlp

if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  python -m pip install -r requirements.txt
fi

python scripts/train_tokenizer.py --config configs/tokenizer.yaml
python scripts/prepare_tinystories.py --config configs/tokenizer.yaml

torchrun --standalone --nproc_per_node=4 scripts/pretrain.py --config configs/pretrain_4gpu.yaml
python scripts/generate.py --checkpoint outputs/checkpoints/pretrain/best.pt --prompt "Once upon a time" --max_new_tokens 100 --out outputs/examples/pretrain_samples.jsonl

torchrun --standalone --nproc_per_node=4 scripts/sft.py --config configs/sft_4gpu.yaml --pretrained outputs/checkpoints/pretrain/best.pt
python scripts/evaluate_sft.py --checkpoint outputs/checkpoints/sft/best.pt --out reports/manual_eval_sft.csv

python scripts/train_reward_model.py --config configs/reward_debug.yaml --sft_checkpoint outputs/checkpoints/sft/best.pt
python scripts/ppo_train.py --config configs/ppo_debug.yaml --sft_checkpoint outputs/checkpoints/sft/best.pt --reward_checkpoint outputs/checkpoints/reward/best.pt
python scripts/evaluate_safety.py --checkpoint outputs/checkpoints/ppo/last.pt --fallback_checkpoint outputs/checkpoints/sft/best.pt --out reports/safety_eval.csv

python scripts/plot_metrics.py
python scripts/make_milestone_report.py

