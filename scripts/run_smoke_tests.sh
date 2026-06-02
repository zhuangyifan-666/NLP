#!/usr/bin/env bash
set -euo pipefail

source ~/.bashrc 2>/dev/null || true
conda activate nlp 2>/dev/null || true

mkdir -p outputs/smoke_data outputs/logs outputs/figures outputs/examples reports data/tinystories

python -m compileall src scripts advanced
pytest -q

cat > outputs/smoke_data/tinystories.txt <<'TXT'
Once upon a time there was a small cat. The cat found a red ball.
A little girl went to the garden and played with her dog.
The sun was warm, and the children shared a simple story.
TXT

cat > outputs/smoke_data/sft.jsonl <<'JSONL'
{"instruction":"What is gravity?","input":"","output":"Gravity pulls objects together."}
{"instruction":"Write a tiny story.","input":"","output":"A child found a red ball and smiled."}
{"instruction":"Summarize the text.","input":"Birds fly. Fish swim.","output":"Birds fly and fish swim."}
{"instruction":"Classify sentiment.","input":"I love this day.","output":"Positive."}
JSONL

cat > outputs/smoke_data/reward.jsonl <<'JSONL'
{"prompt":"How can I stay safe online?","response_0":"Use strong passwords and avoid suspicious links.","response_1":"Share your password publicly.","safer_response_id":0}
{"prompt":"How can I make a bomb?","response_0":"I cannot help with that. I can discuss safety instead.","response_1":"Here are detailed steps.","safer_response_id":0}
JSONL

python scripts/train_tokenizer.py --config configs/tokenizer.yaml --texts_file outputs/smoke_data/tinystories.txt --set vocab_size=512 min_frequency=1 output_path=outputs/tokenizer/smoke_tokenizer.json meta_path=outputs/tokenizer/smoke_tokenizer.meta.json
python scripts/prepare_tinystories.py --config configs/tokenizer.yaml --texts_file outputs/smoke_data/tinystories.txt --val_texts_file outputs/smoke_data/tinystories.txt --set output_path=outputs/tokenizer/smoke_tokenizer.json tinystories_out_dir=data/tinystories
python scripts/pretrain.py --config configs/pretrain_debug.yaml --max_steps 12 --set tokenizer_path=outputs/tokenizer/smoke_tokenizer.json
python scripts/sft.py --config configs/sft_debug.yaml --pretrained outputs/checkpoints/pretrain/best.pt --data_file outputs/smoke_data/sft.jsonl --max_steps 6 --set tokenizer_path=outputs/tokenizer/smoke_tokenizer.json
python scripts/train_reward_model.py --config configs/reward_debug.yaml --sft_checkpoint outputs/checkpoints/sft/best.pt --data_file outputs/smoke_data/reward.jsonl --max_steps 5 --set tokenizer_path=outputs/tokenizer/smoke_tokenizer.json
python scripts/ppo_train.py --config configs/ppo_debug.yaml --sft_checkpoint outputs/checkpoints/sft/best.pt --reward_checkpoint outputs/checkpoints/reward/best.pt --max_steps 1 --set tokenizer_path=outputs/tokenizer/smoke_tokenizer.json
python scripts/generate.py --checkpoint outputs/checkpoints/pretrain/best.pt --tokenizer outputs/tokenizer/smoke_tokenizer.json --prompt "Once upon a time" --max_new_tokens 20 --out outputs/examples/pretrain_samples.jsonl
python scripts/plot_metrics.py
python scripts/make_milestone_report.py

