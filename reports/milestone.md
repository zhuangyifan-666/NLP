# NLP Milestone Report Draft

> This file is generated from actual local artifacts when available. Missing metrics are marked as not yet run.

## 1. Completed Checklist

- [x] Repository scaffold for tokenizer, pretraining, SFT, reward model, PPO, tests, advanced modules.
- [x] From-scratch PyTorch decoder-only Transformer implementation.
- [x] TinyStories tokenizer/data preparation artifacts.
- [x] Pretraining run with validation PPL, CSV logging, checkpoints, and plotting.
- [x] Alpaca SFT run with response-only loss and last-layer fine-tuning logic.
- [x] Reward model and simplified PPO debug runs.
- [x] LoRA and PPO sampling advanced interfaces.
- [ ] Human SFT scoring completed.
- [ ] Human safety scoring completed.

## 2. Environment and Hardware

- Expected env: `conda activate nlp`
- Expected hardware: 4 x RTX 3090
- Mixed precision: fp16 by default; bf16 is not used by default.

## 3. Model Configuration

- Pretrain best checkpoint: step 10000, 6 layers, 512 hidden, 8 heads, block 512
- SFT best checkpoint: step 2000, 6 layers, 512 hidden, 8 heads, block 512

## 4. Pretraining

- Metrics CSV: `present`
- Figure: `present`
- Last logged row: `{"step": "10000", "train_loss": "1.5457179248332977", "val_loss": "1.714336597919464", "val_ppl": "5.552990424748201", "lr": "8.201887236047866e-12"}`

## 5. SFT

- Metrics CSV: `present`
- Figure: `present`
- Manual eval sheet: `present`
- Manual score completion: `0/50 filled`
- Last logged row: `{"step": "2000", "train_loss": "3.785861551761627", "val_loss": "3.9314149141311647", "lr": "6.834904537900144e-11"}`

## 6. RLHF Debug Status

- Reward metrics: `{"step": "10", "train_loss": "0.4081365168094635", "val_loss": "1.0871022641658783", "pairwise_acc": "0.5714285714285714", "lr": "3.8060233744356634e-07"}`
- PPO metrics: `{"step": "2", "reward_mean": "1.1987080574035645", "kl_mean": "-0.002368208020925522", "policy_loss": "-0.011577516794204712", "value_loss": "1.6745532751083374", "entropy": "3.291649103164673", "clipfrac": "0.0"}`
- Safety eval sheet: `present`
- Human safety completion: `0/5 filled`

## 7. Advanced Plan

- `advanced/lora.py`: Linear-layer LoRA wrapper and injection helper.
- `advanced/ppo_sampling.py`: safe prompt oversampling, length limiting, minibatch reuse, dynamic KL beta interfaces.

## 8. Problems and Current Solutions

- Network/data availability: scripts fail with explicit messages and support local synthetic smoke data.
- Compute budget: debug configs are small; 4-GPU configs use fp16 DDP and gradient accumulation.
- Evaluation integrity: missing real metrics remain `not yet run`; manual scores are blank until filled by the user.

## 9. Plan After June 3

1. Fill manual scores in `reports/manual_eval_sft.csv` and `reports/safety_eval.csv`.
2. Improve SFT if manual score is low: train longer, unfreeze more layers, or use more Alpaca samples.
3. Scale reward model/PPO beyond debug after validating reward pairwise accuracy.
4. Add qualitative failure cases and ablation tables to the final report.
5. Prepare final slides with real curves, samples, and limitations.
