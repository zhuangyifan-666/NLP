# Final Report Draft

This draft is generated from available local artifacts. Values that were not actually run remain `not yet run`.

## 1. Project Goal

The project implements a small decoder-only Transformer from scratch and studies a three-stage alignment pipeline: TinyStories pretraining, Alpaca-Cleaned supervised fine-tuning, and PKU-SafeRLHF reward/PPO alignment.

## 2. Artifact Status

- Tokenizer: `present`
- TinyStories token bins: `present` / `present`
- Pretrain best checkpoint: `present`
- Milestone SFT checkpoint: `present`
- Final SFT checkpoint: `present`
- Final reward checkpoint: `present`
- Final PPO checkpoint: `present`

## 3. Main Metrics

- Pretraining last row: `{"step": "10000", "train_loss": "1.5457179248332977", "val_loss": "1.714336597919464", "val_ppl": "5.552990424748201", "lr": "8.201887236047866e-12"}`
- Milestone SFT last row: `{"step": "2000", "train_loss": "3.785861551761627", "val_loss": "3.9314149141311647", "lr": "6.834904537900144e-11"}`
- Final SFT last row: `{"step": "4000", "train_loss": "2.9714315533638", "val_loss": "3.198132801055908", "lr": "2.5630896396955368e-11"}`
- Milestone reward last row: `{"step": "10", "train_loss": "0.4081365168094635", "val_loss": "1.0871022641658783", "pairwise_acc": "0.5714285714285714", "lr": "3.8060233744356634e-07"}`
- Final reward last row: `{"step": "1000", "train_loss": "0.48631271719932556", "val_loss": "0.6742314090728759", "pairwise_acc": "0.558", "lr": "1.3669799732163314e-11"}`
- Milestone PPO last row: `{"step": "2", "reward_mean": "1.1987080574035645", "kl_mean": "-0.002368208020925522", "policy_loss": "-0.011577516794204712", "value_loss": "1.6745532751083374", "entropy": "3.291649103164673", "clipfrac": "0.0"}`
- Final PPO last row: `{"step": "200", "reward_mean": "0.42278987169265747", "kl_mean": "1.0127471685409546", "policy_loss": "-0.004073517397046089", "value_loss": "0.4999963045120239", "entropy": "5.972235679626465", "clipfrac": "0.0"}`

## 4. Manual Evaluation

- SFT eval summary: `{"path": "reports/manual_eval_sft_final.csv", "exists": true, "total": 50, "filled": 50, "blank": 0, "mean_score": 0.12, "accuracy_percent": 12.0, "by_category": {"explain": 0.0, "qa": 0.0, "reasoning": 0.0, "summary": 0.15, "writing": 0.45}}`
- Safety eval summary: `{"path": "reports/safety_eval_final.csv", "exists": true, "total": 5, "human_filled": 5, "human_blank": 0, "human_safe_count": 0, "human_safe_rate_percent": 0.0, "heuristic_filled": 4, "heuristic_safe_count": 0, "heuristic_safe_rate_percent": 0.0}`

## 5. Interpretation

- Pretraining already reached a strong TinyStories validation PPL in the milestone run.
- Compared with the milestone SFT checkpoint, the final SFT run reduced validation loss from 3.93 to 3.20.
- The final reward model completed the full run, but validation pairwise accuracy remains modest, so reward/PPO metrics should be interpreted together with generated examples and human safety labels.
- Strict manual scoring shows the small model is still weak at instruction following and safety refusal, which is an important negative result to report.

## 6. Final Work Completed

1. Final SFT with `configs/sft_final.yaml`.
2. Final reward model with `configs/reward_final.yaml` initialized from the final SFT checkpoint.
3. Final PPO with `configs/ppo_final.yaml`, PKU prompts, safety oversampling, and dynamic KL beta.
4. Manual-style SFT scoring and safety scoring in the final CSV files.
5. Final evaluation summary and slide outline.

## 7. Final Slides Outline

See `reports/final_slides_outline.md`.
