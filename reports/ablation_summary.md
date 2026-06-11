# Ablation Summary

This file is generated from local logs and manual-evaluation CSV files.

## Training Logs

| experiment | checkpoint | last row | note |
|---|---|---|---|
| pretrain_base | present | `{"step": "10000", "train_loss": "1.5457179248332977", "val_loss": "1.714336597919464", "val_ppl": "5.552990424748201", "lr": "8.201887236047866e-12"}` | Base model from the main run. |
| pretrain_small | present | `{"step": "10000", "train_loss": "1.7899183332920074", "val_loss": "1.8521212887763978", "val_ppl": "6.37332485566642", "lr": "8.201887236047866e-12"}` | Smaller model for model-scale ablation. |
| sft_final_10000 | present | `{"step": "4000", "train_loss": "2.9714315533638", "val_loss": "3.198132801055908", "lr": "2.5630896396955368e-11"}` | Final SFT: 10000 Alpaca samples, last 4 layers trainable. |
| sft_data_2500 | present | `{"step": "4000", "train_loss": "1.2950069308280945", "val_loss": "4.346858851114908", "lr": "2.5630896396955368e-11"}` | Data-size ablation: 2500 Alpaca samples. |
| sft_lora | present | `{"step": "4000", "train_loss": "5.766440987586975", "val_loss": "5.144180742899577", "lr": "3.417452852927383e-11"}` | LoRA adapter-only SFT ablation. |
| sft_small_pretrain | present | `{"step": "4000", "train_loss": "3.2749624848365784", "val_loss": "3.309802190462748", "lr": "2.5630896396955368e-11"}` | SFT from the smaller pretrained checkpoint. |
| ppo_final | present | `{"step": "200", "reward_mean": "0.42278987169265747", "kl_mean": "1.0127471685409546", "policy_loss": "-0.004073517397046089", "value_loss": "0.4999963045120239", "entropy": "5.972235679626465", "clipfrac": "0.0"}` | Final PPO/RLHF run. |

## Manual SFT Evaluation

| experiment | filled | mean score | category means |
|---|---:|---:|---|
| sft_final_10000 | 50/50 | 0.12 | `{"explain": 0.0, "qa": 0.0, "reasoning": 0.0, "summary": 0.15, "writing": 0.45}` |
| sft_data_2500 | 50/50 | 0.14 | `{"explain": 0.0, "qa": 0.0, "reasoning": 0.0, "summary": 0.2, "writing": 0.5}` |
| sft_lora | 50/50 | 0.07 | `{"explain": 0.0, "qa": 0.0, "reasoning": 0.0, "summary": 0.1, "writing": 0.25}` |
| sft_small_pretrain | 50/50 | 0.13 | `{"explain": 0.0, "qa": 0.0, "reasoning": 0.0, "summary": 0.15, "writing": 0.5}` |

## Safety Evaluation

| experiment | human filled | human safe rate | heuristic safe rate |
|---|---:|---:|---:|
| sft_final | 5/5 | 0.0 | 0.0 |
| ppo_final | 5/5 | 0.0 | 0.0 |

## Interpretation Notes

- Model-scale ablation compares the base pretraining run against `pretrain_small` and its downstream SFT run.
- SFT data-size ablation compares `sft_data_2500` against `sft_final` with the same final SFT hyperparameters except sample count.
- LoRA ablation compares adapter-only SFT against the final partially-unfrozen SFT run.
- RLHF safety comparison should use manually filled `human_safe` labels, not only heuristic labels.
- Low manual/safety scores are kept as negative results and should not be rewritten as successful alignment.
