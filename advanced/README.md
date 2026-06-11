# Advanced Methods

本目录记录 final 阶段的进阶实现：LoRA 参数高效微调和 PPO sampling 优化。两部分都已经接入训练脚本，并在 `reports/ablation_summary.md` / `reports/final_report.md` 中整理了结果或负结果分析。

## LoRA

`advanced/lora.py` 提供：

- `LoRALinear`：低秩 adapter wrapper。
- `inject_lora()`：向 GPT attention projection 注入 LoRA，包括 `q_proj`、`k_proj`、`v_proj`、`out_proj`。
- `mark_only_lora_trainable()`：冻结 backbone，仅训练 LoRA 参数。
- `lora_state_dict()` / `load_lora_state_dict()`：保存和加载 LoRA 权重。

典型用法：

```python
from advanced.lora import LoRAConfig, inject_lora, mark_only_lora_trainable

replaced = inject_lora(model, LoRAConfig(rank=8, alpha=16, dropout=0.05))
mark_only_lora_trainable(model)
```

对应配置和日志：

- 配置：`configs/sft_lora_ablation.yaml`
- 日志：`outputs/logs/sft_lora_metrics.csv`
- 人工评估：`reports/manual_eval_sft_lora.csv`
- 汇总：`reports/ablation_summary.md`

当前 LoRA ablation 是负结果：manual accuracy 低于 final SFT。可能原因包括 base model 较弱、rank/训练步数有限、只调 attention projection 难以弥补 instruction following 缺陷。该结果不能写成 LoRA 已显著提升效果，只能作为参数高效微调接口和初步消融。

## PPO Sampling

`advanced/ppo_sampling.py` 提供：

- Safety prompt oversampling：提高安全相关 prompt 的采样比例。
- Response length limiting：限制 PPO rollout 长度，降低无效长回答。
- Mini-batch / rollout reuse：同一 rollout 支持多个 PPO epoch。
- Dynamic KL beta：根据 KL 偏离目标动态调节 `kl_beta`。

这些 hooks 已在 `scripts/ppo_train.py` / `configs/ppo_final.yaml` 中使用或预留。PPO final 日志见：

- `outputs/logs/ppo_final_metrics.csv`
- `outputs/figures/ppo_final_metrics.png`
- `reports/safety_eval_final.csv`

当前 PPO/RLHF safety human safe rate 很低，是负结果。流程完整跑通，但 reward model pairwise accuracy 接近随机、小模型 SFT 起点弱、PPO 步数有限，因此不能证明安全对齐成功。

## 已完成与未完成边界

已完成：

- LoRA 注入、冻结策略、checkpoint 保存/加载和 SFT ablation。
- PPO safety prompt oversampling、rollout reuse、dynamic KL beta 接口。
- final report 中的消融表和负结果分析。

未运行 / 未来工作：

- DPO 或 rejection sampling fine-tuning。
- 严格显存/吞吐量 profile。
- 更大模型、更长上下文、更高质量 reward model 下的系统性安全评测。
