# Advanced Methods

This directory contains milestone-ready interfaces for low-resource alignment experiments.

## LoRA

`advanced/lora.py` provides a `LoRALinear` wrapper and `inject_lora()` helper for GPT attention projections: `q_proj`, `k_proj`, `v_proj`, and `out_proj`.

Planned usage for SFT/RLHF:

```python
from advanced.lora import LoRAConfig, inject_lora, mark_only_lora_trainable

replaced = inject_lora(model, LoRAConfig(rank=8, alpha=16, dropout=0.05))
mark_only_lora_trainable(model)
```

Comparison metrics:

- Trainable parameter count
- GPU memory usage
- Training time
- SFT manual accuracy
- RLHF safety answer rate

## PPO Sampling

`advanced/ppo_sampling.py` contains runnable hooks for:

- Safety prompt oversampling
- Response length limiting
- Mini-batch reuse
- Dynamic KL beta adjustment

These hooks are intentionally conservative for the milestone. Full ablation results should be filled only after real runs.

