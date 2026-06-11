from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")

    @classmethod
    def from_dict(cls, data: dict) -> "LoRAConfig":
        return cls(
            rank=int(data.get("rank", cls.rank)),
            alpha=float(data.get("alpha", cls.alpha)),
            dropout=float(data.get("dropout", cls.dropout)),
            target_modules=tuple(data.get("target_modules", cls.target_modules)),
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        return payload


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        delta = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scaling
        return base + delta


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    replaced: list[str] = []
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in cfg.target_modules and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, cfg.rank, cfg.alpha, cfg.dropout))
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                replaced.append(full_name)
    return replaced


def mark_only_lora_trainable(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def merge_lora_weights(model: nn.Module) -> list[str]:
    """Replace LoRALinear modules with ordinary nn.Linear modules for inference."""
    merged: list[str] = []
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, LoRALinear):
                continue
            base = child.base
            merged_linear = nn.Linear(
                base.in_features,
                base.out_features,
                bias=base.bias is not None,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            delta = (child.lora_b @ child.lora_a) * child.scaling
            merged_linear.weight.data.copy_(base.weight.data + delta.to(base.weight.dtype))
            if base.bias is not None and merged_linear.bias is not None:
                merged_linear.bias.data.copy_(base.bias.data)
            setattr(module, child_name, merged_linear)
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            merged.append(full_name)
    return merged
