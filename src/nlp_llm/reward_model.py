from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import GPT, GPTConfig, load_checkpoint, strip_module_prefix


class RewardModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = GPT(config)
        self.reward_head = nn.Linear(config.n_embd, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.backbone(input_ids, return_hidden=True)
        hidden = out["hidden_states"]
        if attention_mask is None:
            idx = torch.full((input_ids.size(0),), input_ids.size(1) - 1, device=input_ids.device, dtype=torch.long)
        else:
            idx = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        final_hidden = hidden[torch.arange(input_ids.size(0), device=input_ids.device), idx]
        return self.reward_head(final_hidden).squeeze(-1)

    def pairwise_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        r_chosen = self(chosen_input_ids, chosen_attention_mask)
        r_rejected = self(rejected_input_ids, rejected_attention_mask)
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        acc = (r_chosen > r_rejected).float().mean()
        return {"loss": loss, "accuracy": acc, "r_chosen": r_chosen, "r_rejected": r_rejected}

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


def load_reward_model_from_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> RewardModel:
    ckpt = load_checkpoint(path, map_location=map_location)
    config = GPTConfig.from_dict(ckpt["config"])
    model = RewardModel(config)
    model.load_state_dict(strip_module_prefix(ckpt["model"]))
    return model


def init_reward_from_gpt_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> RewardModel:
    ckpt = load_checkpoint(path, map_location=map_location)
    config = GPTConfig.from_dict(ckpt["config"])
    model = RewardModel(config)
    model.backbone.load_state_dict(strip_module_prefix(ckpt["model"]), strict=False)
    return model


def reward_checkpoint_payload(model: RewardModel, step: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {"model": model.state_dict(), "config": model.config.to_dict(), "step": step}
    payload.update(extra or {})
    return payload

