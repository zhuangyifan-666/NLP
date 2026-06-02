from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import GPT, GPTConfig


@dataclass
class PPOConfig:
    cliprange: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    kl_beta: float = 0.05
    gamma: float = 1.0


class PolicyWithValue(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.policy = GPT(config)
        self.value_head = nn.Linear(config.n_embd, 1)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        out = self.policy(input_ids, labels=labels, return_hidden=True)
        values = self.value_head(out["hidden_states"]).squeeze(-1)
        out["values"] = values
        return out

    def generate(self, *args, **kwargs):
        return self.policy.generate(*args, **kwargs)


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def compute_ppo_loss(
    logits: torch.Tensor,
    values: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    cfg: PPOConfig,
) -> dict[str, torch.Tensor]:
    new_logprobs = logprobs_from_logits(logits, actions)
    kl = new_logprobs - ref_logprobs
    token_rewards = rewards.unsqueeze(1).expand_as(new_logprobs) - cfg.kl_beta * kl.detach()
    advantages = token_rewards - values.detach()
    valid_adv = advantages[response_mask.bool()]
    adv_std = valid_adv.std(unbiased=False).clamp_min(1e-6) if valid_adv.numel() else torch.tensor(1.0, device=advantages.device)
    advantages = (advantages - masked_mean(advantages, response_mask)) / adv_std
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1.0 - cfg.cliprange, 1.0 + cfg.cliprange)
    policy_loss = masked_mean(torch.maximum(unclipped, clipped), response_mask)
    returns = token_rewards
    value_loss = masked_mean((values - returns) ** 2, response_mask)
    entropy = masked_mean(entropy_from_logits(logits), response_mask)
    loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
    clipfrac = masked_mean((torch.abs(ratio - 1.0) > cfg.cliprange).float(), response_mask)
    return {
        "loss": loss,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
        "kl_mean": masked_mean(kl.detach(), response_mask),
        "clipfrac": clipfrac.detach(),
    }
