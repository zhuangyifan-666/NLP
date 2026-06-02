from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class PPOSamplingConfig:
    safety_prompt_ratio: float = 0.5
    max_new_tokens: int = 64
    minibatch_reuse: int = 2
    target_kl: float = 0.1
    kl_beta: float = 0.05
    kl_increase: float = 1.5
    kl_decrease: float = 0.75


def oversample_safety_prompts(prompts: list[str], safety_keywords: Iterable[str], ratio: float) -> list[str]:
    if not prompts:
        return []
    keywords = tuple(k.lower() for k in safety_keywords)
    safety = [p for p in prompts if any(k in p.lower() for k in keywords)]
    normal = [p for p in prompts if p not in safety]
    if not safety or ratio <= 0:
        return prompts
    target_safety = max(1, int(len(prompts) * ratio))
    out = safety.copy()
    while len(out) < target_safety:
        out.extend(safety)
    out = out[:target_safety] + normal
    return out


def update_kl_beta(current_beta: float, observed_kl: float, target_kl: float, increase: float = 1.5, decrease: float = 0.75) -> float:
    if observed_kl > target_kl * 1.5:
        return current_beta * increase
    if observed_kl < target_kl / 1.5:
        return current_beta * decrease
    return current_beta


def ppo_sampling_hook_description() -> dict[str, str]:
    return {
        "safe_prompt_oversampling": "Increase the proportion of safety-related prompts in PPO rollout batches.",
        "length_limit": "Cap response length to reduce rollout and reward-model cost.",
        "mini_batch_reuse": "Reuse generated rollouts for multiple PPO epochs.",
        "dynamic_kl_beta": "Increase beta when KL is too high and decrease it when updates are too conservative.",
    }

