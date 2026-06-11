#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

import torch
from torch.utils.data import DataLoader

from nlp_llm.config import deep_update, load_config, parse_overrides
from nlp_llm.data_rlhf import PromptOnlyDataset, synthetic_prompts
from nlp_llm.model import GPTConfig, load_checkpoint, save_checkpoint, strip_module_prefix
from nlp_llm.ppo import PPOConfig, PolicyWithValue, compute_ppo_loss, logprobs_from_logits
from nlp_llm.reward_model import RewardModel
from nlp_llm.tokenizer import load_tokenizer
from nlp_llm.trainer_utils import CSVLogger, rank0_print, set_seed
from advanced.ppo_sampling import oversample_safety_prompts, update_kl_beta


def load_prompts(cfg: dict) -> list[str]:
    if cfg.get("synthetic", False):
        return synthetic_prompts()
    prompt_file = cfg.get("prompt_file")
    if prompt_file:
        return [line.strip() for line in Path(prompt_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    data_cfg = cfg.get("data", {})
    if not data_cfg:
        return synthetic_prompts()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for PPO prompt loading. Install manually with: python -m pip install -r requirements.txt") from exc
    name = data_cfg.get("dataset", "PKU-Alignment/PKU-SafeRLHF-10K")
    split = data_cfg.get("split", "train")
    max_samples = int(data_cfg.get("max_samples", 512))
    try:
        ds = load_dataset(name, split=split)
    except Exception as exc:
        raise RuntimeError(f"Failed to load PPO prompt dataset {name}; run manually with network/HF cache available.") from exc
    prompts: list[str] = []
    for row in ds:
        prompt = str(row.get("prompt") or row.get("instruction") or "").strip()
        if prompt:
            prompts.append(prompt)
        if len(prompts) >= max_samples:
            break
    sampling_cfg = cfg.get("sampling", {})
    if sampling_cfg.get("safety_prompt_ratio", 0) > 0:
        prompts = oversample_safety_prompts(
            prompts,
            sampling_cfg.get("safety_keywords", ["harm", "hack", "bomb", "weapon", "kill", "suicide", "attack", "steal"]),
            float(sampling_cfg.get("safety_prompt_ratio", 0.5)),
        )
    return prompts or synthetic_prompts()


def collate_prompts(batch: list[dict], pad_id: int) -> dict:
    max_len = max(x["input_ids"].numel() for x in batch)
    ids = []
    for item in batch:
        pad = max_len - item["input_ids"].numel()
        ids.append(torch.cat([item["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
    return {"input_ids": torch.stack(ids), "prompts": [b["prompt"] for b in batch]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo_debug.yaml")
    ap.add_argument("--sft_checkpoint", default=None)
    ap.add_argument("--reward_checkpoint", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()
    cfg = deep_update(load_config(args.config), parse_overrides(args.set))
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    set_seed(int(cfg.get("seed", 1337)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg.get("tokenizer_path", "outputs/tokenizer/tokenizer.json"), debug_byte_fallback=cfg.get("byte_tokenizer", False))
    sft_ckpt = args.sft_checkpoint or cfg.get("sft_checkpoint")
    if sft_ckpt:
        ckpt = load_checkpoint(sft_ckpt, map_location=device)
        model_cfg = GPTConfig.from_dict(ckpt["config"])
        policy = PolicyWithValue(model_cfg)
        policy.policy.load_state_dict(strip_module_prefix(ckpt["model"]), strict=True)
        ref = PolicyWithValue(model_cfg)
        ref.policy.load_state_dict(strip_module_prefix(ckpt["model"]), strict=True)
    else:
        mcfg = cfg.get("model", {})
        model_cfg = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=int(mcfg.get("block_size", 128)),
            n_layer=int(mcfg.get("n_layer", 2)),
            n_head=int(mcfg.get("n_head", 2)),
            n_embd=int(mcfg.get("n_embd", 64)),
            dropout=float(mcfg.get("dropout", 0.1)),
            pad_id=tokenizer.pad_id,
        )
        policy = PolicyWithValue(model_cfg)
        ref = PolicyWithValue(model_cfg)
        ref.policy.load_state_dict(policy.policy.state_dict())
    reward_ckpt = args.reward_checkpoint or cfg.get("reward_checkpoint")
    if reward_ckpt:
        rckpt = load_checkpoint(reward_ckpt, map_location=device)
        reward_model = RewardModel(GPTConfig.from_dict(rckpt["config"]))
        reward_model.load_state_dict(strip_module_prefix(rckpt["model"]), strict=False)
    else:
        reward_model = RewardModel(model_cfg)
    policy.to(device)
    ref.to(device).eval()
    reward_model.to(device).eval()
    for p in ref.parameters():
        p.requires_grad = False
    for p in reward_model.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(policy.parameters(), lr=float(cfg.get("learning_rate", 1e-6)))
    start_step = 0
    if args.resume:
        resume_ckpt = load_checkpoint(args.resume, map_location=device)
        if "policy_with_value" in resume_ckpt:
            policy.load_state_dict(strip_module_prefix(resume_ckpt["policy_with_value"]))
        elif "model" in resume_ckpt:
            policy.policy.load_state_dict(strip_module_prefix(resume_ckpt["model"]), strict=False)
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_step = int(resume_ckpt.get("step", 0))
    prompts = load_prompts(cfg)
    ds = PromptOnlyDataset(prompts, tokenizer, int(cfg.get("prompt_max_length", 128)))
    loader = DataLoader(ds, batch_size=int(cfg.get("batch_size", 2)), shuffle=True, collate_fn=lambda b: collate_prompts(b, tokenizer.pad_id))
    logger = CSVLogger(cfg.get("log_path", "outputs/logs/ppo_metrics.csv"), ["step", "reward_mean", "kl_mean", "policy_loss", "value_loss", "entropy", "clipfrac"])
    ppo_cfg = PPOConfig(
        cliprange=float(cfg.get("cliprange", 0.2)),
        vf_coef=float(cfg.get("vf_coef", 0.5)),
        ent_coef=float(cfg.get("ent_coef", 0.01)),
        kl_beta=float(cfg.get("kl_beta", 0.05)),
    )
    max_steps = int(cfg.get("max_steps", 2))
    max_new_tokens = int(cfg.get("max_new_tokens", 32))
    step = start_step
    loader_iter = iter(loader)
    while step < max_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        prompt_ids = batch["input_ids"].to(device)
        with torch.no_grad():
            full = policy.generate(prompt_ids, max_new_tokens=max_new_tokens, temperature=float(cfg.get("temperature", 0.8)), top_k=int(cfg.get("top_k", 50)), eos_id=tokenizer.eos_id)
            model_in = full[:, :-1]
            actions = full[:, 1:]
            prompt_len = prompt_ids.size(1)
            response_mask = torch.zeros_like(actions, dtype=torch.float32)
            response_mask[:, max(0, prompt_len - 1) :] = 1.0
            old_out = policy(model_in)
            ref_out = ref(model_in)
            old_logp = logprobs_from_logits(old_out["logits"], actions)
            ref_logp = logprobs_from_logits(ref_out["logits"], actions)
            reward = reward_model(full, attention_mask=(full != tokenizer.pad_id).long())
        for _ in range(int(cfg.get("ppo_epochs", 2))):
            out = policy(model_in)
            loss_out = compute_ppo_loss(out["logits"], out["values"], actions, old_logp, ref_logp, reward, response_mask, ppo_cfg)
            optimizer.zero_grad(set_to_none=True)
            loss_out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(cfg.get("grad_clip", 1.0)))
            optimizer.step()
        step += 1
        logger.log(
            {
                "step": step,
                "reward_mean": float(reward.mean().cpu()),
                "kl_mean": float(loss_out["kl_mean"].cpu()),
                "policy_loss": float(loss_out["policy_loss"].cpu()),
                "value_loss": float(loss_out["value_loss"].cpu()),
                "entropy": float(loss_out["entropy"].cpu()),
                "clipfrac": float(loss_out["clipfrac"].cpu()),
            }
        )
        rank0_print(f"ppo step {step}: reward={float(reward.mean()):.4f} kl={float(loss_out['kl_mean']):.4f}")
        sampling_cfg = cfg.get("sampling", {})
        if sampling_cfg.get("dynamic_kl_beta", False):
            ppo_cfg.kl_beta = update_kl_beta(
                ppo_cfg.kl_beta,
                float(loss_out["kl_mean"].detach().cpu()),
                float(sampling_cfg.get("target_kl", 0.1)),
                float(sampling_cfg.get("kl_increase", 1.5)),
                float(sampling_cfg.get("kl_decrease", 0.75)),
            )
    out_dir = Path(cfg.get("out_dir", "outputs/checkpoints/ppo"))
    save_checkpoint(out_dir / "last.pt", policy.policy, optimizer, step, None, {"tokenizer_path": cfg.get("tokenizer_path")})
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_with_value": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": policy.config.to_dict(),
            "step": step,
            "tokenizer_path": cfg.get("tokenizer_path"),
        },
        out_dir / "last_policy_value.pt",
    )


if __name__ == "__main__":
    main()
