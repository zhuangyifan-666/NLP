#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

import torch
from torch.utils.data import DataLoader

from nlp_llm.config import deep_update, load_config, parse_overrides
from nlp_llm.data_rlhf import PreferenceCollator, PreferenceDataset, read_jsonl, synthetic_preferences
from nlp_llm.model import GPTConfig, load_checkpoint, save_checkpoint
from nlp_llm.reward_model import RewardModel, init_reward_from_gpt_checkpoint
from nlp_llm.tokenizer import load_tokenizer
from nlp_llm.trainer_utils import CSVLogger, cosine_lr, rank0_print, set_optimizer_lr, set_seed


def load_examples(cfg: dict, data_file: str | None) -> list[dict]:
    if data_file:
        return read_jsonl(data_file)
    if cfg.get("synthetic", False):
        return synthetic_preferences()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for PKU-SafeRLHF. Install manually with: python -m pip install -r requirements.txt") from exc
    name = cfg.get("dataset", "PKU-Alignment/PKU-SafeRLHF-10K")
    split = cfg.get("split", "train")
    max_samples = int(cfg.get("max_samples", 2000))
    try:
        ds = load_dataset(name, split=split)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {name}; run manually with network/HF cache available.") from exc
    return [dict(row) for i, row in enumerate(ds) if i < max_samples]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reward_debug.yaml")
    ap.add_argument("--sft_checkpoint", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--data_file", default=None)
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
        model = init_reward_from_gpt_checkpoint(sft_ckpt, map_location=device)
    else:
        mcfg = cfg.get("model", {})
        model = RewardModel(
            GPTConfig(
                vocab_size=tokenizer.vocab_size,
                block_size=int(mcfg.get("block_size", 128)),
                n_layer=int(mcfg.get("n_layer", 2)),
                n_head=int(mcfg.get("n_head", 2)),
                n_embd=int(mcfg.get("n_embd", 64)),
                dropout=float(mcfg.get("dropout", 0.1)),
                pad_id=tokenizer.pad_id,
            )
        )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-5)), weight_decay=float(cfg.get("weight_decay", 0.01)))
    start_step = 0
    best_loss = math.inf
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        best_loss = float(ckpt.get("best_val_loss", best_loss))
    examples = load_examples(cfg.get("data", cfg), args.data_file)
    split = max(1, int(len(examples) * 0.9))
    train_ds = PreferenceDataset(examples[:split], tokenizer, int(cfg.get("max_length", 512)))
    val_ds = PreferenceDataset(examples[split:] or examples[:1], tokenizer, int(cfg.get("max_length", 512)))
    collator = PreferenceCollator(tokenizer.pad_id)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 2)), shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("eval_batch_size", 2)), shuffle=False, collate_fn=collator)
    logger = CSVLogger(cfg.get("log_path", "outputs/logs/reward_metrics.csv"), ["step", "train_loss", "val_loss", "pairwise_acc", "lr"])
    out_dir = Path(cfg.get("out_dir", "outputs/checkpoints/reward"))
    max_steps = int(cfg.get("max_steps", 20))
    eval_interval = int(cfg.get("eval_interval", 10))
    step = start_step
    loader_iter = iter(train_loader)
    while step < max_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model.pairwise_loss(**batch)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
        lr = cosine_lr(step, max_steps, float(cfg.get("learning_rate", 1e-5)), int(cfg.get("warmup_steps", 0)))
        set_optimizer_lr(optimizer, lr)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % eval_interval == 0 or step == max_steps:
            model.eval()
            losses, accs = [], []
            with torch.no_grad():
                for vb in val_loader:
                    vb = {k: v.to(device) for k, v in vb.items()}
                    vo = model.pairwise_loss(**vb)
                    losses.append(float(vo["loss"].cpu()))
                    accs.append(float(vo["accuracy"].cpu()))
            model.train()
            val_loss = sum(losses) / max(1, len(losses))
            val_acc = sum(accs) / max(1, len(accs))
            logger.log({"step": step, "train_loss": float(out["loss"].detach().cpu()), "val_loss": val_loss, "pairwise_acc": val_acc, "lr": lr})
            save_checkpoint(out_dir / "last.pt", model, optimizer, step, best_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint(out_dir / "best.pt", model, optimizer, step, best_loss)
            rank0_print(f"step {step}: loss={float(out['loss']):.4f} val_loss={val_loss:.4f} acc={val_acc:.3f}")


if __name__ == "__main__":
    main()
