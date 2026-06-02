#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from nlp_llm.config import deep_update, load_config, parse_overrides
from nlp_llm.data_pretrain import MemmapTokenDataset, load_memmap_meta
from nlp_llm.model import GPT, GPTConfig, load_checkpoint, save_checkpoint
from nlp_llm.trainer_utils import (
    CSVLogger,
    cleanup_distributed,
    cosine_lr,
    estimate_loss,
    init_distributed,
    perplexity,
    rank0_print,
    set_optimizer_lr,
    set_seed,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_debug.yaml")
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
    ddp, rank, _, _, device = init_distributed()
    data_cfg = cfg.get("data", {})
    meta = load_memmap_meta(data_cfg.get("meta_path", "data/tinystories/meta.json"))
    block_size = int(cfg.get("model", {}).get("block_size", cfg.get("block_size", 512)))
    model_cfg = GPTConfig(
        vocab_size=int(meta["vocab_size"]),
        block_size=block_size,
        n_layer=int(cfg.get("model", {}).get("n_layer", 6)),
        n_head=int(cfg.get("model", {}).get("n_head", 8)),
        n_embd=int(cfg.get("model", {}).get("n_embd", 512)),
        dropout=float(cfg.get("model", {}).get("dropout", 0.1)),
        pad_id=0,
    )
    model = GPT(model_cfg).to(device)
    optimizer = model.configure_optimizers(
        weight_decay=float(cfg.get("weight_decay", 0.1)),
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
    )
    start_step = 0
    best_val_loss = math.inf
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        rank0_print(f"resumed from {args.resume} at step {start_step}")
    if ddp:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    train_ds = MemmapTokenDataset(data_cfg.get("train_bin", "data/tinystories/train.bin"), block_size, meta["dtype"], seed=1337 + rank)
    val_ds = MemmapTokenDataset(data_cfg.get("val_bin", "data/tinystories/val.bin"), block_size, meta["dtype"], seed=2333 + rank)
    loader = DataLoader(train_ds, batch_size=int(cfg.get("per_device_batch_size", 4)), shuffle=False, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("eval_batch_size", cfg.get("per_device_batch_size", 4))), shuffle=False)
    max_steps = int(cfg.get("max_steps", 100))
    grad_accum = int(cfg.get("grad_accum_steps", 1))
    eval_interval = int(cfg.get("eval_interval", 50))
    out_dir = Path(cfg.get("out_dir", "outputs/checkpoints/pretrain"))
    log_path = Path(cfg.get("log_path", "outputs/logs/pretrain_metrics.csv"))
    logger = CSVLogger(log_path, ["step", "train_loss", "val_loss", "val_ppl", "lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("fp16", True)) and device.type == "cuda")
    model.train()
    step = start_step
    loader_iter = iter(loader)
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                out = model(input_ids, labels=labels)
                loss = out["loss"] / grad_accum
            scaler.scale(loss).backward()
            train_loss += float(loss.detach().cpu())
        lr = cosine_lr(step, max_steps, float(cfg.get("learning_rate", 3e-4)), int(cfg.get("warmup_steps", 0)))
        set_optimizer_lr(optimizer, lr)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step % eval_interval == 0 or step == max_steps:
            val_loss = estimate_loss(model.module if hasattr(model, "module") else model, val_loader, device, int(cfg.get("eval_iters", 20)))
            val_ppl = perplexity(val_loss)
            if rank == 0:
                logger.log({"step": step, "train_loss": train_loss, "val_loss": val_loss, "val_ppl": val_ppl, "lr": lr})
                save_checkpoint(out_dir / "last.pt", model, optimizer, step, best_val_loss, {"tokenizer_path": meta.get("tokenizer_path")})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(out_dir / "best.pt", model, optimizer, step, best_val_loss, {"tokenizer_path": meta.get("tokenizer_path")})
                rank0_print(f"step {step}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} ppl={val_ppl:.2f}")
    cleanup_distributed()


if __name__ == "__main__":
    main()

