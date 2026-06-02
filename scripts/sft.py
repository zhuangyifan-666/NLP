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
from nlp_llm.data_sft import SFTCollator, SFTDataset, read_jsonl, synthetic_sft_examples
from nlp_llm.model import GPT, GPTConfig, load_checkpoint, save_checkpoint, strip_module_prefix
from nlp_llm.tokenizer import load_tokenizer
from nlp_llm.trainer_utils import (
    CSVLogger,
    cleanup_distributed,
    cosine_lr,
    estimate_loss,
    init_distributed,
    rank0_print,
    set_optimizer_lr,
    set_seed,
)


def load_examples(cfg: dict, data_file: str | None) -> list[dict]:
    if data_file:
        return read_jsonl(data_file)
    if cfg.get("synthetic", False):
        return synthetic_sft_examples()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for Alpaca. Install manually with: python -m pip install -r requirements.txt") from exc
    name = cfg.get("dataset", "yahma/alpaca-cleaned")
    split = cfg.get("split", "train")
    max_samples = int(cfg.get("max_samples", 2000))
    try:
        ds = load_dataset(name, split=split)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {name}; run this script yourself with network/HF cache available.") from exc
    return [dict(row) for i, row in enumerate(ds) if i < max_samples]


def freeze_for_last_layers(model: GPT, train_last_n_layers: int) -> None:
    if model.lm_head.weight is model.transformer["wte"].weight:
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    for p in model.parameters():
        p.requires_grad = False
    n_layer = len(model.transformer["h"])
    for block in model.transformer["h"][max(0, n_layer - train_last_n_layers) :]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.transformer["ln_f"].parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_debug.yaml")
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--data_file", default=None)
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()
    cfg = deep_update(load_config(args.config), parse_overrides(args.set))
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    set_seed(int(cfg.get("seed", 1337)))
    ddp, rank, _, _, device = init_distributed()
    tokenizer = load_tokenizer(cfg.get("tokenizer_path", "outputs/tokenizer/tokenizer.json"), debug_byte_fallback=cfg.get("byte_tokenizer", False))
    pretrained = args.pretrained or cfg.get("pretrained")
    if pretrained:
        ckpt = load_checkpoint(pretrained, map_location=device)
        model = GPT(GPTConfig.from_dict(ckpt["config"]))
        model.load_state_dict(strip_module_prefix(ckpt["model"]), strict=True)
    else:
        mcfg = cfg.get("model", {})
        model = GPT(
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
    freeze_for_last_layers(model, int(cfg.get("train_last_n_layers", 2)))
    if rank == 0:
        total = model.num_parameters(False)
        trainable = model.num_parameters(True)
        print(f"trainable params: {trainable}/{total} ({trainable / max(1, total):.2%})")
    optimizer = model.configure_optimizers(float(cfg.get("weight_decay", 0.01)), float(cfg.get("learning_rate", 1e-4)))
    start_step = 0
    best_val_loss = math.inf
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(strip_module_prefix(ckpt["model"]), strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
    if ddp:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    examples = load_examples(cfg.get("data", cfg), args.data_file)
    split = max(1, int(len(examples) * 0.9))
    train_ds = SFTDataset(examples[:split], tokenizer, int(cfg.get("max_length", 512)))
    val_ds = SFTDataset(examples[split:] or examples[:1], tokenizer, int(cfg.get("max_length", 512)))
    collator = SFTCollator(tokenizer.pad_id)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("per_device_batch_size", 2)), shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("eval_batch_size", 2)), shuffle=False, collate_fn=collator)
    log_path = Path(cfg.get("log_path", "outputs/logs/sft_metrics.csv"))
    logger = CSVLogger(log_path, ["step", "train_loss", "val_loss", "lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("fp16", True)) and device.type == "cuda")
    out_dir = Path(cfg.get("out_dir", "outputs/checkpoints/sft"))
    max_steps = int(cfg.get("max_steps", 50))
    grad_accum = int(cfg.get("grad_accum_steps", 1))
    eval_interval = int(cfg.get("eval_interval", 25))
    step = start_step
    loader_iter = iter(train_loader)
    model.train()
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                loss = model(input_ids, labels=labels, loss_mask=loss_mask)["loss"] / grad_accum
            scaler.scale(loss).backward()
            train_loss += float(loss.detach().cpu())
        lr = cosine_lr(step, max_steps, float(cfg.get("learning_rate", 1e-4)), int(cfg.get("warmup_steps", 0)))
        set_optimizer_lr(optimizer, lr)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip", 1.0)))
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step % eval_interval == 0 or step == max_steps:
            raw = model.module if hasattr(model, "module") else model
            val_loss = estimate_loss(raw, val_loader, device, int(cfg.get("eval_iters", 10)))
            if rank == 0:
                logger.log({"step": step, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
                save_checkpoint(out_dir / "last.pt", model, optimizer, step, best_val_loss, {"tokenizer_path": cfg.get("tokenizer_path")})
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(out_dir / "best.pt", model, optimizer, step, best_val_loss, {"tokenizer_path": cfg.get("tokenizer_path")})
                rank0_print(f"step {step}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
