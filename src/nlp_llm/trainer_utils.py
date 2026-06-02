from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple[bool, int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return True, rank, world_size, local_rank, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, device


def cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_rank0() -> bool:
    return not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def rank0_print(*args: Any, **kwargs: Any) -> None:
    if is_rank0():
        print(*args, **kwargs)


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: Iterable[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._wrote_header = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow(row)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def cosine_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int = 0, min_lr: float = 0.0) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, max_steps - warmup_steps)))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def perplexity(loss: float) -> float:
    try:
        return float(math.exp(min(20.0, loss)))
    except OverflowError:
        return float("inf")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def plot_metrics(csv_path: str | Path, out_path: str | Path, y_columns: list[str]) -> bool:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return False
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib and pandas are required for plotting. "
            "Install manually with: python -m pip install -r requirements.txt"
        ) from exc
    df = pd.read_csv(csv_path)
    if df.empty:
        return False
    x_col = "step" if "step" in df.columns else df.columns[0]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    for col in y_columns:
        if col in df.columns:
            plt.plot(df[x_col], df[col], label=col)
    plt.xlabel(x_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return True


@torch.no_grad()
def estimate_loss(model, dataloader, device: torch.device, max_batches: int = 20) -> float:
    model.eval()
    meter = AverageMeter()
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            loss_mask = loss_mask.to(device)
        out = model(input_ids, labels=labels, loss_mask=loss_mask)
        meter.update(float(out["loss"].detach().cpu()), input_ids.size(0))
    model.train()
    return meter.avg

