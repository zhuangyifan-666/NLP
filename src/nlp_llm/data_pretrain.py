from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapTokenDataset(Dataset):
    def __init__(self, bin_path: str | Path, block_size: int, dtype: str = "uint16", seed: int = 1337) -> None:
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Token binary not found: {self.bin_path}")
        self.block_size = int(block_size)
        self.dtype = np.dtype(dtype)
        self.data = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        if len(self.data) <= self.block_size + 1:
            raise ValueError(f"Not enough tokens in {self.bin_path}: {len(self.data)}")
        self.seed = int(seed)
        self.length = max(1, len(self.data) - self.block_size - 1)

    def __len__(self) -> int:
        # Random sampling dataset: expose a large virtual length for dataloader iteration.
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + int(idx))
        start = int(rng.integers(0, self.length))
        chunk = np.asarray(self.data[start : start + self.block_size + 1], dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(chunk[:-1].copy()).long(),
            "labels": torch.from_numpy(chunk[1:].copy()).long(),
        }


def load_memmap_meta(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_token_bin(tokens: list[int], out_path: str | Path, dtype: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(tokens, dtype=np.dtype(dtype))
    mem = np.memmap(out_path, dtype=np.dtype(dtype), mode="w+", shape=arr.shape)
    mem[:] = arr[:]
    mem.flush()

