#!/usr/bin/env python
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

import torch
from torch.utils.data import DataLoader

from nlp_llm.data_pretrain import MemmapTokenDataset, load_memmap_meta
from nlp_llm.model import load_gpt_from_checkpoint
from nlp_llm.trainer_utils import estimate_loss, perplexity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--meta", default="data/tinystories/meta.json")
    ap.add_argument("--val_bin", default="data/tinystories/val.bin")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_iters", type=int, default=50)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_gpt_from_checkpoint(args.checkpoint, map_location=device).to(device)
    meta = load_memmap_meta(args.meta)
    ds = MemmapTokenDataset(args.val_bin, model.config.block_size, meta["dtype"])
    dl = DataLoader(ds, batch_size=args.batch_size)
    loss = estimate_loss(model, dl, device, args.eval_iters)
    print(f"val_loss={loss:.6f} val_ppl={perplexity(loss):.3f}")


if __name__ == "__main__":
    main()

