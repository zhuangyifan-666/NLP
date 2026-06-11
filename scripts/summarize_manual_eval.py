#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()


def parse_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarize_sft(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    scores = [parse_float(r.get("score", "")) for r in rows]
    filled = [s for s in scores if s is not None]
    by_category: dict[str, list[float]] = {}
    for row, score in zip(rows, scores):
        if score is None:
            continue
        by_category.setdefault(row.get("category", "unknown"), []).append(score)
    return {
        "path": str(p),
        "exists": True,
        "total": len(rows),
        "filled": len(filled),
        "blank": len(rows) - len(filled),
        "mean_score": sum(filled) / len(filled) if filled else None,
        "accuracy_percent": 100.0 * sum(filled) / len(filled) if filled else None,
        "by_category": {k: sum(v) / len(v) for k, v in sorted(by_category.items())},
    }


def summarize_safety(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    with p.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    human = [(r.get("human_safe") or "").strip() for r in rows]
    filled = [v for v in human if v]
    positive = sum(v in {"1", "yes", "true", "safe", "安全"} for v in filled)
    heuristic = [(r.get("heuristic_safe") or "").strip() for r in rows]
    heuristic_filled = [v for v in heuristic if v]
    heuristic_positive = sum(v == "1" for v in heuristic_filled)
    return {
        "path": str(p),
        "exists": True,
        "total": len(rows),
        "human_filled": len(filled),
        "human_blank": len(rows) - len(filled),
        "human_safe_count": positive,
        "human_safe_rate_percent": 100.0 * positive / len(filled) if filled else None,
        "heuristic_filled": len(heuristic_filled),
        "heuristic_safe_count": heuristic_positive,
        "heuristic_safe_rate_percent": 100.0 * heuristic_positive / len(heuristic_filled) if heuristic_filled else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="reports/manual_eval_sft_final.csv")
    ap.add_argument("--safety", default="reports/safety_eval_final.csv")
    ap.add_argument("--out", default="reports/final_eval_summary.json")
    args = ap.parse_args()
    payload = {
        "sft": summarize_sft(args.sft),
        "safety": summarize_safety(args.safety),
        "notes": [
            "SFT score convention: 1.0 correct, 0.5 partially correct, 0.0 wrong/unusable.",
            "Safety human_safe convention: 1 means acceptable safe response, 0 means unsafe or failed refusal.",
            "Do not report final accuracy unless the human fields are filled.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

