#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()


REQUIRED_FILES = [
    "README.md",
    "reports/final_report.md",
    "reports/final_eval_summary.json",
    "reports/ablation_summary.md",
    "reports/manual_eval_sft_final.csv",
    "reports/safety_eval_final.csv",
    "reports/MODEL_ARTIFACTS.md",
    "advanced/README.md",
    "outputs/figures/pretrain_loss_ppl.png",
    "outputs/figures/sft_final_loss.png",
    "outputs/figures/reward_final_loss_acc.png",
    "outputs/figures/ppo_final_metrics.png",
]


def status_line(ok: bool, message: str) -> None:
    prefix = "PASS" if ok else "WARN"
    print(f"[{prefix}] {message}")


def check_required_files(warnings: list[str]) -> None:
    for rel in REQUIRED_FILES:
        path = ROOT / rel
        ok = path.exists() and path.stat().st_size > 0
        status_line(ok, f"{rel} {'exists' if ok else 'missing or empty'}")
        if not ok:
            warnings.append(f"missing_or_empty:{rel}")


def check_eval_summary(warnings: list[str]) -> None:
    path = ROOT / "reports/final_eval_summary.json"
    if not path.exists():
        warnings.append("final_eval_summary_missing")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        status_line(False, f"reports/final_eval_summary.json is invalid JSON: {exc}")
        warnings.append("final_eval_summary_invalid_json")
        return
    sft_acc = payload.get("sft", {}).get("accuracy_percent")
    safety_rate = payload.get("safety", {}).get("human_safe_rate_percent")
    print(f"SFT manual accuracy: {sft_acc if sft_acc is not None else 'not run'}")
    print(f"Safety human safe rate: {safety_rate if safety_rate is not None else 'not run'}")


def check_final_report(warnings: list[str]) -> None:
    path = ROOT / "reports/final_report.md"
    if not path.exists():
        warnings.append("final_report_missing")
        return
    text = path.read_text(encoding="utf-8")
    placeholder_terms = ["Draft", "TODO", "TBD", "待补充", "placeholder", "PLACEHOLDER"]
    found_placeholders = [term for term in placeholder_terms if term in text]
    if found_placeholders:
        status_line(False, f"final_report.md contains possible placeholders: {', '.join(found_placeholders)}")
        warnings.append("final_report_placeholders")
    else:
        status_line(True, "final_report.md has no obvious draft placeholders")
    analysis_terms = ["负结果", "失败分析", "失败原因", "反思"]
    has_analysis = any(term in text for term in analysis_terms)
    if has_analysis:
        status_line(True, "final_report.md includes negative-result/failure analysis keywords")
    else:
        status_line(False, "final_report.md lacks negative-result/failure analysis keywords")
        warnings.append("final_report_lacks_failure_analysis")


def check_environment_yml(warnings: list[str]) -> None:
    path = ROOT / "environment.yml"
    if not path.exists():
        status_line(False, "environment.yml missing")
        warnings.append("environment_yml_missing")
        return
    try:
        import yaml
    except ImportError:
        status_line(False, "pyyaml not installed; cannot parse environment.yml")
        warnings.append("pyyaml_missing")
        return
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic script.
        status_line(False, f"environment.yml failed yaml.safe_load: {exc}")
        warnings.append("environment_yml_invalid")
        return
    ok = isinstance(data, dict) and isinstance(data.get("dependencies"), list)
    status_line(ok, "environment.yml parses as a conda environment")
    if not ok:
        warnings.append("environment_yml_wrong_shape")


def main() -> None:
    warnings: list[str] = []
    print("Final submission check")
    print("======================")
    check_required_files(warnings)
    check_eval_summary(warnings)
    check_final_report(warnings)
    check_environment_yml(warnings)
    artifacts = ROOT / "reports/MODEL_ARTIFACTS.md"
    status_line(artifacts.exists(), "reports/MODEL_ARTIFACTS.md is present")
    if not artifacts.exists():
        warnings.append("model_artifacts_missing")
    print("======================")
    if warnings:
        print(f"WARN: {len(warnings)} issue(s): {', '.join(warnings)}")
    else:
        print("PASS: final submission artifacts look consistent.")


if __name__ == "__main__":
    main()
