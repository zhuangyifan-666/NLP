#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

try:
    import yaml
except ImportError:  # pragma: no cover - environment.yml includes pyyaml.
    yaml = None


def read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if yaml is None or not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def fnum(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 4) -> str:
    num = fnum(value)
    if num is None:
        return "not run"
    return f"{num:.{digits}f}"


def fmt_percent(value: Any) -> str:
    num = fnum(value)
    if num is None:
        return "not run"
    return f"{num:.2f}%"


def last_row(path: str | Path) -> dict[str, str] | None:
    rows = read_csv(path)
    return rows[-1] if rows else None


def best_row(path: str | Path, key: str) -> dict[str, str] | None:
    rows = [r for r in read_csv(path) if fnum(r.get(key)) is not None]
    return min(rows, key=lambda r: fnum(r.get(key)) or float("inf")) if rows else None


def summarize_sft_csv(path: str | Path) -> dict[str, Any]:
    rows = read_csv(path)
    scores = [fnum(r.get("score")) for r in rows]
    filled = [s for s in scores if s is not None]
    return {
        "path": str(path),
        "total": len(rows),
        "filled": len(filled),
        "mean_score": sum(filled) / len(filled) if filled else None,
        "accuracy_percent": 100.0 * sum(filled) / len(filled) if filled else None,
    }


def summarize_safety_csv(path: str | Path) -> dict[str, Any]:
    rows = read_csv(path)
    labels = [(r.get("human_safe") or "").strip() for r in rows]
    filled = [v for v in labels if v]
    safe = sum(v in {"1", "yes", "true", "safe", "安全"} for v in filled)
    return {
        "path": str(path),
        "total": len(rows),
        "filled": len(filled),
        "safe_count": safe,
        "safe_rate_percent": 100.0 * safe / len(filled) if filled else None,
    }


def clean_response(text: str, limit: int = 420) -> str:
    text = (text or "").strip()
    if "Response:" in text:
        text = text.split("Response:", 1)[1].strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text or "(empty)"


def pick_examples(path: str | Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = read_csv(path)
    positives = [r for r in rows if (fnum(r.get("score")) or 0.0) > 0]
    failures = [r for r in rows if (fnum(r.get("score")) or 0.0) == 0]
    return positives[:3], failures[:5]


def cfg_line(cfg: dict[str, Any], keys: list[str]) -> str:
    parts = []
    for key in keys:
        cur: Any = cfg
        for part in key.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        parts.append(f"{key}={cur if cur is not None else 'not set'}")
    return "; ".join(parts)


def ablation_table(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["| 实验 | 训练指标 | 人工/安全指标 | 备注 |", "|---|---|---|---|", "| ablation | not run | not run | 未生成 ablation_summary.json |"]
    logs = {item["name"]: item for item in summary.get("training_logs", [])}
    manual = {item["name"]: item for item in summary.get("manual_sft", [])}
    safety = {item["name"]: item for item in summary.get("safety", [])}
    rows = ["| 实验 | 训练指标 | 人工/安全指标 | 备注 |", "|---|---|---|---|"]
    specs = [
        ("pretrain_base", "Base pretrain", "基础预训练模型"),
        ("pretrain_small", "Small pretrain", "小模型规模消融"),
        ("sft_final_10000", "Final SFT", "10000 Alpaca 样本，last-4-layer 扩展实验"),
        ("sft_data_2500", "SFT data 2500", "数据量消融"),
        ("sft_lora", "LoRA SFT", "adapter-only 进阶实验"),
        ("sft_small_pretrain", "Small pretrain + SFT", "小模型下游消融"),
        ("ppo_final", "PPO final", "最终 PPO/RLHF"),
    ]
    for key, label, note in specs:
        item = logs.get(key, {})
        last = item.get("last_row") or {}
        if "val_ppl" in last:
            train_metric = f"val_loss={fmt(last.get('val_loss'))}, PPL={fmt(last.get('val_ppl'))}"
        elif key == "ppo_final":
            train_metric = f"reward={fmt(last.get('reward_mean'))}, KL={fmt(last.get('kl_mean'))}"
        elif last:
            train_metric = f"val_loss={fmt(last.get('val_loss'))}"
        else:
            train_metric = "not run"
        if key in manual:
            eval_metric = f"manual={fmt_percent(manual[key].get('accuracy_percent'))}"
        elif key == "ppo_final" and "ppo_final" in safety:
            eval_metric = f"safety={fmt_percent(safety['ppo_final'].get('human_safe_rate_percent'))}"
        else:
            eval_metric = "-"
        rows.append(f"| {label} | {train_metric} | {eval_metric} | {note} |")
    return rows


def format_examples(title: str, examples: list[dict[str, str]]) -> list[str]:
    lines = [f"**{title}**"]
    if not examples:
        return lines + ["- 未观察到稳定成功样例。"]
    for row in examples:
        lines.append(
            f"- score={row.get('score', '')}, category={row.get('category', '')}, prompt={row.get('prompt', '')!r}; "
            f"response: {clean_response(row.get('model_response', ''))}"
        )
    return lines


def main() -> None:
    pre_final = last_row("outputs/logs/pretrain_metrics.csv")
    pre_best = best_row("outputs/logs/pretrain_metrics.csv", "val_loss")
    sft_base = last_row("outputs/logs/sft_metrics.csv")
    sft_final = last_row("outputs/logs/sft_final_metrics.csv")
    reward_final = last_row("outputs/logs/reward_final_metrics.csv")
    ppo_final = last_row("outputs/logs/ppo_final_metrics.csv")
    eval_summary = read_json("reports/final_eval_summary.json") or {}
    ablation_summary = read_json("reports/ablation_summary.json")
    sft_final_summary = eval_summary.get("sft") or summarize_sft_csv("reports/manual_eval_sft_final.csv")
    safety_final_summary = eval_summary.get("safety") or summarize_safety_csv("reports/safety_eval_final.csv")
    safety_rate = safety_final_summary.get("human_safe_rate_percent", safety_final_summary.get("safe_rate_percent"))
    positives, failures = pick_examples("reports/manual_eval_sft_final.csv")

    tokenizer_cfg = read_yaml("configs/tokenizer.yaml")
    pre_cfg = read_yaml("configs/pretrain_4gpu.yaml")
    sft_base_cfg = read_yaml("configs/sft_4gpu.yaml")
    sft_final_cfg = read_yaml("configs/sft_final.yaml")
    reward_cfg = read_yaml("configs/reward_final.yaml")
    ppo_cfg = read_yaml("configs/ppo_final.yaml")

    lines: list[str] = [
        "# 从零开始训练小规模对话模型：Final Report",
        "",
        "## 1. 项目概述",
        "",
        "本项目为 NLP 课程大作业，目标是在有限算力下完整实现一个简化版 LLM 训练与对齐流程。项目由本人单人完成，代码主体使用 PyTorch 自实现，不加载外部 pretrained language model 权重。",
        "",
        "整体流程包含四部分：",
        "",
        "- TinyStories 预训练：训练小规模 decoder-only Transformer 的 next-token prediction 能力。",
        "- Alpaca-Cleaned 指令微调：使用 Alpaca prompt template 和 response-only loss 进行 SFT。",
        "- PKU-SafeRLHF reward model + PPO/RLHF：训练 pairwise reward model，并用 PPO 做安全对齐尝试。",
        "- Advanced：实现 LoRA 参数高效微调接口，并实现 PPO safety prompt oversampling、动态 KL beta 等采样优化。",
        "",
        "需要强调的是，本项目的预训练指标达到课程目标；但 SFT/RLHF 的实际对话与安全效果较弱，报告中按负结果如实分析。",
        "",
        "## 2. 模型与实现",
        "",
        "- **Tokenizer**：基于 TinyStories 训练 BPE tokenizer，默认词表大小为 `16000`，包含 BOS/EOS/PAD/UNK 等特殊 token。配置见 `configs/tokenizer.yaml`，产物位于 `outputs/tokenizer/tokenizer.json`。",
        "- **Backbone**：GPT-style decoder-only Transformer，自实现 causal self-attention、MLP、LayerNorm、位置嵌入和 tied word embeddings。主模型配置为 6 层、512 hidden size、8 heads、block size 512。",
        "- **预训练目标**：标准 next-token prediction，使用 TinyStories token bin 训练并在 validation split 上计算 loss/PPL。",
        "- **SFT**：使用 Alpaca instruction/input/response 模板，loss mask 只覆盖 response tokens，即 response-only loss。",
        "- **冻结策略**：课程 baseline 是冻结 token embedding、pos embedding 和前 `n_layer-2` 层，只训练最后 2 层、final layer norm 和 lm_head。`configs/sft_4gpu.yaml` 对应该设置。`configs/sft_final.yaml` 中 `train_last_n_layers: 4` 是 final 阶段为提升效果进行的扩展实验，不能等同于课程 baseline。",
        "- **Reward model**：在 GPT backbone 顶部加入 scalar reward head，使用 chosen/rejected pairwise ranking loss 训练。",
        "- **PPO/RLHF**：包含 policy model、frozen reference model、frozen reward model、value head、KL penalty 和 clipped policy objective。rollout 中的 old logprob、reference logprob 和 reward 已显式 detach，PPO epoch 内只更新当前 policy 与 value head。",
        "- **Advanced**：LoRA 支持 attention projection adapter 注入和 checkpoint 加载；PPO sampling hooks 支持安全 prompt 过采样、长度限制、rollout reuse 和动态 KL beta。",
        "",
        "## 3. 实验设置",
        "",
        "- **环境**：`conda activate nlp`，PyTorch，fp16 mixed precision，目标硬件为 4 x RTX 3090。",
        "- **数据集**：TinyStories (`roneneldan/TinyStories`)、Alpaca-Cleaned (`yahma/alpaca-cleaned`)、PKU-SafeRLHF (`PKU-Alignment/PKU-SafeRLHF-10K`)。",
        f"- **Tokenizer 配置**：`configs/tokenizer.yaml`; {cfg_line(tokenizer_cfg, ['vocab_size', 'dataset'])}。",
        f"- **Pretrain 配置**：`configs/pretrain_4gpu.yaml`; {cfg_line(pre_cfg, ['model.n_layer', 'model.n_embd', 'model.n_head', 'max_steps', 'learning_rate', 'per_device_batch_size', 'grad_accum_steps'])}。",
        f"- **SFT baseline 配置**：`configs/sft_4gpu.yaml`; {cfg_line(sft_base_cfg, ['data.max_samples', 'train_last_n_layers', 'max_steps', 'learning_rate', 'per_device_batch_size', 'grad_accum_steps'])}。",
        f"- **SFT final 扩展配置**：`configs/sft_final.yaml`; {cfg_line(sft_final_cfg, ['data.max_samples', 'train_last_n_layers', 'max_steps', 'learning_rate', 'per_device_batch_size', 'grad_accum_steps'])}。",
        f"- **Reward 配置**：`configs/reward_final.yaml`; {cfg_line(reward_cfg, ['data.max_samples', 'max_steps', 'learning_rate', 'batch_size'])}。",
        f"- **PPO 配置**：`configs/ppo_final.yaml`; {cfg_line(ppo_cfg, ['data.max_samples', 'max_steps', 'batch_size', 'ppo_epochs', 'kl_beta', 'sampling.safety_prompt_ratio'])}。",
        "",
        "## 4. 预训练结果",
        "",
        "| 指标 | best | final |",
        "|---|---:|---:|",
        f"| validation loss | {fmt(pre_best.get('val_loss') if pre_best else None)} | {fmt(pre_final.get('val_loss') if pre_final else None)} |",
        f"| validation PPL | {fmt(pre_best.get('val_ppl') if pre_best else None)} | {fmt(pre_final.get('val_ppl') if pre_final else None)} |",
        "",
        "曲线文件：`outputs/figures/pretrain_loss_ppl.png`。",
        "",
        f"课程目标要求 PPL < 50；本项目 final PPL 为 {fmt(pre_final.get('val_ppl') if pre_final else None)}，明显达到要求。该结果说明模型在 TinyStories 域内具备较好的语言建模能力。但 TinyStories 主要是儿童故事语域，低 PPL 不代表模型已经具备通用问答、推理和安全拒答能力。",
        "",
        "## 5. 指令微调结果",
        "",
        "SFT 曲线文件：`outputs/figures/sft_final_loss.png`。50 条人工评估表：`reports/manual_eval_sft_final.csv`。",
        "",
        "| 实验 | 冻结策略 | final val_loss | manual accuracy | 说明 |",
        "|---|---|---:|---:|---|",
        f"| SFT baseline | last 2 layers | {fmt(sft_base.get('val_loss') if sft_base else None)} | {fmt_percent(summarize_sft_csv('reports/manual_eval_sft.csv').get('accuracy_percent'))} | 课程要求 baseline，配置 `configs/sft_4gpu.yaml` |",
        f"| SFT final | last 4 layers | {fmt(sft_final.get('val_loss') if sft_final else None)} | {fmt_percent(sft_final_summary.get('accuracy_percent'))} | 扩展实验，配置 `configs/sft_final.yaml` |",
        "",
        f"Final SFT 的人工平均分为 {fmt(sft_final_summary.get('mean_score'))}，accuracy 百分比为 {fmt_percent(sft_final_summary.get('accuracy_percent'))}。该结果较低，不能粉饰为成功对话模型；它更适合作为小模型从故事域迁移到 instruction following 时的负结果。",
        "",
        *format_examples("相对较好的样例（仍然只是部分完成）", positives),
        "",
        *format_examples("代表性失败样例", failures),
        "",
        "失败原因主要包括：TinyStories 预训练域较窄；模型规模小；SFT 数据和训练预算有限；只微调后几层限制了分布迁移能力；英文故事预训练到多样 instruction 的迁移困难；采样解码中仍出现重复、跑题和幻觉。",
        "",
        "## 6. Reward Model 与 PPO/RLHF 结果",
        "",
        "Reward 曲线文件：`outputs/figures/reward_final_loss_acc.png`。PPO 曲线文件：`outputs/figures/ppo_final_metrics.png`。安全评估表：`reports/safety_eval_final.csv`。",
        "",
        "| 模块 | 关键指标 | 数值 |",
        "|---|---|---:|",
        f"| Reward model | final val_loss | {fmt(reward_final.get('val_loss') if reward_final else None)} |",
        f"| Reward model | final pairwise accuracy | {fmt(reward_final.get('pairwise_acc') if reward_final else None)} |",
        f"| PPO | final reward_mean | {fmt(ppo_final.get('reward_mean') if ppo_final else None)} |",
        f"| PPO | final KL mean | {fmt(ppo_final.get('kl_mean') if ppo_final else None)} |",
        f"| Safety eval | human safe rate | {fmt_percent(safety_rate)} |",
        "",
        f"安全回答率为 {fmt_percent(safety_rate)}，是明确的负结果。Reward model 的 pairwise accuracy 约为 {fmt(reward_final.get('pairwise_acc') if reward_final else None)}，接近随机，难以为 PPO 提供可靠优化信号。PPO 只进行了小规模 final run，KL 控制和 reward 优化仍不稳定；同时 SFT 初始对话能力弱，RLHF 无法弥补基础模型能力不足。安全评估 prompt 数量较少，且需要人工确认，因此不能夸大结论。",
        "",
        "## 7. Advanced 与消融实验",
        "",
        "消融汇总文件：`reports/ablation_summary.md` / `reports/ablation_summary.json`。",
        "",
        *ablation_table(ablation_summary),
        "",
        "LoRA 的优点是只训练少量 adapter 参数，理论上可降低显存和存储成本。本项目已实现 LoRA 注入、LoRA checkpoint 加载和 LoRA SFT ablation；但当前 LoRA manual accuracy 只有 7.00%，低于 final SFT。可能原因包括 base model 本身弱、rank 较小、训练信号不足，以及只调 attention projection 难以弥补 instruction following 缺陷。",
        "",
        "PPO sampling 优化已实现安全 prompt 过采样、response 长度限制、同一 rollout 多个 PPO epoch 复用，以及动态 KL beta。当前安全率仍为 0.00%，说明流程实现并不等于效果可靠；后续需要更强 SFT、reward normalization、更高质量 reward model 和更系统的安全数据。",
        "",
        "未运行的方向包括 DPO、rejection sampling fine-tuning 和显存/吞吐量的严格 profile；这些仅作为未来工作，不在本次结果中伪造。",
        "",
        "## 8. 问题、解决方案与反思",
        "",
        "- **预训练成功但 SFT/RLHF 失败**：TinyStories PPL 低说明故事域 next-token modeling 成功，但 instruction following、事实问答和安全拒答是不同能力。",
        "- **低 PPL 不等于好对话能力**：模型能生成流畅故事片段，但面对 QA、reasoning、summary 时经常跑题。",
        "- **小模型 RLHF 困难**：reward model 准确率接近随机时，PPO 优化方向噪声很大；policy 本身弱时，RLHF 容易强化格式或重复模式，而不是安全能力。",
        "- **人工评估必要**：自动 loss/reward 指标无法反映回答是否有用、安全或忠实。最终人工评分揭示了模型真实可用性较低。",
        "- **未来改进**：增大模型和上下文窗口；使用更通用的预训练语料；增加 SFT 数据与步数；提升 reward model 数据规模和准确率；调节 PPO KL beta、reward normalization 和采样策略；尝试 DPO 或 rejection sampling fine-tuning。",
        "",
        "## 9. 复现方式",
        "",
        "环境安装：",
        "",
        "```bash",
        "conda env create -f environment.yml",
        "conda activate nlp",
        "```",
        "",
        "轻量测试：",
        "",
        "```bash",
        "python -m compileall src scripts advanced",
        "pytest -q",
        "python scripts/final_submission_check.py",
        "```",
        "",
        "真实数据和训练命令：",
        "",
        "```bash",
        "python scripts/train_tokenizer.py --config configs/tokenizer.yaml",
        "python scripts/prepare_tinystories.py --config configs/tokenizer.yaml",
        "torchrun --standalone --nproc_per_node=4 scripts/pretrain.py --config configs/pretrain_4gpu.yaml",
        "torchrun --standalone --nproc_per_node=4 scripts/sft.py --config configs/sft_4gpu.yaml --pretrained outputs/checkpoints/pretrain/best.pt",
        "torchrun --standalone --nproc_per_node=4 scripts/sft.py --config configs/sft_final.yaml --pretrained outputs/checkpoints/pretrain/best.pt",
        "python scripts/train_reward_model.py --config configs/reward_final.yaml --sft_checkpoint outputs/checkpoints/sft_final/best.pt",
        "python scripts/ppo_train.py --config configs/ppo_final.yaml --sft_checkpoint outputs/checkpoints/sft_final/best.pt --reward_checkpoint outputs/checkpoints/reward_final/best.pt",
        "python scripts/evaluate_sft.py --checkpoint outputs/checkpoints/sft_final/best.pt --out reports/manual_eval_sft_final.csv --max_new_tokens 96",
        "python scripts/evaluate_safety.py --checkpoint outputs/checkpoints/ppo_final/last.pt --fallback_checkpoint outputs/checkpoints/sft_final/best.pt --out reports/safety_eval_final.csv --max_new_tokens 96",
        "python scripts/summarize_manual_eval.py --sft reports/manual_eval_sft_final.csv --safety reports/safety_eval_final.csv --out reports/final_eval_summary.json",
        "python scripts/summarize_ablation_results.py",
        "python scripts/make_final_report.py",
        "```",
        "",
        "长训需要本地数据、GPU 和 checkpoint。GitHub 主要提交代码、配置、日志、图和报告；大 checkpoint 默认被 `.gitignore` 忽略，见 `reports/MODEL_ARTIFACTS.md`。",
        "",
        "## 10. 结论",
        "",
        "本项目完整实现了 TinyStories 预训练、Alpaca SFT、PKU-SafeRLHF reward model 与 PPO/RLHF 的端到端代码，并补充了 LoRA 与 PPO sampling 的 advanced 方案。预训练 PPL 明确达到课程目标；SFT/RLHF 结果未达到理想对话和安全目标，但保留了完整日志、人工评估、负结果分析和消融对比。最终提交应将该项目定位为“小规模从零训练与对齐流程复现 + 诚实负结果分析”，而不是夸大为可用对话模型。",
        "",
        "## 参考文献",
        "",
        "1. Vaswani et al. Attention Is All You Need. 2017.",
        "2. Radford et al. Improving Language Understanding by Generative Pre-Training. 2018.",
        "3. Brown et al. Language Models are Few-Shot Learners. 2020.",
        "4. Eldan and Li. TinyStories: How Small Can Language Models Be and Still Speak Coherent English? 2023.",
        "5. Taori et al. Alpaca: A Strong, Replicable Instruction-Following Model. 2023.",
        "6. Christiano et al. Deep Reinforcement Learning from Human Preferences. 2017.",
        "7. Ouyang et al. Training language models to follow instructions with human feedback. 2022.",
        "8. Schulman et al. Proximal Policy Optimization Algorithms. 2017.",
        "9. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. 2021.",
        "10. Ji et al. PKU-SafeRLHF: A Safety Alignment Preference Dataset for Llama Family Models. 2024.",
        "11. Hugging Face datasets, tokenizers, transformers documentation.",
        "",
    ]

    out = Path("reports/final_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
