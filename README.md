# NLP LLM Milestone

本仓库用于 NLP 课程大作业：从零开始训练小规模 decoder-only Transformer，并完成预训练、指令微调、奖励模型与 PPO/RLHF 的 milestone 前可交付代码。

保留文件：

- `Proposal_241880230_庄一凡 (1).pdf`：proposal 原文
- `README.md`：当前工程说明

## 当前 Milestone 状态

- 已实现工程脚手架、配置、脚本入口、测试和报告生成器。
- 已实现 PyTorch 从零构建的 GPT-style decoder-only Transformer。
- 已实现 TinyStories tokenizer 训练、token bin 准备、预训练、验证 PPL、checkpoint 和曲线绘制逻辑。
- 已实现 Alpaca-Cleaned SFT 数据处理、response-only loss、冻结 embedding/前 N 层并默认只训练最后 2 层。
- 已实现 reward model pairwise ranking loss 和简化但真实可运行的 PPO debug 代码。
- 已实现 `advanced/lora.py` 和 `advanced/ppo_sampling.py` 接口。
- 长训指标、人工评分、安全率等未实际运行的结果均保持 `not yet run` 或空白，不伪造数值。

## 环境

推荐环境：

```bash
conda activate nlp
python -m pip install -r requirements.txt
```

如果需要重新创建环境：

```bash
conda env create -f environment.yml
conda activate nlp
```

说明：本项目默认 fp16 mixed precision，不默认使用 bf16。4 卡训练使用 PyTorch DDP。

## 目录

```text
configs/                 YAML 配置
src/nlp_llm/             核心 Python 包
scripts/                 训练、评估、绘图、报告脚本
tests/                   pytest 单元测试
advanced/                LoRA 与 PPO 采样优化接口
reports/                 milestone 报告和人工评估表
outputs/logs/            CSV 日志
outputs/figures/         PNG 曲线
outputs/examples/        生成样例
```

大数据、HF cache 和 checkpoint 默认被 `.gitignore` 忽略。

## 本地 Smoke Test

该命令使用脚本内创建的极小合成数据，不需要下载数据集：

```bash
bash scripts/run_smoke_tests.sh
```

它会依次运行：

- `python -m compileall src scripts advanced`
- `pytest -q`
- toy tokenizer / toy TinyStories bin / pretrain debug / SFT debug / reward debug / PPO debug
- 生成样例、绘图、更新 `reports/milestone.md`

## 真实数据准备

以下命令会通过 Hugging Face `datasets` 访问数据集，需要你手动在可联网环境运行：

```bash
python scripts/train_tokenizer.py --config configs/tokenizer.yaml
python scripts/prepare_tinystories.py --config configs/tokenizer.yaml
```

默认数据集：

- TinyStories：`roneneldan/TinyStories`
- Alpaca-Cleaned：`yahma/alpaca-cleaned`
- PKU-SafeRLHF debug：`PKU-Alignment/PKU-SafeRLHF-10K`

## 预训练

单卡/debug：

```bash
python scripts/pretrain.py --config configs/pretrain_debug.yaml
python scripts/evaluate_pretrain.py --checkpoint outputs/checkpoints/pretrain/best.pt
```

4 GPU：

```bash
torchrun --standalone --nproc_per_node=4 scripts/pretrain.py --config configs/pretrain_4gpu.yaml
```

生成样例：

```bash
python scripts/generate.py \
  --checkpoint outputs/checkpoints/pretrain/best.pt \
  --prompt "Once upon a time" \
  --max_new_tokens 100 \
  --out outputs/examples/pretrain_samples.jsonl
```

## 指令微调

默认 SFT 会加载预训练 checkpoint，冻结 token embedding、pos embedding 和前 `n_layer-2` 层，只训练最后 2 层、final ln 和 lm head；若预训练模型使用 tied embeddings，脚本会先复制 lm head 权重以保持 embedding 冻结。

```bash
torchrun --standalone --nproc_per_node=4 scripts/sft.py \
  --config configs/sft_4gpu.yaml \
  --pretrained outputs/checkpoints/pretrain/best.pt
```

生成 50 条人工评估表，`score` 默认空白，需要人工填写：

```bash
python scripts/evaluate_sft.py \
  --checkpoint outputs/checkpoints/sft/best.pt \
  --out reports/manual_eval_sft.csv
```

## Reward Model 和 PPO

Reward model debug：

```bash
python scripts/train_reward_model.py \
  --config configs/reward_debug.yaml \
  --sft_checkpoint outputs/checkpoints/sft/best.pt
```

PPO debug：

```bash
python scripts/ppo_train.py \
  --config configs/ppo_debug.yaml \
  --sft_checkpoint outputs/checkpoints/sft/best.pt \
  --reward_checkpoint outputs/checkpoints/reward/best.pt
```

安全评估表：

```bash
python scripts/evaluate_safety.py \
  --checkpoint outputs/checkpoints/ppo/last.pt \
  --fallback_checkpoint outputs/checkpoints/sft/best.pt \
  --out reports/safety_eval.csv
```

`heuristic_safe` 只是粗略自动标记，最终安全回答率必须人工确认。

## 绘图与报告

```bash
python scripts/plot_metrics.py
python scripts/make_milestone_report.py
```

报告输出到 `reports/milestone.md`。脚本只汇总已有真实 CSV/checkpoint/figure；缺失项显示为 `not yet run`。

## 完整 Milestone Pipeline

脚本已提供，但会安装依赖并下载数据。请你确认环境后手动执行：

```bash
bash scripts/run_milestone_pipeline.sh
```

如果依赖已经装好，不想让脚本执行安装：

```bash
SKIP_INSTALL=1 bash scripts/run_milestone_pipeline.sh
```

## 关键输出

- Pretrain logs：`outputs/logs/pretrain_metrics.csv`
- Pretrain figure：`outputs/figures/pretrain_loss_ppl.png`
- SFT logs：`outputs/logs/sft_metrics.csv`
- SFT manual eval：`reports/manual_eval_sft.csv`
- Reward logs：`outputs/logs/reward_metrics.csv`
- PPO logs：`outputs/logs/ppo_metrics.csv`
- Safety eval：`reports/safety_eval.csv`
- Milestone report：`reports/milestone.md`

## Final 阶段与补充消融

当前 final 主线已经跑通完整链路：final SFT、reward model、PPO/RLHF、人工评分表、安全评估表和 final report 草稿均已有。SFT 与 RLHF 质量仍弱，最终报告应如实报告该负结果。

查看 final 主线命令：

```bash
bash scripts/print_final_training_commands.sh
```

查看 proposal 对应的补充消融与进阶实验命令：

```bash
bash scripts/print_ablation_commands.sh
```

主要 final 配置：

- `configs/sft_final.yaml`
- `configs/reward_final.yaml`
- `configs/ppo_final.yaml`
- `configs/sft_lora_ablation.yaml`
- `configs/sft_data_ablation_2500.yaml`
- `configs/pretrain_small_ablation.yaml`
- `configs/sft_small_pretrain_ablation.yaml`

人工评分汇总：

```bash
python scripts/summarize_manual_eval.py \
  --sft reports/manual_eval_sft_final.csv \
  --safety reports/safety_eval_final.csv \
  --out reports/final_eval_summary.json
```

生成 final report 草稿：

```bash
python scripts/summarize_ablation_results.py
python scripts/make_final_report.py
```

相关输出：

- Final SFT logs：`outputs/logs/sft_final_metrics.csv`
- Final reward logs：`outputs/logs/reward_final_metrics.csv`
- Final PPO logs：`outputs/logs/ppo_final_metrics.csv`
- Final SFT manual eval：`reports/manual_eval_sft_final.csv`
- Final safety eval：`reports/safety_eval_final.csv`
- Final eval summary：`reports/final_eval_summary.json`
- Ablation summary：`reports/ablation_summary.md`
- Final report draft：`reports/final_report.md`
- Final slides outline：`reports/final_slides_outline.md`
