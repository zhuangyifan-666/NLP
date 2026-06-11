# Final 答辩 PPT 大纲

## Slide 1. 标题页

**Key bullets**

- 从零开始训练小规模对话模型
- TinyStories 预训练 + Alpaca SFT + PKU-SafeRLHF Reward/PPO
- 单人项目，4 x RTX 3090 环境完成主要训练与评估

**建议图表**

- 可放项目流程图，或 `outputs/figures/pretrain_loss_ppl.png` 的缩略图作为背景元素。

**演讲备注**

- 开场明确本项目不是调用外部 pretrained LM，而是用自实现 GPT-style backbone 跑通完整训练与对齐流程。

## Slide 2. 任务要求与三阶段流程

**Key bullets**

- 课程要求：TinyStories PPL < 50，Alpaca 指令微调，PKU-SafeRLHF reward model + PPO。
- 额外 advanced：LoRA 与 PPO sampling 优化。
- 输出材料：代码、配置、日志、曲线、人工评估表、final report、slides outline。

**建议图表**

- 三阶段流程图：Pretrain -> SFT -> Reward Model -> PPO -> Evaluation。

**演讲备注**

- 强调所有指标来自已有 CSV/JSON/人工评分表，未运行内容不伪造。

## Slide 3. 模型结构

**Key bullets**

- 自实现 decoder-only Transformer / GPT-style LM。
- 组件：BPE tokenizer、token/position embedding、causal self-attention、MLP、LayerNorm、lm_head。
- reward model 复用 backbone 并添加 scalar reward head。
- PPO 使用 policy、frozen reference、frozen reward model 和 value head。

**建议图表**

- 模型结构框图；可引用代码路径 `src/nlp_llm/model.py`、`src/nlp_llm/reward_model.py`、`src/nlp_llm/ppo.py`。

**演讲备注**

- 说明实现重点在训练流程完整性和可复现性，而不是模型规模。

## Slide 4. 数据集与 Tokenizer

**Key bullets**

- TinyStories：用于从零预训练。
- Alpaca-Cleaned：instruction/input/response 格式，SFT 使用 response-only loss。
- PKU-SafeRLHF：chosen/rejected pairwise 数据训练 reward model，prompt 用于 PPO。
- tokenizer：在 TinyStories 上训练 BPE，并包含 BOS/EOS/PAD/UNK。

**建议图表**

- 数据流图或配置路径列表：`configs/tokenizer.yaml`、`configs/sft_final.yaml`、`configs/reward_final.yaml`、`configs/ppo_final.yaml`。

**演讲备注**

- 点出数据域差异：故事语料预训练迁移到开放指令任务有明显难度。

## Slide 5. TinyStories 预训练结果

**Key bullets**

- 预训练 validation PPL 达到课程目标 PPL < 50。
- final validation loss/PPL 来自 `outputs/logs/pretrain_metrics.csv`。
- 低 PPL 说明故事域语言建模成功，但不等于通用对话能力强。

**建议图表**

- `outputs/figures/pretrain_loss_ppl.png`

**演讲备注**

- 建议展示 best/final PPL 数值，并明确这是本项目最稳定达标的部分。

## Slide 6. Alpaca SFT 方法与结果

**Key bullets**

- baseline：冻结 embedding、pos embedding、前 n_layer-2 层，仅训练最后 2 层、final LN、lm_head。
- final 扩展实验：`configs/sft_final.yaml` 使用 last-4-layer trainable，用于探索效果提升。
- 人工评估 50 条：`reports/manual_eval_sft_final.csv`。
- SFT manual accuracy 较低，是负结果。

**建议图表**

- `outputs/figures/sft_final_loss.png`
- 人工评估类别均值表，可从 `reports/final_eval_summary.json` 摘取。

**演讲备注**

- 必须区分课程 baseline 和 final 扩展实验，避免被理解为 baseline 改成了 last 4 layers。

## Slide 7. SFT 样例分析

**Key bullets**

- 展示 2-3 个相对部分成功样例。
- 展示 3-5 个失败样例：跑题、重复、格式不稳、回答不完整。
- 失败原因：小模型、预训练域窄、SFT 预算有限、只微调后几层、解码策略不稳。

**建议图表**

- 从 `reports/manual_eval_sft_final.csv` 截取样例表格。

**演讲备注**

- 这一页用来证明报告不是只看 loss，而是做了人工质量检查。

## Slide 8. Reward Model 与 PPO 方法

**Key bullets**

- reward model：chosen/rejected pairwise ranking loss。
- PPO：old logprob、reference logprob、reward 在 rollout 阶段 detach。
- reference model 和 reward model frozen/eval；policy 与 value head 参与更新。
- 记录 reward、KL、policy loss、value loss、entropy、clipfrac。

**建议图表**

- `outputs/figures/reward_final_loss_acc.png`
- PPO 算法流程图或 `scripts/ppo_train.py` 关键逻辑摘要。

**演讲备注**

- 可以说明最终修复了 old/ref/reward 计算图复用风险，代码现在更符合 PPO 语义。

## Slide 9. Safety Eval 与 RLHF 结果

**Key bullets**

- PPO 曲线来自 `outputs/figures/ppo_final_metrics.png`。
- safety eval 表：`reports/safety_eval_final.csv`。
- human safe rate 为低/零结果，不能夸大为成功对齐。
- reward accuracy 接近随机时，PPO 信号质量不足。

**建议图表**

- `outputs/figures/ppo_final_metrics.png`
- 安全评估汇总表：safe count / total。

**演讲备注**

- 强调“流程跑通”和“安全效果可靠”是两件事，本项目诚实报告负结果。

## Slide 10. Advanced：LoRA 与 PPO Sampling

**Key bullets**

- LoRA：注入 attention projection adapter，只训练 adapter 参数。
- PPO sampling：安全 prompt 过采样、长度限制、rollout reuse、动态 KL beta。
- 当前结果显示 LoRA 与 PPO sampling 没有显著提升最终人工/安全指标。
- 价值在于接口完整，可作为后续扩展基础。

**建议图表**

- `advanced/README.md`
- `reports/ablation_summary.md` 中 LoRA 与 PPO 行。

**演讲备注**

- 不要把 advanced 说成已经解决问题，只说实现了方案并做了初步消融。

## Slide 11. 消融实验

**Key bullets**

- base pretrain vs small pretrain。
- final SFT vs 2500 data ablation。
- LoRA SFT ablation。
- PPO final 与 safety eval。
- 指标：loss/PPL/manual accuracy/safety rate。

**建议图表**

- `reports/ablation_summary.md`
- 可制作一张紧凑对比表。

**演讲备注**

- 解释消融结论：预训练达标不直接保证 SFT；LoRA 节省参数但当前质量弱；小模型和数据预算是主要瓶颈。

## Slide 12. 总结与未来工作

**Key bullets**

- 完整实现三阶段训练代码和 advanced 接口。
- 预训练 PPL 达标。
- SFT/RLHF 效果未达理想目标，但提供了完整负结果、人工评估和消融分析。
- 未来方向：更通用预训练数据、更大模型/上下文、更强 SFT、更可靠 reward model、DPO/rejection sampling。

**建议图表**

- 最终结果汇总表：pretrain PPL、SFT accuracy、reward pairwise accuracy、PPO reward/KL、safety safe rate。

**演讲备注**

- 收束到课程价值：该项目展示了从零训练和对齐流程中的工程实现、实验失败原因和可复现材料。
