# 从零开始训练小规模对话模型：Final Report

## 1. 项目概述

本项目为 NLP 课程大作业，目标是在有限算力下完整实现一个简化版 LLM 训练与对齐流程。项目由本人单人完成，代码主体使用 PyTorch 自实现，不加载外部 pretrained language model 权重。

整体流程包含四部分：

- TinyStories 预训练：训练小规模 decoder-only Transformer 的 next-token prediction 能力。
- Alpaca-Cleaned 指令微调：使用 Alpaca prompt template 和 response-only loss 进行 SFT。
- PKU-SafeRLHF reward model + PPO/RLHF：训练 pairwise reward model，并用 PPO 做安全对齐尝试。
- Advanced：实现 LoRA 参数高效微调接口，并实现 PPO safety prompt oversampling、动态 KL beta 等采样优化。

需要强调的是，本项目的预训练指标达到课程目标；但 SFT/RLHF 的实际对话与安全效果较弱，报告中按负结果如实分析。

## 2. 模型与实现

- **Tokenizer**：基于 TinyStories 训练 BPE tokenizer，默认词表大小为 `16000`，包含 BOS/EOS/PAD/UNK 等特殊 token。配置见 `configs/tokenizer.yaml`，产物位于 `outputs/tokenizer/tokenizer.json`。
- **Backbone**：GPT-style decoder-only Transformer，自实现 causal self-attention、MLP、LayerNorm、位置嵌入和 tied word embeddings。主模型配置为 6 层、512 hidden size、8 heads、block size 512。
- **预训练目标**：标准 next-token prediction，使用 TinyStories token bin 训练并在 validation split 上计算 loss/PPL。
- **SFT**：使用 Alpaca instruction/input/response 模板，loss mask 只覆盖 response tokens，即 response-only loss。
- **冻结策略**：课程 baseline 是冻结 token embedding、pos embedding 和前 `n_layer-2` 层，只训练最后 2 层、final layer norm 和 lm_head。`configs/sft_4gpu.yaml` 对应该设置。`configs/sft_final.yaml` 中 `train_last_n_layers: 4` 是 final 阶段为提升效果进行的扩展实验，不能等同于课程 baseline。
- **Reward model**：在 GPT backbone 顶部加入 scalar reward head，使用 chosen/rejected pairwise ranking loss 训练。
- **PPO/RLHF**：包含 policy model、frozen reference model、frozen reward model、value head、KL penalty 和 clipped policy objective。rollout 中的 old logprob、reference logprob 和 reward 已显式 detach，PPO epoch 内只更新当前 policy 与 value head。
- **Advanced**：LoRA 支持 attention projection adapter 注入和 checkpoint 加载；PPO sampling hooks 支持安全 prompt 过采样、长度限制、rollout reuse 和动态 KL beta。

## 3. 实验设置

- **环境**：`conda activate nlp`，PyTorch，fp16 mixed precision，目标硬件为 4 x RTX 3090。
- **数据集**：TinyStories (`roneneldan/TinyStories`)、Alpaca-Cleaned (`yahma/alpaca-cleaned`)、PKU-SafeRLHF (`PKU-Alignment/PKU-SafeRLHF-10K`)。
- **Tokenizer 配置**：`configs/tokenizer.yaml`; vocab_size=16000; dataset=roneneldan/TinyStories。
- **Pretrain 配置**：`configs/pretrain_4gpu.yaml`; model.n_layer=6; model.n_embd=512; model.n_head=8; max_steps=10000; learning_rate=0.0003; per_device_batch_size=12; grad_accum_steps=4。
- **SFT baseline 配置**：`configs/sft_4gpu.yaml`; data.max_samples=5000; train_last_n_layers=2; max_steps=2000; learning_rate=0.0001; per_device_batch_size=8; grad_accum_steps=4。
- **SFT final 扩展配置**：`configs/sft_final.yaml`; data.max_samples=10000; train_last_n_layers=4; max_steps=4000; learning_rate=0.00015; per_device_batch_size=8; grad_accum_steps=4。
- **Reward 配置**：`configs/reward_final.yaml`; data.max_samples=5000; max_steps=1000; learning_rate=5e-06; batch_size=4。
- **PPO 配置**：`configs/ppo_final.yaml`; data.max_samples=512; max_steps=200; batch_size=2; ppo_epochs=2; kl_beta=0.05; sampling.safety_prompt_ratio=0.6。

## 4. 预训练结果

| 指标 | best | final |
|---|---:|---:|
| validation loss | 1.7143 | 1.7143 |
| validation PPL | 5.5530 | 5.5530 |

曲线文件：`outputs/figures/pretrain_loss_ppl.png`。

课程目标要求 PPL < 50；本项目 final PPL 为 5.5530，明显达到要求。该结果说明模型在 TinyStories 域内具备较好的语言建模能力。但 TinyStories 主要是儿童故事语域，低 PPL 不代表模型已经具备通用问答、推理和安全拒答能力。

## 5. 指令微调结果

SFT 曲线文件：`outputs/figures/sft_final_loss.png`。50 条人工评估表：`reports/manual_eval_sft_final.csv`。

| 实验 | 冻结策略 | final val_loss | manual accuracy | 说明 |
|---|---|---:|---:|---|
| SFT baseline | last 2 layers | 3.9314 | 11.00% | 课程要求 baseline，配置 `configs/sft_4gpu.yaml` |
| SFT final | last 4 layers | 3.1981 | 12.00% | 扩展实验，配置 `configs/sft_final.yaml` |

Final SFT 的人工平均分为 0.1200，accuracy 百分比为 12.00%。该结果较低，不能粉饰为成功对话模型；它更适合作为小模型从故事域迁移到 instruction following 时的负结果。

**相对较好的样例（仍然只是部分完成）**
- score=0.5, category=writing, prompt='Write a short story about friendship.'; response: Once upon a time, there was a young girl named Lily. She lived in a small village with her family. One night, she felt lonely and couldn't help but wonder about her dreams. When her mom said that she would miss a new night by her side, and she felt scared. She did not want to miss out on her own. Lily spent her days with her family, from school to school. She watched the TV and listened with her friends, and
- score=0.5, category=writing, prompt='Write a short story about friendship.'; response: Once upon a time, there was a little girl named Lily who loved to explore the world. One day, she found herself in a small stream where she found herself lost by the streamsion. She was no longer afraid of it. She was fascinated by the stream, asking her where she was going when she found herself lost from the streams. As she explored the stream, she met a new stream where she was stuck from the stream. The stream...
- score=0.5, category=summary, prompt='Summarize the text.'; response: The text is a process that involves relaxation sleep and relaxation. They include creating catiovas, sharing, and sharing, and the amount of sleep, and the energy that will make them a great choice. These three ways to stay connected with each other is the way we communicate, work, and connect with each other.

**代表性失败样例**
- score=0, category=qa, prompt='What is gravity?'; response: Ob, also known as Arx, is a form of gravity, that is known as the Mapolallo, which is the first major with his name, the second majorousian, and the second term, is 222-608, 02-565, where 55 is the second considered s was 15. To create a more revenue, we could all make revenue
- score=0, category=explain, prompt='Explain why the sky is blue in simple words.'; response: The sky is blue and blue in the sky, making blue and dark lines.
- score=0, category=summary, prompt='Summarize the text.'; response: The text is a subjective sentence, which means that it's best to be vig or adapted aspects.
- score=0, category=reasoning, prompt='If A is taller than B and B is taller than C, who is tallest?'; response: The Big B is an AI language model and do not have been able to create inno relationships based on how many people have. However, B is a subset of machine that is designed to be a vitalal development of several people, including the development of the United States, the Jemb, and the B is better known for historical and overviewy.
- score=0, category=qa, prompt='What is gravity?'; response: A gravity is the Network of the United States, which is a documentational Detworkb, and It is a type of artificial, that has a base, with a base for 7, 10, 15, 99,9,9868, 98,8,0,84,0000,000. It is the Network of the United States and was

失败原因主要包括：TinyStories 预训练域较窄；模型规模小；SFT 数据和训练预算有限；只微调后几层限制了分布迁移能力；英文故事预训练到多样 instruction 的迁移困难；采样解码中仍出现重复、跑题和幻觉。

## 6. Reward Model 与 PPO/RLHF 结果

Reward 曲线文件：`outputs/figures/reward_final_loss_acc.png`。PPO 曲线文件：`outputs/figures/ppo_final_metrics.png`。安全评估表：`reports/safety_eval_final.csv`。

| 模块 | 关键指标 | 数值 |
|---|---|---:|
| Reward model | final val_loss | 0.6742 |
| Reward model | final pairwise accuracy | 0.5580 |
| PPO | final reward_mean | 0.4228 |
| PPO | final KL mean | 1.0127 |
| Safety eval | human safe rate | 0.00% |

安全回答率为 0.00%，是明确的负结果。Reward model 的 pairwise accuracy 约为 0.5580，接近随机，难以为 PPO 提供可靠优化信号。PPO 只进行了小规模 final run，KL 控制和 reward 优化仍不稳定；同时 SFT 初始对话能力弱，RLHF 无法弥补基础模型能力不足。安全评估 prompt 数量较少，且需要人工确认，因此不能夸大结论。

## 7. Advanced 与消融实验

消融汇总文件：`reports/ablation_summary.md` / `reports/ablation_summary.json`。

| 实验 | 训练指标 | 人工/安全指标 | 备注 |
|---|---|---|---|
| Base pretrain | val_loss=1.7143, PPL=5.5530 | - | 基础预训练模型 |
| Small pretrain | val_loss=1.8521, PPL=6.3733 | - | 小模型规模消融 |
| Final SFT | val_loss=3.1981 | manual=12.00% | 10000 Alpaca 样本，last-4-layer 扩展实验 |
| SFT data 2500 | val_loss=4.3469 | manual=14.00% | 数据量消融 |
| LoRA SFT | val_loss=5.1442 | manual=7.00% | adapter-only 进阶实验 |
| Small pretrain + SFT | val_loss=3.3098 | manual=13.00% | 小模型下游消融 |
| PPO final | reward=0.4228, KL=1.0127 | safety=0.00% | 最终 PPO/RLHF |

LoRA 的优点是只训练少量 adapter 参数，理论上可降低显存和存储成本。本项目已实现 LoRA 注入、LoRA checkpoint 加载和 LoRA SFT ablation；但当前 LoRA manual accuracy 只有 7.00%，低于 final SFT。可能原因包括 base model 本身弱、rank 较小、训练信号不足，以及只调 attention projection 难以弥补 instruction following 缺陷。

PPO sampling 优化已实现安全 prompt 过采样、response 长度限制、同一 rollout 多个 PPO epoch 复用，以及动态 KL beta。当前安全率仍为 0.00%，说明流程实现并不等于效果可靠；后续需要更强 SFT、reward normalization、更高质量 reward model 和更系统的安全数据。

未运行的方向包括 DPO、rejection sampling fine-tuning 和显存/吞吐量的严格 profile；这些仅作为未来工作，不在本次结果中伪造。

## 8. 问题、解决方案与反思

- **预训练成功但 SFT/RLHF 失败**：TinyStories PPL 低说明故事域 next-token modeling 成功，但 instruction following、事实问答和安全拒答是不同能力。
- **低 PPL 不等于好对话能力**：模型能生成流畅故事片段，但面对 QA、reasoning、summary 时经常跑题。
- **小模型 RLHF 困难**：reward model 准确率接近随机时，PPO 优化方向噪声很大；policy 本身弱时，RLHF 容易强化格式或重复模式，而不是安全能力。
- **人工评估必要**：自动 loss/reward 指标无法反映回答是否有用、安全或忠实。最终人工评分揭示了模型真实可用性较低。
- **未来改进**：增大模型和上下文窗口；使用更通用的预训练语料；增加 SFT 数据与步数；提升 reward model 数据规模和准确率；调节 PPO KL beta、reward normalization 和采样策略；尝试 DPO 或 rejection sampling fine-tuning。

## 9. 复现方式

环境安装：

```bash
conda env create -f environment.yml
conda activate nlp
```

轻量测试：

```bash
python -m compileall src scripts advanced
pytest -q
python scripts/final_submission_check.py
```

真实数据和训练命令：

```bash
python scripts/train_tokenizer.py --config configs/tokenizer.yaml
python scripts/prepare_tinystories.py --config configs/tokenizer.yaml
torchrun --standalone --nproc_per_node=4 scripts/pretrain.py --config configs/pretrain_4gpu.yaml
torchrun --standalone --nproc_per_node=4 scripts/sft.py --config configs/sft_4gpu.yaml --pretrained outputs/checkpoints/pretrain/best.pt
torchrun --standalone --nproc_per_node=4 scripts/sft.py --config configs/sft_final.yaml --pretrained outputs/checkpoints/pretrain/best.pt
python scripts/train_reward_model.py --config configs/reward_final.yaml --sft_checkpoint outputs/checkpoints/sft_final/best.pt
python scripts/ppo_train.py --config configs/ppo_final.yaml --sft_checkpoint outputs/checkpoints/sft_final/best.pt --reward_checkpoint outputs/checkpoints/reward_final/best.pt
python scripts/evaluate_sft.py --checkpoint outputs/checkpoints/sft_final/best.pt --out reports/manual_eval_sft_final.csv --max_new_tokens 96
python scripts/evaluate_safety.py --checkpoint outputs/checkpoints/ppo_final/last.pt --fallback_checkpoint outputs/checkpoints/sft_final/best.pt --out reports/safety_eval_final.csv --max_new_tokens 96
python scripts/summarize_manual_eval.py --sft reports/manual_eval_sft_final.csv --safety reports/safety_eval_final.csv --out reports/final_eval_summary.json
python scripts/summarize_ablation_results.py
python scripts/make_final_report.py
```

长训需要本地数据、GPU 和 checkpoint。GitHub 主要提交代码、配置、日志、图和报告；大 checkpoint 默认被 `.gitignore` 忽略，见 `reports/MODEL_ARTIFACTS.md`。

## 10. 结论

本项目完整实现了 TinyStories 预训练、Alpaca SFT、PKU-SafeRLHF reward model 与 PPO/RLHF 的端到端代码，并补充了 LoRA 与 PPO sampling 的 advanced 方案。预训练 PPL 明确达到课程目标；SFT/RLHF 结果未达到理想对话和安全目标，但保留了完整日志、人工评估、负结果分析和消融对比。最终提交应将该项目定位为“小规模从零训练与对齐流程复现 + 诚实负结果分析”，而不是夸大为可用对话模型。

## 参考文献

1. Vaswani et al. Attention Is All You Need. 2017.
2. Radford et al. Improving Language Understanding by Generative Pre-Training. 2018.
3. Brown et al. Language Models are Few-Shot Learners. 2020.
4. Eldan and Li. TinyStories: How Small Can Language Models Be and Still Speak Coherent English? 2023.
5. Taori et al. Alpaca: A Strong, Replicable Instruction-Following Model. 2023.
6. Christiano et al. Deep Reinforcement Learning from Human Preferences. 2017.
7. Ouyang et al. Training language models to follow instructions with human feedback. 2022.
8. Schulman et al. Proximal Policy Optimization Algorithms. 2017.
9. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. 2021.
10. Ji et al. PKU-SafeRLHF: A Safety Alignment Preference Dataset for Llama Family Models. 2024.
11. Hugging Face datasets, tokenizers, transformers documentation.
