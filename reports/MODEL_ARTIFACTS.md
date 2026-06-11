# Model Artifacts and Checkpoint Submission Notes

本项目的 GitHub 提交以代码、配置、日志、曲线图、人工评估表和报告为主。大模型 checkpoint、数据缓存和 tokenizer 产物默认被 `.gitignore` 忽略，不应强行加入 git。

## 不提交到 GitHub 的大文件

- `outputs/checkpoints/` 下的所有 `.pt` checkpoint。
- `outputs/tokenizer/` 下的 tokenizer 训练产物。
- `data/` 下的真实数据文件。
- Hugging Face cache、conda/venv 环境、临时训练缓存。

如果课程最终要求提交模型权重，建议通过课程平台压缩包或网盘单独提交，并在提交说明中引用下表路径、大小和 sha256。不要使用 `git add -f` 强行提交 checkpoint。

## 主要 Checkpoint 路径

| 模块 | 本地路径 | 当前状态 | size bytes | sha256 |
|---|---|---:|---:|---|
| Pretrain best | `outputs/checkpoints/pretrain/best.pt` | found | 328535133 | `521367a3fa973236310deef143b3605ed61b7009592b1373edbfd487b3d5c60d` |
| Pretrain last | `outputs/checkpoints/pretrain/last.pt` | found | 328535133 | `c261fdde997e64ca355c8310dba8e6b5302360a8784119721589390e0bdbfeab` |
| SFT baseline best | `outputs/checkpoints/sft/best.pt` | found | 258284135 | `7dd798781006709dff80ca8c9797942b51168e41b55d0a2495adeac8d79ff01f` |
| SFT baseline last | `outputs/checkpoints/sft/last.pt` | found | 258284135 | `9e7f22af62803f49c543cfb08dfb420be1b6957e89441c9ce8b66914aae16da6` |
| SFT final best | `outputs/checkpoints/sft_final/best.pt` | found | 308744775 | `d07c0cf1617b5c48f51efc413f6ef43b2c2abd714ac3c9a504bc3b5b112f8a3e` |
| SFT final last | `outputs/checkpoints/sft_final/last.pt` | found | 308744775 | `924144a17ae8b9621fcadc46cd8f9b19f0388f7d16782cf98e96ec42eff44a46` |
| Reward baseline best | `outputs/checkpoints/reward/best.pt` | found | 328544949 | `b764adcb9af98972332da9281d9800314a3ee139425311a74c95a177d365d36c` |
| Reward baseline last | `outputs/checkpoints/reward/last.pt` | found | 328544949 | `a10b9c5344d5ce558543c154ad81e49308f51632f2084ad2cd34c5bb532717d8` |
| Reward final best | `outputs/checkpoints/reward_final/best.pt` | found | 328544949 | `a9db92780368a36f8baea7577cbdc75c057580ab0e05dbf21ed8a3b622f83058` |
| Reward final last | `outputs/checkpoints/reward_final/last.pt` | found | 328544949 | `5071a6eaecee369441fc52b5fa3d71c349e19eafb3dc9287f17e5c3fbc56c4d5` |
| PPO baseline policy | `outputs/checkpoints/ppo/last.pt` | found | 328540607 | `f03c6fde37518c95d50e9aa6fed43d6996bc21a479e5925474b8852f8b0637e4` |
| PPO baseline policy+value | `outputs/checkpoints/ppo/last_policy_value.pt` | found | 328569403 | `4087f4405d6bf2941215c150ae81c454a32dc42f0d3fb90fcae063cba8b9eac9` |
| PPO final policy | `outputs/checkpoints/ppo_final/last.pt` | found | 328540607 | `ffb5f0e1719ee1e89afe79b3e1df3e68924a8843b1796b7547bddba4822b765d` |
| PPO final policy+value | `outputs/checkpoints/ppo_final/last_policy_value.pt` | found | 328569403 | `971dc94bcc48358f4acc039914e9f9550ab5730e631642810c7fe630267bcde4` |

## 消融实验 Checkpoint

| 实验 | 本地路径 | 当前状态 | size bytes | sha256 |
|---|---|---:|---:|---|
| Small pretrain best | `outputs/checkpoints/pretrain_small/best.pt` | found | 161340253 | `7c435226f757d033423f36baeab4a45ff04ec64f1e6791811732f35bd517a6b7` |
| Small pretrain last | `outputs/checkpoints/pretrain_small/last.pt` | found | 161340253 | `486777efa4faf1546246b35a961b279706da9abb23825ac7995a42f96a626785` |
| SFT data 2500 best | `outputs/checkpoints/sft_data_2500/best.pt` | found | 308744775 | `3c32ebb19dd0e207d8dab4d0454f22f065f0a285fbf06b6e67d59239de3806c4` |
| SFT data 2500 last | `outputs/checkpoints/sft_data_2500/last.pt` | found | 308744775 | `5253f16a37469fdc71ee08d5cf56257806ae3e80aaabbea6da8559615d237fca` |
| LoRA SFT best | `outputs/checkpoints/sft_lora/best.pt` | found | 111918905 | `ffe5a4f79ba35f9950e105bee32b3e33bad62596d2a1347de4dcea77f29c0609` |
| LoRA SFT last | `outputs/checkpoints/sft_lora/last.pt` | found | 111918905 | `d3cb5727994282640dae5494029f20bb4b3ee7b784badfb0fe7249100a069699` |
| Small pretrain + SFT best | `outputs/checkpoints/sft_small_pretrain/best.pt` | found | 184342823 | `b2aa4d1cb53976eded2aef244307a1d4b3eecf7b3dd4fbcbac6b2b530d0e8f66` |
| Small pretrain + SFT last | `outputs/checkpoints/sft_small_pretrain/last.pt` | found | 184342823 | `7b07df92595cb231bbc298e3b5e609835ec5a7290dffeb96d31613fd1b8bfd74` |

## 提交建议

1. GitHub 仓库提交代码、配置、日志、图、CSV、Markdown 报告和 LaTeX/PDF 报告。
2. checkpoint 如需提交，单独压缩 `outputs/checkpoints/pretrain/`、`outputs/checkpoints/sft_final/`、`outputs/checkpoints/reward_final/`、`outputs/checkpoints/ppo_final/`。
3. 单独提交权重时，同时附上本文件，方便助教核对文件完整性。
4. 若重新训练后 checkpoint 变化，需要重新计算 sha256：`find outputs/checkpoints -maxdepth 2 -type f -exec sha256sum {} +`。
