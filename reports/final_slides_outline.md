# Final Slides Outline

1. Title and project motivation
2. Goal: from-scratch small decoder-only Transformer plus alignment pipeline
3. System architecture and repository structure
4. Tokenizer and TinyStories data preparation
5. Decoder-only Transformer implementation details
6. Pretraining setup and validation PPL curve
7. Pretraining generated examples and analysis
8. SFT data format and response-only loss
9. SFT results: milestone vs final run
10. Reward model design and pairwise ranking loss
11. PPO/RLHF implementation details
12. Safety evaluation prompts and manual scoring protocol
13. Advanced methods: LoRA PEFT and PPO safety-prompt sampling
14. Ablation results/status: model scale, SFT data size, LoRA, RLHF before/after
15. Failure cases: weak instruction following, weak reward accuracy, failed safety refusal
16. Final conclusions and future work
