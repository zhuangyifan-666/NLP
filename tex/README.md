# Milestone LaTeX Report

This folder is designed to compile independently from the rest of the project.

Files:

- `milestone_report.tex`: main LaTeX source
- `figures/pretrain_loss_ppl.png`
- `figures/sft_loss.png`
- `figures/reward_loss_acc.png`
- `figures/ppo_metrics.png`

Recommended compile command:

```bash
xelatex milestone_report.tex
xelatex milestone_report.tex
```

The source uses `ctexart`, so a TeX distribution with Chinese support is required. The current server did not expose `xelatex`, so the source and figures are prepared here for standalone compilation on a machine with TeX Live/MacTeX installed.

