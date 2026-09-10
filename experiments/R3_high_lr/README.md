# R3_high_lr — Failure injection: excessive learning rate

> **Status: NOT RUN.** This file is the pre-registered plan. Every section from
> "Observed behavior" onward is filled in only from real run output. Nothing here may be
> reported as a result until it has actually executed.

## Objective

Record the concrete, measurable signature of a learning rate that is too high, so it is recognisable instantly in future runs.

## Hypothesis

Raising LR 3-5x above the stable baseline produces a joint spike in grad norm, KL and clip fraction, with importance ratios pushed far from 1.0, before reward visibly collapses.

## Why this experiment exists

To know the difference between diverging and learning fast, measured on this hardware and this model rather than taken from a textbook.

## Independent variables

Actor learning rate: 3-5x the stable baseline. Stopped the moment divergence is unambiguous.

## Controlled variables

Model Qwen/Qwen3-8B, dataset BytedTsinghua-SIA/DAPO-Math-17k, validation AIME 2024,
seed policy, seed, prompt distribution, validation protocol and hardware are held fixed unless
listed as independent above.

## Hardware

4 x NVIDIA H20 96GB, full NV18 NVLink mesh. See [../../docs/hardware_environment.md](../../docs/hardware_environment.md).

## Model

Qwen/Qwen3-8B post-trained release. Revision recorded in [../../manifests/model.md](../../manifests/model.md).

## Dataset

See [../../manifests/dataset.md](../../manifests/dataset.md).

## Full training command

```bash
# TO BE FILLED from the actual launch
```

## Resolved configuration

resolved_config.yaml in this directory, written by the trainer rather than by hand.

## Expected behavior

Grad norm and KL rise together within a handful of updates; clipfrac saturates; entropy moves sharply in one direction.

## Metrics

results_summary.md plus this run CSV/JSONL under analysis/. Figures are regenerated
from those files by [../../scripts/plotting/](../../scripts/plotting/) into figures/generated/.

## Observed behavior

NOT RUN.

## Failure / incident

NOT RUN.

## Root-cause hypotheses

NOT RUN.

## Evidence

NOT RUN.

## Confirmed root cause

NOT RUN.

## Fix

NOT RUN.

## Post-fix evidence

NOT RUN.

## Conclusion

NOT RUN.

## What I learned

NOT RUN.

## Next experiment

R3_verifier_timeout.
