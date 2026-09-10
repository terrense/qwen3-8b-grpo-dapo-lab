# R0_grpo_smoke — GRPO smoke test

> **Status: NOT RUN.** This file is the pre-registered plan. Every section from
> "Observed behavior" onward is filled in only from real run output. Nothing here may be
> reported as a result until it has actually executed.

## Objective

Prove the end-to-end VeRL GRPO pipeline executes on this node, and that one optimizer update data flow is fully accounted for. This is a functional test, not a science run.

## Hypothesis

With the official Qwen3-8B FSDP GRPO example as the base config, scaled down to a tiny batch, 2-5 optimizer updates complete without OOM and every stage of the update produces sane statistics.

## Why this experiment exists

Every later run inherits this pipeline. A masking, old-logprob or advantage bug found here would silently corrupt R1, R2 and R3 alike. It is far cheaper to catch it in a 5-update run than 150 updates in.

## Independent variables

None. Batch sizes are reduced only to make the test fast.

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

Loss, KL, entropy and grad norm are finite and non-degenerate; importance ratio sits near 1.0 on the first inner epoch; prompt tokens contribute no policy gradient; padding is masked.

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

R1_grpo_baseline, once the full data flow is verified and per-stage peak VRAM is known.
