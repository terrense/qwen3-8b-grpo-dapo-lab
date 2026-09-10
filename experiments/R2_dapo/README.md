# R2_dapo — DAPO matched-budget

> **Status: NOT RUN.** This file is the pre-registered plan. Every section from
> "Observed behavior" onward is filled in only from real run output. Nothing here may be
> reported as a result until it has actually executed.

## Objective

Compare full DAPO against the R1 vanilla GRPO baseline under a matched budget, and explain the mechanism behind every difference observed.

## Hypothesis

Dynamic sampling raises the mixed-group ratio and therefore the effective batch, at a real cost in generated tokens and rollout wall-clock. Clip-higher slows entropy collapse.

## Why this experiment exists

To learn what each of the four DAPO mechanisms actually changes in the dynamics, instead of accepting a single reward number as the result.

## Independent variables

Clip-Higher (asymmetric clipping), Dynamic Sampling, token-level policy-gradient loss aggregation, and overlong reward shaping, as implemented in the current source rather than from memory of the paper.

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

Higher mixed-group ratio and lower zero-std ratio than R1, bought with more generated tokens per optimizer update.

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

R3 failure injections, branched from whichever configuration proved stable.
