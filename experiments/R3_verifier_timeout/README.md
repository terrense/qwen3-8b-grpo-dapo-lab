# R3_verifier_timeout — Failure injection: verifier timeout contamination

> **Status: NOT RUN.** This file is the pre-registered plan. Every section from
> "Observed behavior" onward is filled in only from real run output. Nothing here may be
> reported as a result until it has actually executed.

## Objective

Quantify how a ~5 percent verifier failure rate poisons the reward signal when mishandled, and verify that correct handling recovers it.

## Hypothesis

Mapping timeout to reward 0 injects false negatives that inflate all-wrong groups, raise the zero-std group ratio and bias advantages, degrading training in a way indistinguishable from policy failure. Masking the affected samples instead removes the bias.

## Why this experiment exists

Verifier reliability is the most under-instrumented part of most RL stacks, and this failure mode is invisible unless the verifier reports distinct states rather than a bare reward.

## Independent variables

Verifier failure handling: phase 1 timeout maps to reward 0 (deliberately wrong) versus phase 2 timeout maps to invalid/masked sample with bounded retry. Failures are injected deterministically by seed/hash so both phases see an identical failure set.

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

Phase 1 shows a depressed reward mean and an elevated all-wrong group ratio versus the clean baseline; phase 2 restores both.

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

Consolidation into analysis/FINAL_RL_ENGINEERING_REPORT.md.
