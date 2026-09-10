# Live Run Status

_Updated 2026-09-10T14:05:40 · health **YELLOW**_

| | |
|---|---|
| Run | `R1_grpo_baseline` (`R1_grpo_baseline-20260910-134534`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **10** |
| Updates recorded | 10 |
| Elapsed | 1206 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.2656 |
| KL | -0.00001 |
| entropy | 0.3094 |
| clip_fraction | 0.0001 |
| grad_norm | 0.2354 |
| mixed_group_ratio | 0.750 |
| zero_std_group_ratio | 0.250 |
| effective_signal_fraction | 0.750 |
| response_length_mean | 2188.1 |
| truncation_rate | 0.008 |
| rollout time (s) | 44.4 |
| actor update time (s) | 39.9 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[87.161, 92.306, 92.296, 92.296]`
disk / : 1.27%   disk data: 3.06%

## Incidents

1 recorded. Latest: `INC-20260910-140540-RESPONSE_LENGTH_MEAN_ROBUST_Z`

Last checkpoint: `none`
Git commit: `c814527`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
