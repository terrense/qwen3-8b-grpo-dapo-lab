# Live Run Status

_Updated 2026-09-10T14:11:40 · health **YELLOW**_

| | |
|---|---|
| Run | `R1_grpo_baseline` (`R1_grpo_baseline-20260910-134534`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **14** |
| Updates recorded | 14 |
| Elapsed | 1566 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.2344 |
| KL | 0.00001 |
| entropy | 0.3703 |
| clip_fraction | 0.0001 |
| grad_norm | 0.2011 |
| mixed_group_ratio | 0.500 |
| zero_std_group_ratio | 0.500 |
| effective_signal_fraction | 0.500 |
| response_length_mean | 2115.3 |
| truncation_rate | 0.008 |
| rollout time (s) | 54.4 |
| actor update time (s) | 39.0 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.917, 92.062, 91.968, 91.968]`
disk / : 1.27%   disk data: 3.06%

## Incidents

2 recorded. Latest: `INC-20260910-141140-RESPONSE_LENGTH_MEAN_ROBUST_Z`

Last checkpoint: `none`
Git commit: `630c3b4`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
