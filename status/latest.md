# Live Run Status

_Updated 2026-09-11T18:23:29 · health **YELLOW**_

| | |
|---|---|
| Run | `C_aggonly` (`C_aggonly-20260911-174244`) |
| Algorithm | GRPO_seqagg |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **17** |
| Updates recorded | 17 |
| Elapsed | 2445 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.3594 |
| KL | -0.00004 |
| entropy | 0.3677 |
| clip_fraction | 0.0001 |
| grad_norm | 0.1755 |
| mixed_group_ratio | 0.438 |
| zero_std_group_ratio | 0.562 |
| effective_signal_fraction | 0.438 |
| response_length_mean | 2755.6 |
| truncation_rate | unavailable |
| rollout time (s) | 62.4 |
| actor update time (s) | 50.9 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.745, 91.968, 92.017, 92.017]`
disk / : 1.27%   disk data: 3.09%

## Incidents

2 recorded. Latest: `INC-20260911-182329-RESPONSE_LENGTH_MEAN_ROBUST_Z`

Last checkpoint: `none`
Git commit: `8f4e45d`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
