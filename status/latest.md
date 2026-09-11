# Live Run Status

_Updated 2026-09-11T18:17:29 · health **YELLOW**_

| | |
|---|---|
| Run | `C_aggonly` (`C_aggonly-20260911-174244`) |
| Algorithm | GRPO_seqagg |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **14** |
| Updates recorded | 14 |
| Elapsed | 2085 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.2656 |
| KL | 0.00001 |
| entropy | 0.3397 |
| clip_fraction | 0.0001 |
| grad_norm | 0.2160 |
| mixed_group_ratio | 0.625 |
| zero_std_group_ratio | 0.375 |
| effective_signal_fraction | 0.625 |
| response_length_mean | 2300.6 |
| truncation_rate | unavailable |
| rollout time (s) | 52.4 |
| actor update time (s) | 42.0 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.857, 92.079, 92.019, 92.019]`
disk / : 1.27%   disk data: 3.09%

## Incidents

1 recorded. Latest: `INC-20260911-181729-RESPONSE_LENGTH_MEAN_ROBUST_Z`

Last checkpoint: `none`
Git commit: `d5ff62c`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
