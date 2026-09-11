# Live Run Status

_Updated 2026-09-11T17:22:40 · health **YELLOW**_

| | |
|---|---|
| Run | `B_gspo` (`B_gspo-20260911-164853`) |
| Algorithm | GSPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **14** |
| Updates recorded | 14 |
| Elapsed | 2027 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.2969 |
| KL | 0.00000 |
| entropy | 0.3377 |
| clip_fraction | 0.0000 |
| grad_norm | 0.1620 |
| mixed_group_ratio | 0.375 |
| zero_std_group_ratio | 0.625 |
| effective_signal_fraction | 0.375 |
| response_length_mean | 2376.9 |
| truncation_rate | unavailable |
| rollout time (s) | 58.4 |
| actor update time (s) | 43.3 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.673, 91.968, 92.089, 92.089]`
disk / : 1.27%   disk data: 3.08%

## Incidents

2 recorded. Latest: `INC-20260911-172239-T_STEP_ROBUST_Z`

Last checkpoint: `none`
Git commit: `0a6bc92`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
