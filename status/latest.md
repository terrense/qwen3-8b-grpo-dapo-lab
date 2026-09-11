# Live Run Status

_Updated 2026-09-11T19:47:55 · health **YELLOW**_

| | |
|---|---|
| Run | `E_drgrpo_scaled` (`E_drgrpo_scaled-20260911-193309`) |
| Algorithm | DrGRPO_fixed |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **7** |
| Updates recorded | 7 |
| Elapsed | 886 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | 0.0781 |
| KL | 0.00002 |
| entropy | 0.2640 |
| clip_fraction | 0.0001 |
| grad_norm | 0.1752 |
| mixed_group_ratio | 0.500 |
| zero_std_group_ratio | 0.500 |
| effective_signal_fraction | 0.500 |
| response_length_mean | 1848.3 |
| truncation_rate | unavailable |
| rollout time (s) | 46.4 |
| actor update time (s) | 33.7 |

## System

GPU util: `[7, 0, 0, 0]`
GPU mem (GiB): `[30.931, 36.226, 36.226, 40.864]`
disk / : 1.27%   disk data: 3.09%

## Incidents

1 recorded. Latest: `INC-20260911-194754-KL_ROBUST_Z`

Last checkpoint: `none`
Git commit: `ce9f6b6`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
