# Live Run Status

_Updated 2026-09-11T19:49:55 · health **YELLOW**_

| | |
|---|---|
| Run | `E_drgrpo_scaled` (`E_drgrpo_scaled-20260911-193309`) |
| Algorithm | DrGRPO_fixed |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **8** |
| Updates recorded | 8 |
| Elapsed | 1006 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | 0.0312 |
| KL | 0.00000 |
| entropy | 0.2786 |
| clip_fraction | 0.0002 |
| grad_norm | 0.1848 |
| mixed_group_ratio | 0.625 |
| zero_std_group_ratio | 0.375 |
| effective_signal_fraction | 0.625 |
| response_length_mean | 1790.8 |
| truncation_rate | unavailable |
| rollout time (s) | 48.4 |
| actor update time (s) | 33.2 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.855, 92.15, 92.03, 92.03]`
disk / : 1.27%   disk data: 3.09%

## Incidents

2 recorded. Latest: `INC-20260911-194954-T_ROLLOUT_ROBUST_Z`

Last checkpoint: `none`
Git commit: `9b01406`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
