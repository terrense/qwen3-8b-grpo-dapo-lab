# Live Run Status

_Updated 2026-09-11T20:05:14 · health **GREEN**_

| | |
|---|---|
| Run | `E_drgrpo_scaled` (`E_drgrpo_scaled-20260911-193309`) |
| Algorithm | DrGRPO_fixed |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **12** |
| Updates recorded | 12 |
| Elapsed | 1926 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.0781 |
| KL | -0.00006 |
| entropy | 0.3219 |
| clip_fraction | 0.0001 |
| grad_norm | 0.1873 |
| mixed_group_ratio | 0.625 |
| zero_std_group_ratio | 0.375 |
| effective_signal_fraction | 0.625 |
| response_length_mean | 1636.6 |
| truncation_rate | unavailable |
| rollout time (s) | 32.3 |
| actor update time (s) | 30.0 |

## System

GPU util: `[99, 99, 100, 100]`
GPU mem (GiB): `[66.034, 66.835, 66.894, 66.874]`
disk / : 1.27%   disk data: 3.09%

## Incidents

2 recorded. Latest: `INC-20260911-194954-T_ROLLOUT_ROBUST_Z`

Last checkpoint: `none`
Git commit: `90df986`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
