# Live Run Status

_Updated 2026-09-11T19:09:13 · health **YELLOW**_

| | |
|---|---|
| Run | `D_drgrpo` (`D_drgrpo-20260911-183747`) |
| Algorithm | DrGRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **13** |
| Updates recorded | 13 |
| Elapsed | 1886 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.1719 |
| KL | 0.00002 |
| entropy | 0.0824 |
| clip_fraction | 0.0001 |
| grad_norm | 0.0439 |
| mixed_group_ratio | 0.688 |
| zero_std_group_ratio | 0.312 |
| effective_signal_fraction | 0.688 |
| response_length_mean | 1980.6 |
| truncation_rate | unavailable |
| rollout time (s) | 52.4 |
| actor update time (s) | 36.3 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.751, 91.974, 91.976, 91.976]`
disk / : 1.27%   disk data: 3.09%

## Incidents

1 recorded. Latest: `INC-20260911-190913-T_ROLLOUT_ROBUST_Z`

Last checkpoint: `none`
Git commit: `e32335a`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
