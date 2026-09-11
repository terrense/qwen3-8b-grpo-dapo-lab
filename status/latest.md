# Live Run Status

_Updated 2026-09-11T19:16:53 · health **YELLOW**_

| | |
|---|---|
| Run | `D_drgrpo` (`D_drgrpo-20260911-183747`) |
| Algorithm | DrGRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **17** |
| Updates recorded | 17 |
| Elapsed | 2346 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.2344 |
| KL | 0.00002 |
| entropy | 0.1208 |
| clip_fraction | 0.0001 |
| grad_norm | 0.0387 |
| mixed_group_ratio | 0.375 |
| zero_std_group_ratio | 0.625 |
| effective_signal_fraction | 0.375 |
| response_length_mean | 2494.9 |
| truncation_rate | unavailable |
| rollout time (s) | 54.4 |
| actor update time (s) | 45.6 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.728, 91.95, 91.933, 91.933]`
disk / : 1.27%   disk data: 3.09%

## Incidents

4 recorded. Latest: `INC-20260911-191653-RESPONSE_LENGTH_MEAN_ROBUST_Z`

Last checkpoint: `none`
Git commit: `abd1d26`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
