# Live Run Status

_Updated 2026-09-11T19:11:33 · health **YELLOW**_

| | |
|---|---|
| Run | `D_drgrpo` (`D_drgrpo-20260911-183747`) |
| Algorithm | DrGRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **14** |
| Updates recorded | 14 |
| Elapsed | 2027 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.2500 |
| KL | 0.00000 |
| entropy | 0.1036 |
| clip_fraction | 0.0001 |
| grad_norm | 0.0330 |
| mixed_group_ratio | 0.312 |
| zero_std_group_ratio | 0.688 |
| effective_signal_fraction | 0.312 |
| response_length_mean | 2455.7 |
| truncation_rate | unavailable |
| rollout time (s) | 58.4 |
| actor update time (s) | 44.6 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.782, 92.005, 92.087, 92.087]`
disk / : 1.27%   disk data: 3.09%

## Incidents

3 recorded. Latest: `INC-20260911-191133-T_ROLLOUT_ROBUST_Z`

Last checkpoint: `none`
Git commit: `bb82fcc`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
