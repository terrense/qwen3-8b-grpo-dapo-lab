# Live Run Status

_Updated 2026-09-11T20:03:34 · health **GREEN**_

| | |
|---|---|
| Run | `E_drgrpo_scaled` (`E_drgrpo_scaled-20260911-193309`) |
| Algorithm | DrGRPO_fixed |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **11** |
| Updates recorded | 11 |
| Elapsed | 1826 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.0312 |
| KL | 0.00001 |
| entropy | 0.2266 |
| clip_fraction | 0.0002 |
| grad_norm | 0.1407 |
| mixed_group_ratio | 0.500 |
| zero_std_group_ratio | 0.500 |
| effective_signal_fraction | 0.500 |
| response_length_mean | 1632.3 |
| truncation_rate | unavailable |
| rollout time (s) | 36.4 |
| actor update time (s) | 30.3 |

## System

GPU util: `[100, 99, 100, 100]`
GPU mem (GiB): `[64.431, 66.132, 66.152, 66.112]`
disk / : 1.27%   disk data: 3.09%

## Incidents

2 recorded. Latest: `INC-20260911-194954-T_ROLLOUT_ROBUST_Z`

Last checkpoint: `none`
Git commit: `b96b65c`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
