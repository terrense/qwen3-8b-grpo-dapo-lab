# Live Run Status

_Updated 2026-09-11T20:25:34 · health **GREEN**_

| | |
|---|---|
| Run | `E_drgrpo_scaled` (`E_drgrpo_scaled-20260911-193309`) |
| Algorithm | DrGRPO_fixed |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **19** |
| Updates recorded | 19 |
| Elapsed | 3146 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.0781 |
| KL | -0.00004 |
| entropy | 0.2210 |
| clip_fraction | 0.0002 |
| grad_norm | 0.1755 |
| mixed_group_ratio | 0.625 |
| zero_std_group_ratio | 0.375 |
| effective_signal_fraction | 0.625 |
| response_length_mean | 1901.1 |
| truncation_rate | unavailable |
| rollout time (s) | 48.4 |
| actor update time (s) | 35.0 |

## System

GPU util: `[0, 0, 100, 100]`
GPU mem (GiB): `[87.136, 92.431, 92.503, 92.501]`
disk / : 1.27%   disk data: 3.1%

## Incidents

2 recorded. Latest: `INC-20260911-194954-T_ROLLOUT_ROBUST_Z`

Last checkpoint: `none`
Git commit: `d99213f`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
