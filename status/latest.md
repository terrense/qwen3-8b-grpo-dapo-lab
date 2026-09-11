# Live Run Status

_Updated 2026-09-11T19:39:34 · health **GREEN**_

| | |
|---|---|
| Run | `E_drgrpo_scaled` (`E_drgrpo_scaled-20260911-193309`) |
| Algorithm | DrGRPO_fixed |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **1** |
| Updates recorded | 1 |
| Elapsed | 386 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.3438 |
| KL | -0.00002 |
| entropy | 0.3272 |
| clip_fraction | 0.0001 |
| grad_norm | 0.1633 |
| mixed_group_ratio | 0.562 |
| zero_std_group_ratio | 0.438 |
| effective_signal_fraction | 0.562 |
| response_length_mean | 1767.9 |
| truncation_rate | unavailable |
| rollout time (s) | 46.4 |
| actor update time (s) | 34.3 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[68.554, 73.654, 73.614, 73.632]`
disk / : 1.27%   disk data: 3.09%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `f02e769`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
