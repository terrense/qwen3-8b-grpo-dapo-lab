# Live Run Status

_Updated 2026-09-11T18:52:53 · health **GREEN**_

| | |
|---|---|
| Run | `D_drgrpo` (`D_drgrpo-20260911-183747`) |
| Algorithm | DrGRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **7** |
| Updates recorded | 7 |
| Elapsed | 906 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | 0.0469 |
| KL | -0.00001 |
| entropy | 0.0671 |
| clip_fraction | 0.0002 |
| grad_norm | 0.0403 |
| mixed_group_ratio | 0.562 |
| zero_std_group_ratio | 0.438 |
| effective_signal_fraction | 0.562 |
| response_length_mean | 1776.2 |
| truncation_rate | unavailable |
| rollout time (s) | 48.4 |
| actor update time (s) | 32.8 |

## System

GPU util: `[99, 99, 99, 100]`
GPU mem (GiB): `[54.23, 58.73, 58.73, 58.73]`
disk / : 1.27%   disk data: 3.09%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `a24ffde`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
