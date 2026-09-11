# Live Run Status

_Updated 2026-09-11T19:08:13 · health **GREEN**_

| | |
|---|---|
| Run | `D_drgrpo` (`D_drgrpo-20260911-183747`) |
| Algorithm | DrGRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **12** |
| Updates recorded | 12 |
| Elapsed | 1826 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.0625 |
| KL | 0.00002 |
| entropy | 0.0884 |
| clip_fraction | 0.0001 |
| grad_norm | 0.0448 |
| mixed_group_ratio | 0.562 |
| zero_std_group_ratio | 0.438 |
| effective_signal_fraction | 0.562 |
| response_length_mean | 1692.2 |
| truncation_rate | unavailable |
| rollout time (s) | 34.4 |
| actor update time (s) | 30.9 |

## System

GPU util: `[99, 99, 99, 99]`
GPU mem (GiB): `[41.554, 46.972, 46.952, 46.757]`
disk / : 1.27%   disk data: 3.09%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `2219784`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
