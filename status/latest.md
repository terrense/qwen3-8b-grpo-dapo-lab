# Live Run Status

_Updated 2026-09-11T11:29:03 · health **GREEN**_

| | |
|---|---|
| Run | `M20_dapo` (`M20_dapo-20260911-111356`) |
| Algorithm | DAPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **6** |
| Updates recorded | 6 |
| Elapsed | 907 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.1261 |
| KL | 0.00002 |
| entropy | 0.3736 |
| clip_fraction | 0.0001 |
| grad_norm | 0.2878 |
| mixed_group_ratio | 1.000 |
| zero_std_group_ratio | 0.000 |
| effective_signal_fraction | 1.000 |
| response_length_mean | 1884.0 |
| truncation_rate | unavailable |
| rollout time (s) | 63.4 |
| actor update time (s) | 35.4 |

## System

GPU util: `[99, 99, 99, 99]`
GPU mem (GiB): `[34.62, 40.234, 40.273, 40.273]`
disk / : 1.27%   disk data: 3.07%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `52751e8`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
