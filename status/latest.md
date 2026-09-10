# Live Run Status

_Updated 2026-09-10T17:22:06 · health **GREEN**_

| | |
|---|---|
| Run | `R2_dapo_smoke` (`R2_dapo_smoke-20260910-171600`) |
| Algorithm | DAPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **1** |
| Updates recorded | 1 |
| Elapsed | 366 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.0473 |
| KL | -0.00002 |
| entropy | 0.3446 |
| clip_fraction | 0.0001 |
| grad_norm | 0.2879 |
| mixed_group_ratio | 1.000 |
| zero_std_group_ratio | 0.000 |
| effective_signal_fraction | 1.000 |
| response_length_mean | 1742.8 |
| truncation_rate | unavailable |
| rollout time (s) | 73.1 |
| actor update time (s) | 32.3 |

## System

GPU util: `[99, 99, 99, 100]`
GPU mem (GiB): `[63.648, 69.183, 69.261, 69.515]`
disk / : 1.27%   disk data: 3.06%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `c3cd12b`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
