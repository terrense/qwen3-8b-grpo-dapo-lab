# Live Run Status

_Updated 2026-09-11T10:52:34 · health **GREEN**_

| | |
|---|---|
| Run | `M20_gspo` (`M20_gspo-20260911-103728`) |
| Algorithm | GSPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **7** |
| Updates recorded | 7 |
| Elapsed | 907 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.0625 |
| KL | 0.00002 |
| entropy | 0.3398 |
| clip_fraction | 0.0000 |
| grad_norm | 0.2675 |
| mixed_group_ratio | 0.562 |
| zero_std_group_ratio | 0.438 |
| effective_signal_fraction | 0.562 |
| response_length_mean | 1686.9 |
| truncation_rate | unavailable |
| rollout time (s) | 38.4 |
| actor update time (s) | 31.6 |

## System

GPU util: `[99, 100, 99, 99]`
GPU mem (GiB): `[65.128, 66.132, 66.171, 66.191]`
disk / : 1.27%   disk data: 3.07%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `5625719`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
