# Live Run Status

_Updated 2026-09-11T09:05:51 · health **GREEN**_

| | |
|---|---|
| Run | `R_gspo_smoke` (`R_gspo_smoke-20260911-085807`) |
| Algorithm | GSPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **3** |
| Updates recorded | 3 |
| Elapsed | 465 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.4062 |
| KL | -0.00000 |
| entropy | 0.4437 |
| clip_fraction | 0.0000 |
| grad_norm | 0.2611 |
| mixed_group_ratio | 0.500 |
| zero_std_group_ratio | 0.500 |
| effective_signal_fraction | 0.500 |
| response_length_mean | 1809.5 |
| truncation_rate | unavailable |
| rollout time (s) | 36.4 |
| actor update time (s) | 33.4 |

## System

GPU util: `[24, 0, 11, 0]`
GPU mem (GiB): `[43.521, 48.665, 48.665, 44.027]`
disk / : 1.27%   disk data: 3.06%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `5982b31`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
