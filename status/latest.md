# Live Run Status

_Updated 2026-09-11T17:57:49 · health **GREEN**_

| | |
|---|---|
| Run | `C_aggonly` (`C_aggonly-20260911-174244`) |
| Algorithm | GRPO_seqagg |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **7** |
| Updates recorded | 7 |
| Elapsed | 905 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | 0.0312 |
| KL | 0.00005 |
| entropy | 0.2766 |
| clip_fraction | 0.0002 |
| grad_norm | 0.2535 |
| mixed_group_ratio | 0.750 |
| zero_std_group_ratio | 0.250 |
| effective_signal_fraction | 0.750 |
| response_length_mean | 1796.8 |
| truncation_rate | unavailable |
| rollout time (s) | 44.4 |
| actor update time (s) | 33.1 |

## System

GPU util: `[99, 99, 99, 99]`
GPU mem (GiB): `[44.406, 49.706, 49.784, 49.628]`
disk / : 1.27%   disk data: 3.08%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `295ea70`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
