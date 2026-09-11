# Live Run Status

_Updated 2026-09-11T10:20:58 · health **YELLOW**_

| | |
|---|---|
| Run | `M20_grpo` (`M20_grpo-20260911-100052`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **10** |
| Updates recorded | 10 |
| Elapsed | 1207 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.1719 |
| KL | 0.00002 |
| entropy | 0.3020 |
| clip_fraction | 0.0002 |
| grad_norm | 0.1776 |
| mixed_group_ratio | 0.562 |
| zero_std_group_ratio | 0.438 |
| effective_signal_fraction | 0.562 |
| response_length_mean | 2299.9 |
| truncation_rate | unavailable |
| rollout time (s) | 52.4 |
| actor update time (s) | 42.3 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.937, 92.159, 92.441, 92.441]`
disk / : 1.27%   disk data: 3.06%

## Incidents

1 recorded. Latest: `INC-20260911-102058-RESPONSE_LENGTH_MEAN_ROBUST_Z`

Last checkpoint: `none`
Git commit: `cc0fbef`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
