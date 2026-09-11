# Live Run Status

_Updated 2026-09-11T10:15:58 · health **GREEN**_

| | |
|---|---|
| Run | `M20_grpo` (`M20_grpo-20260911-100052`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **8** |
| Updates recorded | 8 |
| Elapsed | 906 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.1562 |
| KL | 0.00003 |
| entropy | 0.2620 |
| clip_fraction | 0.0002 |
| grad_norm | 0.1772 |
| mixed_group_ratio | 0.438 |
| zero_std_group_ratio | 0.562 |
| effective_signal_fraction | 0.438 |
| response_length_mean | 1613.1 |
| truncation_rate | unavailable |
| rollout time (s) | 46.4 |
| actor update time (s) | 30.7 |

## System

GPU util: `[99, 100, 100, 99]`
GPU mem (GiB): `[66.458, 65.878, 65.917, 65.937]`
disk / : 1.27%   disk data: 3.06%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `8e62f17`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
