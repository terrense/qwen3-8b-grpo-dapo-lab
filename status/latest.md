# Live Run Status

_Updated 2026-09-11T10:05:38 · health **GREEN**_

| | |
|---|---|
| Run | `M20_grpo` (`M20_grpo-20260911-100052`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **1** |
| Updates recorded | 1 |
| Elapsed | 286 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.4688 |
| KL | -0.00000 |
| entropy | 0.3527 |
| clip_fraction | 0.0001 |
| grad_norm | 0.2370 |
| mixed_group_ratio | 0.625 |
| zero_std_group_ratio | 0.375 |
| effective_signal_fraction | 0.625 |
| response_length_mean | 1666.9 |
| truncation_rate | unavailable |
| rollout time (s) | 38.4 |
| actor update time (s) | 31.0 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.743, 91.968, 91.968, 91.968]`
disk / : 1.27%   disk data: 3.06%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `ad45a32`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
