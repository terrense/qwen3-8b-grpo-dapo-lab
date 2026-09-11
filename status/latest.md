# Live Run Status

_Updated 2026-09-11T10:37:18 · health **GREEN**_

| | |
|---|---|
| Run | `M20_grpo` (`M20_grpo-20260911-100052`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **20** |
| Updates recorded | 20 |
| Elapsed | 2187 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | 0.3281 |
| KL | 0.00000 |
| entropy | 0.2986 |
| clip_fraction | 0.0002 |
| grad_norm | 0.2736 |
| mixed_group_ratio | 0.562 |
| zero_std_group_ratio | 0.438 |
| effective_signal_fraction | 0.562 |
| response_length_mean | 1429.8 |
| truncation_rate | unavailable |
| rollout time (s) | 28.3 |
| actor update time (s) | 26.4 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[87.179, 92.402, 92.255, 92.255]`
disk / : 1.27%   disk data: 3.07%

## Incidents

2 recorded. Latest: `INC-20260911-103438-RESPONSE_LENGTH_MEAN_ROBUST_Z`

Last checkpoint: `none`
Git commit: `3895488`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
