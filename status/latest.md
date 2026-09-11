# Live Run Status

_Updated 2026-09-11T10:34:38 · health **YELLOW**_

| | |
|---|---|
| Run | `M20_grpo` (`M20_grpo-20260911-100052`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **19** |
| Updates recorded | 19 |
| Elapsed | 2027 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.2031 |
| KL | 0.00000 |
| entropy | 0.2936 |
| clip_fraction | 0.0001 |
| grad_norm | 0.2351 |
| mixed_group_ratio | 0.562 |
| zero_std_group_ratio | 0.438 |
| effective_signal_fraction | 0.562 |
| response_length_mean | 2239.7 |
| truncation_rate | unavailable |
| rollout time (s) | 54.4 |
| actor update time (s) | 41.2 |

## System

GPU util: `[0, 56, 67, 83]`
GPU mem (GiB): `[53.062, 53.128, 53.128, 53.128]`
disk / : 1.27%   disk data: 3.07%

## Incidents

2 recorded. Latest: `INC-20260911-103438-RESPONSE_LENGTH_MEAN_ROBUST_Z`

Last checkpoint: `none`
Git commit: `37de981`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
