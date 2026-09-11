# Live Run Status

_Updated 2026-09-11T11:43:03 · health **YELLOW**_

| | |
|---|---|
| Run | `M20_dapo` (`M20_dapo-20260911-111356`) |
| Algorithm | DAPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **13** |
| Updates recorded | 13 |
| Elapsed | 1748 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | 0.0156 |
| KL | -0.00003 |
| entropy | 0.3927 |
| clip_fraction | 0.0001 |
| grad_norm | 0.2780 |
| mixed_group_ratio | 1.000 |
| zero_std_group_ratio | 0.000 |
| effective_signal_fraction | 1.000 |
| response_length_mean | 2156.3 |
| truncation_rate | unavailable |
| rollout time (s) | 109.3 |
| actor update time (s) | 39.5 |

## System

GPU util: `[13, 47, 0, 43]`
GPU mem (GiB): `[83.987, 89.64, 89.603, 89.603]`
disk / : 1.27%   disk data: 3.07%

## Incidents

1 recorded. Latest: `INC-20260911-114303-T_STEP_ROBUST_Z`

Last checkpoint: `none`
Git commit: `2fcff20`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
