# Live Run Status

_Updated 2026-09-11T11:59:23 · health **GREEN**_

| | |
|---|---|
| Run | `M20_dapo` (`M20_dapo-20260911-111356`) |
| Algorithm | DAPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **19** |
| Updates recorded | 19 |
| Elapsed | 2727 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | 0.1250 |
| KL | -0.00004 |
| entropy | 0.3706 |
| clip_fraction | 0.0001 |
| grad_norm | 0.2711 |
| mixed_group_ratio | 1.000 |
| zero_std_group_ratio | 0.000 |
| effective_signal_fraction | 1.000 |
| response_length_mean | 2074.6 |
| truncation_rate | unavailable |
| rollout time (s) | 80.9 |
| actor update time (s) | 38.1 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[84.349, 90.001, 89.894, 89.894]`
disk / : 1.27%   disk data: 3.07%

## Incidents

1 recorded. Latest: `INC-20260911-114303-T_STEP_ROBUST_Z`

Last checkpoint: `none`
Git commit: `b5d2821`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
