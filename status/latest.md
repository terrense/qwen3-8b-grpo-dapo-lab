# Live Run Status

_Updated 2026-09-11T16:29:18 · health **YELLOW**_

| | |
|---|---|
| Run | `A_grpo` (`A_grpo-20260911-155511`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **14** |
| Updates recorded | 14 |
| Elapsed | 2047 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.2344 |
| KL | -0.00002 |
| entropy | 0.3421 |
| clip_fraction | 0.0001 |
| grad_norm | 0.1591 |
| mixed_group_ratio | 0.500 |
| zero_std_group_ratio | 0.500 |
| effective_signal_fraction | 0.500 |
| response_length_mean | 2406.0 |
| truncation_rate | unavailable |
| rollout time (s) | 50.4 |
| actor update time (s) | 43.8 |

## System

GPU util: `[78, 68, 82, 79]`
GPU mem (GiB): `[53.109, 53.128, 53.128, 53.128]`
disk / : 1.27%   disk data: 3.08%

## Incidents

1 recorded. Latest: `INC-20260911-162918-RESPONSE_LENGTH_MEAN_ROBUST_Z`

Last checkpoint: `none`
Git commit: `e6fc3dc`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
