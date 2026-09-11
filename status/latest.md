# Live Run Status

_Updated 2026-09-11T16:34:58 · health **YELLOW**_

| | |
|---|---|
| Run | `A_grpo` (`A_grpo-20260911-155511`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **17** |
| Updates recorded | 17 |
| Elapsed | 2388 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.3438 |
| KL | -0.00003 |
| entropy | 0.3937 |
| clip_fraction | 0.0001 |
| grad_norm | 0.1325 |
| mixed_group_ratio | 0.250 |
| zero_std_group_ratio | 0.750 |
| effective_signal_fraction | 0.250 |
| response_length_mean | 2578.6 |
| truncation_rate | unavailable |
| rollout time (s) | 58.4 |
| actor update time (s) | 47.4 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.847, 92.07, 92.136, 92.136]`
disk / : 1.27%   disk data: 3.08%

## Incidents

6 recorded. Latest: `INC-20260911-163458-RESPONSE_LENGTH_MEAN_ROBUST_Z`

Last checkpoint: `none`
Git commit: `ecf7fbf`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
