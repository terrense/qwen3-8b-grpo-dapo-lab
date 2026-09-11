# Live Run Status

_Updated 2026-09-11T16:30:58 · health **YELLOW**_

| | |
|---|---|
| Run | `A_grpo` (`A_grpo-20260911-155511`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **15** |
| Updates recorded | 15 |
| Elapsed | 2148 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.0938 |
| KL | -0.00004 |
| entropy | 0.3745 |
| clip_fraction | 0.0001 |
| grad_norm | 0.1401 |
| mixed_group_ratio | 0.188 |
| zero_std_group_ratio | 0.812 |
| effective_signal_fraction | 0.188 |
| response_length_mean | 1745.5 |
| truncation_rate | unavailable |
| rollout time (s) | 36.4 |
| actor update time (s) | 32.2 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.821, 92.044, 92.071, 92.071]`
disk / : 1.27%   disk data: 3.08%

## Incidents

3 recorded. Latest: `INC-20260911-163058-EFFECTIVE_SIGNAL_LOW`

Last checkpoint: `none`
Git commit: `4de4c95`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
