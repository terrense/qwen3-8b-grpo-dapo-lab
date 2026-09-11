# Live Run Status

_Updated 2026-09-11T16:48:18 · health **YELLOW**_

| | |
|---|---|
| Run | `A_grpo` (`A_grpo-20260911-155511`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **19** |
| Updates recorded | 19 |
| Elapsed | 3187 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.0312 |
| KL | 0.00003 |
| entropy | 0.2604 |
| clip_fraction | 0.0002 |
| grad_norm | 0.3194 |
| mixed_group_ratio | 0.875 |
| zero_std_group_ratio | 0.125 |
| effective_signal_fraction | 0.875 |
| response_length_mean | 1957.6 |
| truncation_rate | unavailable |
| rollout time (s) | 46.4 |
| actor update time (s) | 35.9 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[87.224, 92.446, 92.4, 92.4]`
disk / : 1.27%   disk data: 3.08%

## Incidents

7 recorded. Latest: `INC-20260911-164818-TRAINING_STALL`

Last checkpoint: `none`
Git commit: `788736c`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
