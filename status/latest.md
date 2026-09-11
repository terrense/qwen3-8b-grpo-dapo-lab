# Live Run Status

_Updated 2026-09-11T16:25:18 · health **GREEN**_

| | |
|---|---|
| Run | `A_grpo` (`A_grpo-20260911-155511`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **11** |
| Updates recorded | 11 |
| Elapsed | 1807 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | 0.0312 |
| KL | -0.00006 |
| entropy | 0.2911 |
| clip_fraction | 0.0001 |
| grad_norm | 0.2100 |
| mixed_group_ratio | 0.562 |
| zero_std_group_ratio | 0.438 |
| effective_signal_fraction | 0.562 |
| response_length_mean | 1768.7 |
| truncation_rate | unavailable |
| rollout time (s) | 40.4 |
| actor update time (s) | 32.7 |

## System

GPU util: `[99, 99, 99, 99]`
GPU mem (GiB): `[65.265, 66.503, 66.484, 66.503]`
disk / : 1.27%   disk data: 3.08%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `39f6c52`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
