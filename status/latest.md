# Live Run Status

_Updated 2026-09-11T16:10:18 · health **GREEN**_

| | |
|---|---|
| Run | `A_grpo` (`A_grpo-20260911-155511`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **7** |
| Updates recorded | 7 |
| Elapsed | 907 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.0469 |
| KL | 0.00003 |
| entropy | 0.2996 |
| clip_fraction | 0.0001 |
| grad_norm | 0.2302 |
| mixed_group_ratio | 0.688 |
| zero_std_group_ratio | 0.312 |
| effective_signal_fraction | 0.688 |
| response_length_mean | 1868.2 |
| truncation_rate | unavailable |
| rollout time (s) | 48.4 |
| actor update time (s) | 34.2 |

## System

GPU util: `[100, 100, 99, 100]`
GPU mem (GiB): `[46.652, 51.601, 51.64, 51.601]`
disk / : 1.27%   disk data: 3.08%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `f343b03`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
