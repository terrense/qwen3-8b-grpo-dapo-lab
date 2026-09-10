# Live Run Status

_Updated 2026-09-10T13:27:09 · health **GREEN**_

| | |
|---|---|
| Run | `R0_grpo_smoke` (`R0_grpo_smoke-20260910-132022`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **1** |
| Updates recorded | 1 |
| Elapsed | 406 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.1875 |
| KL | 0.00000 |
| entropy | 0.3430 |
| clip_fraction | 0.0000 |
| grad_norm | 0.2547 |
| mixed_group_ratio | unavailable |
| zero_std_group_ratio | unavailable |
| effective_signal_fraction | unavailable |
| response_length_mean | 1532.2 |
| truncation_rate | 0.031 |
| rollout time (s) | 18.3 |
| actor update time (s) | 8.3 |

## System

GPU util: `[99, 100, 99, 100]`
GPU mem (GiB): `[52.732, 53.736, 53.716, 53.697]`
disk / : 1.27%   disk data: 3.05%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `9d22a75`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
