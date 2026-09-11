# Live Run Status

_Updated 2026-09-11T17:04:19 · health **GREEN**_

| | |
|---|---|
| Run | `B_gspo` (`B_gspo-20260911-164853`) |
| Algorithm | GSPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **7** |
| Updates recorded | 7 |
| Elapsed | 927 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | 0.0938 |
| KL | -0.00000 |
| entropy | 0.2772 |
| clip_fraction | 0.0000 |
| grad_norm | 0.2233 |
| mixed_group_ratio | 0.562 |
| zero_std_group_ratio | 0.438 |
| effective_signal_fraction | 0.562 |
| response_length_mean | 1647.7 |
| truncation_rate | unavailable |
| rollout time (s) | 38.4 |
| actor update time (s) | 30.6 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[65.995, 71.562, 71.855, 71.659]`
disk / : 1.27%   disk data: 3.08%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `de3b602`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
