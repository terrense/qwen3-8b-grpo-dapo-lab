# Live Run Status

_Updated 2026-09-11T11:24:23 · health **GREEN**_

| | |
|---|---|
| Run | `M20_dapo` (`M20_dapo-20260911-111356`) |
| Algorithm | DAPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **4** |
| Updates recorded | 4 |
| Elapsed | 627 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.0938 |
| KL | 0.00006 |
| entropy | 0.2958 |
| clip_fraction | 0.0001 |
| grad_norm | 0.2640 |
| mixed_group_ratio | 1.000 |
| zero_std_group_ratio | 0.000 |
| effective_signal_fraction | 1.000 |
| response_length_mean | 1749.5 |
| truncation_rate | unavailable |
| rollout time (s) | 63.4 |
| actor update time (s) | 33.0 |

## System

GPU util: `[0, 0, 100, 100]`
GPU mem (GiB): `[84.027, 89.679, 89.665, 89.665]`
disk / : 1.27%   disk data: 3.07%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `aee5f01`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
