# Live Run Status

_Updated 2026-09-11T10:57:34 · health **YELLOW**_

| | |
|---|---|
| Run | `M20_gspo` (`M20_gspo-20260911-103728`) |
| Algorithm | GSPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **10** |
| Updates recorded | 10 |
| Elapsed | 1207 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.2031 |
| KL | -0.00002 |
| entropy | 0.3007 |
| clip_fraction | 0.0000 |
| grad_norm | 0.1639 |
| mixed_group_ratio | 0.375 |
| zero_std_group_ratio | 0.625 |
| effective_signal_fraction | 0.375 |
| response_length_mean | 2213.4 |
| truncation_rate | unavailable |
| rollout time (s) | 40.3 |
| actor update time (s) | 40.3 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[87.306, 92.45, 92.292, 92.292]`
disk / : 1.27%   disk data: 3.07%

## Incidents

1 recorded. Latest: `INC-20260911-105734-RESPONSE_LENGTH_MEAN_ROBUST_Z`

Last checkpoint: `none`
Git commit: `91dec6b`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
