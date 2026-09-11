# Live Run Status

_Updated 2026-09-11T11:13:34 · health **GREEN**_

| | |
|---|---|
| Run | `M20_gspo` (`M20_gspo-20260911-103728`) |
| Algorithm | GSPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **20** |
| Updates recorded | 20 |
| Elapsed | 2167 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | 0.4375 |
| KL | 0.00002 |
| entropy | 0.2759 |
| clip_fraction | 0.0000 |
| grad_norm | 0.2338 |
| mixed_group_ratio | 0.375 |
| zero_std_group_ratio | 0.625 |
| effective_signal_fraction | 0.375 |
| response_length_mean | 1540.9 |
| truncation_rate | unavailable |
| rollout time (s) | 32.3 |
| actor update time (s) | 28.3 |

## System

GPU util: `[0, 0, 100, 100]`
GPU mem (GiB): `[87.308, 92.452, 92.505, 92.505]`
disk / : 1.27%   disk data: 3.07%

## Incidents

2 recorded. Latest: `INC-20260911-111054-RESPONSE_LENGTH_MEAN_ROBUST_Z`

Last checkpoint: `none`
Git commit: `9b51e0e`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
