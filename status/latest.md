# Live Run Status

_Updated 2026-09-11T11:10:54 · health **YELLOW**_

| | |
|---|---|
| Run | `M20_gspo` (`M20_gspo-20260911-103728`) |
| Algorithm | GSPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **19** |
| Updates recorded | 19 |
| Elapsed | 2007 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.3594 |
| KL | 0.00002 |
| entropy | 0.3032 |
| clip_fraction | 0.0000 |
| grad_norm | 0.2466 |
| mixed_group_ratio | 0.562 |
| zero_std_group_ratio | 0.438 |
| effective_signal_fraction | 0.562 |
| response_length_mean | 2256.9 |
| truncation_rate | unavailable |
| rollout time (s) | 50.4 |
| actor update time (s) | 41.3 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.786, 91.931, 91.931, 91.931]`
disk / : 1.27%   disk data: 3.07%

## Incidents

2 recorded. Latest: `INC-20260911-111054-RESPONSE_LENGTH_MEAN_ROBUST_Z`

Last checkpoint: `none`
Git commit: `64bf10e`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
