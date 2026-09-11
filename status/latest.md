# Live Run Status

_Updated 2026-09-11T19:30:13 · health **GREEN**_

| | |
|---|---|
| Run | `D_drgrpo` (`D_drgrpo-20260911-183747`) |
| Algorithm | DrGRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **19** |
| Updates recorded | 19 |
| Elapsed | 3146 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | 0.0469 |
| KL | 0.00002 |
| entropy | 0.0647 |
| clip_fraction | 0.0002 |
| grad_norm | 0.0451 |
| mixed_group_ratio | 0.625 |
| zero_std_group_ratio | 0.375 |
| effective_signal_fraction | 0.625 |
| response_length_mean | 1992.0 |
| truncation_rate | unavailable |
| rollout time (s) | 44.4 |
| actor update time (s) | 36.6 |

## System

GPU util: `[0, 0, 100, 100]`
GPU mem (GiB): `[87.22, 92.443, 92.728, 92.726]`
disk / : 1.27%   disk data: 3.09%

## Incidents

4 recorded. Latest: `INC-20260911-191653-RESPONSE_LENGTH_MEAN_ROBUST_Z`

Last checkpoint: `none`
Git commit: `4470e34`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
