# Live Run Status

_Updated 2026-09-11T17:42:19 · health **GREEN**_

| | |
|---|---|
| Run | `B_gspo` (`B_gspo-20260911-164853`) |
| Algorithm | GSPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **19** |
| Updates recorded | 19 |
| Elapsed | 3207 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.0469 |
| KL | -0.00002 |
| entropy | 0.2687 |
| clip_fraction | 0.0000 |
| grad_norm | 0.2207 |
| mixed_group_ratio | 0.625 |
| zero_std_group_ratio | 0.375 |
| effective_signal_fraction | 0.625 |
| response_length_mean | 2032.4 |
| truncation_rate | unavailable |
| rollout time (s) | 50.4 |
| actor update time (s) | 37.5 |

## System

GPU util: `[100, 100, 0, 0]`
GPU mem (GiB): `[87.443, 92.737, 92.45, 92.45]`
disk / : 1.27%   disk data: 3.08%

## Incidents

5 recorded. Latest: `INC-20260911-174139-TRAINING_STALL`

Last checkpoint: `none`
Git commit: `26c3dd3`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
