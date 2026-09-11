# Live Run Status

_Updated 2026-09-11T18:37:29 · health **GREEN**_

| | |
|---|---|
| Run | `C_aggonly` (`C_aggonly-20260911-174244`) |
| Algorithm | GRPO_seqagg |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **20** |
| Updates recorded | 20 |
| Elapsed | 3285 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.1094 |
| KL | -0.00002 |
| entropy | 0.3124 |
| clip_fraction | 0.0001 |
| grad_norm | 0.1908 |
| mixed_group_ratio | 0.562 |
| zero_std_group_ratio | 0.438 |
| effective_signal_fraction | 0.562 |
| response_length_mean | 2520.0 |
| truncation_rate | unavailable |
| rollout time (s) | 58.4 |
| actor update time (s) | 46.4 |

## System

GPU util: `[0, 0, 0, 0]`
GPU mem (GiB): `[87.222, 92.445, 92.788, 92.788]`
disk / : 1.27%   disk data: 3.09%

## Incidents

3 recorded. Latest: `INC-20260911-183649-TRAINING_STALL`

Last checkpoint: `none`
Git commit: `0dd767d`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
