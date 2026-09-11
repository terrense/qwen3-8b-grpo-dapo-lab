# Live Run Status

_Updated 2026-09-11T18:36:49 · health **YELLOW**_

| | |
|---|---|
| Run | `C_aggonly` (`C_aggonly-20260911-174244`) |
| Algorithm | GRPO_seqagg |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **19** |
| Updates recorded | 19 |
| Elapsed | 3246 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.0625 |
| KL | -0.00000 |
| entropy | 0.2569 |
| clip_fraction | 0.0002 |
| grad_norm | 0.2262 |
| mixed_group_ratio | 0.750 |
| zero_std_group_ratio | 0.250 |
| effective_signal_fraction | 0.750 |
| response_length_mean | 2072.5 |
| truncation_rate | unavailable |
| rollout time (s) | 48.4 |
| actor update time (s) | 37.9 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[87.222, 92.445, 92.788, 92.788]`
disk / : 1.27%   disk data: 3.09%

## Incidents

3 recorded. Latest: `INC-20260911-183649-TRAINING_STALL`

Last checkpoint: `none`
Git commit: `4771e64`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
