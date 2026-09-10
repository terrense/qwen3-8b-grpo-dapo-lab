# Live Run Status

_Updated 2026-09-10T14:21:40 · health **GREEN**_

| | |
|---|---|
| Run | `R1_grpo_baseline` (`R1_grpo_baseline-20260910-134534`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **19** |
| Updates recorded | 19 |
| Elapsed | 2166 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.4062 |
| KL | -0.00002 |
| entropy | 0.3319 |
| clip_fraction | 0.0001 |
| grad_norm | 0.2515 |
| mixed_group_ratio | 0.562 |
| zero_std_group_ratio | 0.438 |
| effective_signal_fraction | 0.562 |
| response_length_mean | 2147.1 |
| truncation_rate | 0.008 |
| rollout time (s) | 50.4 |
| actor update time (s) | 39.3 |

## System

GPU util: `[0, 0, 100, 100]`
GPU mem (GiB): `[87.292, 92.437, 92.4, 92.4]`
disk / : 1.27%   disk data: 3.06%

## Incidents

3 recorded. Latest: `INC-20260910-141700-GRAD_NORM_ROBUST_Z`

Last checkpoint: `none`
Git commit: `bb5103e`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
