# Live Run Status

_Updated 2026-09-10T14:17:00 · health **YELLOW**_

| | |
|---|---|
| Run | `R1_grpo_baseline` (`R1_grpo_baseline-20260910-134534`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **18** |
| Updates recorded | 18 |
| Elapsed | 1886 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | 0.0625 |
| KL | -0.00002 |
| entropy | 0.3584 |
| clip_fraction | 0.0002 |
| grad_norm | 0.5185 |
| mixed_group_ratio | 0.688 |
| zero_std_group_ratio | 0.312 |
| effective_signal_fraction | 0.688 |
| response_length_mean | 1192.2 |
| truncation_rate | 0.008 |
| rollout time (s) | 24.3 |
| actor update time (s) | 22.6 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.857, 92.001, 92.146, 92.146]`
disk / : 1.27%   disk data: 3.06%

## Incidents

3 recorded. Latest: `INC-20260910-141700-GRAD_NORM_ROBUST_Z`

Last checkpoint: `none`
Git commit: `65db07c`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
