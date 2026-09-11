# Live Run Status

_Updated 2026-09-11T18:13:09 · health **GREEN**_

| | |
|---|---|
| Run | `C_aggonly` (`C_aggonly-20260911-174244`) |
| Algorithm | GRPO_seqagg |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **11** |
| Updates recorded | 11 |
| Elapsed | 1825 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.0156 |
| KL | -0.00005 |
| entropy | 0.2699 |
| clip_fraction | 0.0001 |
| grad_norm | 0.2105 |
| mixed_group_ratio | 0.500 |
| zero_std_group_ratio | 0.500 |
| effective_signal_fraction | 0.500 |
| response_length_mean | 1717.2 |
| truncation_rate | unavailable |
| rollout time (s) | 38.4 |
| actor update time (s) | 31.5 |

## System

GPU util: `[99, 99, 100, 99]`
GPU mem (GiB): `[66.339, 66.21, 66.249, 66.249]`
disk / : 1.27%   disk data: 3.09%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `fc2db4e`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
