# Live Run Status

_Updated 2026-09-11T17:28:20 · health **YELLOW**_

| | |
|---|---|
| Run | `B_gspo` (`B_gspo-20260911-164853`) |
| Algorithm | GSPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **17** |
| Updates recorded | 17 |
| Elapsed | 2367 s |
| Health | **YELLOW** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.3125 |
| KL | -0.00000 |
| entropy | 0.3903 |
| clip_fraction | 0.0000 |
| grad_norm | 0.1947 |
| mixed_group_ratio | 0.500 |
| zero_std_group_ratio | 0.500 |
| effective_signal_fraction | 0.500 |
| response_length_mean | 2552.3 |
| truncation_rate | unavailable |
| rollout time (s) | 54.4 |
| actor update time (s) | 46.6 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.636, 91.931, 92.013, 92.013]`
disk / : 1.27%   disk data: 3.08%

## Incidents

4 recorded. Latest: `INC-20260911-172819-T_STEP_ROBUST_Z`

Last checkpoint: `none`
Git commit: `4a5d3b7`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
