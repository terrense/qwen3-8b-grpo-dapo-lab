# Live Run Status

_Updated 2026-09-10T14:01:00 · health **GREEN**_

| | |
|---|---|
| Run | `R1_grpo_baseline` (`R1_grpo_baseline-20260910-134534`) |
| Algorithm | GRPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **8** |
| Updates recorded | 8 |
| Elapsed | 926 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | -0.1250 |
| KL | 0.00003 |
| entropy | 0.2707 |
| clip_fraction | 0.0001 |
| grad_norm | 0.2307 |
| mixed_group_ratio | 0.562 |
| zero_std_group_ratio | 0.438 |
| effective_signal_fraction | 0.562 |
| response_length_mean | 1608.4 |
| truncation_rate | 0.008 |
| rollout time (s) | 34.4 |
| actor update time (s) | 29.5 |

## System

GPU util: `[100, 100, 100, 100]`
GPU mem (GiB): `[86.874, 92.019, 91.974, 91.974]`
disk / : 1.27%   disk data: 3.06%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `b013f88`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
