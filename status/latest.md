# Live Run Status

_Updated 2026-09-11T17:19:19 · health **GREEN**_

| | |
|---|---|
| Run | `B_gspo` (`B_gspo-20260911-164853`) |
| Algorithm | GSPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| Current optimizer step | **12** |
| Updates recorded | 12 |
| Elapsed | 1827 s |
| Health | **GREEN** |

## Latest update

| metric | value |
|---|---|
| reward_mean | 0.0000 |
| KL | -0.00004 |
| entropy | 0.3816 |
| clip_fraction | 0.0000 |
| grad_norm | 0.3514 |
| mixed_group_ratio | 0.562 |
| zero_std_group_ratio | 0.438 |
| effective_signal_fraction | 0.562 |
| response_length_mean | 1697.9 |
| truncation_rate | unavailable |
| rollout time (s) | 40.4 |
| actor update time (s) | 31.6 |

## System

GPU util: `[0, 0, 100, 100]`
GPU mem (GiB): `[86.638, 91.933, 92.159, 92.159]`
disk / : 1.27%   disk data: 3.08%

## Incidents

0 recorded. Latest: `none`

Last checkpoint: `none`
Git commit: `03dc16b`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
