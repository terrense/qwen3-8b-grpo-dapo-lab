# INC-20260911-162918-RESPONSE_LENGTH_MEAN_ROBUST_Z

**Level:** YELLOW
**Rule:** `response_length_mean_robust_z`
**Metric:** `response_length_mean`
**Step:** 14
**Timestamp:** 2026-09-11T16:29:18

## Observed symptom

response_length_mean=2406.015625 is +4.7 robust-z from the rolling median of the last 13 updates

Value at detection: `2406.015625`
Robust z-score vs rolling median: `4.71`


## Detection rule

`response_length_mean_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=9 | reward_mean=0.03125 | kl=-1.654883180890465e-05 | entropy=0.2975699007511139 | clip_fraction=0.00015975177393556805 | grad_norm=0.21992754191160202 | response_length_mean=1972.4921875 | zero_std_group_ratio=0.3125 | effective_signal_fraction=0.6875 | t_rollout=38.3571104221046 | t_actor_update=35.40944502875209
global_step=10 | reward_mean=0.09375 | kl=-3.492479675060167e-05 | entropy=0.3102119266986847 | clip_fraction=6.492622014775407e-05 | grad_norm=0.21653076261281967 | response_length_mean=1730.421875 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=44.37116274237633 | t_actor_update=32.08560228720307
global_step=11 | reward_mean=0.03125 | kl=-5.559628721130139e-05 | entropy=0.29106247425079346 | clip_fraction=0.0001452918304494233 | grad_norm=0.20998407155275345 | response_length_mean=1768.6875 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=40.37379898130894 | t_actor_update=32.7178374491632
global_step=12 | reward_mean=-0.046875 | kl=-5.105393188387097e-05 | entropy=0.4024648666381836 | clip_fraction=0.0001573630133862025 | grad_norm=0.29083656519651413 | response_length_mean=1712.375 | zero_std_group_ratio=0.3125 | effective_signal_fraction=0.6875 | t_rollout=32.39371655881405 | t_actor_update=31.436130687594414
global_step=13 | reward_mean=-0.28125 | kl=1.1058449194134568e-05 | entropy=0.3351052403450012 | clip_fraction=0.00014996995696492377 | grad_norm=0.2732975706458092 | response_length_mean=1916.4453125 | zero_std_group_ratio=0.25 | effective_signal_fraction=0.75 | t_rollout=48.37846424430609 | t_actor_update=35.64451490342617
```

## Metrics at the event

```
global_step=14 | reward_mean=-0.234375 | kl=-2.0474693059213678e-05 | entropy=0.34207940101623535 | clip_fraction=7.721060319454409e-05 | grad_norm=0.1591385081410408 | response_length_mean=2406.015625 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=50.37621448934078 | t_actor_update=43.75125392526388
```

## Initial hypotheses

Work the checklist in [`docs/observability/diagnosis_playbook.md`](../../../../docs/observability/diagnosis_playbook.md)
in order — environment/verifier, data batch composition, truncation, policy-version
freshness, old-logprob correctness, token masking, reward normalisation, optimizer,
LR, and only then KL/clipping.

## Evidence collected

- `metric_window.csv` — 14 updates around the event
- `system_window.csv` — system telemetry, last 600s
- `last_log_lines.txt` — trainer log tail (200 lines)
- `nvidia_smi.txt`, `process_snapshot.txt`
- `resolved_config.yaml`, `run_manifest_snapshot.json` (when available)

## Root cause

PENDING

## Fix

PENDING

## Post-fix evidence

PENDING
