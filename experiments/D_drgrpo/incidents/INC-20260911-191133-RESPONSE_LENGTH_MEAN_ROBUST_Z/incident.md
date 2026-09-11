# INC-20260911-191133-RESPONSE_LENGTH_MEAN_ROBUST_Z

**Level:** YELLOW
**Rule:** `response_length_mean_robust_z`
**Metric:** `response_length_mean`
**Step:** 14
**Timestamp:** 2026-09-11T19:11:33

## Observed symptom

response_length_mean=2455.71875 is +8.4 robust-z from the rolling median of the last 13 updates

Value at detection: `2455.71875`
Robust z-score vs rolling median: `8.43`


## Detection rule

`response_length_mean_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=9 | reward_mean=0.015625 | kl=1.0214331226165996e-05 | entropy=0.07295192778110504 | clip_fraction=0.00016877730286068982 | grad_norm=0.04645455442368984 | response_length_mean=1927.578125 | zero_std_group_ratio=0.125 | effective_signal_fraction=0.875 | t_rollout=40.35613425076008 | t_actor_update=34.753928162157536
global_step=10 | reward_mean=0.046875 | kl=-3.948678056531207e-06 | entropy=0.0626949742436409 | clip_fraction=0.00020394785860844422 | grad_norm=0.03122872579842806 | response_length_mean=1681.4609375 | zero_std_group_ratio=0.375 | effective_signal_fraction=0.625 | t_rollout=46.41218003258109 | t_actor_update=31.71690656989813
global_step=11 | reward_mean=0.03125 | kl=4.693784397602485e-05 | entropy=0.06314451992511749 | clip_fraction=0.00012675092420977307 | grad_norm=0.027351864613592625 | response_length_mean=1734.765625 | zero_std_group_ratio=0.625 | effective_signal_fraction=0.375 | t_rollout=38.41803064197302 | t_actor_update=32.09515957534313
global_step=12 | reward_mean=-0.0625 | kl=2.213015881125102e-05 | entropy=0.08841685205698013 | clip_fraction=0.00014408186211767315 | grad_norm=0.04481813125312328 | response_length_mean=1692.15625 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=34.35443186759949 | t_actor_update=30.942734714597464
global_step=13 | reward_mean=-0.171875 | kl=2.3617621877747297e-05 | entropy=0.0823981836438179 | clip_fraction=9.189916818286292e-05 | grad_norm=0.043869366869330406 | response_length_mean=1980.609375 | zero_std_group_ratio=0.3125 | effective_signal_fraction=0.6875 | t_rollout=52.3699898570776 | t_actor_update=36.316316973418
```

## Metrics at the event

```
global_step=14 | reward_mean=-0.25 | kl=4.159075501775078e-06 | entropy=0.10362868010997772 | clip_fraction=5.233478509580891e-05 | grad_norm=0.032994926907122135 | response_length_mean=2455.71875 | zero_std_group_ratio=0.6875 | effective_signal_fraction=0.3125 | t_rollout=58.38026152551174 | t_actor_update=44.59707336872816
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
