# INC-20260911-191653-RESPONSE_LENGTH_MEAN_ROBUST_Z

**Level:** YELLOW
**Rule:** `response_length_mean_robust_z`
**Metric:** `response_length_mean`
**Step:** 17
**Timestamp:** 2026-09-11T19:16:53

## Observed symptom

response_length_mean=2494.890625 is +4.2 robust-z from the rolling median of the last 16 updates

Value at detection: `2494.890625`
Robust z-score vs rolling median: `4.19`


## Detection rule

`response_length_mean_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=12 | reward_mean=-0.0625 | kl=2.213015881125102e-05 | entropy=0.08841685205698013 | clip_fraction=0.00014408186211767315 | grad_norm=0.04481813125312328 | response_length_mean=1692.15625 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=34.35443186759949 | t_actor_update=30.942734714597464
global_step=13 | reward_mean=-0.171875 | kl=2.3617621877747297e-05 | entropy=0.0823981836438179 | clip_fraction=9.189916818286292e-05 | grad_norm=0.043869366869330406 | response_length_mean=1980.609375 | zero_std_group_ratio=0.3125 | effective_signal_fraction=0.6875 | t_rollout=52.3699898570776 | t_actor_update=36.316316973418
global_step=14 | reward_mean=-0.25 | kl=4.159075501775078e-06 | entropy=0.10362868010997772 | clip_fraction=5.233478509580891e-05 | grad_norm=0.032994926907122135 | response_length_mean=2455.71875 | zero_std_group_ratio=0.6875 | effective_signal_fraction=0.3125 | t_rollout=58.38026152551174 | t_actor_update=44.59707336872816
global_step=15 | reward_mean=-0.15625 | kl=1.8791847949728435e-05 | entropy=0.08760986477136612 | clip_fraction=7.626233218616107e-05 | grad_norm=0.033988600596785545 | response_length_mean=1921.7421875 | zero_std_group_ratio=0.5625 | effective_signal_fraction=0.4375 | t_rollout=44.368423249572515 | t_actor_update=35.190548334270716
global_step=16 | reward_mean=0.140625 | kl=-7.206155260064406e-06 | entropy=0.08130694180727005 | clip_fraction=0.00027650391166389454 | grad_norm=0.05394689738750458 | response_length_mean=1886.546875 | zero_std_group_ratio=0.125 | effective_signal_fraction=0.875 | t_rollout=40.35988187044859 | t_actor_update=34.393510423600674
```

## Metrics at the event

```
global_step=17 | reward_mean=-0.234375 | kl=1.9554519237630075e-05 | entropy=0.12079127877950668 | clip_fraction=0.00012206559358673985 | grad_norm=0.03874293714761734 | response_length_mean=2494.890625 | zero_std_group_ratio=0.625 | effective_signal_fraction=0.375 | t_rollout=54.37522393837571 | t_actor_update=45.59542439877987
```

## Initial hypotheses

Work the checklist in [`docs/observability/diagnosis_playbook.md`](../../../../docs/observability/diagnosis_playbook.md)
in order — environment/verifier, data batch composition, truncation, policy-version
freshness, old-logprob correctness, token masking, reward normalisation, optimizer,
LR, and only then KL/clipping.

## Evidence collected

- `metric_window.csv` — 16 updates around the event
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
