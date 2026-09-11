# INC-20260911-102058-RESPONSE_LENGTH_MEAN_ROBUST_Z

**Level:** YELLOW
**Rule:** `response_length_mean_robust_z`
**Metric:** `response_length_mean`
**Step:** 10
**Timestamp:** 2026-09-11T10:20:58

## Observed symptom

response_length_mean=2299.8671875 is +6.1 robust-z from the rolling median of the last 9 updates

Value at detection: `2299.8671875`
Robust z-score vs rolling median: `6.08`


## Detection rule

`response_length_mean_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=5 | reward_mean=-0.1875 | kl=1.6927497767937894e-05 | entropy=0.23578892648220062 | clip_fraction=8.887194530871056e-05 | grad_norm=0.19708006083965302 | response_length_mean=1723.9375 | zero_std_group_ratio=0.5625 | effective_signal_fraction=0.4375 | t_rollout=48.36955415084958 | t_actor_update=31.921732306480408
global_step=6 | reward_mean=0.1875 | kl=-9.0890691808454e-06 | entropy=0.29520151019096375 | clip_fraction=0.00024071802727121394 | grad_norm=0.2917567789554596 | response_length_mean=1455.5234375 | zero_std_group_ratio=0.125 | effective_signal_fraction=0.875 | t_rollout=30.348320595920086 | t_actor_update=26.580287896096706
global_step=7 | reward_mean=-0.078125 | kl=-5.548426997847855e-07 | entropy=0.36519548296928406 | clip_fraction=0.0001803160932922765 | grad_norm=0.2877662181854248 | response_length_mean=1493.1171875 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=30.346693336963654 | t_actor_update=27.94768112897873
global_step=8 | reward_mean=-0.15625 | kl=2.5218408154614735e-05 | entropy=0.26196256279945374 | clip_fraction=0.00015943499101922498 | grad_norm=0.17715346813201904 | response_length_mean=1613.0703125 | zero_std_group_ratio=0.5625 | effective_signal_fraction=0.4375 | t_rollout=46.3648579120636 | t_actor_update=30.749237403273582
global_step=9 | reward_mean=-0.15625 | kl=4.0412773955722514e-05 | entropy=0.3341643512248993 | clip_fraction=0.00015772408960401663 | grad_norm=0.251394659280777 | response_length_mean=1652.03125 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=46.371601831167936 | t_actor_update=31.73167872056365
```

## Metrics at the event

```
global_step=10 | reward_mean=-0.171875 | kl=2.40020649471262e-05 | entropy=0.30201274156570435 | clip_fraction=0.00015506264571740758 | grad_norm=0.177559494972229 | response_length_mean=2299.8671875 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=52.37194440886378 | t_actor_update=42.330374639481306
```

## Initial hypotheses

Work the checklist in [`docs/observability/diagnosis_playbook.md`](../../../../docs/observability/diagnosis_playbook.md)
in order — environment/verifier, data batch composition, truncation, policy-version
freshness, old-logprob correctness, token masking, reward normalisation, optimizer,
LR, and only then KL/clipping.

## Evidence collected

- `metric_window.csv` — 10 updates around the event
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
