# INC-20260910-141700-GRAD_NORM_ROBUST_Z

**Level:** YELLOW
**Rule:** `grad_norm_robust_z`
**Metric:** `grad_norm`
**Step:** 18
**Timestamp:** 2026-09-10T14:17:00

## Observed symptom

grad_norm=0.5184504240751266 is +5.5 robust-z from the rolling median of the last 17 updates

Value at detection: `0.5184504240751266`
Robust z-score vs rolling median: `5.51`


## Detection rule

`grad_norm_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=13 | reward_mean=0.171875 | kl=-4.905358991891262e-06 | entropy=0.31510356068611145 | clip_fraction=0.00014548951730830595 | grad_norm=0.22632461786270142 | response_length_mean=1591.3125 | truncation_rate=0.0078125 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=28.348288543522358 | t_actor_update=28.640123546123505
global_step=14 | reward_mean=-0.234375 | kl=1.1460655514383689e-05 | entropy=0.3702550232410431 | clip_fraction=0.00011572215771593619 | grad_norm=0.20105402171611786 | response_length_mean=2115.34375 | truncation_rate=0.0078125 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=54.37376821413636 | t_actor_update=38.989438984543085
global_step=15 | reward_mean=0.03125 | kl=1.3034779385634465e-05 | entropy=0.36637142300605774 | clip_fraction=0.00021083847605041228 | grad_norm=0.27505070716142654 | response_length_mean=1788.3359375 | truncation_rate=0.0078125 | zero_std_group_ratio=0.3125 | effective_signal_fraction=0.6875 | t_rollout=32.3476640060544 | t_actor_update=32.41146018356085
global_step=16 | reward_mean=0.109375 | kl=9.030292858369648e-06 | entropy=0.2910960018634796 | clip_fraction=0.00022808672747487435 | grad_norm=0.24120991677045822 | response_length_mean=1566.296875 | truncation_rate=0.0078125 | zero_std_group_ratio=0.375 | effective_signal_fraction=0.625 | t_rollout=38.35257479920983 | t_actor_update=28.895293936133385
global_step=17 | reward_mean=0.328125 | kl=2.7671848556565237e-05 | entropy=0.2980288863182068 | clip_fraction=0.00026073747721966356 | grad_norm=0.2576696574687958 | response_length_mean=1342.984375 | truncation_rate=0.0078125 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=28.342985339462757 | t_actor_update=24.87773536890745
```

## Metrics at the event

```
global_step=18 | reward_mean=0.0625 | kl=-2.1589140487776604e-05 | entropy=0.35837557911872864 | clip_fraction=0.00024938289789133705 | grad_norm=0.5184504240751266 | response_length_mean=1192.2265625 | truncation_rate=0.0078125 | zero_std_group_ratio=0.3125 | effective_signal_fraction=0.6875 | t_rollout=24.33597357943654 | t_actor_update=22.61376391351223
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
