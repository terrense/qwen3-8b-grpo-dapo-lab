# INC-20260910-141140-RESPONSE_LENGTH_MEAN_ROBUST_Z

**Level:** YELLOW
**Rule:** `response_length_mean_robust_z`
**Metric:** `response_length_mean`
**Step:** 14
**Timestamp:** 2026-09-10T14:11:40

## Observed symptom

response_length_mean=2115.34375 is +5.3 robust-z from the rolling median of the last 13 updates

Value at detection: `2115.34375`
Robust z-score vs rolling median: `5.3`


## Detection rule

`response_length_mean_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=9 | reward_mean=-0.1875 | kl=-9.220324500347488e-06 | entropy=0.341254323720932 | clip_fraction=0.00016498915010743076 | grad_norm=0.306532546877861 | response_length_mean=1665.625 | truncation_rate=0.0078125 | zero_std_group_ratio=0.25 | effective_signal_fraction=0.75 | t_rollout=40.389205541461706 | t_actor_update=31.086185909807682
global_step=10 | reward_mean=-0.265625 | kl=-9.19265494303545e-06 | entropy=0.3094458281993866 | clip_fraction=0.00013155156352695485 | grad_norm=0.2354121282696724 | response_length_mean=2188.109375 | truncation_rate=0.0078125 | zero_std_group_ratio=0.25 | effective_signal_fraction=0.75 | t_rollout=44.35808978974819 | t_actor_update=39.853838447481394
global_step=11 | reward_mean=0.125 | kl=-1.8726126540968835e-05 | entropy=0.38999059796333313 | clip_fraction=0.00018753468430077191 | grad_norm=0.25863051414489746 | response_length_mean=1615.0 | truncation_rate=0.0078125 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=34.40260545164347 | t_actor_update=30.245607297867537
global_step=12 | reward_mean=0.0 | kl=-3.106643725914182e-05 | entropy=0.28067320585250854 | clip_fraction=0.0002040405988736893 | grad_norm=0.2827133610844612 | response_length_mean=1697.4609375 | truncation_rate=0.0078125 | zero_std_group_ratio=0.375 | effective_signal_fraction=0.625 | t_rollout=30.350323017686605 | t_actor_update=30.322958894073963
global_step=13 | reward_mean=0.171875 | kl=-4.905358991891262e-06 | entropy=0.31510356068611145 | clip_fraction=0.00014548951730830595 | grad_norm=0.22632461786270142 | response_length_mean=1591.3125 | truncation_rate=0.0078125 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=28.348288543522358 | t_actor_update=28.640123546123505
```

## Metrics at the event

```
global_step=14 | reward_mean=-0.234375 | kl=1.1460655514383689e-05 | entropy=0.3702550232410431 | clip_fraction=0.00011572215771593619 | grad_norm=0.20105402171611786 | response_length_mean=2115.34375 | truncation_rate=0.0078125 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=54.37376821413636 | t_actor_update=38.989438984543085
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
