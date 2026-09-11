# INC-20260911-105734-RESPONSE_LENGTH_MEAN_ROBUST_Z

**Level:** YELLOW
**Rule:** `response_length_mean_robust_z`
**Metric:** `response_length_mean`
**Step:** 10
**Timestamp:** 2026-09-11T10:57:34

## Observed symptom

response_length_mean=2213.359375 is +5.6 robust-z from the rolling median of the last 9 updates

Value at detection: `2213.359375`
Robust z-score vs rolling median: `5.59`


## Detection rule

`response_length_mean_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=5 | reward_mean=-0.15625 | kl=-1.4010068980496726e-05 | entropy=0.2537659704685211 | clip_fraction=0.0 | grad_norm=0.2939658910036087 | response_length_mean=1699.2421875 | zero_std_group_ratio=0.25 | effective_signal_fraction=0.75 | t_rollout=46.36061515286565 | t_actor_update=31.730163596570492
global_step=6 | reward_mean=0.125 | kl=-5.3005516065240954e-05 | entropy=0.277549684047699 | clip_fraction=0.0 | grad_norm=0.2703757584095001 | response_length_mean=1454.6875 | zero_std_group_ratio=0.3125 | effective_signal_fraction=0.6875 | t_rollout=36.351540952920914 | t_actor_update=26.936474196612835
global_step=7 | reward_mean=-0.0625 | kl=1.7296298409519295e-05 | entropy=0.3398197591304779 | clip_fraction=0.0 | grad_norm=0.26745806634426117 | response_length_mean=1686.8671875 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=38.35101280361414 | t_actor_update=31.563385855406523
global_step=8 | reward_mean=-0.125 | kl=2.8191903282959174e-05 | entropy=0.25567910075187683 | clip_fraction=0.0 | grad_norm=0.20518256723880768 | response_length_mean=1646.71875 | zero_std_group_ratio=0.5625 | effective_signal_fraction=0.4375 | t_rollout=40.398606926202774 | t_actor_update=30.291284080594778
global_step=9 | reward_mean=-0.140625 | kl=-1.1764546115955454e-05 | entropy=0.34807682037353516 | clip_fraction=0.0 | grad_norm=0.27455244958400726 | response_length_mean=1583.9140625 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=38.35667223855853 | t_actor_update=29.766290422528982
```

## Metrics at the event

```
global_step=10 | reward_mean=-0.203125 | kl=-1.9672261714731576e-05 | entropy=0.30070775747299194 | clip_fraction=0.0 | grad_norm=0.1639334186911583 | response_length_mean=2213.359375 | zero_std_group_ratio=0.625 | effective_signal_fraction=0.375 | t_rollout=40.34968816116452 | t_actor_update=40.297325640916824
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
