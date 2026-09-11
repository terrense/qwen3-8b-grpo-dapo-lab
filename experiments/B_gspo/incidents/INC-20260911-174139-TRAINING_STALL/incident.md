# INC-20260911-174139-TRAINING_STALL

**Level:** YELLOW
**Rule:** `training_stall`
**Metric:** `global_step`
**Step:** 19
**Timestamp:** 2026-09-11T17:41:39

## Observed symptom

no new optimizer step for 600s (adaptive threshold 600s = max(3x median update, 600s))

Value at detection: `19`



## Detection rule

`training_stall` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=14 | reward_mean=-0.296875 | kl=4.58798126601323e-07 | entropy=0.3376767039299011 | clip_fraction=0.0 | grad_norm=0.16201301664113998 | response_length_mean=2376.9140625 | zero_std_group_ratio=0.625 | effective_signal_fraction=0.375 | t_rollout=58.3813852109015 | t_actor_update=43.31100440770388
global_step=15 | reward_mean=-0.125 | kl=-1.7480514486578613e-05 | entropy=0.34087350964546204 | clip_fraction=0.0 | grad_norm=0.15469012036919594 | response_length_mean=1809.1953125 | zero_std_group_ratio=0.625 | effective_signal_fraction=0.375 | t_rollout=44.36967374384403 | t_actor_update=33.6256600022316
global_step=16 | reward_mean=0.09375 | kl=-1.490263503001188e-05 | entropy=0.3252444863319397 | clip_fraction=0.0 | grad_norm=0.2755322754383087 | response_length_mean=1856.25 | zero_std_group_ratio=0.3125 | effective_signal_fraction=0.6875 | t_rollout=50.37097604572773 | t_actor_update=34.513098914176226
global_step=17 | reward_mean=-0.3125 | kl=-4.597625320457155e-06 | entropy=0.3903157711029053 | clip_fraction=0.0 | grad_norm=0.19470329955220222 | response_length_mean=2552.296875 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=54.37679713591933 | t_actor_update=46.62357620149851
global_step=18 | reward_mean=0.015625 | kl=-2.0426715764187975e-05 | entropy=0.37579160928726196 | clip_fraction=0.0 | grad_norm=0.26482222974300385 | response_length_mean=1856.2421875 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=42.36222492530942 | t_actor_update=34.12850560620427
```

## Metrics at the event

```
global_step=19 | reward_mean=-0.046875 | kl=-2.0682647800640552e-05 | entropy=0.26873958110809326 | clip_fraction=0.0 | grad_norm=0.220686636865139 | response_length_mean=2032.4140625 | zero_std_group_ratio=0.375 | effective_signal_fraction=0.625 | t_rollout=50.3652253523469 | t_actor_update=37.53342678025365
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
