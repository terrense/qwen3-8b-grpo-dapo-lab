# INC-20260911-163458-RESPONSE_LENGTH_MEAN_ROBUST_Z

**Level:** YELLOW
**Rule:** `response_length_mean_robust_z`
**Metric:** `response_length_mean`
**Step:** 17
**Timestamp:** 2026-09-11T16:34:58

## Observed symptom

response_length_mean=2578.5859375 is +5.2 robust-z from the rolling median of the last 16 updates

Value at detection: `2578.5859375`
Robust z-score vs rolling median: `5.21`


## Detection rule

`response_length_mean_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=12 | reward_mean=-0.046875 | kl=-5.105393188387097e-05 | entropy=0.4024648666381836 | clip_fraction=0.0001573630133862025 | grad_norm=0.29083656519651413 | response_length_mean=1712.375 | zero_std_group_ratio=0.3125 | effective_signal_fraction=0.6875 | t_rollout=32.39371655881405 | t_actor_update=31.436130687594414
global_step=13 | reward_mean=-0.28125 | kl=1.1058449194134568e-05 | entropy=0.3351052403450012 | clip_fraction=0.00014996995696492377 | grad_norm=0.2732975706458092 | response_length_mean=1916.4453125 | zero_std_group_ratio=0.25 | effective_signal_fraction=0.75 | t_rollout=48.37846424430609 | t_actor_update=35.64451490342617
global_step=14 | reward_mean=-0.234375 | kl=-2.0474693059213678e-05 | entropy=0.34207940101623535 | clip_fraction=7.721060319454409e-05 | grad_norm=0.1591385081410408 | response_length_mean=2406.015625 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=50.37621448934078 | t_actor_update=43.75125392526388
global_step=15 | reward_mean=-0.09375 | kl=-3.6391720868778066e-05 | entropy=0.37445035576820374 | clip_fraction=6.588529845430457e-05 | grad_norm=0.14014475792646408 | response_length_mean=1745.5390625 | zero_std_group_ratio=0.8125 | effective_signal_fraction=0.1875 | t_rollout=36.36248426511884 | t_actor_update=32.15696446225047
global_step=16 | reward_mean=0.0625 | kl=2.5979528572861454e-05 | entropy=0.32513898611068726 | clip_fraction=0.00014437146182899596 | grad_norm=0.23526539653539658 | response_length_mean=1864.1171875 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=50.387399442493916 | t_actor_update=34.081021286547184
```

## Metrics at the event

```
global_step=17 | reward_mean=-0.34375 | kl=-3.338814158837522e-05 | entropy=0.39368289709091187 | clip_fraction=6.662805299129104e-05 | grad_norm=0.1324857398867607 | response_length_mean=2578.5859375 | zero_std_group_ratio=0.75 | effective_signal_fraction=0.25 | t_rollout=58.38428305462003 | t_actor_update=47.41960859671235
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
