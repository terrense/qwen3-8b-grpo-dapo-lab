# INC-20260911-182329-RESPONSE_LENGTH_MEAN_ROBUST_Z

**Level:** YELLOW
**Rule:** `response_length_mean_robust_z`
**Metric:** `response_length_mean`
**Step:** 17
**Timestamp:** 2026-09-11T18:23:29

## Observed symptom

response_length_mean=2755.6484375 is +4.0 robust-z from the rolling median of the last 16 updates

Value at detection: `2755.6484375`
Robust z-score vs rolling median: `4.03`


## Detection rule

`response_length_mean_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=12 | reward_mean=0.0625 | kl=4.321686446928652e-05 | entropy=0.3718275725841522 | clip_fraction=0.00016547724226256832 | grad_norm=0.25927501916885376 | response_length_mean=1673.6640625 | zero_std_group_ratio=0.375 | effective_signal_fraction=0.625 | t_rollout=44.3526823297143 | t_actor_update=32.14912939444184
global_step=13 | reward_mean=-0.15625 | kl=-1.943239931279095e-05 | entropy=0.3451754152774811 | clip_fraction=9.918875366565771e-05 | grad_norm=0.23142782598733902 | response_length_mean=2007.1640625 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=50.36082151904702 | t_actor_update=36.72901030629873
global_step=14 | reward_mean=-0.265625 | kl=1.322668890679779e-05 | entropy=0.3397117257118225 | clip_fraction=8.989179013951798e-05 | grad_norm=0.21603482216596603 | response_length_mean=2300.5546875 | zero_std_group_ratio=0.375 | effective_signal_fraction=0.625 | t_rollout=52.3633099719882 | t_actor_update=41.99770010262728
global_step=15 | reward_mean=-0.1875 | kl=1.932520535774529e-05 | entropy=0.3359094262123108 | clip_fraction=9.423057508683996e-05 | grad_norm=0.19749534130096436 | response_length_mean=2023.0546875 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=46.36168426647782 | t_actor_update=37.0955865085125
global_step=16 | reward_mean=0.03125 | kl=-7.047882078836665e-06 | entropy=0.3236340880393982 | clip_fraction=0.00018043336149276001 | grad_norm=0.24696137011051178 | response_length_mean=1958.59375 | zero_std_group_ratio=0.375 | effective_signal_fraction=0.625 | t_rollout=42.35294355079532 | t_actor_update=35.634336825460196
```

## Metrics at the event

```
global_step=17 | reward_mean=-0.359375 | kl=-4.1714662529557245e-05 | entropy=0.36769652366638184 | clip_fraction=7.021210634169013e-05 | grad_norm=0.17553916200995445 | response_length_mean=2755.6484375 | zero_std_group_ratio=0.5625 | effective_signal_fraction=0.4375 | t_rollout=62.39077754691243 | t_actor_update=50.86324157565832
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
