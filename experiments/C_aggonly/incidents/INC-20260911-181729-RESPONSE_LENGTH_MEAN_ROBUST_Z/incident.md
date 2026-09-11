# INC-20260911-181729-RESPONSE_LENGTH_MEAN_ROBUST_Z

**Level:** YELLOW
**Rule:** `response_length_mean_robust_z`
**Metric:** `response_length_mean`
**Step:** 14
**Timestamp:** 2026-09-11T18:17:29

## Observed symptom

response_length_mean=2300.5546875 is +4.9 robust-z from the rolling median of the last 13 updates

Value at detection: `2300.5546875`
Robust z-score vs rolling median: `4.95`


## Detection rule

`response_length_mean_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=9 | reward_mean=-0.046875 | kl=-2.857679135104263e-06 | entropy=0.2941012978553772 | clip_fraction=0.00019352874096512096 | grad_norm=0.25336113572120667 | response_length_mean=2119.6328125 | zero_std_group_ratio=0.25 | effective_signal_fraction=0.75 | t_rollout=48.36923759430647 | t_actor_update=38.08037879317999
global_step=10 | reward_mean=0.0625 | kl=-1.7849693335847405e-05 | entropy=0.2715272009372711 | clip_fraction=0.00013816330465488136 | grad_norm=0.22777851670980453 | response_length_mean=1709.0859375 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=32.34203789010644 | t_actor_update=31.47299624606967
global_step=11 | reward_mean=-0.015625 | kl=-4.858684837927285e-05 | entropy=0.26987671852111816 | clip_fraction=0.00010935277191492787 | grad_norm=0.2105427160859108 | response_length_mean=1717.2109375 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=38.36486475169659 | t_actor_update=31.530048821121454
global_step=12 | reward_mean=0.0625 | kl=4.321686446928652e-05 | entropy=0.3718275725841522 | clip_fraction=0.00016547724226256832 | grad_norm=0.25927501916885376 | response_length_mean=1673.6640625 | zero_std_group_ratio=0.375 | effective_signal_fraction=0.625 | t_rollout=44.3526823297143 | t_actor_update=32.14912939444184
global_step=13 | reward_mean=-0.15625 | kl=-1.943239931279095e-05 | entropy=0.3451754152774811 | clip_fraction=9.918875366565771e-05 | grad_norm=0.23142782598733902 | response_length_mean=2007.1640625 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=50.36082151904702 | t_actor_update=36.72901030629873
```

## Metrics at the event

```
global_step=14 | reward_mean=-0.265625 | kl=1.322668890679779e-05 | entropy=0.3397117257118225 | clip_fraction=8.989179013951798e-05 | grad_norm=0.21603482216596603 | response_length_mean=2300.5546875 | zero_std_group_ratio=0.375 | effective_signal_fraction=0.625 | t_rollout=52.3633099719882 | t_actor_update=41.99770010262728
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
