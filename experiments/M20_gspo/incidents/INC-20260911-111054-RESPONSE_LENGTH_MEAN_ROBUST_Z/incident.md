# INC-20260911-111054-RESPONSE_LENGTH_MEAN_ROBUST_Z

**Level:** YELLOW
**Rule:** `response_length_mean_robust_z`
**Metric:** `response_length_mean`
**Step:** 19
**Timestamp:** 2026-09-11T11:10:54

## Observed symptom

response_length_mean=2256.8984375 is +4.0 robust-z from the rolling median of the last 18 updates

Value at detection: `2256.8984375`
Robust z-score vs rolling median: `4.03`


## Detection rule

`response_length_mean_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=14 | reward_mean=-0.109375 | kl=-1.3872350848487258e-05 | entropy=0.35176849365234375 | clip_fraction=0.0 | grad_norm=0.22716522216796875 | response_length_mean=1971.328125 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=44.36464321613312 | t_actor_update=36.41251276060939
global_step=15 | reward_mean=0.125 | kl=-2.297642147652823e-05 | entropy=0.3267538845539093 | clip_fraction=0.0 | grad_norm=0.2509043738245964 | response_length_mean=1801.875 | zero_std_group_ratio=0.375 | effective_signal_fraction=0.625 | t_rollout=36.35550071299076 | t_actor_update=32.897204510867596
global_step=16 | reward_mean=-0.046875 | kl=-1.5149614000620204e-05 | entropy=0.2653578519821167 | clip_fraction=0.0 | grad_norm=0.2701319754123688 | response_length_mean=1607.6171875 | zero_std_group_ratio=0.375 | effective_signal_fraction=0.625 | t_rollout=40.43523773550987 | t_actor_update=29.570729285478592
global_step=17 | reward_mean=0.359375 | kl=-1.3955673239252064e-05 | entropy=0.26181477308273315 | clip_fraction=0.0 | grad_norm=0.2962905168533325 | response_length_mean=1392.5625 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=38.355517867952585 | t_actor_update=26.058740869164467
global_step=18 | reward_mean=0.125 | kl=-4.1072116800933145e-06 | entropy=0.3241255283355713 | clip_fraction=0.0 | grad_norm=0.4252918064594269 | response_length_mean=1345.3203125 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=28.384702894836664 | t_actor_update=25.553657293319702
```

## Metrics at the event

```
global_step=19 | reward_mean=-0.359375 | kl=1.5032626571098717e-05 | entropy=0.30319496989250183 | clip_fraction=0.0 | grad_norm=0.2465992346405983 | response_length_mean=2256.8984375 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=50.37072657421231 | t_actor_update=41.33571203798056
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
