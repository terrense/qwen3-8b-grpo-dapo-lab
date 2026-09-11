# INC-20260911-172239-RESPONSE_LENGTH_MEAN_ROBUST_Z

**Level:** YELLOW
**Rule:** `response_length_mean_robust_z`
**Metric:** `response_length_mean`
**Step:** 14
**Timestamp:** 2026-09-11T17:22:39

## Observed symptom

response_length_mean=2376.9140625 is +6.6 robust-z from the rolling median of the last 13 updates

Value at detection: `2376.9140625`
Robust z-score vs rolling median: `6.59`


## Detection rule

`response_length_mean_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=9 | reward_mean=-0.203125 | kl=3.0999897603578574e-05 | entropy=0.2798762917518616 | clip_fraction=0.0 | grad_norm=0.22114261984825134 | response_length_mean=1923.3515625 | zero_std_group_ratio=0.25 | effective_signal_fraction=0.75 | t_rollout=36.3521593362093 | t_actor_update=34.71122531592846
global_step=10 | reward_mean=0.078125 | kl=3.456394586009992e-05 | entropy=0.26776599884033203 | clip_fraction=0.0 | grad_norm=0.2814696952700615 | response_length_mean=1720.484375 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=44.37108999863267 | t_actor_update=32.03460555151105
global_step=11 | reward_mean=0.09375 | kl=8.639030113499757e-06 | entropy=0.27261072397232056 | clip_fraction=0.0 | grad_norm=0.2022116780281067 | response_length_mean=1641.609375 | zero_std_group_ratio=0.5625 | effective_signal_fraction=0.4375 | t_rollout=48.449895560741425 | t_actor_update=31.38540717959404
global_step=12 | reward_mean=0.0 | kl=-3.7554465961875394e-05 | entropy=0.3816106617450714 | clip_fraction=0.0 | grad_norm=0.35137222707271576 | response_length_mean=1697.9375 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=40.360168132930994 | t_actor_update=31.59568899869919
global_step=13 | reward_mean=-0.109375 | kl=-6.627140891168892e-05 | entropy=0.35682058334350586 | clip_fraction=0.0 | grad_norm=0.24432548135519028 | response_length_mean=1748.359375 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=40.37087306752801 | t_actor_update=32.4125188626349
```

## Metrics at the event

```
global_step=14 | reward_mean=-0.296875 | kl=4.58798126601323e-07 | entropy=0.3376767039299011 | clip_fraction=0.0 | grad_norm=0.16201301664113998 | response_length_mean=2376.9140625 | zero_std_group_ratio=0.625 | effective_signal_fraction=0.375 | t_rollout=58.3813852109015 | t_actor_update=43.31100440770388
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
