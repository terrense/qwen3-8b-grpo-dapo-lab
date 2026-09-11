# INC-20260911-103438-RESPONSE_LENGTH_MEAN_ROBUST_Z

**Level:** YELLOW
**Rule:** `response_length_mean_robust_z`
**Metric:** `response_length_mean`
**Step:** 19
**Timestamp:** 2026-09-11T10:34:38

## Observed symptom

response_length_mean=2239.7109375 is +5.0 robust-z from the rolling median of the last 18 updates

Value at detection: `2239.7109375`
Robust z-score vs rolling median: `5.02`


## Detection rule

`response_length_mean_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=14 | reward_mean=-0.109375 | kl=3.873669470522145e-05 | entropy=0.3521260619163513 | clip_fraction=0.00014145786053632037 | grad_norm=0.21038009226322174 | response_length_mean=1989.71875 | zero_std_group_ratio=0.5625 | effective_signal_fraction=0.4375 | t_rollout=46.36774395406246 | t_actor_update=36.666022308170795
global_step=15 | reward_mean=0.203125 | kl=2.0900563185932697e-05 | entropy=0.3365101218223572 | clip_fraction=0.0002347385398024926 | grad_norm=0.26516710221767426 | response_length_mean=1737.0703125 | zero_std_group_ratio=0.3125 | effective_signal_fraction=0.6875 | t_rollout=38.35471073165536 | t_actor_update=32.05062888562679
global_step=16 | reward_mean=0.015625 | kl=-2.1204627273618826e-05 | entropy=0.2862550616264343 | clip_fraction=0.00025412791092094267 | grad_norm=0.26675326377153397 | response_length_mean=1562.875 | zero_std_group_ratio=0.375 | effective_signal_fraction=0.625 | t_rollout=34.35309550911188 | t_actor_update=28.675437502563
global_step=17 | reward_mean=0.328125 | kl=8.696213853909285e-06 | entropy=0.3029800355434418 | clip_fraction=0.00024869250592018943 | grad_norm=0.19549734890460968 | response_length_mean=1449.265625 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=36.36557274311781 | t_actor_update=26.833979956805706
global_step=18 | reward_mean=0.328125 | kl=2.3882051550572214e-05 | entropy=0.32108473777770996 | clip_fraction=0.0002005701717280317 | grad_norm=0.3861839920282364 | response_length_mean=1279.2109375 | zero_std_group_ratio=0.375 | effective_signal_fraction=0.625 | t_rollout=28.34835122525692 | t_actor_update=24.401681065559387
```

## Metrics at the event

```
global_step=19 | reward_mean=-0.203125 | kl=1.1052591162297176e-06 | entropy=0.2936002314090729 | clip_fraction=0.0001006443312689953 | grad_norm=0.2350800856947899 | response_length_mean=2239.7109375 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=54.377830903977156 | t_actor_update=41.22613522410393
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
