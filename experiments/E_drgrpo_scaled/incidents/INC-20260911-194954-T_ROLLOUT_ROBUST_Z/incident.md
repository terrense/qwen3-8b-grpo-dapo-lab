# INC-20260911-194954-T_ROLLOUT_ROBUST_Z

**Level:** YELLOW
**Rule:** `t_rollout_robust_z`
**Metric:** `t_rollout`
**Step:** 8
**Timestamp:** 2026-09-11T19:49:55

## Observed symptom

t_rollout=48.367967408150434 is +112.5 robust-z from the rolling median of the last 7 updates

Value at detection: `48.367967408150434`
Robust z-score vs rolling median: `112.53`


## Detection rule

`t_rollout_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=3 | reward_mean=-0.46875 | kl=-1.824102332648181e-05 | entropy=0.3553938567638397 | clip_fraction=5.771448240921018e-05 | grad_norm=0.16950258612632751 | response_length_mean=1955.0078125 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=44.36041585728526 | t_actor_update=37.392739068716764
global_step=4 | reward_mean=-0.09375 | kl=2.0949772533640498e-05 | entropy=0.24460768699645996 | clip_fraction=9.196723704008036e-05 | grad_norm=0.1634281426668167 | response_length_mean=1666.359375 | zero_std_group_ratio=0.3125 | effective_signal_fraction=0.6875 | t_rollout=46.371352173388004 | t_actor_update=31.39823703095317
global_step=5 | reward_mean=-0.015625 | kl=-2.149262945749797e-05 | entropy=0.26476800441741943 | clip_fraction=0.00018846634702640586 | grad_norm=0.20192071050405502 | response_length_mean=1704.1171875 | zero_std_group_ratio=0.125 | effective_signal_fraction=0.875 | t_rollout=30.34441814571619 | t_actor_update=30.967726048082113
global_step=6 | reward_mean=-0.3125 | kl=-1.451880848435394e-05 | entropy=0.2818373441696167 | clip_fraction=0.00015817550320207374 | grad_norm=0.20302365720272064 | response_length_mean=1680.3203125 | zero_std_group_ratio=0.25 | effective_signal_fraction=0.75 | t_rollout=46.3612158857286 | t_actor_update=32.23388513177633
global_step=7 | reward_mean=0.078125 | kl=2.2035671804587764e-05 | entropy=0.26395779848098755 | clip_fraction=0.0001316406301157258 | grad_norm=0.17521021887660027 | response_length_mean=1848.328125 | zero_std_group_ratio=0.5 | effective_signal_fraction=0.5 | t_rollout=46.373244162648916 | t_actor_update=33.70860982686281
```

## Metrics at the event

```
global_step=8 | reward_mean=0.03125 | kl=3.5676871448231395e-06 | entropy=0.2786373198032379 | clip_fraction=0.00016874601942618028 | grad_norm=0.184847179800272 | response_length_mean=1790.7734375 | zero_std_group_ratio=0.375 | effective_signal_fraction=0.625 | t_rollout=48.367967408150434 | t_actor_update=33.23520312458277
```

## Initial hypotheses

Work the checklist in [`docs/observability/diagnosis_playbook.md`](../../../../docs/observability/diagnosis_playbook.md)
in order — environment/verifier, data batch composition, truncation, policy-version
freshness, old-logprob correctness, token masking, reward normalisation, optimizer,
LR, and only then KL/clipping.

## Evidence collected

- `metric_window.csv` — 8 updates around the event
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
