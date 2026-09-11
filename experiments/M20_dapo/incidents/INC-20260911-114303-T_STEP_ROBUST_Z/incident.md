# INC-20260911-114303-T_STEP_ROBUST_Z

**Level:** YELLOW
**Rule:** `t_step_robust_z`
**Metric:** `t_step`
**Step:** 13
**Timestamp:** 2026-09-11T11:43:03

## Observed symptom

t_step=163.70084330067039 is +4.3 robust-z from the rolling median of the last 12 updates

Value at detection: `163.70084330067039`
Robust z-score vs rolling median: `4.34`


## Detection rule

`t_step_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=8 | reward_mean=0.046875 | kl=-4.4034480197296944e-05 | entropy=0.34535250067710876 | clip_fraction=0.00010978344653267413 | grad_norm=0.3062981218099594 | response_length_mean=1518.78125 | zero_std_group_ratio=0.0 | effective_signal_fraction=1.0 | t_rollout=64.24667801335454 | t_actor_update=27.67669190093875
global_step=9 | reward_mean=0.15625 | kl=2.931546760009951e-05 | entropy=0.3104524612426758 | clip_fraction=0.0001368030807498144 | grad_norm=0.35399432480335236 | response_length_mean=1262.8671875 | zero_std_group_ratio=0.0 | effective_signal_fraction=1.0 | t_rollout=51.99750545993447 | t_actor_update=23.576386857777834
global_step=10 | reward_mean=-0.046875 | kl=-2.0642399931602995e-05 | entropy=0.29235848784446716 | clip_fraction=0.000144801517308224 | grad_norm=0.2692812532186508 | response_length_mean=1754.953125 | zero_std_group_ratio=0.0625 | effective_signal_fraction=0.9375 | t_rollout=78.86132263019681 | t_actor_update=32.091973543167114
global_step=11 | reward_mean=0.046875 | kl=-2.2816129103375715e-05 | entropy=0.28499433398246765 | clip_fraction=0.0001033176133660163 | grad_norm=0.23913776874542236 | response_length_mean=1964.890625 | zero_std_group_ratio=0.0 | effective_signal_fraction=1.0 | t_rollout=66.4930972456932 | t_actor_update=36.67240938171744
global_step=12 | reward_mean=0.09375 | kl=8.717220578091656e-05 | entropy=0.3074379563331604 | clip_fraction=0.00017324248619843274 | grad_norm=0.3718007206916809 | response_length_mean=1601.171875 | zero_std_group_ratio=0.0 | effective_signal_fraction=1.0 | t_rollout=87.59418372437358 | t_actor_update=29.03269987180829
```

## Metrics at the event

```
global_step=13 | reward_mean=0.0155792236328125 | kl=-3.132063193334034e-05 | entropy=0.3927077651023865 | clip_fraction=0.00011857001209136797 | grad_norm=0.27804891020059586 | response_length_mean=2156.3046875 | zero_std_group_ratio=0.0 | effective_signal_fraction=1.0 | t_rollout=109.28300446644425 | t_actor_update=39.4720222838223
```

## Initial hypotheses

Work the checklist in [`docs/observability/diagnosis_playbook.md`](../../../../docs/observability/diagnosis_playbook.md)
in order — environment/verifier, data batch composition, truncation, policy-version
freshness, old-logprob correctness, token masking, reward normalisation, optimizer,
LR, and only then KL/clipping.

## Evidence collected

- `metric_window.csv` — 13 updates around the event
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
