# INC-20260910-140540-RESPONSE_LENGTH_MEAN_ROBUST_Z

**Level:** YELLOW
**Rule:** `response_length_mean_robust_z`
**Metric:** `response_length_mean`
**Step:** 10
**Timestamp:** 2026-09-10T14:05:40

## Observed symptom

response_length_mean=2188.109375 is +6.2 robust-z from the rolling median of the last 9 updates

Value at detection: `2188.109375`
Robust z-score vs rolling median: `6.16`


## Detection rule

`response_length_mean_robust_z` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
global_step=5 | reward_mean=-0.28125 | kl=-1.810004937397025e-05 | entropy=0.22545349597930908 | clip_fraction=5.0717918611553614e-05 | grad_norm=0.2179325707256794 | response_length_mean=1804.2734375 | truncation_rate=0.0234375 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=50.392642606049776 | t_actor_update=34.16155535727739
global_step=6 | reward_mean=0.34375 | kl=2.4713042421353748e-05 | entropy=0.305340975522995 | clip_fraction=0.0001676934443821665 | grad_norm=0.2861437052488327 | response_length_mean=1384.453125 | truncation_rate=0.0078125 | zero_std_group_ratio=0.3125 | effective_signal_fraction=0.6875 | t_rollout=28.350442733615637 | t_actor_update=25.526083558797836
global_step=7 | reward_mean=0.078125 | kl=-2.9148652629373828e-05 | entropy=0.3664397597312927 | clip_fraction=0.00018549183732829988 | grad_norm=0.2879910320043564 | response_length_mean=1803.25 | truncation_rate=0.0078125 | zero_std_group_ratio=0.25 | effective_signal_fraction=0.75 | t_rollout=42.36393387243152 | t_actor_update=33.540443036705256
global_step=8 | reward_mean=-0.125 | kl=3.236113752791425e-05 | entropy=0.27066776156425476 | clip_fraction=0.00012427335695974762 | grad_norm=0.23071860522031784 | response_length_mean=1608.3984375 | truncation_rate=0.0078125 | zero_std_group_ratio=0.4375 | effective_signal_fraction=0.5625 | t_rollout=34.35660845041275 | t_actor_update=29.540890265256166
global_step=9 | reward_mean=-0.1875 | kl=-9.220324500347488e-06 | entropy=0.341254323720932 | clip_fraction=0.00016498915010743076 | grad_norm=0.306532546877861 | response_length_mean=1665.625 | truncation_rate=0.0078125 | zero_std_group_ratio=0.25 | effective_signal_fraction=0.75 | t_rollout=40.389205541461706 | t_actor_update=31.086185909807682
```

## Metrics at the event

```
global_step=10 | reward_mean=-0.265625 | kl=-9.19265494303545e-06 | entropy=0.3094458281993866 | clip_fraction=0.00013155156352695485 | grad_norm=0.2354121282696724 | response_length_mean=2188.109375 | truncation_rate=0.0078125 | zero_std_group_ratio=0.25 | effective_signal_fraction=0.75 | t_rollout=44.35808978974819 | t_actor_update=39.853838447481394
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
