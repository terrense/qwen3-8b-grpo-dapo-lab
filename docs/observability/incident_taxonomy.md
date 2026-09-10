# Incident Taxonomy

Three levels. Level says how urgent, **not** what is wrong — the cause is determined by
a human using [`diagnosis_playbook.md`](diagnosis_playbook.md).

| level | meaning | automatic action |
|---|---|---|
| **GREEN** | nothing tripped | none |
| **YELLOW** | something anomalous; training continues | freeze an evidence bundle |
| **RED** | hard failure, or a signal that makes the update meaningless | freeze bundle, surface immediately |

**No level ever changes a training parameter.** The detector observes; a human decides.
Auto-tuning on an alert would make the alert unfalsifiable.

## RED — hard failures

| rule | trigger | why it is RED |
|---|---|---|
| `non_finite_metric` | NaN/Inf in `policy_loss`, `reward_mean`, `kl`, `entropy`, `grad_norm` | the update is already corrupt; continuing writes garbage into the weights |
| `no_effective_signal` | `effective_signal_fraction == 0` | every group has zero reward variance, so the GRPO advantage is identically zero — the step teaches nothing |
| `cuda_oom` | `CUDA out of memory` / `torch.OutOfMemoryError` in the trainer log | |
| `nccl_error`, `nccl_timeout` | `NCCL error`, watchdog collective timeout | |
| `ray_actor_death`, `ray_oom` | `RayActorError`, actor died, Ray OOM | |
| `vllm_crash` | `EngineCore encountered a fatal error`, `AsyncEngineDeadError` | |
| `trainer_process_gone` | training PID no longer alive | |
| `system_disk_full` | `/` above 90% | 30 GB system disk; overflow corrupts the run |
| `data_disk_full` | data disk above 95% | |
| non-zero trainer exit | propagated by `run_with_observer.sh` | the exit code is never swallowed |

## YELLOW — anomalies worth explaining

Adaptive (robust z-score above 4.0 against a rolling median of the last 20 updates,
MAD-scaled, abstaining until 6 updates of history exist):

`reward_mean` (either direction), `kl` up, `entropy` down, `clip_fraction` up,
`grad_norm` up, `response_length_mean` up, `truncation_rate` up,
`zero_std_group_ratio` up, `t_rollout` up, `t_step` up.

Absolute heuristics:

| rule | threshold |
|---|---|
| `truncation_rate_above_heuristic` | `> 0.30` |
| `zero_std_group_ratio_above_heuristic` | `> 0.70` |
| `effective_signal_low` | `< 0.30` |
| `clip_fraction_above_heuristic` | `> 0.40` |
| `training_stall` | no new step for `max(3 x median update time, 10 min)` |

Thresholds are **diagnostic heuristics, not theoretical constants** — see
[`metric_dictionary.md`](metric_dictionary.md).

## Why the stall rule is adaptive

A fixed "5 minutes without a step" alarm is wrong for RL: a single GRPO update with long
rollouts can legitimately exceed it. The threshold is derived from the run's own observed
median update time, with a 10-minute floor for the first few updates when no history
exists yet.

## Incident bundle

Every firing creates `experiments/<RUN_ID>/incidents/INC-<timestamp>-<TYPE>/`:

```
incident.md                 symptom, rule, step, metrics before/at the event
finding.json                the raw detector finding
metric_window.csv           ~15 updates either side
system_window.csv           telemetry, last 600 s
last_log_lines.txt          trainer log tail (200 lines)
nvidia_smi.txt              GPU state at capture time
process_snapshot.txt        ps aux
ray_status.txt              when Ray is up
resolved_config.yaml        the config actually in force
run_manifest_snapshot.json  provenance
trajectories.jsonl          sampled rollouts, when small enough
```

`incident.md` is generated with **`Root cause: PENDING`**, **`Fix: PENDING`**,
**`Post-fix evidence: PENDING`**.

This is deliberate. An automatically generated root cause would be a guess wearing the
costume of a finding, and this repository's whole purpose is separating real causes from
plausible-sounding ones. A human fills those three fields in, and the confirmed result is
promoted to [`analysis/incident_log.md`](../../analysis/incident_log.md).

## Relationship to the failure taxonomy

Incidents are *detections*. Causes fall into six families, and the mapping is
deliberately many-to-many — which is exactly why the detector does not guess:

| family | examples seen or expected |
|---|---|
| **Data** | DAPO-Math-17k 100x prompt repetition; duplicate prompts inside one batch |
| **Reward / verifier** | parse failure counted as a wrong answer; the 300-char extraction window; trailing-period false negative |
| **Policy** | entropy collapse, reward hacking |
| **Optimization** | LR too high; grad-norm and KL spiking together |
| **Rollout** | truncation contamination; length explosion; KV-cache pressure |
| **Infrastructure** | OOM, NCCL, Ray/vLLM death, disk pressure, the network faults in INC-001/002/003 |

A single symptom (reward down) can come from any of the six. That is the entire reason
this system records the other five families rather than only the reward curve.
