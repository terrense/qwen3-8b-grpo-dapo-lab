# Run Report — R2_dapo_smoke

**Run UID:** `R2_dapo_smoke-20260910-171600`
**Outcome:** **COMPLETED**
**Duration:** 6.4 min
**Optimizer updates recorded:** 2
**Generated:** 2026-09-10T17:24:40

## Configuration

| | |
|---|---|
| Algorithm | DAPO |
| Model | Qwen/Qwen3-8B |
| Dataset | BytedTsinghua-SIA/DAPO-Math-17k |
| VeRL commit | `1252cc71aa5bd82e5604322064d69bfe6454c660` |
| Lab commit at start | `639497a64a6b1b0d4ab0cc6891f64d181d76c1a9` |
| Seed | 20260910 |
| Started | 2026-09-10T17:16:00 |
| Exit code | `0` |

Launch command: `command.txt`. When present, `launch_overrides.json` records overrides extracted from the actual trainer log; it is not a full resolved configuration. Check `resolved_config.yaml` before treating it as evidence (R1 originally contained only a placeholder).

## Outcome

The launcher recorded trainer exit code 0. Exit code alone does not rule out teardown warnings; see the analysis report.

## Training Dynamics

| metric | trajectory |
|---|---|
| policy loss | first 0.0260 -> last -0.0166 (min -0.0166, max 0.0260, mean 0.0047) |
| learning rate | first 1.00e-06 -> last 1.00e-06 (min 1.00e-06, max 1.00e-06, mean 1.00e-06) |
| grad norm | first 0.2879 -> last 0.2918 (min 0.2879, max 0.2918, mean 0.2899) |

## Reward & Group Signal

| metric | trajectory |
|---|---|
| reward mean | first -0.0473 -> last 0.0469 (min -0.0473, max 0.0469, mean -0.0002) |
| reward std | first 0.9969 -> last 0.9851 (min 0.9851, max 0.9969, mean 0.9910) |
| mixed group ratio | first 1.000 -> last 1.000 (min 1.000, max 1.000, mean 1.000) |
| all-correct group ratio | first 0.000 -> last 0.000 (min 0.000, max 0.000, mean 0.000) |
| all-wrong group ratio | first 0.000 -> last 0.000 (min 0.000, max 0.000, mean 0.000) |
| zero-std group ratio | first 0.000 -> last 0.000 (min 0.000, max 0.000, mean 0.000) |
| effective signal fraction | first 1.000 -> last 1.000 (min 1.000, max 1.000, mean 1.000) |

Zero-variance groups produce an identically zero GRPO advantage, so
`effective_signal_fraction` is the fraction of the nominal prompt batch that
actually contributed gradient. See `figures/04_group_signal.png`.

## Policy Dynamics

| metric | trajectory |
|---|---|
| KL (`actor/ppo_kl`) | first -0.00002 -> last -0.00007 (min -0.00007, max -0.00002, mean -0.00004) |
| entropy | first 0.3446 -> last 0.3406 (min 0.3406, max 0.3446, mean 0.3426) |
| PPO objective clip fraction | first 0.0001 -> last 0.0002 (min 0.0001, max 0.0002, mean 0.0001) |
| directional lower clip (not emitted) | unavailable |
| directional upper clip (not emitted) | unavailable |
| advantage mean | first -0.0245 -> last 0.0147 (min -0.0245, max 0.0147, mean -0.0049) |

See `figures/02_policy_dynamics.png`, `figures/03_ratio_grad.png`.

## Generation Dynamics

| metric | trajectory |
|---|---|
| response length mean | first 1742.8 -> last 1571.3 (min 1571.3, max 1742.8, mean 1657.1) |
| response length max | first 6260 -> last 5658 (min 5658, max 6260, mean 5959) |
| configured-cap hit proxy | unavailable |
| exact truncation / finish reason | unavailable |

See `figures/05_length_dynamics.png`.

## Systems Performance

| stage | mean share of update wall time |
|---|---|
| rollout | 62.2 % |
| verifier | 0.0 % |
| logprob | unavailable |
| actor update | 26.5 % |
| weight sync | 3.4 % |
| other | 0.5 % |

| metric | trajectory |
|---|---|
| tokens/sec | first 507 -> last 492 (min 492, max 507, mean 499) |
| step wall time (s) | first 119.4 -> last 112.4 (min 112.4, max 119.4, mean 115.9) |
| peak VRAM GPU0 (GiB) | first 84.3 -> last 84.3 (min 84.3, max 84.3, mean 84.3) |

**Policy provenance / staleness:** unavailable. For synchronous GRPO this should
be a constant; a drifting value would mean rollouts are being consumed by a
different policy version than the one that generated them.

See `figures/06_system_timing.png`, `figures/07_gpu_memory_util.png`.

## Incidents

None detected by the flight recorder during this run.

## Root Causes

Not applicable — no incidents.

## Recovery / Fixes

Not applicable.

## Representative Trajectories

4 sampled trajectories in `trajectories/audit.jsonl`
(high-reward / low-reward / mixed-group / longest per sampled update).

## What Changed During the Run

No restarts. Metrics are never stitched across a restart.

## What We Learned

TO BE COMPLETED BY A HUMAN — cite steps, metrics, figures and incident IDs.

## Open Questions

TO BE COMPLETED BY A HUMAN.

## Next Experiment

TO BE DECIDED.

---

### Figures

- `figures/01_reward_validation.png`
- `figures/02_policy_dynamics.png`
- `figures/03_ratio_grad.png`
- `figures/04_group_signal.png`
- `figures/05_length_dynamics.png`
- `figures/06_system_timing.png`
- `figures/07_gpu_memory_util.png`

### Metric availability note

Fields that the current VeRL build does not emit are recorded as empty in
`metrics/training_metrics.csv` and rendered as `unavailable` here. They are never
substituted with `0`. See `docs/observability/metric_dictionary.md`.
