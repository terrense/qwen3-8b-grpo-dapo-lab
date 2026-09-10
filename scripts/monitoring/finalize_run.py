#!/usr/bin/env python3
"""End-of-run finalization: rebuild metrics, regenerate figures, write RUN_REPORT.md.

The report states what actually happened. Sections with no data say so; nothing
is inferred, and no root cause is invented. Conclusions cite step numbers,
metric values, figures and incident IDs.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric_collector import collect  # noqa: E402


def load_csv(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        rr = {}
        for k, v in r.items():
            if v is None or v == "":
                rr[k] = None
            else:
                try:
                    rr[k] = float(v)
                except ValueError:
                    rr[k] = v
        out.append(rr)
    return out


def col(rows, k):
    return [r[k] for r in rows if isinstance(r.get(k), (int, float))]


def stat_line(rows, k, fmt="{:.4f}"):
    v = col(rows, k)
    if not v:
        return "unavailable"
    if len(v) == 1:
        return fmt.format(v[0])
    return (f"first {fmt.format(v[0])} -> last {fmt.format(v[-1])} "
            f"(min {fmt.format(min(v))}, max {fmt.format(max(v))}, "
            f"mean {fmt.format(statistics.mean(v))})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--lab", default="/root/autodl-tmp/rl_lab")
    ap.add_argument("--verl-jsonl", required=True)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--exit-code", type=int, default=None)
    a = ap.parse_args()

    rows = collect(a.run_dir, a.verl_jsonl, a.dump_dir)
    subprocess.run([sys.executable,
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "plot_live_metrics.py"),
                    "--run-dir", a.run_dir], check=False)
    rows = load_csv(os.path.join(a.run_dir, "metrics", "training_metrics.csv"))

    man_path = os.path.join(a.run_dir, "run_manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    man["end_ts"] = time.time()
    man["exit_code"] = a.exit_code
    man["updates_recorded"] = len(rows)
    with open(man_path, "w") as f:
        json.dump(man, f, indent=2, default=str)

    inc_dir = os.path.join(a.run_dir, "incidents")
    incidents = sorted(os.listdir(inc_dir)) if os.path.isdir(inc_dir) else []

    traj = os.path.join(a.run_dir, "trajectories", "audit.jsonl")
    n_traj = sum(1 for _ in open(traj)) if os.path.exists(traj) else 0

    figs = []
    fd = os.path.join(a.run_dir, "figures")
    if os.path.isdir(fd):
        figs = sorted(f for f in os.listdir(fd) if f.endswith(".png"))

    dur = (man.get("end_ts", 0) - man.get("start_ts", 0)) or 0
    outcome = ("COMPLETED" if a.exit_code == 0 else
               f"FAILED (exit {a.exit_code})" if a.exit_code is not None else "UNKNOWN")

    # timing attribution averages
    pct_keys = [("pct_rollout", "rollout"), ("pct_reward", "verifier"),
                ("pct_logprob", "logprob"), ("pct_actor_update", "actor update"),
                ("pct_weight_sync", "weight sync"), ("pct_other", "other")]
    timing_lines = []
    for k, lab in pct_keys:
        v = col(rows, k)
        if v:
            timing_lines.append(f"| {lab} | {statistics.mean(v):.1f} % |")
    timing_tbl = ("| stage | mean share of update wall time |\n|---|---|\n"
                  + "\n".join(timing_lines)) if timing_lines else \
                 "Stage timing attribution unavailable."

    stale = col(rows, "policy_staleness_steps")
    stale_txt = (f"observed values {sorted(set(int(x) for x in stale))} over {len(stale)} updates"
                 if stale else "unavailable")

    md = f"""# Run Report — {man.get('run_id', 'unknown')}

**Run UID:** `{man.get('run_uid', 'unknown')}`
**Outcome:** **{outcome}**
**Duration:** {dur / 60:.1f} min
**Optimizer updates recorded:** {len(rows)}
**Generated:** {time.strftime('%Y-%m-%dT%H:%M:%S')}

## Configuration

| | |
|---|---|
| Algorithm | {man.get('algorithm', 'unavailable')} |
| Model | {man.get('model', 'unavailable')} |
| Dataset | {man.get('dataset', 'unavailable')} |
| VeRL commit | `{man.get('verl_commit', 'unavailable')}` |
| Lab commit at start | `{man.get('lab_commit', 'unavailable')}` |
| Seed | {man.get('seed', 'unavailable')} |
| Started | {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(man.get('start_ts', 0)))} |
| Exit code | `{a.exit_code}` |

Full resolved config: `resolved_config.yaml`. Launch command: `command.txt`.

## Outcome

{'The trainer exited cleanly.' if a.exit_code == 0 else
 f'The trainer exited non-zero (`{a.exit_code}`). The exit code is preserved verbatim by the launcher and is NOT swallowed.' if a.exit_code else
 'Exit status unavailable.'}

## Training Dynamics

| metric | trajectory |
|---|---|
| policy loss | {stat_line(rows, 'policy_loss')} |
| learning rate | {stat_line(rows, 'learning_rate', '{:.2e}')} |
| grad norm | {stat_line(rows, 'grad_norm')} |

## Reward & Group Signal

| metric | trajectory |
|---|---|
| reward mean | {stat_line(rows, 'reward_mean')} |
| reward std | {stat_line(rows, 'reward_std')} |
| mixed group ratio | {stat_line(rows, 'mixed_group_ratio', '{:.3f}')} |
| all-correct group ratio | {stat_line(rows, 'all_correct_group_ratio', '{:.3f}')} |
| all-wrong group ratio | {stat_line(rows, 'all_wrong_group_ratio', '{:.3f}')} |
| zero-std group ratio | {stat_line(rows, 'zero_std_group_ratio', '{:.3f}')} |
| effective signal fraction | {stat_line(rows, 'effective_signal_fraction', '{:.3f}')} |

Zero-variance groups produce an identically zero GRPO advantage, so
`effective_signal_fraction` is the fraction of the nominal prompt batch that
actually contributed gradient. See `figures/04_group_signal.png`.

## Policy Dynamics

| metric | trajectory |
|---|---|
| KL (`actor/ppo_kl`) | {stat_line(rows, 'kl', '{:.5f}')} |
| entropy | {stat_line(rows, 'entropy')} |
| clip fraction (total) | {stat_line(rows, 'clip_fraction')} |
| clip fraction lower | {stat_line(rows, 'clip_fraction_low')} |
| clip fraction upper | {stat_line(rows, 'clip_fraction_high')} |
| advantage mean | {stat_line(rows, 'advantage_mean')} |

See `figures/02_policy_dynamics.png`, `figures/03_ratio_grad.png`.

## Generation Dynamics

| metric | trajectory |
|---|---|
| response length mean | {stat_line(rows, 'response_length_mean', '{:.1f}')} |
| response length max | {stat_line(rows, 'response_length_max', '{:.0f}')} |
| truncation rate | {stat_line(rows, 'truncation_rate', '{:.4f}')} |

See `figures/05_length_dynamics.png`.

## Systems Performance

{timing_tbl}

| metric | trajectory |
|---|---|
| tokens/sec | {stat_line(rows, 'tokens_per_second', '{:.0f}')} |
| step wall time (s) | {stat_line(rows, 't_step', '{:.1f}')} |
| peak VRAM GPU0 (GiB) | {stat_line(rows, 'peak_vram_gpu0', '{:.1f}')} |

**Policy provenance / staleness:** {stale_txt}. For synchronous GRPO this should
be a constant; a drifting value would mean rollouts are being consumed by a
different policy version than the one that generated them.

See `figures/06_system_timing.png`, `figures/07_gpu_memory_util.png`.

## Incidents

{chr(10).join(f'- `{i}` — see `incidents/{i}/incident.md`' for i in incidents)
 if incidents else 'None detected by the flight recorder during this run.'}

## Root Causes

{'PENDING — each incident bundle carries a `Root cause: PENDING` field that a human must complete after investigation. The detector deliberately does not guess.' if incidents else 'Not applicable — no incidents.'}

## Recovery / Fixes

{'PENDING' if incidents else 'Not applicable.'}

## Representative Trajectories

{n_traj} sampled trajectories in `trajectories/audit.jsonl`
(high-reward / low-reward / mixed-group / longest per sampled update).

## What Changed During the Run

{'Restart boundaries are marked in the figures. ' if any(r.get('restart_epoch') for r in rows) else 'No restarts. '}Metrics are never stitched across a restart.

## What We Learned

TO BE COMPLETED BY A HUMAN — cite steps, metrics, figures and incident IDs.

## Open Questions

TO BE COMPLETED BY A HUMAN.

## Next Experiment

{man.get('next_experiment', 'TO BE DECIDED.')}

---

### Figures

{chr(10).join(f'- `figures/{f}`' for f in figs) if figs else 'No figures generated (no metric rows).'}

### Metric availability note

Fields that the current VeRL build does not emit are recorded as empty in
`metrics/training_metrics.csv` and rendered as `unavailable` here. They are never
substituted with `0`. See `docs/observability/metric_dictionary.md`.
"""
    out = os.path.join(a.run_dir, "RUN_REPORT.md")
    with open(out, "w") as f:
        f.write(md)
    print(f"RUN_REPORT -> {out}  ({len(rows)} updates, {len(incidents)} incidents)")


if __name__ == "__main__":
    main()
