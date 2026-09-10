#!/usr/bin/env python3
"""Freeze evidence when an incident fires.

Creates experiments/<RUN_ID>/incidents/INC-<ts>-<TYPE>/ containing everything a
human needs to diagnose the event *after* it is gone: the metric window either
side of the event, the system telemetry window, the tail of the trainer log, a
GPU and process snapshot, and the resolved config.

Root cause is deliberately left as PENDING. The detector observes; it does not
diagnose. Auto-filling a plausible-sounding root cause would defeat the entire
purpose of this log.
"""

import csv
import json
import os
import shutil
import subprocess
import time

WINDOW_UPDATES = 15
WINDOW_SECONDS = 600


def _run(cmd, timeout=20):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout
    except Exception as e:  # noqa: BLE001
        return f"<failed: {type(e).__name__}: {e}>"


def _tail(path, n):
    if not os.path.exists(path):
        return f"<missing {path}>"
    try:
        out = subprocess.run(["tail", "-n", str(n), path], capture_output=True,
                             text=True, timeout=20).stdout
        return out
    except Exception as e:  # noqa: BLE001
        return f"<failed: {e}>"


def capture(run_dir, finding, metric_rows, trainer_log, resolved_config=None,
            run_manifest=None, trajectories=None):
    ts = time.strftime("%Y%m%d-%H%M%S")
    itype = finding.get("rule", "unknown").upper()
    inc_id = f"INC-{ts}-{itype}"
    d = os.path.join(run_dir, "incidents", inc_id)
    os.makedirs(d, exist_ok=True)

    step = finding.get("step")

    # ---- metric window around the event ----
    win = metric_rows
    if step is not None:
        idx = next((i for i, r in enumerate(metric_rows) if r.get("global_step") == step), None)
        if idx is not None:
            win = metric_rows[max(0, idx - WINDOW_UPDATES): idx + 2]
    if win:
        cols = list(win[0].keys())
        with open(os.path.join(d, "metric_window.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in win:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})

    # ---- system telemetry window ----
    sysj = os.path.join(run_dir, "telemetry", "system.jsonl")
    now = time.time()
    if os.path.exists(sysj):
        recs = []
        with open(sysj) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if r.get("ts", 0) >= now - WINDOW_SECONDS:
                    recs.append(r)
        if recs:
            flat = []
            for r in recs:
                base = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
                for g in r.get("gpus", []) or []:
                    for k, v in g.items():
                        if k != "index":
                            base[f"gpu{g['index']}_{k}"] = v
                for dk in ("disk_root", "disk_data"):
                    dd = r.get(dk) or {}
                    base[f"{dk}_used_pct"] = dd.get("used_pct")
                flat.append(base)
            cols = sorted({k for r in flat for k in r})
            with open(os.path.join(d, "system_window.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(flat)

    # ---- raw evidence ----
    with open(os.path.join(d, "last_log_lines.txt"), "w") as f:
        f.write(_tail(trainer_log, 200))
    with open(os.path.join(d, "nvidia_smi.txt"), "w") as f:
        f.write(_run("nvidia-smi"))
    with open(os.path.join(d, "process_snapshot.txt"), "w") as f:
        f.write(_run("ps aux --sort=-%mem | head -40"))
    rs = _run("ray status 2>/dev/null")
    if rs.strip():
        with open(os.path.join(d, "ray_status.txt"), "w") as f:
            f.write(rs)
    for src, dst in ((resolved_config, "resolved_config.yaml"),
                     (run_manifest, "run_manifest_snapshot.json")):
        if src and os.path.exists(src):
            shutil.copy(src, os.path.join(d, dst))
    if trajectories and os.path.exists(trajectories):
        if os.path.getsize(trajectories) < 4 * 1024 * 1024:
            shutil.copy(trajectories, os.path.join(d, "trajectories.jsonl"))

    # ---- incident.md ----
    before = win[:-1] if len(win) > 1 else []
    at = win[-1] if win else {}

    def fmt(r, keys):
        return " | ".join(f"{k}={r.get(k)}" for k in keys if r.get(k) is not None)

    keys = ["global_step", "reward_mean", "kl", "entropy", "clip_fraction", "grad_norm",
            "response_length_mean", "truncation_rate", "zero_std_group_ratio",
            "effective_signal_fraction", "t_rollout", "t_actor_update"]

    md = f"""# {inc_id}

**Level:** {finding.get('level')}
**Rule:** `{finding.get('rule')}`
**Metric:** `{finding.get('metric')}`
**Step:** {step}
**Timestamp:** {time.strftime('%Y-%m-%dT%H:%M:%S')}

## Observed symptom

{finding.get('detail', '(no detail)')}

Value at detection: `{finding.get('value')}`
{f"Robust z-score vs rolling median: `{finding['z']}`" if 'z' in finding else ''}
{f"Heuristic threshold: `{finding['threshold']}`" if 'threshold' in finding else ''}

## Detection rule

`{finding.get('rule')}` — see [`docs/observability/incident_taxonomy.md`](../../../../docs/observability/incident_taxonomy.md)
for what this rule does and does not imply. Thresholds are diagnostic heuristics,
not theoretical constants.

## Metrics before the event

```
{chr(10).join(fmt(r, keys) for r in before[-5:]) or '(no prior updates in window)'}
```

## Metrics at the event

```
{fmt(at, keys) or '(unavailable)'}
```

## Initial hypotheses

Work the checklist in [`docs/observability/diagnosis_playbook.md`](../../../../docs/observability/diagnosis_playbook.md)
in order — environment/verifier, data batch composition, truncation, policy-version
freshness, old-logprob correctness, token masking, reward normalisation, optimizer,
LR, and only then KL/clipping.

## Evidence collected

- `metric_window.csv` — {len(win)} updates around the event
- `system_window.csv` — system telemetry, last {WINDOW_SECONDS}s
- `last_log_lines.txt` — trainer log tail (200 lines)
- `nvidia_smi.txt`, `process_snapshot.txt`
- `resolved_config.yaml`, `run_manifest_snapshot.json` (when available)

## Root cause

PENDING

## Fix

PENDING

## Post-fix evidence

PENDING
"""
    with open(os.path.join(d, "incident.md"), "w") as f:
        f.write(md)

    with open(os.path.join(d, "finding.json"), "w") as f:
        json.dump(finding, f, indent=2, default=str)
    return inc_id, d
