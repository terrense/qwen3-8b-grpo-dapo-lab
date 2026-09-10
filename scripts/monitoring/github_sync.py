#!/usr/bin/env python3
"""Sync small, reproducible research evidence to GitHub.

What goes up: configs, run manifests, compact metric CSV/JSONL, incident
bundles, sampled trajectories, figures, run reports, docs.

What never goes up: weights, optimizer checkpoints, raw datasets, HF cache,
full rollout dumps, large logs. Those stay on the data disk and are referenced
by run id / step / path / hash.

README is only touched inside the bounded region between STATUS_START and
STATUS_END, so a monitoring daemon can never clobber prose.
"""

import argparse
import json
import os
import subprocess
import time

MAX_FILE_BYTES = 5 * 1024 * 1024
STATUS_START = "<!-- STATUS_START -->"
STATUS_END = "<!-- STATUS_END -->"


def sh(cmd, cwd=None, timeout=180):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                          text=True, timeout=timeout)


def update_readme_status(lab, status_json):
    readme = os.path.join(lab, "README.md")
    if not os.path.exists(readme) or not os.path.exists(status_json):
        return False
    st = json.load(open(status_json))
    txt = open(readme).read()
    if STATUS_START not in txt or STATUS_END not in txt:
        return False

    def v(k, fmt="{}"):
        x = st.get(k)
        return fmt.format(x) if isinstance(x, (int, float)) else "—"

    block = f"""{STATUS_START}
| Stage | Status |
|---|---|
| Infrastructure validation | **PASS** |
| CUDA 13 / torch 2.11 stack | **PASS** |
| Qwen3-8B via vLLM 0.24 | **PASS** |
| 4-GPU NCCL (349.8 GB/s busBW) | **PASS** |
| RLVR verifier gate | {st.get('verifier_gate', '**PASS**')} |
| Flight recorder | **ACTIVE** |
| R0 — GRPO smoke | {st.get('r0_status', '**IN PROGRESS**')} |
| R1 — Vanilla GRPO | **NOT RUN** |
| R2 — DAPO | **NOT RUN** |
| R3 — Failure injection | **NOT RUN** |

**Live run** `{st.get('run_id', '—')}` · step **{st.get('current_step', '—')}** ·
health **{st.get('health', '—')}** · reward {v('reward_mean', '{:.4f}')} ·
KL {v('kl', '{:.5f}')} · entropy {v('entropy', '{:.4f}')} ·
effective signal {v('effective_signal_fraction', '{:.2f}')} ·
incidents {st.get('n_incidents', 0)}

_Auto-updated {st.get('updated', '')} by `scripts/monitoring/github_sync.py`. Full status: [`status/latest.md`](status/latest.md)._
{STATUS_END}"""

    pre = txt.split(STATUS_START)[0]
    post = txt.split(STATUS_END)[1]
    open(readme, "w").write(pre + block + post)
    return True


def stage_paths(lab, run_dir):
    """Stage only small text/figure artifacts."""
    rel = os.path.relpath(run_dir, lab)
    wanted = []
    for sub in ("metrics", "incidents", "trajectories", "figures"):
        p = os.path.join(run_dir, sub)
        if os.path.isdir(p):
            wanted.append(os.path.join(rel, sub))
    for f in ("run_manifest.json", "RUN_REPORT.md", "resolved_config.yaml", "command.txt"):
        if os.path.exists(os.path.join(run_dir, f)):
            wanted.append(os.path.join(rel, f))
    wanted += ["status", "docs", "scripts", "configs", "analysis", "manifests", "README.md"]
    return [w for w in wanted if os.path.exists(os.path.join(lab, w))]


def drop_oversized(lab):
    """Unstage anything unexpectedly large -- a hard guard against dump leakage."""
    r = sh("git diff --cached --name-only", cwd=lab)
    dropped = []
    for name in r.stdout.splitlines():
        p = os.path.join(lab, name)
        try:
            if os.path.isfile(p) and os.path.getsize(p) > MAX_FILE_BYTES:
                sh(f"git restore --staged -- {name!r}", cwd=lab)
                dropped.append((name, os.path.getsize(p)))
        except OSError:
            pass
    return dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default="/root/autodl-tmp/rl_lab")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--reason", default="periodic")
    ap.add_argument("--message", default=None)
    a = ap.parse_args()

    lab = a.lab
    lock = os.path.join(lab, "tmp", ".github_sync.lock")
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    # crude single-flight guard so overlapping syncs cannot race on the index
    if os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 300:
        print("[sync] another sync in flight; skipping")
        return
    open(lock, "w").write(str(time.time()))

    try:
        status_json = os.path.join(lab, "status", "latest.json")
        update_readme_status(lab, status_json)

        paths = stage_paths(lab, a.run_dir) if a.run_dir else ["status", "docs", "scripts",
                                                               "analysis", "manifests", "README.md"]
        for p in paths:
            sh(f"git add -A -- {p!r}", cwd=lab)

        dropped = drop_oversized(lab)
        for n, s in dropped:
            print(f"[sync] DROPPED oversized {n} ({s} bytes)")

        st = sh("git diff --cached --name-only", cwd=lab)
        if not st.stdout.strip():
            print("[sync] nothing to commit")
            return

        run_id = os.path.basename(a.run_dir) if a.run_dir else "lab"
        msg = a.message or f"obs({run_id}): {a.reason} telemetry sync"
        body = (f"{msg}\n\nAutomated flight-recorder sync ({a.reason}).\n"
                f"Files: {len(st.stdout.strip().splitlines())}\n\n"
                "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>\n")
        p = subprocess.run(["git", "commit", "-q", "-F", "-"], cwd=lab, input=body,
                           capture_output=True, text=True, timeout=120)
        if p.returncode != 0 and "nothing to commit" not in (p.stdout + p.stderr):
            print(f"[sync] commit failed: {p.stdout}{p.stderr}")
            return
        env = os.environ.copy()
        push = subprocess.run(["git", "push", "-q", "origin", "main"], cwd=lab,
                              capture_output=True, text=True, timeout=300, env=env)
        if push.returncode != 0:
            print(f"[sync] push failed: {push.stdout}{push.stderr}")
        else:
            sha = sh("git rev-parse --short HEAD", cwd=lab).stdout.strip()
            print(f"[sync] pushed {sha} ({a.reason})")
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


if __name__ == "__main__":
    main()
