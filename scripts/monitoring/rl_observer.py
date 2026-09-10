#!/usr/bin/env python3
"""RL flight recorder -- supervisor loop.

Runs alongside (never inside) the trainer. Every cycle it:
  1. re-reads VeRL's structured metric sink and rebuilds the canonical metrics,
  2. runs the incident detector over new updates and the latest system sample,
  3. freezes an incident bundle when something fires,
  4. samples diagnostic trajectories from the rollout dump,
  5. rewrites status/latest.{json,md},
  6. periodically triggers plots + a GitHub sync.

Hard rule: this process may observe, record, derive, alert and capture evidence.
It may NOT modify reward, advantage, optimizer, LR, KL coefficient, rollout.n,
batch size, or restart training to paper over a failure.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from incident_capture import capture  # noqa: E402
from incident_detector import GREEN, RED, YELLOW, IncidentDetector, scan_log_for_fatal  # noqa: E402
from metric_collector import collect  # noqa: E402


def read_last_system(run_dir):
    p = os.path.join(run_dir, "telemetry", "system.jsonl")
    if not os.path.exists(p):
        return {}
    last = {}
    try:
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            for line in f.read().decode(errors="ignore").splitlines():
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)
                    except Exception:  # noqa: BLE001
                        pass
    except Exception:  # noqa: BLE001
        pass
    return last


def sample_trajectories(dump_dir, step, out_path, max_chars=4000):
    """Save a handful of diagnostically useful rollouts, not the whole dump."""
    if not dump_dir:
        return 0
    src = None
    for cand in (os.path.join(dump_dir, f"{step}.jsonl"), os.path.join(dump_dir, f"step_{step}.jsonl")):
        if os.path.exists(cand):
            src = cand
            break
    if src is None:
        return 0
    rows = []
    try:
        with open(src) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception:  # noqa: BLE001
        return 0
    if not rows:
        return 0

    def sc(r):
        s = r.get("score")
        if isinstance(s, dict):
            s = s.get("score")
        try:
            return float(s)
        except (TypeError, ValueError):
            return None

    def olen(r):
        return len(str(r.get("output", "")))

    scored = [(sc(r), r) for r in rows]
    scored = [(s, r) for s, r in scored if s is not None]
    picks = {}
    if scored:
        picks["high_reward"] = max(scored, key=lambda x: x[0])[1]
        picks["low_reward"] = min(scored, key=lambda x: x[0])[1]
    picks["longest_response"] = max(rows, key=olen)
    # group-mixed representative: a prompt whose samples disagree
    groups = {}
    for s, r in scored:
        groups.setdefault(r.get("input"), []).append((s, r))
    mixed = [g for g in groups.values() if len({x[0] for x in g}) > 1]
    if mixed:
        picks["mixed_group_representative"] = mixed[0][0][1]

    n = 0
    with open(out_path, "a") as f:
        for kind, r in picks.items():
            s = sc(r)
            grp = groups.get(r.get("input"), [])
            gsc = [x[0] for x in grp]
            rec = {
                "step": step, "kind": kind,
                "prompt": str(r.get("input", ""))[:max_chars],
                "response": str(r.get("output", ""))[:max_chars],
                "ground_truth": r.get("gts") or r.get("gt"),
                "reward": s,
                "response_length_chars": olen(r),
                "group_rewards": gsc,
                "group_mean": statistics.mean(gsc) if gsc else None,
                "group_std": statistics.pstdev(gsc) if len(gsc) > 1 else 0.0,
            }
            for k, v in r.items():
                if k in ("acc", "pred", "state", "verifier_status"):
                    rec[k] = v
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_status(run_dir, status_dir, manifest, rows, health, last_sys, incidents, git_sha):
    last = rows[-1] if rows else {}
    gpus = last_sys.get("gpus", []) or []
    started = manifest.get("start_ts", time.time())
    st = {
        "run_id": manifest.get("run_id"),
        "run_uid": manifest.get("run_uid"),
        "algorithm": manifest.get("algorithm"),
        "model": manifest.get("model"),
        "dataset": manifest.get("dataset"),
        "current_step": last.get("global_step"),
        "updates_recorded": len(rows),
        "elapsed_s": round(time.time() - started, 1),
        "health": health,
        "reward_mean": last.get("reward_mean"),
        "kl": last.get("kl"),
        "entropy": last.get("entropy"),
        "clip_fraction": last.get("clip_fraction"),
        "grad_norm": last.get("grad_norm"),
        "mixed_group_ratio": last.get("mixed_group_ratio"),
        "zero_std_group_ratio": last.get("zero_std_group_ratio"),
        "effective_signal_fraction": last.get("effective_signal_fraction"),
        "response_length_mean": last.get("response_length_mean"),
        "response_length_p95": last.get("response_length_p95"),
        "truncation_rate": last.get("truncation_rate"),
        "t_rollout": last.get("t_rollout"),
        "t_actor_update": last.get("t_actor_update"),
        "gpu_util_pct": [g.get("util_gpu_pct") for g in gpus],
        "gpu_mem_used_gib": [g.get("mem_used_gib") for g in gpus],
        "disk_root_pct": (last_sys.get("disk_root") or {}).get("used_pct"),
        "disk_data_pct": (last_sys.get("disk_data") or {}).get("used_pct"),
        "n_incidents": len(incidents),
        "latest_incident": incidents[-1] if incidents else None,
        "last_checkpoint": manifest.get("last_checkpoint"),
        "git_commit": git_sha,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(status_dir, exist_ok=True)
    with open(os.path.join(status_dir, "latest.json"), "w") as f:
        json.dump(st, f, indent=2, default=str)

    def g(k, fmt="{}"):
        v = st.get(k)
        return fmt.format(v) if isinstance(v, (int, float)) else "unavailable"

    md = f"""# Live Run Status

_Updated {st['updated']} · health **{health}**_

| | |
|---|---|
| Run | `{st['run_id']}` (`{st['run_uid']}`) |
| Algorithm | {st['algorithm']} |
| Model | {st['model']} |
| Dataset | {st['dataset']} |
| Current optimizer step | **{st['current_step']}** |
| Updates recorded | {st['updates_recorded']} |
| Elapsed | {st['elapsed_s']:.0f} s |
| Health | **{health}** |

## Latest update

| metric | value |
|---|---|
| reward_mean | {g('reward_mean', '{:.4f}')} |
| KL | {g('kl', '{:.5f}')} |
| entropy | {g('entropy', '{:.4f}')} |
| clip_fraction | {g('clip_fraction', '{:.4f}')} |
| grad_norm | {g('grad_norm', '{:.4f}')} |
| mixed_group_ratio | {g('mixed_group_ratio', '{:.3f}')} |
| zero_std_group_ratio | {g('zero_std_group_ratio', '{:.3f}')} |
| effective_signal_fraction | {g('effective_signal_fraction', '{:.3f}')} |
| response_length_mean | {g('response_length_mean', '{:.1f}')} |
| truncation_rate | {g('truncation_rate', '{:.3f}')} |
| rollout time (s) | {g('t_rollout', '{:.1f}')} |
| actor update time (s) | {g('t_actor_update', '{:.1f}')} |

## System

GPU util: `{st['gpu_util_pct']}`
GPU mem (GiB): `{st['gpu_mem_used_gib']}`
disk / : {st['disk_root_pct']}%   disk data: {st['disk_data_pct']}%

## Incidents

{len(incidents)} recorded. Latest: `{st['latest_incident'] or 'none'}`

Last checkpoint: `{st['last_checkpoint'] or 'none'}`
Git commit: `{git_sha}`

_Values shown as `unavailable` are metrics the current VeRL build does not emit.
They are never substituted with 0._
"""
    with open(os.path.join(status_dir, "latest.md"), "w") as f:
        f.write(md)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--lab", default="/root/autodl-tmp/rl_lab")
    ap.add_argument("--verl-jsonl", required=True)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--trainer-log", required=True)
    ap.add_argument("--resolved-config", default=None)
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--traj-every", type=int, default=1)
    ap.add_argument("--sync-every-updates", type=int, default=20)
    ap.add_argument("--sync-every-seconds", type=float, default=900.0)
    ap.add_argument("--no-git", action="store_true")
    a = ap.parse_args()

    run_dir = a.run_dir
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))

    det = IncidentDetector()
    handled_steps, incidents = set(), []
    traj_path = os.path.join(run_dir, "trajectories", "audit.jsonl")
    os.makedirs(os.path.dirname(traj_path), exist_ok=True)
    status_dir = os.path.join(a.lab, "status")

    last_sync_t, last_sync_step = 0.0, -1
    last_step_seen, last_step_change_t = None, time.time()
    update_durations = []
    fatal_reported = set()

    def git_sha():
        try:
            return subprocess.run(["git", "-C", a.lab, "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    while True:
        cycle_t = time.time()
        try:
            rows = collect(run_dir, a.verl_jsonl, a.dump_dir,
                           restart_epoch=manifest.get("restart_epoch", 0))
        except Exception as e:  # noqa: BLE001
            rows = []
            print(f"[observer] collect failed: {e}", flush=True)

        health = GREEN
        new_findings = []
        for r in rows:
            s = r.get("global_step")
            if s in handled_steps:
                continue
            handled_steps.add(s)
            lvl, fs = det.check_update(r)
            if lvl == RED:
                health = RED
            elif lvl == YELLOW and health != RED:
                health = YELLOW
            new_findings.extend(fs)
            if r.get("t_step"):
                update_durations.append(r["t_step"])
            if a.traj_every and s is not None and s % a.traj_every == 0:
                try:
                    sample_trajectories(a.dump_dir, s, traj_path)
                except Exception as e:  # noqa: BLE001
                    print(f"[observer] traj sample failed: {e}", flush=True)

        # system rules
        last_sys = read_last_system(run_dir)
        if last_sys:
            lvl, fs = det.check_system(last_sys)
            if lvl == RED:
                health = RED
            new_findings.extend(fs)

        # fatal patterns in the trainer log
        try:
            tail = subprocess.run(["tail", "-n", "400", a.trainer_log],
                                  capture_output=True, text=True, timeout=20).stdout
            for f in scan_log_for_fatal(tail):
                if f["rule"] not in fatal_reported:
                    fatal_reported.add(f["rule"])
                    new_findings.append(f)
                    health = RED
        except Exception:  # noqa: BLE001
            pass

        # adaptive stall detection
        cur_step = rows[-1].get("global_step") if rows else None
        if cur_step != last_step_seen:
            last_step_seen, last_step_change_t = cur_step, time.time()
        else:
            thr = IncidentDetector.stall_threshold_s(update_durations)
            stalled_for = time.time() - last_step_change_t
            if stalled_for > thr and "stall" not in fatal_reported:
                fatal_reported.add("stall")
                new_findings.append(dict(
                    level=YELLOW, rule="training_stall", metric="global_step",
                    value=cur_step, step=cur_step,
                    detail=f"no new optimizer step for {stalled_for:.0f}s "
                           f"(adaptive threshold {thr:.0f}s = max(3x median update, 600s))"))
                health = RED if health == RED else YELLOW

        for f in new_findings:
            try:
                inc_id, _ = capture(run_dir, f, rows, a.trainer_log,
                                    resolved_config=a.resolved_config,
                                    run_manifest=manifest_path, trajectories=traj_path)
                incidents.append(inc_id)
                print(f"[observer] {f['level']} incident {inc_id}: {f.get('detail')}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[observer] capture failed: {e}", flush=True)

        sha = git_sha()
        try:
            write_status(run_dir, status_dir, manifest, rows, health, last_sys, incidents, sha)
        except Exception as e:  # noqa: BLE001
            print(f"[observer] status write failed: {e}", flush=True)

        # periodic plots + sync -- never every cycle, GitHub is not a TSDB
        want_sync = (
            (cur_step is not None and last_sync_step >= 0
             and cur_step - last_sync_step >= a.sync_every_updates)
            or (time.time() - last_sync_t >= a.sync_every_seconds)
            or bool(new_findings)
        )
        if last_sync_step < 0:
            last_sync_step = cur_step if cur_step is not None else 0
            last_sync_t = time.time()
        elif want_sync and rows:
            last_sync_step = cur_step if cur_step is not None else last_sync_step
            last_sync_t = time.time()
            _bg(f"{sys.executable} {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_live_metrics.py')} "
                f"--run-dir {run_dir}")
            if not a.no_git:
                _bg(f"{sys.executable} {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'github_sync.py')} "
                    f"--lab {a.lab} --run-dir {run_dir} --reason periodic")

        time.sleep(max(1.0, a.interval - (time.time() - cycle_t)))


def _bg(cmd):
    try:
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception as e:  # noqa: BLE001
        print(f"[observer] bg launch failed: {e}", flush=True)


if __name__ == "__main__":
    main()
