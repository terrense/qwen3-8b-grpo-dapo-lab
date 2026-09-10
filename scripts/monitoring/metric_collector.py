#!/usr/bin/env python3
"""Per-optimizer-update RL metric collector.

Reads VeRL's built-in structured sink (the `file` logger backend, enabled with
trainer.logger=[...,"file"] and VERL_FILE_LOGGER_PATH) rather than scraping
stdout. Each line there is {"step": N, "data": {...}} with VeRL's own metric
names, so this module never has to guess.

It then:
  * maps VeRL's real key names onto the canonical schema in
    docs/observability/metric_dictionary.md,
  * derives group-signal statistics from the rollout dump (rollout_data_dir),
  * derives update-time attribution from timing_s/*,
  * marks anything VeRL does not emit as UNAVAILABLE -- never as 0.

Nothing here feeds back into training.
"""

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict

UNAVAILABLE = None  # explicit: absent metric is None, never 0.0

# ---------------------------------------------------------------------------
# Canonical name -> VeRL key. Verified against verl @ 1252cc71:
#   verl/trainer/ppo/metric_utils.py, verl/workers/**/*.py, ray_trainer.py _timer
# A canonical field with no VeRL source stays UNAVAILABLE unless derived below.
# ---------------------------------------------------------------------------
DIRECT_MAP = {
    # optimization
    "policy_loss": "actor/pg_loss",
    "entropy": "actor/entropy",
    "entropy_loss": "actor/entropy_loss",
    "grad_norm": "actor/grad_norm",
    "mfu": "actor/mfu",
    "kl": "actor/ppo_kl",                      # in-batch approx KL (old vs current)
    "kl_penalty_coeff": "actor/reward_kl_penalty_coeff",
    "kl_penalty": "actor/reward_kl_penalty",
    "clip_fraction_high": "actor/pg_clipfrac",        # upper clip
    "clip_fraction_low": "actor/pg_clipfrac_lower",   # lower clip
    # reward / score
    "reward_mean": "critic/rewards/mean",
    "reward_max": "critic/rewards/max",
    "reward_min": "critic/rewards/min",
    "score_mean": "critic/score/mean",
    "score_max": "critic/score/max",
    "score_min": "critic/score/min",
    # advantage
    "advantage_mean": "critic/advantages/mean",
    "advantage_max": "critic/advantages/max",
    "advantage_min": "critic/advantages/min",
    # generation
    "response_length_mean": "response_length/mean",
    "response_length_max": "response_length/max",
    "response_length_min": "response_length/min",
    "truncation_rate": "response_length/clip_ratio",   # fraction hitting the cap
    "prompt_length_mean": "prompt_length/mean",
    "prompt_length_max": "prompt_length/max",
    "prompt_length_clip_ratio": "prompt_length/clip_ratio",
    "aborted_ratio": "response/aborted_ratio",
    # throughput
    "tokens_per_second": "perf/throughput",
    "time_per_step": "perf/time_per_step",
    "total_num_tokens": "perf/total_num_tokens",
    # timing (verl emits timing_s/<name>; see ray_trainer _timer call sites)
    "t_rollout": "timing_s/gen",
    "t_old_logprob": "timing_s/old_log_prob",
    "t_ref_logprob": "timing_s/ref",
    "t_values": "timing_s/values",
    "t_adv": "timing_s/adv",
    "t_reward": "timing_s/reward",
    "t_actor_update": "timing_s/update_actor",
    "t_weight_sync": "timing_s/update_weights",
    "t_checkpoint": "timing_s/save_checkpoint",
    "t_testing": "timing_s/testing",
    "t_step": "timing_s/step",
    # policy provenance -- VeRL publishes these natively; do NOT re-derive them.
    "policy_staleness_steps": "training/off_policy/trajectory_staleness/mean",
    "policy_staleness_max": "training/off_policy/trajectory_staleness/max",
    "trajectory_span_mean": "training/off_policy/trajectory_spans/mean",
    # weight-sync correctness: rollout policy vs training policy agreement
    "rollout_train_prob_corr": "training/rollout_actor_probs_pearson_corr",
    "rollout_train_prob_diff_mean": "training/rollout_probs_diff_mean",
    "rollout_train_prob_diff_max": "training/rollout_probs_diff_max",
    "rollout_train_kl": "rollout_corr/kl",
    "actor_peak_mem_gb": "actor/perf/max_memory_allocated_gb",
    "actor_reserved_mem_gb": "actor/perf/max_memory_reserved_gb",
    "mfu_actor": "perf/mfu/actor",
    "kl_loss": "actor/kl_loss",
    # ---- value-function arms (PPO/GAE, and VAPO if it is ever implemented) ----
    # These stay UNAVAILABLE for critic-free algorithms, which is the correct
    # reading: the quantity does not exist, it is not zero.
    "value_mean": "critic/values/mean",
    "value_max": "critic/values/max",
    "value_min": "critic/values/min",
    "returns_mean": "critic/returns/mean",
    "returns_max": "critic/returns/max",
    "returns_min": "critic/returns/min",
    "explained_variance": "critic/vf_explained_var",
    "value_loss": "critic/vf_loss",
    "value_clip_fraction": "critic/vf_clipfrac",
    "t_critic_update": "timing_s/update_critic",
    "t_values": "timing_s/values",
}

TIMING_FIELDS = ["t_rollout", "t_old_logprob", "t_ref_logprob", "t_values", "t_adv",
                 "t_reward", "t_actor_update", "t_weight_sync", "t_checkpoint", "t_testing"]

CSV_COLUMNS = [
    "global_step", "wall_iso", "restart_epoch",
    "policy_loss", "learning_rate", "grad_norm", "entropy", "kl",
    "clip_fraction", "clip_fraction_low", "clip_fraction_high",
    "reward_mean", "reward_std", "reward_max", "reward_min",
    "advantage_mean", "advantage_std", "advantage_min", "advantage_max",
    "n_prompts", "rollout_n", "n_groups",
    "all_correct_groups", "all_wrong_groups", "mixed_groups", "zero_std_groups",
    "all_correct_group_ratio", "all_wrong_group_ratio", "mixed_group_ratio",
    "zero_std_group_ratio", "effective_signal_groups", "effective_signal_fraction",
    "group_reward_std_mean",
    "verifier_correct", "verifier_incorrect", "verifier_parse_failure",
    "verifier_exception", "verifier_truncated",
    "response_length_mean", "response_length_median", "response_length_p95",
    "response_length_max", "truncation_rate", "eos_rate",
    "total_num_tokens", "tokens_per_second",
    "t_rollout", "t_old_logprob", "t_ref_logprob", "t_adv", "t_reward",
    "t_actor_update", "t_weight_sync", "t_checkpoint", "t_step",
    "pct_rollout", "pct_reward", "pct_logprob", "pct_actor_update",
    "pct_weight_sync", "pct_other",
    "rollout_policy_step", "consumer_update_step", "policy_staleness_steps",
    "policy_staleness_max",
    "trajectory_span_mean",
    "rollout_train_prob_corr",
    "rollout_train_prob_diff_mean",
    "rollout_train_prob_diff_max",
    "rollout_train_kl",
    "actor_peak_mem_gb",
    "actor_reserved_mem_gb",
    "mfu_actor",
    "kl_loss",
    "value_mean",
    "value_max",
    "value_min",
    "returns_mean",
    "returns_max",
    "returns_min",
    "explained_variance",
    "value_loss",
    "value_clip_fraction",
    "t_critic_update",
    "peak_vram_gpu0", "peak_vram_gpu1", "peak_vram_gpu2", "peak_vram_gpu3",
]


def _f(d, k):
    v = d.get(k, UNAVAILABLE)
    if v is None:
        return UNAVAILABLE
    try:
        v = float(v)
    except (TypeError, ValueError):
        return UNAVAILABLE
    return v if math.isfinite(v) else v  # keep NaN/Inf: the detector needs to see them


def find_lr(data):
    """VeRL does not publish a canonical actor LR key; probe plausible ones."""
    for k in ("actor/lr", "actor/learning_rate", "actor/optim/lr", "lr", "learning_rate"):
        if k in data:
            return _f(data, k)
    return UNAVAILABLE


def group_stats_from_rollout_dump(dump_dir, step):
    """Derive GRPO group signal from VeRL's own rollout dump (rollout_data_dir).

    Non-invasive: the trainer writes this file itself when rollout_data_dir is set.
    Returns UNAVAILABLE-filled dict when the dump is absent.
    """
    empty = {k: UNAVAILABLE for k in (
        "n_prompts", "rollout_n", "n_groups", "all_correct_groups", "all_wrong_groups",
        "mixed_groups", "zero_std_groups", "all_correct_group_ratio",
        "all_wrong_group_ratio", "mixed_group_ratio", "zero_std_group_ratio",
        "effective_signal_groups", "effective_signal_fraction", "group_reward_std_mean",
        "reward_std", "advantage_std", "response_length_median", "response_length_p95",
        "eos_rate", "verifier_correct", "verifier_incorrect", "verifier_parse_failure",
        "verifier_exception", "verifier_truncated")}
    if not dump_dir:
        return empty
    path = None
    for cand in (os.path.join(dump_dir, f"{step}.jsonl"),
                 os.path.join(dump_dir, f"step_{step}.jsonl")):
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        return empty

    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception:  # noqa: BLE001
        return empty
    if not rows:
        return empty

    # group by prompt text (the dump's stable per-prompt identity)
    groups = defaultdict(list)
    for r in rows:
        key = r.get("input") or r.get("prompt") or r.get("id")
        groups[key].append(r)

    def score_of(r):
        s = r.get("score")
        if isinstance(s, dict):
            s = s.get("score")
        try:
            return float(s)
        except (TypeError, ValueError):
            return None

    ac = aw = mx = zs = 0
    stds, all_scores = [], []
    for _, rs in groups.items():
        sc = [x for x in (score_of(r) for r in rs) if x is not None]
        if not sc:
            continue
        all_scores.extend(sc)
        sd = statistics.pstdev(sc) if len(sc) > 1 else 0.0
        stds.append(sd)
        hi = max(sc)
        ncor = sum(1 for x in sc if x == hi and hi > 0)
        if sd == 0.0:
            zs += 1
            if hi > 0:
                ac += 1
            else:
                aw += 1
        else:
            mx += 1
    n_groups = len([g for g in groups.values() if g])
    if n_groups == 0:
        return empty

    out = dict(empty)
    out.update(
        n_prompts=n_groups,
        rollout_n=round(len(rows) / n_groups, 3),
        n_groups=n_groups,
        all_correct_groups=ac, all_wrong_groups=aw, mixed_groups=mx, zero_std_groups=zs,
        all_correct_group_ratio=ac / n_groups,
        all_wrong_group_ratio=aw / n_groups,
        mixed_group_ratio=mx / n_groups,
        zero_std_group_ratio=zs / n_groups,
        effective_signal_groups=n_groups - zs,
        effective_signal_fraction=(n_groups - zs) / n_groups,
        group_reward_std_mean=statistics.mean(stds) if stds else UNAVAILABLE,
        reward_std=statistics.pstdev(all_scores) if len(all_scores) > 1 else UNAVAILABLE,
    )
    return out


def build_row(step, data, dump_dir, restart_epoch, prev_step, wall_iso, gpu_peak):
    row = {c: UNAVAILABLE for c in CSV_COLUMNS}
    row["global_step"] = step
    row["wall_iso"] = wall_iso
    row["restart_epoch"] = restart_epoch

    for canon, verl_key in DIRECT_MAP.items():
        if canon in row or canon in TIMING_FIELDS:
            row[canon] = _f(data, verl_key)
    row["learning_rate"] = find_lr(data)

    lo, hi = row.get("clip_fraction_low"), row.get("clip_fraction_high")
    if lo is not None and hi is not None:
        row["clip_fraction"] = lo + hi
    elif hi is not None:
        row["clip_fraction"] = hi

    row.update({k: v for k, v in group_stats_from_rollout_dump(dump_dir, step).items()
                if v is not UNAVAILABLE})

    # update-time attribution
    tstep = row.get("t_step")
    if tstep and tstep > 0:
        def pct(x):
            return round(100.0 * x / tstep, 2) if isinstance(x, (int, float)) else UNAVAILABLE
        row["pct_rollout"] = pct(row.get("t_rollout") or 0.0)
        row["pct_reward"] = pct(row.get("t_reward") or 0.0)
        lp = (row.get("t_old_logprob") or 0.0) + (row.get("t_ref_logprob") or 0.0)
        row["pct_logprob"] = pct(lp)
        row["pct_actor_update"] = pct(row.get("t_actor_update") or 0.0)
        row["pct_weight_sync"] = pct(row.get("t_weight_sync") or 0.0)
        accounted = sum(v for v in (row.get("t_rollout"), row.get("t_reward"),
                                    row.get("t_actor_update"), row.get("t_weight_sync"),
                                    row.get("t_checkpoint"), lp)
                        if isinstance(v, (int, float)))
        row["pct_other"] = round(max(0.0, 100.0 - 100.0 * accounted / tstep), 2)

    # policy provenance. VeRL measures staleness itself
    # (training/off_policy/trajectory_staleness), mapped above -- that value is
    # authoritative. Deriving it from step deltas gave a WRONG answer (0 then 1)
    # while VeRL reported a true 0 for both updates, so the derivation is gone.
    row["consumer_update_step"] = step
    if row.get("policy_staleness_steps") is not None:
        row["rollout_policy_step"] = step - row["policy_staleness_steps"]

    for i in range(4):
        row[f"peak_vram_gpu{i}"] = gpu_peak.get(i, UNAVAILABLE)
    return row


def gpu_peaks_between(system_jsonl, t_lo, t_hi):
    peaks = {}
    if not os.path.exists(system_jsonl):
        return peaks
    try:
        with open(system_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                ts = r.get("ts")
                if ts is None or not (t_lo <= ts <= t_hi):
                    continue
                for g in r.get("gpus", []) or []:
                    i, m = g.get("index"), g.get("mem_used_gib")
                    if i is not None and m is not None:
                        peaks[i] = max(peaks.get(i, 0.0), m)
    except Exception:  # noqa: BLE001
        pass
    return peaks


def collect(run_dir, verl_jsonl, dump_dir, restart_epoch=0):
    """One-shot collection: read the whole VeRL sink, emit jsonl + csv."""
    mdir = os.path.join(run_dir, "metrics")
    os.makedirs(mdir, exist_ok=True)
    out_jsonl = os.path.join(mdir, "training_metrics.jsonl")
    out_csv = os.path.join(mdir, "training_metrics.csv")
    sysj = os.path.join(run_dir, "telemetry", "system.jsonl")

    if not os.path.exists(verl_jsonl):
        return []

    records = []
    with open(verl_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue

    rows, prev_step = [], None
    import time as _t
    for rec in records:
        step = rec.get("step")
        data = rec.get("data", {}) or {}
        now = _t.time()
        peaks = gpu_peaks_between(sysj, now - 3600, now + 1)
        rows.append(build_row(step, data, dump_dir, restart_epoch, prev_step,
                              _t.strftime("%Y-%m-%dT%H:%M:%S"), peaks))
        prev_step = step

    with open(out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_COLUMNS})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--verl-jsonl", required=True)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--restart-epoch", type=int, default=0)
    a = ap.parse_args()
    rows = collect(a.run_dir, a.verl_jsonl, a.dump_dir, a.restart_epoch)
    print(f"collected {len(rows)} update rows -> {a.run_dir}/metrics/")


if __name__ == "__main__":
    main()
