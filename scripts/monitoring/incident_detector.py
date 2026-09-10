#!/usr/bin/env python3
"""Incident detection for the RL flight recorder.

Three levels: GREEN / YELLOW / RED.

Philosophy
----------
Early in a run the natural scale of reward, KL, entropy and grad norm is
unknown, so hard thresholds on those would fire constantly or never. Continuous
metrics are therefore judged by a *robust, adaptive* rule: a rolling window of
recent updates, a median, and a MAD-based robust z-score. Only quantities with
genuine semantic meaning (NaN/Inf, disk full, process death) get absolute rules.

Every threshold here is a DIAGNOSTIC HEURISTIC, not a theoretical constant --
see docs/observability/metric_dictionary.md.

The detector NEVER changes training. It emits findings; a human decides.
"""

import math
import statistics
from collections import deque

GREEN, YELLOW, RED = "GREEN", "YELLOW", "RED"

WINDOW = 20          # rolling window of updates for adaptive rules
MIN_HISTORY = 6      # below this, adaptive rules abstain (scale not yet known)
ROBUST_Z = 4.0       # MAD-based z above which a jump is "anomalous"

# Absolute rules with defensible semantics (still heuristics, documented as such).
ABS_RULES = {
    "truncation_rate_warn": 0.30,
    "zero_std_group_ratio_warn": 0.70,
    "effective_signal_fraction_warn": 0.30,   # below this, most groups teach nothing
    "system_disk_pct_red": 90.0,
    "data_disk_pct_red": 95.0,
    "clip_fraction_warn": 0.40,
}

# Metrics watched by the adaptive rule, and the direction that is suspicious.
ADAPTIVE = {
    "reward_mean": "both",
    "kl": "up",
    "entropy": "down",
    "clip_fraction": "up",
    "grad_norm": "up",
    "response_length_mean": "up",
    "truncation_rate": "up",
    "zero_std_group_ratio": "up",
    "t_rollout": "up",
    "t_step": "up",
}


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def robust_z(history, value):
    """MAD-based robust z-score. Returns None when the scale is undetermined."""
    h = [x for x in history if _finite(x)]
    if len(h) < MIN_HISTORY or not _finite(value):
        return None
    med = statistics.median(h)
    mad = statistics.median([abs(x - med) for x in h])
    if mad == 0:
        # degenerate spread: fall back to a relative check against the median
        if med == 0:
            return None
        rel = abs(value - med) / abs(med)
        return ROBUST_Z + 1 if rel > 0.5 else 0.0
    return (value - med) / (1.4826 * mad)


class IncidentDetector:
    def __init__(self):
        self.hist = {k: deque(maxlen=WINDOW) for k in ADAPTIVE}
        self.seen = set()

    # ---------------- per-update rules ----------------
    def check_update(self, row):
        """Returns (level, [findings]) for one optimizer-update metric row."""
        findings = []
        level = GREEN
        step = row.get("global_step")

        # ---- RED: non-finite core quantities ----
        for k in ("policy_loss", "reward_mean", "kl", "entropy", "grad_norm"):
            v = row.get(k)
            if v is not None and isinstance(v, (int, float)) and not math.isfinite(v):
                findings.append(dict(level=RED, rule="non_finite_metric", metric=k,
                                     value=str(v), step=step,
                                     detail=f"{k} is {v} at step {step}"))
                level = RED

        # ---- RED: effectively dead learning signal ----
        esf = row.get("effective_signal_fraction")
        if _finite(esf) and esf == 0.0:
            findings.append(dict(level=RED, rule="no_effective_signal",
                                 metric="effective_signal_fraction", value=esf, step=step,
                                 detail="every group has zero reward variance; GRPO advantage "
                                        "is identically zero, so this update teaches nothing"))
            level = RED

        # ---- YELLOW: absolute heuristics ----
        for metric, key in (("truncation_rate", "truncation_rate_warn"),
                            ("zero_std_group_ratio", "zero_std_group_ratio_warn"),
                            ("clip_fraction", "clip_fraction_warn")):
            v = row.get(metric)
            if _finite(v) and v > ABS_RULES[key]:
                findings.append(dict(level=YELLOW, rule=f"{metric}_above_heuristic",
                                     metric=metric, value=v, step=step,
                                     threshold=ABS_RULES[key],
                                     detail=f"{metric}={v:.3f} exceeds heuristic "
                                            f"{ABS_RULES[key]}"))
                level = RED if level == RED else YELLOW
        if _finite(esf) and 0.0 < esf < ABS_RULES["effective_signal_fraction_warn"]:
            findings.append(dict(level=YELLOW, rule="effective_signal_low",
                                 metric="effective_signal_fraction", value=esf, step=step,
                                 threshold=ABS_RULES["effective_signal_fraction_warn"],
                                 detail=f"only {esf:.1%} of groups carry gradient signal"))
            level = RED if level == RED else YELLOW

        # ---- YELLOW: adaptive robust-z rules ----
        for metric, direction in ADAPTIVE.items():
            v = row.get(metric)
            z = robust_z(self.hist[metric], v)
            if z is not None:
                bad = (direction == "up" and z > ROBUST_Z) or \
                      (direction == "down" and z < -ROBUST_Z) or \
                      (direction == "both" and abs(z) > ROBUST_Z)
                if bad:
                    findings.append(dict(level=YELLOW, rule=f"{metric}_robust_z",
                                         metric=metric, value=v, step=step, z=round(z, 2),
                                         detail=f"{metric}={v} is {z:+.1f} robust-z from the "
                                                f"rolling median of the last "
                                                f"{len(self.hist[metric])} updates"))
                    level = RED if level == RED else YELLOW
            if _finite(v):
                self.hist[metric].append(v)

        # de-duplicate identical (rule, step)
        out = []
        for f in findings:
            key = (f["rule"], f.get("step"))
            if key not in self.seen:
                self.seen.add(key)
                out.append(f)
        return level, out

    # ---------------- system rules ----------------
    def check_system(self, sysrec):
        findings = []
        level = GREEN
        for name, key, thr in (("system_disk", "disk_root", ABS_RULES["system_disk_pct_red"]),
                               ("data_disk", "disk_data", ABS_RULES["data_disk_pct_red"])):
            d = sysrec.get(key) or {}
            p = d.get("used_pct")
            if _finite(p) and p > thr:
                findings.append(dict(level=RED, rule=f"{name}_full", metric=name, value=p,
                                     threshold=thr, detail=f"{name} at {p:.1f}% (> {thr}%)"))
                level = RED
        if sysrec.get("train_pid_alive") is False:
            findings.append(dict(level=RED, rule="trainer_process_gone",
                                 metric="train_pid_alive", value=False,
                                 detail="training PID is no longer alive"))
            level = RED
        return level, findings

    # ---------------- stall rule ----------------
    @staticmethod
    def stall_threshold_s(update_durations):
        """Adaptive stall threshold: max(3 x median update time, 10 minutes).

        A fixed '5 minutes with no step' alarm is wrong for RL -- a single GRPO
        update with long rollouts can legitimately exceed it.
        """
        d = [x for x in update_durations if _finite(x) and x > 0]
        if len(d) < 3:
            return 600.0
        return max(3.0 * statistics.median(d), 600.0)


def scan_log_for_fatal(text):
    """Hard infrastructure failures that are unambiguous in the trainer log."""
    pats = [
        ("cuda_oom", "CUDA out of memory"),
        ("cuda_oom", "torch.OutOfMemoryError"),
        ("nccl_error", "NCCL error"),
        ("nccl_timeout", "Watchdog caught collective operation timeout"),
        ("ray_actor_death", "RayActorError"),
        ("ray_actor_death", "The actor died unexpectedly"),
        ("ray_oom", "ray.exceptions.OutOfMemoryError"),
        ("vllm_crash", "EngineCore encountered a fatal error"),
        ("vllm_crash", "AsyncEngineDeadError"),
        ("assertion", "AssertionError"),
    ]
    hits = []
    for rule, needle in pats:
        if needle in text:
            hits.append(dict(level=RED, rule=rule, metric="trainer_log", value=needle,
                             detail=f"trainer log contains {needle!r}"))
    return hits
