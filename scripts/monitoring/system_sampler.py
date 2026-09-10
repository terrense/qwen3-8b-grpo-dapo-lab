#!/usr/bin/env python3
"""System telemetry sampler for the RL flight recorder.

Samples NVML + psutil every SAMPLE_INTERVAL seconds and appends one JSON object
per sample to <run_dir>/telemetry/system.jsonl.

Design constraints:
  * Observer, never trainer -- this process only reads.
  * Cheap: NVML/psutil handles are opened once; no subprocess per sample.
  * Never crash the run: every sampler failure is recorded in-band and skipped.
"""

import argparse
import json
import os
import shutil
import time

import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML = True
    _HANDLES = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
except Exception:  # noqa: BLE001
    _NVML = False
    _HANDLES = []


def _nvml_int(fn, *a, default=None):
    try:
        return fn(*a)
    except Exception:  # noqa: BLE001
        return default


def gpu_sample():
    out = []
    for i, h in enumerate(_HANDLES):
        rec = {"index": i}
        try:
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            rec["util_gpu_pct"] = u.gpu
            rec["util_mem_pct"] = u.memory
        except Exception:  # noqa: BLE001
            rec["util_gpu_pct"] = None
            rec["util_mem_pct"] = None
        try:
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            rec["mem_used_gib"] = round(m.used / 1024**3, 3)
            rec["mem_total_gib"] = round(m.total / 1024**3, 3)
        except Exception:  # noqa: BLE001
            rec["mem_used_gib"] = rec["mem_total_gib"] = None
        rec["temp_c"] = _nvml_int(pynvml.nvmlDeviceGetTemperature, h, pynvml.NVML_TEMPERATURE_GPU)
        pw = _nvml_int(pynvml.nvmlDeviceGetPowerUsage, h)
        rec["power_w"] = round(pw / 1000.0, 1) if pw is not None else None
        pl = _nvml_int(pynvml.nvmlDeviceGetEnforcedPowerLimit, h)
        rec["power_limit_w"] = round(pl / 1000.0, 1) if pl is not None else None
        out.append(rec)
    return out


def disk(path):
    try:
        u = shutil.disk_usage(path)
        return {"total_gib": round(u.total / 1024**3, 2),
                "used_gib": round(u.used / 1024**3, 2),
                "free_gib": round(u.free / 1024**3, 2),
                "used_pct": round(100.0 * u.used / u.total, 2)}
    except Exception:  # noqa: BLE001
        return None


def proc_snapshot(train_pid):
    alive = None
    if train_pid:
        try:
            alive = psutil.pid_exists(int(train_pid)) and \
                psutil.Process(int(train_pid)).status() != psutil.STATUS_ZOMBIE
        except Exception:  # noqa: BLE001
            alive = None
    names = {"ray": 0, "vllm": 0, "python": 0}
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
        except Exception:  # noqa: BLE001
            continue
        if "raylet" in cl or "ray::" in cl or "/ray/" in cl:
            names["ray"] += 1
        if "vllm" in cl.lower() or "EngineCore" in cl:
            names["vllm"] += 1
        if (p.info.get("name") or "").startswith("python"):
            names["python"] += 1
    return {"train_pid_alive": alive, "proc_counts": names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--train-pid", default=None)
    args = ap.parse_args()

    out_dir = os.path.join(args.run_dir, "telemetry")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "system.jsonl")

    psutil.cpu_percent(interval=None)  # prime the counter
    while True:
        t0 = time.time()
        try:
            vm = psutil.virtual_memory()
            rec = {
                "ts": t0,
                "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                "gpus": gpu_sample(),
                "cpu_pct": psutil.cpu_percent(interval=None),
                "load1": os.getloadavg()[0],
                "ram_used_gib": round((vm.total - vm.available) / 1024**3, 2),
                "ram_available_gib": round(vm.available / 1024**3, 2),
                "ram_total_gib": round(vm.total / 1024**3, 2),
                "disk_root": disk("/"),
                "disk_data": disk("/root/autodl-tmp"),
                "disk_shm": disk("/dev/shm"),
                **proc_snapshot(args.train_pid),
            }
        except Exception as e:  # noqa: BLE001 -- sampler must never kill the run
            rec = {"ts": t0, "sampler_error": f"{type(e).__name__}: {e}"}
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        time.sleep(max(0.0, args.interval - (time.time() - t0)))


if __name__ == "__main__":
    main()
