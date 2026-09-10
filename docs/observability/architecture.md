# RL Flight Recorder — Architecture

A flight recorder for RL training, not a GPU monitor. It follows one signal chain end
to end and records what actually happened at every stage, so that when a run improves,
stalls or collapses, the change can be explained rather than guessed at.

No Grafana, no Prometheus, no web UI, no database. Python + `psutil` + NVML + VeRL's
own structured logger.

## The one rule

> **The observer may observe, record, derive, alert and capture evidence.
> It may never train.**

It is forbidden from modifying reward, advantage, optimizer state, learning rate, KL
coefficient, `rollout.n`, batch size, or from restarting training to paper over a
failure. It may stop its own wrapper on a hard infrastructure failure. It may not
"intelligently tune" anything. An observer that edits the experiment is no longer
measuring it.

## Signal chain under observation

```
DAPO-Math-17k
   -> vLLM rollout            (policy version, latency, tokens, finish_reason)
   -> RLVR verifier           (state split: correct / incorrect / parse / exception / truncated)
   -> group rewards           (per-prompt mean, std, composition)
   -> GRPO advantage          (distribution summary, zero-variance detection)
   -> old vs current logprob  (importance ratio; see limitations)
   -> clipping / KL / entropy
   -> FSDP actor update       (grad norm, loss, timing)
   -> weight sync             (timing, policy version bump)
   -> next rollout
```

## Components

| file | role |
|---|---|
| `scripts/training/run_with_observer.sh` | run supervisor: provenance, sampler, trainer, teardown, finalize, sync |
| `scripts/monitoring/system_sampler.py` | 10s NVML + psutil telemetry -> `telemetry/system.jsonl` |
| `scripts/monitoring/metric_collector.py` | VeRL's JSONL sink -> canonical per-update metrics (CSV + JSONL) |
| `scripts/monitoring/incident_detector.py` | GREEN / YELLOW / RED rules, adaptive + absolute |
| `scripts/monitoring/incident_capture.py` | freezes an evidence bundle when a rule fires |
| `scripts/monitoring/rl_observer.py` | supervisor loop tying the above together, writes `status/` |
| `scripts/monitoring/plot_live_metrics.py` | figures, regenerable from committed CSVs |
| `scripts/monitoring/finalize_run.py` | end-of-run metrics rebuild + `RUN_REPORT.md` |
| `scripts/monitoring/github_sync.py` | pushes small evidence; guards against large-file leakage |

## Where metrics come from

The collector reads **VeRL's own structured sink** — the `file` logger backend
(`verl/utils/tracking.py:529-548`), enabled with `trainer.logger=[...,"file"]` and
`VERL_FILE_LOGGER_PATH`. Each line is `{"step": N, "data": {...}}` carrying VeRL's real
metric names.

This is deliberate: **stdout regex scraping is fragile and would silently rot** when
upstream changes a print format. The JSONL sink is a supported interface.

Group-signal statistics are derived from VeRL's **own rollout dump**
(`trainer.rollout_data_dir`), which the trainer writes itself. Nothing in the algorithm
path is patched to obtain them.

### Honest limitation

VeRL at `1252cc71` emits `actor/pg_clipfrac`, `actor/pg_clipfrac_lower` and
`actor/ppo_kl`, but **not** importance-ratio quantiles. Those therefore read
`unavailable` and their figure panel says so. Obtaining them requires a small in-trainer
instrumentation hook, which is deferred rather than faked. **A metric that VeRL does not
emit is recorded as empty and rendered `unavailable` — never substituted with `0`.**

## Cadence

| what | when | why |
|---|---|---|
| system telemetry | every 10 s | cheap; NVML/psutil handles opened once, no subprocess per sample |
| training metrics | every optimizer update | the natural unit of RL progress |
| observer cycle | every 20 s | re-collect, detect, write status |
| trajectory audit | every N updates | full rollouts would fill the disk |
| plots | on sync | asynchronous, off the training path |
| GitHub sync | run start, incident, ~20 updates or ~15 min, checkpoint, run end | GitHub is not a time-series database |

High-frequency data stays local in `logs/` and `experiments/<RUN_ID>/telemetry/`.
Only small, reproducible, non-sensitive artifacts are pushed.

## Overhead

The sampler is one process waking every 10 s to read NVML counters and a few `psutil`
fields; the observer wakes every 20 s to re-read a small JSONL and rewrite two status
files. Plotting and git push are launched detached, off the training path. Measured
overhead is reported in each `RUN_REPORT.md`. **Target: well under 1% — if monitoring
ever costs more than ~2-3% of training throughput, it gets simplified, not tolerated.**

## Restart handling

`run_with_observer.sh` increments `restart_epoch` in `run_manifest.json` on every
relaunch of the same `RUN_ID`. The plotter draws an explicit vertical `RESTART` marker
at each boundary. **Two restarted runs are never stitched into one continuous curve** —
a stitched curve is a fabricated one.
