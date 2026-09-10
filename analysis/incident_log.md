# Incident Log

Every anomaly during this project gets an entry here, in the fixed form:

**Observed symptom -> Hypotheses -> Evidence checked -> Root cause -> Fix -> Post-fix evidence -> Lesson.**

The discipline this file enforces: *a reward curve is not ground truth.* Before touching a
hyperparameter, walk the checklist in
[`../docs/debugging_playbook.md`](../docs/debugging_playbook.md) — environment/verifier, data batch
composition, truncation, policy-version freshness, old-logprob correctness, token masking, reward
normalisation, optimizer, LR, and only then KL/clipping.

Incidents are numbered `INC-nnn` in the order they were hit, whatever stage they belong to.

---

## INC-001 — `uv sync` cannot fetch `flash-attn`: split-routed network

**Date / stage:** 2026-09-10, environment build (pre-R0).
**Run / step:** n/a — no training started.

### Observed symptom

```
x Failed to download `flash-attn==2.8.3`
|-> Request failed after 3 retries in 88.6s
|-> Failed to fetch:
|   `https://github.com/verl-project/verl-wheelhouse/releases/download/
|    flash-attention-v2.8.3/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl`
|-> client error (Connect)
`-> operation timed out
```

The same `uv sync` had already successfully downloaded `torch`, `torchvision` and `torchaudio`
from `download.pytorch.org`, so the network was clearly not down.

### Hypotheses

1. `verl-wheelhouse` index is offline.
2. Wheel URL is wrong / asset missing.
3. The node's network reaches some hosts and not others.
4. Transient timeout, retry would fix it.

### Evidence checked

Reachability matrix, measured with `curl -o /dev/null -w "%{http_code}"`, both with and without
AutoDL's `/etc/network_turbo` academic proxy:

| Host | direct | via proxy |
|---|---|---|
| `github.com` | **000** | 200 |
| `huggingface.co` | **000** | 200 |
| `verl-project.github.io` (the wheelhouse *index*) | 200 | 200 |
| `download.pytorch.org` | **200** | **000** |
| `pypi.org` | 200 | 200 |

That killed hypotheses 1, 2 and 4. The wheelhouse *index page* is on `github.io` and resolves
fine — but the **wheels themselves redirect to `github.com/.../releases/download/...`**, and
`github.com` is unreachable without the proxy.

### Root cause

The node is **split-routed**: `github.com` / `huggingface.co` are reachable *only through* the
academic proxy, while `download.pytorch.org` is reachable *only by bypassing* it. `uv sync` needs
both hosts in a single invocation, so neither proxy-on nor proxy-off can succeed alone.

### Fix

Proxy on, with a `no_proxy` carve-out for the one host the proxy blocks:

```bash
source /etc/network_turbo
export no_proxy="$no_proxy,download.pytorch.org"
export NO_PROXY="$no_proxy"
```

Persisted as `lab_proxy_on` in the (machine-local, uncommitted) `env.sh`.

### Post-fix evidence

`download.pytorch.org/whl/cu130/torch/` -> 200 (bypassing proxy) **and**
the `flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl` release asset -> 200 (through proxy), in the
same shell. `uv sync` then reported `Downloaded flash-attn`.

### Lesson

"Install failed" on a rented GPU node in a restricted-network region is far more often a
**routing** fault than a dependency fault. Build the host reachability matrix *before* editing
version pins — the tempting wrong move here was to delete the `flash-attn` pin or drop the `fsdp`
extra, which would have silently changed the training stack to work around a network problem.

---

## INC-002 — `uv sync` git dependencies reset: proxy not inherited by `git`

**Date / stage:** 2026-09-10, environment build (pre-R0).
**Run / step:** n/a.

### Observed symptom

After fixing INC-001, `uv sync` got further and then failed on a *different* dependency:

```
x Failed to download and build `mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@641a5a01...`
`-> process didn't exit successfully: `/usr/bin/git fetch --force --update-head-ok
    'https://github.com/ISEEKYAN/mbridge.git' ...` (exit status: 128)
    --- stderr
    fatal: unable to access 'https://github.com/ISEEKYAN/mbridge.git/':
    Recv failure: Connection reset by peer
```

then, after a retry, the identical error on `https://github.com/Ascend/TransferQueue.git`.

### Hypotheses

1. Those repos are unreachable from this node.
2. The pinned revision no longer exists.
3. The proxy env vars are not reaching the `git` subprocess uv spawns.
4. Transient proxy flakiness under uv's parallel fetches.

### Evidence checked

- `git ls-remote https://github.com/Ascend/TransferQueue.git` run three times in a row with the
  proxy exported: **all three succeeded**, exit 0. Same for `mbridge`. Kills (1) and (2).
- Earlier in the session, `git clone` of `verl`, `verl-recipe` and `DAPO` through the same proxy
  succeeded — so `git` *can* use the proxy when the env vars are set in that shell.
- The failures happened while uv had many parallel downloads in flight, which points at (4);
  and the error is `Recv failure: Connection reset by peer` (connection established, then reset)
  rather than a connect timeout — consistent with proxy load-shedding.

Both (3) and (4) were live, so both were addressed.

### Root cause

Two compounding causes: the spawned `git` subprocess did not reliably pick up the proxy from the
environment, **and** the academic proxy resets connections when uv saturates it with parallel
fetches.

### Fix

1. Make the proxy a property of `git` itself rather than of the environment, scoped to GitHub only
   so `download.pytorch.org` is unaffected:
   ```bash
   git config --global http.https://github.com/.proxy http://<academic-proxy>
   ```
2. Reduce proxy pressure and retry with resume, via
   [`../scripts/preflight/uv_sync_retry.sh`](../scripts/preflight/uv_sync_retry.sh):
   `UV_CONCURRENT_DOWNLOADS=4`, `UV_HTTP_TIMEOUT=180`, bounded retry loop. Because every attempt
   resumes from `UV_CACHE_DIR` on the data disk, attempts **accumulate** progress instead of
   restarting from zero.

### Post-fix evidence

Next attempt reported `Downloaded torch`, `Downloaded flash-attn`,
`Building mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@641a5a01...` and proceeded to
download `vllm` — i.e. it passed both previously-fatal git dependencies.

### Lesson

Env-var proxy configuration is not inherited as reliably as it looks once a tool spawns
subprocesses. For anything that shells out to `git`, configure the proxy **in git's own config**,
and scope it per-host so you do not accidentally route a host the proxy blocks. Separately: an
install step that is *resumable* turns a flaky network from a blocker into a delay — the retry loop
was worth more than any single fix.

---

## Template for future entries

```markdown
## INC-nnn — one-line title

**Date / stage:**            **Run / step:**

### Observed symptom
(metric, log excerpt, step number — not a paraphrase)

### Hypotheses
1. ...

### Evidence checked
(what was measured, and which hypotheses it killed)

### Root cause

### Fix

### Post-fix evidence
(the same metric/log, after)

### Lesson
```

---

## INC-003 — `uv sync` at 0.1 MB/s: PyPI unreachable at speed, and `--frozen` pins the mirror

**Date / stage:** 2026-09-10, environment build (pre-R0).
**Run / step:** n/a.

### Observed symptom

After INC-001 and INC-002 were fixed, `uv sync` ran without errors but crawled. Measured by
sampling `du -sb` on `UV_CACHE_DIR`:

```
cache delta over 60s: 6 MB   =>  0.10 MB/s
```

~5 GB of wheels remained (vllm 266 MB, cudnn 349 MB, cublas 403 MB, flashinfer-cubin 427 MB, ...).
At that rate the environment build alone was ~1.5 hours of paid GPU time.

### Hypotheses

1. The node's network is simply slow.
2. The academic proxy is throttling.
3. PyPI specifically is slow from this node.
4. uv is misconfigured / not parallelising.

### Evidence checked

**(1) is false.** The Qwen3-8B download had just pulled 16 GB from `hf-mirror.com` in ~9 minutes,
about 30 MB/s. The node has bandwidth.

**(3) is true, dramatically.** Same 30-40 MB byte-range request for the *same* `vllm-0.24.0` wheel,
against five hosts:

| source | throughput |
|---|---|
| `mirrors.aliyun.com` | **63.01 MB/s** |
| `mirrors.bfsu.edu.cn` | 48.14 MB/s |
| `repo.huaweicloud.com` | 8.95 MB/s |
| `pypi.tuna.tsinghua.edu.cn` | 6.57 MB/s |
| `mirrors.ustc.edu.cn` | 0.83 MB/s |
| `files.pythonhosted.org` **direct** | **0.04 MB/s** |
| `files.pythonhosted.org` via proxy | 0.05 MB/s |

**(2) is partly true but not the cause.** Setting `UV_DEFAULT_INDEX` to a fast mirror changed
nothing — throughput stayed at 0.4-1.2 MB/s. Socket-level proof, by mapping the uv process's
`/proc/<pid>/fd` socket inodes against `/proc/net/tcp`:

- with the proxy exported: 12 established connections, **all** to `<internal-proxy>:<port>` (the proxy);
- with the proxy unset: connections to `151.101.64.223:443` — Fastly, i.e.
  `files.pythonhosted.org`.

So uv was still fetching from PyPI while `UV_DEFAULT_INDEX` pointed at a mirror.

### Root cause

Two independent causes stacked:

1. `files.pythonhosted.org` is effectively unusable from this node (~40 KB/s), whether direct or
   proxied.
2. **`uv sync --frozen` ignores `UV_DEFAULT_INDEX`.** `uv.lock` records a fully-qualified
   `url = "https://files.pythonhosted.org/packages/..."` for every one of its 1377 PyPI artifacts,
   and `--frozen` means "install exactly what the lock says" — including the host. An index
   override only affects *resolution*, and `--frozen` skips resolution entirely.

### Fix

Retarget the lock's mirror host, leaving resolution untouched. The Aliyun mirror uses the identical
`packages/<a>/<b>/<hash>/<file>` layout, so it is a pure host substitution:

```bash
cp uv.lock system/uv.lock.upstream.bak          # revertible; also tracked in verl's git
sed -i 's|https://files\.pythonhosted\.org/|https://mirrors.aliyun.com/pypi/|g' uv.lock
```

Why this is safe rather than a version change:

- **No version, package or dependency edge is modified.** Verified by round-tripping the
  substitution and diffing against the upstream lock: zero non-URL differences.
- **All 1413 `sha256` hashes are untouched** (count identical before and after), and uv verifies
  every downloaded artifact against them. A mirror serving anything other than the exact upstream
  bytes would fail the hash check, so package identity stays cryptographically enforced.
- Fully reversible: `git checkout uv.lock` in the verl repo, or the backup above.

Then run with the proxy on and `no_proxy` covering the mirror, so PyPI artifacts go direct to
Aliyun while GitHub-hosted artifacts still traverse the proxy.

### Post-fix evidence

```
cache delta 100s: 4661 MB  =>  46.62 MB/s
```

**0.10 MB/s -> 46.62 MB/s, a ~460x speedup.** `uv sync` then completed with no errors, producing an
11 GB venv, and every validation gate that followed (torch cu130, flash-attn kernel, vLLM, 4-rank
NCCL) passed on the resulting environment.

### Lesson

Two lessons, and the second is the more general one:

1. When a download is slow in a restricted-network region, **measure per-host throughput before
   changing any configuration**. The fix was a mirror, and the mirrors differ from each other by
   75x — picking one without measuring would likely have picked `ustc` (0.83 MB/s) or `tuna`
   (6.57 MB/s) over `aliyun` (63 MB/s).
2. **Know which knob a flag disables.** `UV_DEFAULT_INDEX` looked like the obvious fix and was
   silently a no-op, because `--frozen` bypasses resolution. Two attempts were spent on it before
   the socket dump settled the question. Guessing at configuration is slower than measuring what a
   process is actually connected to.

---

## INC-004 — probe crashed in scoring *after* the expensive arm had generated

**Date / stage:** 2026-09-10, pre-R0 length-budget probe.
**Run / step:** n/a — diagnostic tooling, not training.

### Observed symptom

The length-budget probe ran arm A (thinking mode, `max_response_length=16384`) to
completion on the GPU — roughly 12 minutes of H20 time at 100% utilization — and then
died before writing a single result:

```
Traceback (most recent call last):
  File ".../length_budget_probe.py", line 89, in main
    row = dict(n_tokens=len(c.token_ids), truncated=trunc,
TypeError: dict() got multiple values for keyword argument 'state'
WARNING [core_client.py:702] [shutdown] MPClient: engine core exited unexpectedly
```

### Hypotheses

1. vLLM engine crash / OOM at the 16384 context (the `MPClient` shutdown warning is the
   most visually alarming line in the log).
2. A bug in the probe's own scoring code.

### Evidence checked

- The traceback is a plain `TypeError` in **our** script, above the vLLM shutdown line.
  The `MPClient: engine core exited unexpectedly` message is the *consequence* of the
  parent process dying, not the cause — engine teardown follows the crash.
- No `CUDA out of memory`, no OOM killer entry.
- GPU had been sitting at 100% / 460 W with 90.5 GiB in use and had produced output
  normally right up to the end.

Hypothesis 1 is dead: the scary infrastructure-looking line was downstream of an
ordinary application bug.

### Root cause

`score()` returns a dict that already contains a `state` key. The row builder did
`dict(..., state=<derived>, **s)`, so `state` was supplied twice — once explicitly and
once via `**s`. Python rejects that at call time. The failure was unconditional and
would have fired on the very first scored sample.

### Fix

Pop the key before splatting:

```python
s = dict(score(c.text, gt))
base_state = s.pop("state")          # avoid duplicate kwarg with **s
row = dict(n_tokens=..., truncated=trunc,
           state="TRUNCATED" if trunc and base_state == "PARSE_FAILURE" else base_state,
           **s)
```

and, more importantly, added a `_selftest()` that exercises the scoring + row-building
path on two synthetic samples **before** the model is loaded or any token is generated.

### Post-fix evidence

Relaunch printed `[selftest] scoring/row path OK` within seconds of start, before vLLM
initialisation.

### Lesson

Two, and the second is the expensive one:

1. **Read the traceback, not the scariest line.** `MPClient: engine core exited
   unexpectedly` looks like a vLLM/CUDA infrastructure failure and would have sent a
   diagnosis down the wrong path. It was a consequence of the parent dying. This is the
   same misattribution pattern the whole lab exists to study, encountered in our own
   tooling.
2. **Any code path that runs *after* an expensive GPU stage must be exercised *before*
   it.** The bug was deterministic and would have been caught in under a second by a
   synthetic sample. Instead it cost a full generation arm. Post-processing code is
   exactly where this happens, because it only executes once the expensive part is
   already done — cheap smoke-test first, then spend the GPU.


---

## INC-005 — R1 post-run reporting semantics and provenance gaps

**Date / stage:** 2026-09-10, R1 post-run audit. **Run:** R1_grpo_baseline, updates 1–20.

### Observed symptom
Width-hit fractions were labeled truncation even at update 10 with maximum 6625 < configured 8192.
Missing verifier timing became 0%. Validation at updates 10/20 was omitted from the CSV/plot.
Vanilla PPO objective clipping and dual clipping were mislabeled upper/lower.
The resolved-config file was a placeholder; the manifest dataset hash differed from the selected deduplicated file.
The final public status still showed update 19 after 20 updates completed.

### Hypotheses
1. Training/configuration malfunction.
2. Instrumentation mappings and metadata capture fail to represent the underlying run.

### Evidence checked
Source reads: metric_utils.py uses response tensor width; core_algos.py distinguishes objective and dual clipping.
Actual shell xtrace confirms 8192, mini-batch 8 < batch 16, and deduplicated training path.
Native metrics have two validation results. All 2560 saved outputs reproduce their original scores.
Report regeneration previously overwrote end_ts with audit time; the original recorded end is now preserved.
See analysis/R1_baseline_report.md and analysis/R1_audit.json for values and limitations.

### Root cause
PENDING

### Fix
Reporting-only mapping corrections; keep exact truncation unavailable and show a labeled cap-hit proxy.
Preserve missingness, native clipping meanings, actual validation points and original end timestamp.
Retain original manifest provenance; supplement with actual xtrace overrides and post-run dataset hashes.
Mark status completed. No training hyperparameter or verifier changes.

### Post-fix evidence
CPU synthetic reporting tests passed; regenerated 20-row metrics and figures.
Validation is 0.495 / 0.485 only at steps 10 / 20; verifier timing remains unavailable.
Configured-cap proxy is 5/2560 versus 22/2560 tensor-width hits.
Native training metric records and all detector root-cause fields were preserved.

### Lesson
Audit an instrument's source definition before interpreting a label; missing metadata cannot be reconstructed as a historical measurement.

---

## INC-006 — DataLoader signal exception during R1 exit cleanup

**Date / stage:** 2026-09-10, R1 termination. **Run / step:** R1_grpo_baseline, after 20/20 progress.

### Observed symptom
Exit-phase traceback: weakref._exitfunc -> torch.library._del_library ->
torch DataLoader signal handler -> RuntimeError: DataLoader worker killed by signal.
Ray debug messages report child exits with code 1.

### Hypotheses
1. Worker termination during normal distributed teardown.
2. External kill, including resource pressure.
3. Another cleanup lifecycle problem.

### Evidence checked
The original traceback is in the weakref exit finalizer. All 20 native metric rows and rollout dumps exist.
Final validation exists; launcher explicitly records trainer exit code 0 and completed finalization.
No native scalar is nonfinite. GPU allocations are released at post-run audit.
A complete kernel/cgroup kill history was not captured; the signal's cause is not established.

### Root cause
PENDING

### Fix
No training restart or environmental change. Preserve the anomaly and qualify the exit-0 report.

### Post-fix evidence
NOT RUN — no causal fix applied. Existing completed run artifacts were independently checked.

### Lesson
Distinguish top-level exit code, child exits and shutdown traceback context; neither hide the traceback nor reclassify completed measurements without evidence.
