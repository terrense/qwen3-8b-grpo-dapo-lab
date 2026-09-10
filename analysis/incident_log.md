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
