# Figures

Every figure here is **regenerated from the committed metric files** by the scripts in
[`../scripts/plotting/`](../scripts/plotting/). Nothing is hand-drawn and nothing is exported from
a WandB dashboard, so any figure can be rebuilt from this repository alone.

`generated/` holds the output. Core figures:

- reward vs step
- validation accuracy vs step
- KL vs step
- entropy vs step
- clip fraction vs step (total, low, high)
- gradient norm vs step
- response length vs step (mean / median / p95 / max)
- zero-std group ratio vs step
- rollout time vs actor-update time
- GRPO vs DAPO comparison
- failure-experiment comparisons

Raw traces are always plotted alongside any smoothed curve — a smoothed-only plot hides exactly
the spikes this project exists to study. Restart boundaries are drawn as explicit vertical
markers; runs are never stitched into a single continuous line.

**NOT RUN** — no figures generated yet.
