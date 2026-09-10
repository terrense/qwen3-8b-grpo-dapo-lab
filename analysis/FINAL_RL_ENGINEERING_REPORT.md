# Final RL Engineering Report

**Status: NOT RUN.** This is the report skeleton. Every answer below must cite a real metric, log
line, trajectory, step number, config value, run id or commit SHA from this project. Textbook
description is explicitly *not* an acceptable substitute, and any question still unanswered stays
marked `NOT RUN` rather than being filled with plausible generalities.

The deliverable of this project is this document, not a checkpoint.

---

## Scope

GRPO systems pilot / rehearsal on the released post-trained `Qwen/Qwen3-8B`, using VeRL on
4 x NVIDIA H20 96GB. **Not** the formal `M3` of the medical paper
(`Qwen3-8B -> SFT M1 -> DPO M2 -> GRPO M3`).

---

## Questions to answer

1. **What is the real data flow of this VeRL GRPO pipeline?** — NOT RUN
2. **When are the rollout policy and the training policy synchronised?** — NOT RUN
3. **When is `old_logprob` produced or recomputed?** — NOT RUN
4. **How is the response token mask actually constructed?** — NOT RUN
5. **Why must prompt / tool / environment tokens stay out of the policy loss?** — NOT RUN
6. **How is the GRPO group advantage computed in this code?** — NOT RUN
7. **How many zero-variance groups actually occurred?** — NOT RUN
8. **What did dynamic sampling do to the effective batch?** — NOT RUN
9. **What did `rollout.n` change from an engineering standpoint?** — NOT RUN
10. **What did `max_response_length` do to KV cache, throughput and truncation?** — NOT RUN
11. **When KL rose, what happened to reward?** — NOT RUN
12. **What did falling entropy mean, and did performance rise with it?** — NOT RUN
13. **What does a high clip fraction actually indicate here?** — NOT RUN
14. **What were the concrete symptoms of an excessive learning rate?** — NOT RUN
15. **How did verifier failure contaminate the reward?** — NOT RUN
16. **What are the checkpoint/resume traps?** — NOT RUN
17. **Where was the bottleneck when GPU utilisation was low?** — NOT RUN
18. **How was wall time split between rollout and actor update?** — NOT RUN
19. **What was the clearest dynamics change from vanilla GRPO to DAPO?** — NOT RUN
20. **Which apparent "training failures" were really systems / data / verifier faults?** — NOT RUN

---

## Findings available so far

Only infrastructure findings exist at this point; no training has run.

### The environment contract was the first real trap

The current VeRL checkout pins a **CUDA 13.0 / torch 2.11.0 / vLLM 0.24.0** world, while the node
ships torch 2.8.0+cu128. The node driver (580.105.08) advertises CUDA 13.0, so the pinned stack is
valid here. Upgrading the ambient torch in place would have been the wrong move; the official
`uv sync --frozen` isolated venv keeps both worlds intact. Recorded in
[`../docs/rl_pipeline_notes.md`](../docs/rl_pipeline_notes.md).

### Two install failures, neither of which was a dependency problem

Both were network-routing faults wearing a dependency error's clothing — see
[`incident_log.md`](incident_log.md), INC-001 and INC-002. This is already an instance of question
20, before a single training step has run: the error text named `flash-attn` and `mbridge`, and the
actual causes were a split-routed proxy and a subprocess not inheriting proxy configuration.

### Measured interconnect

351 GB/s busBW on a 1 GB 4-rank all-reduce, 0.01 % rank spread. Establishes that any later
throughput problem is *not* the interconnect — a baseline worth having before blaming NCCL for a
slow run.
