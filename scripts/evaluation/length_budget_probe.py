"""Length-budget probe: what max_response_length does Qwen3-8B actually need here?

The 64x8 preflight rollout showed 87.1% truncation at max_response_length=4096,
with every response opening a <think> block. This probe tests the two candidate
fixes side by side on the SAME prompts and the SAME verifier, so the R1 config
is chosen from measurement instead of intuition.

Arms:
  A  think_16384      thinking on (template default), cap 16384
  B  nothink_4096     thinking off (enable_thinking=False), cap 4096

Deliberately NOT tested: loosening the verifier. The verifier contract is not
the thing that is wrong here.
"""

import json
import os
import statistics
import time
from collections import Counter

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score import default_compute_score

MODEL = "/root/autodl-tmp/rl_lab/models/Qwen3-8B"
PARQUET = "/root/autodl-tmp/rl_lab/data/raw/DAPO-Math-17k/data/dapo-math-17k.parquet"
OUT = "/root/autodl-tmp/rl_lab/analysis/length_budget_probe.json"

N_PROMPTS, G, SEED = 16, 4, 20260910
ARMS = [("A_think_16384", True, 16384), ("B_nothink_4096", False, 4096)]
INVALID = (None, "[INVALID]", "")


def score(sol, gt):
    try:
        r = default_compute_score(data_source="math_dapo", solution_str=sol, ground_truth=gt)
    except Exception as e:  # noqa: BLE001
        return dict(score=None, acc=None, pred=None, state="VERIFIER_EXCEPTION", err=repr(e))
    st = "OK_CORRECT" if r.get("acc") else ("PARSE_FAILURE" if r.get("pred") in INVALID
                                            else "OK_INCORRECT")
    return dict(score=r.get("score"), acc=bool(r.get("acc")), pred=r.get("pred"), state=st)


def _selftest():
    """Exercise the scoring+row path on one synthetic sample before any GPU work.

    Added after a `dict() got multiple values for keyword argument 'state'` bug
    crashed the probe AFTER a full arm had already generated -- wasting the
    expensive half and keeping none of it.
    """
    s = dict(score("Answer: 7", "7"))
    base_state = s.pop("state")
    row = dict(n_tokens=3, truncated=False, state=base_state, **s)
    assert row["score"] == 1.0 and row["state"] == "OK_CORRECT", row
    s2 = dict(score("no answer here", "7"))
    b2 = s2.pop("state")
    _ = dict(n_tokens=3, truncated=True,
             state="TRUNCATED" if b2 == "PARSE_FAILURE" else b2, **s2)
    print("[selftest] scoring/row path OK")


def main():
    _selftest()
    import random
    random.seed(SEED)
    t = pq.read_table(PARQUET, columns=["reward_model", "extra_info", "prompt"])
    rm, ei, pr = (t.column(c).to_pylist() for c in ("reward_model", "extra_info", "prompt"))
    seen = {}
    for r, e, p in zip(rm, ei, pr):
        seen.setdefault(e["index"], (r["ground_truth"], p))
    keys = sorted(seen)
    picked = random.sample(keys, N_PROMPTS)

    tok = AutoTokenizer.from_pretrained(MODEL)
    results = {}
    max_cap = max(c for _, _, c in ARMS)
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=2048 + max_cap + 256, tensor_parallel_size=1, seed=SEED)

    for arm, thinking, cap in ARMS:
        texts, gts = [], []
        for k in picked:
            gt, chat = seen[k]
            chat = [dict(role=m["role"], content=m["content"]) for m in chat]
            try:
                txt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=thinking)
            except TypeError:
                txt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            texts.append(txt)
            gts.append(gt)

        sp = SamplingParams(n=G, temperature=1.0, top_p=1.0, max_tokens=cap)
        t0 = time.time()
        outs = llm.generate(texts, sp)
        dt = time.time() - t0

        rows, groups, ntok = [], [], 0
        for gt, o in zip(gts, outs):
            grp = []
            for c in o.outputs:
                ntok += len(c.token_ids)
                s = dict(score(c.text, gt))
                trunc = c.finish_reason == "length"
                base_state = s.pop("state")          # avoid duplicate kwarg with **s
                row = dict(n_tokens=len(c.token_ids), truncated=trunc,
                           state="TRUNCATED" if trunc and base_state == "PARSE_FAILURE"
                           else base_state,
                           **s)
                rows.append(row)
                grp.append(row)
            groups.append(grp)

        n = len(rows)
        lens = sorted(r["n_tokens"] for r in rows)
        ac = sum(1 for g in groups if all(r["score"] == 1.0 for r in g))
        aw = sum(1 for g in groups if all(r["score"] != 1.0 for r in g))
        zs = sum(1 for g in groups if len({r["score"] for r in g}) == 1)
        results[arm] = dict(
            arm=arm, thinking=thinking, cap=cap, n_prompts=len(groups), G=G, n=n,
            wall_s=round(dt, 1), out_tokens=ntok, tok_per_s=round(ntok / dt),
            accuracy=sum(1 for r in rows if r["score"] == 1.0) / n,
            reward_mean=statistics.mean([r["score"] for r in rows if r["score"] is not None]),
            truncation_rate=sum(1 for r in rows if r["truncated"]) / n,
            len_mean=statistics.mean(lens), len_median=statistics.median(lens),
            len_p95=lens[int(0.95 * (len(lens) - 1))], len_max=max(lens),
            all_correct_group_ratio=ac / len(groups), all_wrong_group_ratio=aw / len(groups),
            mixed_group_ratio=(len(groups) - zs) / len(groups),
            zero_std_group_ratio=zs / len(groups),
            states=dict(Counter(r["state"] for r in rows)),
        )
        r = results[arm]
        print(f"\n===== {arm}  (thinking={thinking}, cap={cap}) =====")
        print(f"  wall {r['wall_s']}s  {r['tok_per_s']} tok/s  out_tokens {r['out_tokens']}")
        print(f"  accuracy            {r['accuracy']:6.2%}   reward_mean {r['reward_mean']:+.4f}")
        print(f"  truncation_rate     {r['truncation_rate']:6.2%}")
        print(f"  len mean/med/p95/max {r['len_mean']:.0f}/{r['len_median']:.0f}/"
              f"{r['len_p95']}/{r['len_max']}")
        print(f"  zero_std_group_ratio {r['zero_std_group_ratio']:6.2%}   "
              f"mixed {r['mixed_group_ratio']:6.2%}")
        print(f"  states {r['states']}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")
    print("LENGTH_BUDGET_PROBE_DONE")


if __name__ == "__main__":
    main()
