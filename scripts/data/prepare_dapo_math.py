"""Prepare DAPO-Math-17k for VeRL, with the 100x repetition removed.

The upstream parquet ships 1,791,700 rows but only 17,917 unique prompts -- each
repeated EXACTLY 100 times, keyed by extra_info.index. Feeding that to VeRL raw
would make `total_epochs=1` mean 100 passes over the same prompts, and would put
duplicate prompts in a single batch, which in GRPO silently double-weights them
(two independent groups for one problem).

So we deduplicate on extra_info.index and write an explicit, shuffled train split
plus a held-out validation split. The upstream schema is preserved verbatim --
VeRL's RLHFDataset reads `prompt`, `data_source`, `reward_model`, `extra_info`
and all four already exist with the right shapes, so no schema is invented here.
"""

import argparse
import hashlib
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

SRC = "/root/autodl-tmp/rl_lab/data/raw/DAPO-Math-17k/data/dapo-math-17k.parquet"
OUTDIR = "/root/autodl-tmp/rl_lab/data/dapo_math_17k"
SEED = 20260910


def sha256_16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-train", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--smoke-train", type=int, default=64,
                    help="tiny split for R0 pipeline smoke tests")
    a = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    t = pq.read_table(SRC)
    print(f"source rows        : {t.num_rows}")
    print(f"source columns     : {t.schema.names}")

    ei = t.column("extra_info").to_pylist()
    idx = [r["index"] for r in ei]

    # keep the FIRST occurrence of each unique prompt
    seen, keep = set(), []
    for i, k in enumerate(idx):
        if k not in seen:
            seen.add(k)
            keep.append(i)
    print(f"unique prompts     : {len(keep)}  (repeat factor "
          f"{t.num_rows / len(keep):.1f}x removed)")

    dedup = t.take(keep)

    # deterministic shuffle
    import random
    random.seed(SEED)
    order = list(range(dedup.num_rows))
    random.shuffle(order)
    dedup = dedup.take(order)

    n_val = a.n_val
    val = dedup.slice(0, n_val)
    rest = dedup.slice(n_val)
    train = rest if a.n_train == 0 else rest.slice(0, a.n_train)
    smoke = rest.slice(0, a.smoke_train)

    paths = {}
    for name, tbl in (("train", train), ("val", val), ("smoke_train", smoke)):
        p = os.path.join(OUTDIR, f"{name}.parquet")
        pq.write_table(tbl, p)
        paths[name] = dict(path=p, rows=tbl.num_rows,
                           bytes=os.path.getsize(p), sha256_16=sha256_16(p))
        print(f"  {name:12s} {tbl.num_rows:6d} rows -> {p}")

    manifest = dict(
        source_parquet=SRC, source_rows=t.num_rows, unique_prompts=len(keep),
        repeat_factor=round(t.num_rows / len(keep), 3),
        dedup_key="extra_info.index", seed=SEED,
        columns=t.schema.names, splits=paths,
        note=("Deduplicated on extra_info.index. Upstream ships each prompt exactly "
              "100x; using the raw file would make 1 epoch mean 100 passes and would "
              "place duplicate prompts in one GRPO batch."),
    )
    mp = os.path.join(OUTDIR, "manifest.json")
    json.dump(manifest, open(mp, "w"), indent=2)
    print(f"\nmanifest -> {mp}")

    # show one record so the schema is confirmed from the real file, not assumed
    r = train.slice(0, 1).to_pylist()[0]
    print("\nsample record keys:", list(r.keys()))
    print("  data_source :", r["data_source"])
    print("  prompt[0]   :", {k: (v[:90] + "...") if isinstance(v, str) and len(v) > 90 else v
                              for k, v in r["prompt"][0].items()})
    print("  reward_model:", r["reward_model"])
    print("  extra_info  :", r["extra_info"])
    print("DATA_PREP_DONE")


if __name__ == "__main__":
    main()
