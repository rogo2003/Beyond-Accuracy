"""10 — Table 14a: image-level cross-benchmark near-duplicate overlap.

Runs locally (no GPU, no images): only needs the four *_phash_group_split.csv files.
For every ordered pair (target, source): the share of target images with at least one
near-duplicate (64-bit DCT pHash, Hamming distance ≤ 6) anywhere in the source dataset,
and the number with an exact match (distance 0).

Usage:  python audit/local/10_overlap_table14a.py [--splits "Output files"] [--out audit/results]
"""
import argparse, os
import numpy as np, pandas as pd

THR = 6
DATASETS = ["D1", "D2", "D3", "D4"]
# Values printed in the revised manuscript (Table 14a): target -> source -> (near-dup %, exact n)
EXPECTED = {
    "D1": {"D2": (67.8, 2029), "D3": (70.2, 235), "D4": (86.9, 2704)},
    "D2": {"D1": (34.9, 1989), "D3": (29.5, 158), "D4": (80.5, 4792)},
    "D3": {"D1": (79.6, 240), "D2": (61.9, 162), "D4": (66.0, 173)},
    "D4": {"D1": (42.1, 2757), "D2": (69.7, 4811), "D3": (26.6, 172)},
}

POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def to_u64(s):
    return np.array([int(x) for x in s], dtype=np.uint64)

def min_hamming(q, ref, chunk=256):
    out = np.empty(len(q), dtype=np.int64)
    for i in range(0, len(q), chunk):
        b = q[i:i + chunk]
        x = np.ascontiguousarray(b[:, None] ^ ref[None, :]).view(np.uint8)
        out[i:i + chunk] = POP8[x.reshape(len(b), len(ref), 8)].sum(-1).min(1)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="Output files")
    ap.add_argument("--out", default="audit/results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = {k: pd.read_csv(os.path.join(a.splits, f"{k}_phash_group_split.csv"), dtype={"phash": str}) for k in DATASETS}
    h = {k: to_u64(v.phash) for k, v in df.items()}
    rows, ok = [], True
    for t in DATASETS:
        for s in DATASETS:
            if s == t: continue
            m = min_hamming(h[t], h[s])
            near, exact = (m <= THR), (m == 0)
            r = {"target": t, "source": s, "n_target": len(m), "near_dup_n": int(near.sum()),
                 "near_dup_pct": round(near.mean() * 100, 1), "exact_n": int(exact.sum()),
                 "median_min_hamming": int(np.median(m)), "max_min_hamming": int(m.max())}
            exp = EXPECTED[t][s]
            r["matches_manuscript"] = (r["near_dup_pct"], r["exact_n"]) == exp
            ok &= r["matches_manuscript"]
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "table14a_overlap.csv"), index=False)
    wide = res.assign(cell=res.near_dup_pct.map("{:.1f}%".format) + " (" + res.exact_n.map("{:,}".format) + ")") \
              .pivot(index="target", columns="source", values="cell").fillna("—")
    print(wide.to_string())
    print("\nAll cells match the revised manuscript:", "YES" if ok else "NO — see table14a_overlap.csv")

if __name__ == "__main__":
    main()
