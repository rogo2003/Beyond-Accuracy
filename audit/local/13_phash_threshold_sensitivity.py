"""13 — Supplementary Table S1: sensitivity of duplicate grouping to the pHash Hamming threshold δ.

Runs locally (no GPU, no images): only needs the four *_phash_group_split.csv files.
For δ = 0…10, images whose 64-bit pHash differ by ≤ δ bits are linked and grouped by
connected components (equivalent to the notebook's union-find). Reported per dataset and δ:
number of groups, share of images in multi-image groups, size of the largest group
(chaining indicator), and number of groups that contain more than one tumor label
(an objective sign that non-duplicates have been merged).

Check: at δ = 6 the group counts must equal the `dup_group` counts saved by the notebook.

Usage:  python audit/local/13_phash_threshold_sensitivity.py [--splits "Output files"] [--out audit/results]
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

DATASETS = ["D1", "D2", "D3", "D4"]
POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def to_u64(s):
    return np.array([int(x) for x in s], dtype=np.uint64)

def close_pairs(h, max_d=10, chunk=256):
    """All pairs (i < j) with Hamming distance ≤ max_d."""
    n = len(h); I, J, D = [], [], []
    for i in range(0, n, chunk):
        b = h[i:i + chunk]
        dist = POP8[np.ascontiguousarray(b[:, None] ^ h[None, :]).view(np.uint8).reshape(len(b), n, 8)].sum(-1)
        ii, jj = np.nonzero(dist <= max_d); keep = (ii + i) < jj
        I.append(ii[keep] + i); J.append(jj[keep]); D.append(dist[ii[keep], jj[keep]])
    return np.concatenate(I), np.concatenate(J), np.concatenate(D)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="Output files")
    ap.add_argument("--out", default="audit/results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows, ok = [], True
    for ds in DATASETS:
        df = pd.read_csv(os.path.join(a.splits, f"{ds}_phash_group_split.csv"), dtype={"phash": str})
        n, lab = len(df), df.label.values
        I, J, D = close_pairs(to_u64(df.phash))
        for d in range(11):
            m = D <= d
            k, cc = connected_components(coo_matrix((np.ones(m.sum()), (I[m], J[m])), shape=(n, n)), directed=False)
            sizes = np.bincount(cc); multi = np.nonzero(sizes > 1)[0]
            mixed = int(sum(len(set(lab[cc == c])) > 1 for c in multi))
            rows.append({"dataset": ds, "delta": d, "n_images": n, "n_groups": k,
                         "pct_images_in_dup_groups": round(sizes[sizes > 1].sum() / n * 100, 1),
                         "largest_group": int(sizes.max()), "label_mixed_groups": mixed, "linked_pairs": int(m.sum())})
        g6 = [r for r in rows if r["dataset"] == ds and r["delta"] == 6][0]["n_groups"]
        match = g6 == df.dup_group.nunique(); ok &= match
        print(f"{ds}: δ=6 groups {g6} vs notebook {df.dup_group.nunique()} → {'MATCH' if match else 'MISMATCH'}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "phash_threshold_sensitivity.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n" + res.pivot(index="delta", columns="dataset", values="largest_group").add_prefix("largest_").join(
          res.pivot(index="delta", columns="dataset", values="label_mixed_groups").add_prefix("mixed_")).to_string())
    print("\nCheck against notebook groups:", "PASS" if ok else "FAIL")

if __name__ == "__main__":
    main()
