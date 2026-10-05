# ═══════════════════════════════════════════════════════════════════════
# 04 — Temperature sensitivity + near-duplicate example pairs
#      (run after 00_load_models.py in the same session; ~5–10 min on a T4)
#
# A. Temperature: NLL and ECE (15 bins) of each calibrated model as a function of
#    T ∈ [0.5, 2.0], on the calibration split (where T* was fitted) and on the test split.
#    Check: the T minimising calibration NLL must reproduce the stored T* (±0.02).
# B. Near-duplicate examples (Supplementary Fig. for Reviewer 8, Comment 8.2):
#    one random within-dataset image pair at pHash Hamming distance δ = 0, 3, 6, 7, 8
#    for each dataset, shown as the raw released images with their labels.
# Outputs → /kaggle/working/audit/: temperature_sensitivity.csv/.png,
#           duplicate_pair_examples.csv/.png
# ═══════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

OUT = "/kaggle/working/audit"; os.makedirs(OUT, exist_ok=True)
T_GRID = np.round(np.linspace(0.5, 2.0, 151), 3)
PAIR_DELTAS = [0, 3, 6, 7, 8]

# ── A. Temperature sensitivity ──────────────────────────────────────────
@torch.no_grad()
def raw_logits(model, paths, bs=64):
    dl = DataLoader(PathDataset(paths), batch_size=bs, num_workers=2)
    return torch.cat([model(x.to(DEVICE)).float().cpu() for x in dl])

def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)

rows, gate = [], []
for ds in ["D1", "D2", "D3", "D4"]:
    model, T_star = models[ds]
    for part in ["cal", "test"]:
        sub = splits[ds][splits[ds].split == part]
        y = sub.label.map({c: i for i, c in enumerate(CLASSES[ds])}).values
        z = raw_logits(model, sub.path); yt = torch.as_tensor(y)
        for T in T_GRID:
            p = torch.softmax(z / T, 1)
            rows.append({"dataset": ds, "split": part, "T": float(T), "n": len(y),
                         "nll": float(torch.nn.functional.cross_entropy(z / T, yt)),
                         "ece": ece(p.numpy(), y)})
    r = pd.DataFrame([q for q in rows if q["dataset"] == ds and q["split"] == "cal"])
    T_min = float(r.loc[r.nll.idxmin(), "T"])
    gate.append({"dataset": ds, "T_star_checkpoint": round(T_star, 4), "T_argmin_cal_nll": T_min,
                 "match": abs(T_min - T_star) <= 0.02})
temp = pd.DataFrame(rows); temp.to_csv(f"{OUT}/temperature_sensitivity.csv", index=False)
gate = pd.DataFrame(gate); gate.to_csv(f"{OUT}/temperature_gate.csv", index=False)
print("=== Temperature: argmin of calibration NLL vs stored T* ===")
print(gate.to_string(index=False))
# relative change of test NLL / ECE across T ∈ [0.8, 1.2] (robustness near the optimum)
near = temp[(temp.split == "test") & temp["T"].between(0.8, 1.2)].groupby("dataset")[["nll", "ece"]].agg(["min", "max"])
print("\nTest NLL / ECE range for T in [0.8, 1.2]:\n" + near.round(4).to_string())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for ds, col in zip(["D1", "D2", "D3", "D4"], ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]):
    for k, metric in enumerate(["nll", "ece"]):
        for part, ls in [("cal", "-"), ("test", "--")]:
            r = temp[(temp.dataset == ds) & (temp.split == part)]
            ax[k].plot(r["T"], r[metric], ls, color=col, lw=1.8, label=f"{ds} {part}" if k == 0 else None)
        ax[k].axvline(models[ds][1], color=col, lw=0.8, alpha=0.5)
ax[0].set_ylabel("Negative log-likelihood"); ax[1].set_ylabel("ECE (15 bins)")
for a_ in ax: a_.set_xlabel("Temperature T"); a_.grid(alpha=0.3)
ax[0].legend(frameon=False, ncol=2, fontsize=8)
ax[0].set_title("(a) NLL vs T (solid: calibration, dashed: test)", loc="left", fontsize=10)
ax[1].set_title("(b) ECE vs T; vertical lines = fitted T*", loc="left", fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/temperature_sensitivity.png", dpi=300); plt.close()

# ── B. Near-duplicate example pairs ─────────────────────────────────────
POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def pairs_at_distance(df, d, max_scan=None, seed=0):
    h = np.array([int(x) for x in df.phash], dtype=np.uint64); n = len(h)
    I, J = [], []
    for i in range(0, n, 256):
        b = h[i:i + 256]
        dist = POP8[np.ascontiguousarray(b[:, None] ^ h[None, :]).view(np.uint8).reshape(len(b), n, 8)].sum(-1)
        ii, jj = np.nonzero(dist == d); keep = (ii + i) < jj
        I.append(ii[keep] + i); J.append(jj[keep])
    I, J = np.concatenate(I), np.concatenate(J)
    if not len(I): return None
    k = np.random.default_rng(seed).integers(len(I))
    return int(I[k]), int(J[k]), len(I)

sel = []
for ds in ["D1", "D2", "D3", "D4"]:
    df = splits[ds].reset_index(drop=True)
    for d in PAIR_DELTAS:
        r = pairs_at_distance(df, d)
        if r is None: continue
        i, j, npairs = r
        sel.append({"dataset": ds, "delta": d, "n_pairs_at_delta": npairs,
                    "path_a": df.path[i], "label_a": df.label[i], "path_b": df.path[j], "label_b": df.label[j]})
sel = pd.DataFrame(sel); sel.to_csv(f"{OUT}/duplicate_pair_examples.csv", index=False)

fig, ax = plt.subplots(len(PAIR_DELTAS), 8, figsize=(16, 2.3 * len(PAIR_DELTAS)))
for ri, d in enumerate(PAIR_DELTAS):
    for di, ds in enumerate(["D1", "D2", "D3", "D4"]):
        row = sel[(sel.dataset == ds) & (sel.delta == d)]
        for k in range(2):
            a_ = ax[ri][2 * di + k]; a_.axis("off")
            if not len(row): continue
            r = row.iloc[0]; path, lab = (r.path_a, r.label_a) if k == 0 else (r.path_b, r.label_b)
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            a_.imshow(img); flag = "  ≠ label" if r.label_a != r.label_b else ""
            a_.set_title(f"{ds} δ={d} — {lab.replace('_', ' ')}{flag if k == 1 else ''}", fontsize=7.5)
plt.tight_layout(); plt.savefig(f"{OUT}/duplicate_pair_examples.png", dpi=200); plt.close()
print("\n=== Near-duplicate example pairs ===")
print(sel[["dataset", "delta", "n_pairs_at_delta", "label_a", "label_b"]].to_string(index=False))
print(f"\nSaved to {OUT}/")
