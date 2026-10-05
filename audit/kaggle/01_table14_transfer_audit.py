# ═══════════════════════════════════════════════════════════════════════
# 01 — Re-run all six Table 14 transfers under three input pipelines,
#          stratified by near-duplicate contamination
#          (run after 00_load_models.py in the same session)
# ═══════════════════════════════════════════════════════════════════════
import json
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score

CLASSES["D2"] = ["glioma", "meningioma", "no_tumor", "pituitary"]
if "D2" not in models: models["D2"] = load_calibrated("D2")
IN_DOMAIN = {"D2": 0.9766, "D3": 0.9532, "D4": 0.9666}          # published test acc
PUBLISHED = {("D2", "D4"): 79.30, ("D4", "D2"): 63.55, ("D2", "D3"): 27.42,
             ("D4", "D3"): 31.77, ("D3", "D2"): 73.00, ("D3", "D4"): 69.25}
THR, SEEN = 6, {"train", "val", "cal"}
OUT = "/kaggle/working"

def read_split(ds):
    hits = glob.glob(f"/kaggle/input/**/{ds}_phash_group_split.csv", recursive=True)
    assert hits, f"{ds}_phash_group_split.csv not found — upload it with the checkpoints"
    df = pd.read_csv(hits[0], dtype={"phash": str})
    assert os.path.exists(df.path.iloc[0]), f"{ds} image paths don't resolve — attach the raw dataset"
    return df

# ── Three input pipelines
def load_published(path):      # exactly what cross-gen.ipynb did (no crop, no median blur)
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

# D3 (denizkavi1/brain-tumor) PNGs are viridis pseudo-colour renderings (RGBA), not grayscale.
# Harmonised pipeline = bring the target into the SOURCE model's representation:
#   source D2/D4 (grayscale-trained) → invert viridis on colour targets back to intensity
#   source D3 (viridis-trained)      → apply viridis to grayscale targets
from matplotlib import colormaps
from scipy.spatial import cKDTree
VIR = np.round(colormaps["viridis"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
_q = np.stack(np.meshgrid(*[np.arange(0, 256, 2)] * 3, indexing="ij"), -1).reshape(-1, 3)
VIR_INV = cKDTree(VIR.astype(np.float32)).query(_q.astype(np.float32))[1] \
            .astype(np.uint8).reshape(128, 128, 128)            # RGB (step 2) → viridis index
del _q

def is_pseudocolour(rgb):
    return np.abs(rgb[..., 0].astype(np.int16) - rgb[..., 2]).mean() > 3

def to_gray(rgb):
    if not is_pseudocolour(rgb): return rgb[..., 1]
    q = rgb >> 1
    return VIR_INV[q[..., 0], q[..., 1], q[..., 2]]

def make_harmonised(src):
    colour = (src == "D3")
    def fn(path):
        g = to_gray(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
        img = VIR[g] if colour else cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        img = crop_brain_contour(np.ascontiguousarray(img))
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        return cv2.medianBlur(img, MEDIAN_K)
    return fn

# Sanity check of the viridis hypothesis on 50 raw D3 files and 50 D4 files
def lut_dist(path):
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)
    return np.median(cKDTree(VIR.astype(np.float32)).query(rgb[::97])[0])
for ds in ["D3", "D4"]:
    sample = read_split(ds).path.sample(50, random_state=0)
    print(f"[CHECK] {ds}: pseudo-colour {np.mean([is_pseudocolour(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)) for p in sample]):.0%}, "
          f"median distance to viridis LUT = {np.median([lut_dist(p) for p in sample]):.2f} (≈0 ⇒ viridis)")

LOADERS = {"published": load_published, "training": load_and_enhance, "harmonised": None}

class LoaderDS(Dataset):
    def __init__(self, paths, fn): self.p, self.fn = list(paths), fn
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return EVAL_TF(self.fn(self.p[i]))

@torch.no_grad()
def get_logits(src, paths, fn):
    m, _ = models[src]
    dl = DataLoader(LoaderDS(paths, fn), batch_size=64, num_workers=2)
    return torch.cat([m(x.to(DEVICE)).float().cpu() for x in dl]).numpy()

# ── Contamination strata (pHash near-duplicates of what the source model saw)
POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def to_u64(s): return np.array([int(x) for x in s], dtype=np.uint64)
def min_hamming(q, ref, chunk=256):
    out = np.empty(len(q), dtype=np.int64)
    for i in range(0, len(q), chunk):
        b = q[i:i+chunk]
        x = np.ascontiguousarray(b[:, None] ^ ref[None, :]).view(np.uint8)
        out[i:i+chunk] = POP8[x.reshape(len(b), len(ref), 8)].sum(-1).min(1)
    return out

# ── Table 14 metrics (+ balanced accuracy / macro-F1 / bootstrap CI)
rng = np.random.default_rng(42)
def ece_score(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e

def table_metrics(probs, y, k):
    n = len(y)
    if n == 0: return {"n": 0}
    pred = probs.argmax(1); correct = (pred == y).astype(np.float32)
    boot = correct[rng.integers(0, n, (2000, n), dtype=np.int32)].mean(1)
    present = np.unique(y)
    onehot = np.eye(k)[y]
    try: auc = roc_auc_score(onehot, probs, multi_class="ovr", average="macro")
    except ValueError: auc = np.nan
    return {"n": n, "acc": correct.mean() * 100,
            "acc_lo": np.percentile(boot, 2.5) * 100, "acc_hi": np.percentile(boot, 97.5) * 100,
            "bal_acc": np.mean([(pred[y == c] == c).mean() for c in present]) * 100,
            "f1_weighted": f1_score(y, pred, average="weighted", zero_division=0) * 100,
            "f1_macro": f1_score(y, pred, labels=present, average="macro", zero_division=0) * 100,
            "auc": auc * 100, "kappa": cohen_kappa_score(y, pred),
            "brier": float(np.mean(np.sum((probs - onehot) ** 2, 1))), "ece": ece_score(probs, y)}

splits = {ds: read_split(ds) for ds in ["D2", "D3", "D4"]}
rows, pred_frames = [], []
for (src, tgt), pub in PUBLISHED.items():
    s_df, t_df = splits[src], splits[tgt]
    shared = [c for c in CLASSES[tgt] if c in CLASSES[src]]
    idx = [CLASSES[src].index(c) for c in shared]
    t = t_df[t_df.label.isin(shared)].reset_index(drop=True).copy()
    q, ref = to_u64(t.phash), to_u64(s_df.phash)
    t["min_ham_seen"] = min_hamming(q, ref[s_df.split.isin(SEEN).values])
    t["min_ham_any"] = min_hamming(q, ref)
    t["stratum"] = np.select([t.min_ham_seen <= THR, t.min_ham_any <= THR],
                             ["exposed", "overlap_unseen"], "clean")
    y_all = t.label.map({c: i for i, c in enumerate(shared)}).values
    T_src = models[src][1]
    for lname, fn in LOADERS.items():
        if lname == "harmonised": fn = make_harmonised(src)
        # published pipeline only needs the test split (reproduction check)
        sub = t[t.split == "test"] if lname == "published" else t
        print(f"[{src}->{tgt}] {lname:10s} predicting {len(sub)} images ...")
        logits = get_logits(src, sub.path, fn)[:, idx] / T_src   # restrict, then temperature
        probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
        y = y_all[sub.index.values]
        pf = sub[["path", "label", "split", "stratum", "min_ham_seen"]].copy()
        pf["pred"] = np.array(shared)[probs.argmax(1)]
        pf["conf"] = probs.max(1)
        pf[["source", "target", "pipeline"]] = src, tgt, lname
        pred_frames.append(pf)
        for scope in (["test"] if lname == "published" else ["test", "full"]):
            in_scope = (sub.split == "test").values if scope == "test" else np.ones(len(sub), bool)
            for sname, smask in [("ALL", np.ones(len(sub), bool)),
                                 ("exposed", (sub.stratum == "exposed").values),
                                 ("not_exposed", (sub.stratum != "exposed").values),
                                 ("clean", (sub.stratum == "clean").values)]:
                m = in_scope & smask
                r = {"source": src, "target": tgt, "pipeline": lname, "scope": scope, "stratum": sname,
                     **table_metrics(probs[m], y[m], len(shared))}
                if "acc" in r: r["delta_acc"] = IN_DOMAIN[src] * 100 - r["acc"]
                rows.append(r)

res = pd.DataFrame(rows)

# ── Reproduction gate: 'published' pipeline must reproduce every Table 14 accuracy
print("\n=== REPRODUCTION GATE (published pipeline, test split, ALL) ===")
g = res[(res.pipeline == "published") & (res.stratum == "ALL")]
gate_ok = True
for _, r in g.iterrows():
    pub = PUBLISHED[(r.source, r.target)]
    ok = abs(r.acc - pub) <= 0.5          # allow ~1-3 images of GPU/library noise
    gate_ok &= ok
    print(f"  {r.source}->{r.target}: {r.acc:6.2f}% vs published {pub:6.2f}%  {'OK' if ok else 'MISMATCH'}")
print("GATE:", "PASS" if gate_ok else "FAIL — send me this output")

pd.set_option("display.width", 250)
cols = ["source", "target", "pipeline", "scope", "stratum", "n", "acc", "acc_lo", "acc_hi",
        "bal_acc", "f1_weighted", "auc", "kappa", "ece"]
print("\n=== TABLE 14 RE-RUN (test split, ALL) ===")
print(res[(res.scope == "test") & (res.stratum == "ALL")][cols].round(2).to_string(index=False))
print("\n=== CONTAMINATION STRATA (full target, training + harmonised pipelines) ===")
print(res[(res.scope == "full")][cols].round(2).to_string(index=False))

res.to_csv(f"{OUT}/table14_rerun_results.csv", index=False)
pd.concat(pred_frames).to_csv(f"{OUT}/table14_rerun_predictions.csv.gz", index=False)
json.dump({"threshold": THR, "seen_splits": sorted(SEEN), "gate_pass": bool(gate_ok),
           "published": {f"{a}->{b}": v for (a, b), v in PUBLISHED.items()}},
          open(f"{OUT}/table14_rerun_meta.json", "w"), indent=2)
print(f"\nSaved to {OUT}/table14_rerun_results.csv, table14_rerun_predictions.csv.gz, table14_rerun_meta.json")
