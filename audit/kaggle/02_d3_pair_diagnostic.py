# ═══════════════════════════════════════════════════════════════════════
# 02 — D3/D4 near-identical pair diagnostic (Supplementary Fig. S1)
#      (run after 00_load_models.py in the same session)
#
# Question: why does the D4 model fail on D3 images that are near-copies of
# its own training images? Selects 300 same-label D3/D4 pairs with pHash
# Hamming distance ≤ 2 (D3 image exposed to the D4 model's train/val/cal data),
# then compares the D4 model on each D3 image vs its D4 twin, with the original
# pipeline and with the raw image intensity-rescaled to 0–255.
#
# Expected (published revision): D4 twins 100.0%, D3 copies 43.3% (92.0% predicted
# meningioma), D3 rescaled 89.7%, D4 rescaled 99.7%. Raw format: D3 = 4-channel
# uint8 PNG (viridis pseudo-colour), D4 = 3-channel JPEG.
# ═══════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

OUT = "/kaggle/working/audit"; os.makedirs(OUT, exist_ok=True)
THR, SEEN = 6, {"train", "val", "cal"}

# ── Pair selection from the split CSVs (reproduces the published 300 pairs)
POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def to_u64(s): return np.array([int(x) for x in s], dtype=np.uint64)
def nearest(q, ref, chunk=256):
    md, arg = np.empty(len(q), dtype=np.int64), np.empty(len(q), dtype=np.int64)
    for i in range(0, len(q), chunk):
        b = q[i:i + chunk]
        dist = POP8[np.ascontiguousarray(b[:, None] ^ ref[None, :]).view(np.uint8).reshape(len(b), len(ref), 8)].sum(-1)
        md[i:i + chunk], arg[i:i + chunk] = dist.min(1), dist.argmin(1)
    return md, arg

d3, d4 = splits["D3"].reset_index(drop=True), splits["D4"].reset_index(drop=True)
q, r4 = to_u64(d3.phash), to_u64(d4.phash)
min_any, arg_any = nearest(q, r4)
min_seen, _ = nearest(q, r4[d4.split.isin(SEEN).values])
pairs = d3.assign(min_hamming_to_D4=min_any, nearest_D4_path=d4.path.values[arg_any],
                  d4_label=d4.label.values[arg_any])
pairs = pairs[(min_seen <= THR) & (pairs.min_hamming_to_D4 <= 2) & (pairs.label == pairs.d4_label)]
pairs = pairs.sample(min(300, len(pairs)), random_state=0)
print(f"{len(pairs)} near-identical D3/D4 pairs (Hamming <= 2, same label, D3 image exposed)")

# ── 1. Raw file format of each side
def raw_stats(path):
    r = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return {"dtype": str(r.dtype), "ch": 1 if r.ndim == 2 else r.shape[2],
            "h": r.shape[0], "w": r.shape[1], "max": float(r.max()),
            "mean_raw": float(r.mean()), "mean_8bit": float(cv2.imread(path).mean())}
print("\n=== RAW FILE FORMAT ===")
for name, col in [("D3", "path"), ("D4", "nearest_D4_path")]:
    s = pd.DataFrame([raw_stats(x) for x in pairs[col]])
    print(f"{name}: dtype={s.dtype.value_counts().to_dict()} channels={s.ch.value_counts().to_dict()} "
          f"size~{int(s.h.median())}x{int(s.w.median())} mean(raw)={s.mean_raw.mean():.1f} "
          f"mean(as cv2 8-bit)={s.mean_8bit.mean():.1f}")

# ── 2. D4 model on D3 image vs its D4 twin, two loaders
SHARED = [CLASSES["D4"].index(c) for c in CLASSES["D3"]]
class LoaderDS(Dataset):
    def __init__(self, paths, fn): self.p, self.fn = list(paths), fn
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return EVAL_TF(self.fn(self.p[i]))

@torch.no_grad()
def pred_d4model(paths, fn):
    m, temp = models["D4"]
    dl = DataLoader(LoaderDS(paths, fn), batch_size=64, num_workers=2)
    pr = torch.cat([torch.softmax(m(x.to(DEVICE)) / temp, 1).cpu() for x in dl]).numpy()
    return np.array(CLASSES["D3"])[pr[:, SHARED].argmax(1)]

def load_rescaled(path):     # percentile-rescale the RAW file (channel mean) to 0–255, then the training pipeline
    r = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if r.ndim == 3: r = r[..., :3].mean(-1)
    lo, hi = np.percentile(r, [0.5, 99.5])
    g = (np.clip((r - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
    img = crop_brain_contour(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return cv2.medianBlur(img, MEDIAN_K)

y = pairs.label.values; rows = []
for img_set, col in [("D3 image", "path"), ("D4 near-twin", "nearest_D4_path")]:
    for loader_name, fn in [("original pipeline", load_and_enhance), ("rescaled raw", load_rescaled)]:
        pr = pred_d4model(pairs[col], fn)
        rows.append({"images": img_set, "loader": loader_name, "acc": (pr == y).mean(),
                     **{f"pred_{c}": (pr == c).mean() for c in CLASSES["D3"]}})
res = pd.DataFrame(rows)
print("\n=== D4 MODEL ON NEAR-IDENTICAL PAIRS ===")
print(res.round(3).to_string(index=False))
res.to_csv(f"{OUT}/D3D4_pair_diagnostic.csv", index=False)
pairs[["path", "nearest_D4_path", "label", "min_hamming_to_D4"]].to_csv(f"{OUT}/D3D4_pair_list.csv", index=False)

# ── 3. Supplementary Fig. S1: four pairs side by side
fig, ax = plt.subplots(4, 4, figsize=(12, 12))
for i, (_, r) in enumerate(pairs.head(4).iterrows()):
    for j, (path, fn, title) in enumerate([(r.path, load_and_enhance, "D3 original"),
                                           (r.nearest_D4_path, load_and_enhance, "D4 twin original"),
                                           (r.path, load_rescaled, "D3 rescaled"),
                                           (r.nearest_D4_path, load_rescaled, "D4 twin rescaled")]):
        ax[i, j].imshow(fn(path)); ax[i, j].set_title(f"{title}\n{r.label}", fontsize=9); ax[i, j].axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/D3D4_pair_diagnostic.png", dpi=110); plt.close()
print(f"\nSaved to {OUT}/")
