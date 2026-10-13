# ═══════════════════════════════════════════════════════════════════════
# 09 — Grad-CAM++ coefficient correction (second revision, Reviewer 4)
#          (run after 00_load_models.py in the same session)
#
#  03_xai_rerun.py computed the Grad-CAM++ coefficient as
#      α_ij = g_ij² / (2·g_ij² + Σ_ab A_ab·g_ab³)
#  whereas Chattopadhay et al. (2018, Eq. 19) define
#      α_ij = g_ij² / (2·g_ij² + (Σ_ab A_ab)·g_ij³)
#  (g = ∂y^c/∂A^k, per channel k). Everything else is unchanged.
#
#  This script
#   1. checks the vectorised corrected formula against Eq. 19 written out location by location;
#   2. GATE: recomputes SHAP, IG and Grad-CAM++ with the formula of 03_xai_rerun.py on the same
#      750 images and must reproduce the per-image IoU behind Table 14 of the revised manuscript
#      (xai_rerun_per_image.csv), i.e. confirms that this implementation produced Table 14 / Fig. 8;
#   3. recomputes Grad-CAM++ with the corrected formula (same models, images, target layer and
#      top-10% masks) and writes drop-in replacements for the local scripts 11 and 12.
#  Deletion–insertion ranks pixels by SHAP only, so its AUCs (Fig. 7, Fig. S3) are carried over.
#
#  Kaggle inputs: as for 00, plus xai_rerun_per_image.csv from the repository root.
# ═══════════════════════════════════════════════════════════════════════
import time, shap
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DATASETS = ["D1", "D2", "D3", "D4"]      # trim this list to run fewer datasets
K        = 50                            # images per class, as in 03
TOP_K    = 0.10                          # top-10% masks
IG_STEPS = 100
SEED     = 42
TOL      = 0.005                         # gate: per-class mean IoU must match within this
OUT      = "/kaggle/working/xai_gcpp_fix"
os.makedirs(OUT, exist_ok=True)

CLASSES["D1"] = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASSES["D2"] = ["glioma", "meningioma", "no_tumor", "pituitary"]
PUBLISHED_IN_DOMAIN = {"D1": 0.9294, "D2": 0.9766, "D3": 0.9532, "D4": 0.9666}

hits = glob.glob("/kaggle/input/**/xai_rerun_per_image.csv", recursive=True)
assert hits, "attach xai_rerun_per_image.csv (repository root) as a Kaggle dataset"
REPORTED = pd.read_csv(hits[0])
REPORTED = REPORTED[REPORTED.dataset.isin(DATASETS)].reset_index(drop=True)

def read_split_csv(ds):
    hits = glob.glob(f"/kaggle/input/**/{ds}_phash_group_split.csv", recursive=True)
    assert hits, f"{ds}_phash_group_split.csv not found"
    return pd.read_csv(hits[0], dtype={"phash": str})

class TempScaled(nn.Module):                 # calibrated model, as used for SHAP in 03
    def __init__(self, m, T): super().__init__(); self.m, self.T = m, T
    def forward(self, x): return self.m(x) / self.T

def alpha_reported(g, a):                    # 03_xai_rerun.py, used for the revised manuscript
    return g.pow(2) / (2 * g.pow(2) + (a * g.pow(3)).sum((2, 3), keepdim=True) + 1e-7)

def alpha_corrected(g, a):                   # Chattopadhay et al. (2018), Eq. 19
    return g.pow(2) / (2 * g.pow(2) + a.sum((2, 3), keepdim=True) * g.pow(3) + 1e-7)

# ── 1. Formula check on random tensors (float64): vectorised vs Eq. 19 location by location
def alpha_eq19_loop(g, a):
    out = torch.empty_like(g)
    for k in range(g.shape[1]):
        s = a[0, k].sum()
        for i in range(g.shape[2]):
            for j in range(g.shape[3]):
                gij = g[0, k, i, j]
                out[0, k, i, j] = gij ** 2 / (2 * gij ** 2 + s * gij ** 3 + 1e-7)
    return out

_r = torch.Generator().manual_seed(SEED)
tg = torch.randn(1, 6, 7, 7, generator=_r, dtype=torch.float64) * 0.1
ta = torch.randn(1, 6, 7, 7, generator=_r, dtype=torch.float64).abs()
ref = alpha_eq19_loop(tg, ta)
FORMULA_OK = torch.allclose(alpha_corrected(tg, ta), ref, rtol=1e-9, atol=1e-12)
print(f"[formula] corrected == Eq. 19: {FORMULA_OK}; "
      f"reported == Eq. 19: {torch.allclose(alpha_reported(tg, ta), ref, rtol=1e-9, atol=1e-12)} "
      f"(max |difference| {float((alpha_reported(tg, ta) - ref).abs().max()):.3g})")
assert FORMULA_OK, "corrected formula does not match Eq. 19 — stop"

class GradCAMpp:
    """One forward/backward pass per image; returns the map under both coefficient formulas."""
    def __init__(self, model):
        self.model, self.layer = model, model.backbone.blocks[-1]   # same target layer as 03
        self.layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        self.layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))
    def __call__(self, x, c):
        self.model.zero_grad(); self.model(x)[0, c].backward()
        with torch.no_grad():
            g, a = self.grads, self.acts
            return self._cam(alpha_reported(g, a), g, a), self._cam(alpha_corrected(g, a), g, a)
    @staticmethod
    def _cam(alpha, g, a):
        n_bad = int((~torch.isfinite(alpha)).sum())                 # reported, expected 0
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        w = (alpha * F.relu(g)).sum((2, 3), keepdim=True)           # ReLU on gradients, as in 03
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=IMG_SIZE, mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        return (cam / cam.max() if cam.max() > 0 else cam), n_bad

def integrated_gradients(model, x, c, steps=IG_STEPS, chunk=34):
    alphas = torch.linspace(0, 1, steps + 1, device=DEVICE).view(-1, 1, 1, 1)
    grads = []
    for i in range(0, steps + 1, chunk):
        xi = (alphas[i:i + chunk] * x).detach().requires_grad_(True)
        model.zero_grad(); model(xi)[:, c].sum().backward()
        grads.append(xi.grad.detach())
    ig = (x * torch.cat(grads).mean(0, keepdim=True)).abs().sum(1).squeeze().cpu().numpy()
    return ig / ig.max() if ig.max() > 0 else ig

def top_mask(m):
    return m >= np.percentile(m, 100 * (1 - TOP_K))

def iou(a, b):
    u = (a | b).sum(); return float((a & b).sum() / u) if u else 0.0

def spearman(a, b):
    return float(spearmanr(a.ravel(), b.ravel())[0]) if a.max() > 0 and b.max() > 0 else np.nan

rows = []
for ds in DATASETS:
    t0 = time.time()
    if ds not in models: models[ds] = load_calibrated(ds)
    raw, T = models[ds]; raw.eval()
    cal = TempScaled(raw, T).eval()
    for p_ in raw.parameters(): p_.requires_grad_(True)
    cls = CLASSES[ds]
    te = read_split_csv(ds); te = te[te.split == "test"].reset_index(drop=True)
    y = te.label.map({c: i for i, c in enumerate(cls)}).values
    X = torch.stack([EVAL_TF(load_and_enhance(p)) for p in te.path])
    with torch.no_grad():
        pred = torch.cat([cal(X[i:i + 64].to(DEVICE)).argmax(1).cpu() for i in range(0, len(X), 64)]).numpy()
    acc = (pred == y).mean()
    print(f"\n[{ds}] in-domain acc {acc:.4f} (published {PUBLISHED_IN_DOMAIN[ds]:.4f})")
    assert abs(acc - PUBLISHED_IN_DOMAIN[ds]) < 0.01, "model/preprocessing mismatch — stop"

    gcam = GradCAMpp(raw)
    rng = np.random.default_rng(SEED)                                # same SHAP background as 03
    bg = X[rng.choice(len(X), min(50, len(X)), replace=False)].to(DEVICE)
    explainer = shap.GradientExplainer(cal, bg)
    fig8 = {}
    for ci, c in enumerate(cls):
        idx = np.where((y == ci) & (pred == ci))[0][:K]               # same images as 03
        xb = X[idx].to(DEVICE)
        sv, ranked = explainer.shap_values(xb, ranked_outputs=1, nsamples=200, rseed=SEED)
        to_np = lambda a: a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
        sv = to_np(sv[0] if isinstance(sv, list) else sv)
        if sv.ndim == 5: sv = sv[..., 0]
        assert (to_np(ranked).ravel() == ci).all(), "ranked output != true class"
        for k, j in enumerate(idx):
            x = X[j:j + 1].to(DEVICE)
            sh = np.abs(sv[k]).mean(0); sh = sh / (sh.max() + 1e-8)
            (g_rep, bad_rep), (g_cor, bad_cor) = gcam(x, ci)
            ig = integrated_gradients(raw, x, ci)
            ms, mr, mc, mi = top_mask(sh), top_mask(g_rep), top_mask(g_cor), top_mask(ig)
            rows.append({"dataset": ds, "class": c, "test_row": int(j), "path": te.path[j],
                         "SI": iou(ms, mi),
                         "SG_reported": iou(ms, mr), "GI_reported": iou(mr, mi), "flat_reported": bool(g_rep.max() == 0),
                         "SG_corrected": iou(ms, mc), "GI_corrected": iou(mc, mi), "flat_corrected": bool(g_cor.max() == 0),
                         "gc_mask_iou_reported_vs_corrected": iou(mr, mc),
                         "gc_spearman_reported_vs_corrected": spearman(g_rep, g_cor),
                         "nonfinite_alpha_reported": bad_rep, "nonfinite_alpha_corrected": bad_cor})
            if k == 0: fig8[c] = (X[j], sh, g_rep, g_cor, ig)
        r = pd.DataFrame([q for q in rows if q["dataset"] == ds and q["class"] == c])
        print(f"  {c:11s} n={len(r):2d} | reported formula SG={r.SG_reported.mean():.3f} GI={r.GI_reported.mean():.3f} "
              f"flat={int(r.flat_reported.sum())} | corrected SG={r.SG_corrected.mean():.3f} GI={r.GI_corrected.mean():.3f} "
              f"flat={int(r.flat_corrected.sum())} | map IoU rep-vs-cor {r.gc_mask_iou_reported_vs_corrected.mean():.3f}")

    # Fig. 8 replacement (same layout as 03), with the corrected Grad-CAM++
    fig, ax = plt.subplots(len(fig8), 4, figsize=(12, 3 * len(fig8)), squeeze=False)
    for r_, (c, (img_t, sh, g_rep, g_cor, ig)) in enumerate(fig8.items()):
        img = (img_t.permute(1, 2, 0).numpy() * np.array(NORM_STD) + np.array(NORM_MEAN)).clip(0, 1)
        for k_, (m_, cm, ttl) in enumerate([(img, None, "Input"), (sh, "bwr", "SHAP"), (g_cor, "jet", "Grad-CAM++"), (ig, "hot", "IG")]):
            ax[r_][k_].imshow(m_, cmap=cm); ax[r_][k_].set_title(f"{c} — {ttl}", fontsize=9); ax[r_][k_].axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/{ds}_xai_examples.png", dpi=200); plt.close()
    # Side-by-side comparison of the two formulas (for the response letter / supplement)
    fig, ax = plt.subplots(len(fig8), 3, figsize=(9, 3 * len(fig8)), squeeze=False)
    for r_, (c, (img_t, sh, g_rep, g_cor, ig)) in enumerate(fig8.items()):
        img = (img_t.permute(1, 2, 0).numpy() * np.array(NORM_STD) + np.array(NORM_MEAN)).clip(0, 1)
        for k_, (m_, cm, ttl) in enumerate([(img, None, "Input"), (g_rep, "jet", "Grad-CAM++ (revision 1)"), (g_cor, "jet", "Grad-CAM++ (Eq. 19)")]):
            ax[r_][k_].imshow(m_, cmap=cm); ax[r_][k_].set_title(f"{c} — {ttl}", fontsize=9); ax[r_][k_].axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/{ds}_gradcampp_formula_comparison.png", dpi=200); plt.close()
    print(f"  [{ds}] done in {(time.time() - t0) / 60:.1f} min")

new = pd.DataFrame(rows)
new.to_csv(f"{OUT}/gradcampp_fix_per_image.csv", index=False)

# ── 2. GATE: the formula of 03 must reproduce the reported per-image IoU on the same images
m = REPORTED.merge(new, on=["dataset", "class", "test_row"], how="outer", indicator=True)
assert (m._merge == "both").all(), "image selection differs from the reported run — send me the output"
gate = m.groupby(["dataset", "class"], sort=False).agg(
    SG_reported=("SG", "mean"), SG_reproduced=("SG_reported", "mean"),
    GI_reported=("GI", "mean"), GI_reproduced=("GI_reported", "mean"),
    SI_reported=("SI_x", "mean"), SI_reproduced=("SI_y", "mean"),
    flat_reported=("gc_fixed_flat", "sum"), flat_reproduced=("flat_reported", "sum")).reset_index()
gate["match"] = ((gate.SG_reported - gate.SG_reproduced).abs() < TOL) & ((gate.GI_reported - gate.GI_reproduced).abs() < TOL) \
              & ((gate.SI_reported - gate.SI_reproduced).abs() < TOL) & (gate.flat_reported == gate.flat_reproduced)
gate.to_csv(f"{OUT}/gradcampp_fix_gate.csv", index=False)
pd.set_option("display.width", 250)
print("\n=== GATE: formula of 03_xai_rerun.py vs reported per-image IoU (per-class means) ===")
print(gate.round(4).to_string(index=False))
print("per-image max |diff|: SG {:.4f}  GI {:.4f}  SI {:.4f}".format(
    (m.SG - m.SG_reported).abs().max(), (m.GI - m.GI_reported).abs().max(), (m.SI_x - m.SI_y).abs().max()))
print("GATE:", "PASS" if gate.match.all() else "FAIL — send me this table")

# ── 3. Drop-in replacement for 11_tables_and_stats.py / 12_figures.py (corrected Grad-CAM++;
#       SI and deletion–insertion columns unchanged from the reported file)
out = REPORTED.copy()
c_ = m.set_index(["dataset", "class", "test_row"])
key = pd.MultiIndex.from_frame(out[["dataset", "class", "test_row"]])
out["SG"] = c_.loc[key, "SG_corrected"].values
out["GI"] = c_.loc[key, "GI_corrected"].values
out["gc_fixed_flat"] = c_.loc[key, "flat_corrected"].values
out.to_csv(f"{OUT}/xai_rerun_per_image.csv", index=False)

print("\n=== EFFECT OF THE CORRECTION (per-class means) ===")
eff = new.groupby(["dataset", "class"], sort=False).agg(
    SG_reported=("SG_reported", "mean"), SG_corrected=("SG_corrected", "mean"),
    GI_reported=("GI_reported", "mean"), GI_corrected=("GI_corrected", "mean"),
    flat_reported=("flat_reported", "sum"), flat_corrected=("flat_corrected", "sum"),
    map_iou=("gc_mask_iou_reported_vs_corrected", "mean"), map_spearman=("gc_spearman_reported_vs_corrected", "median"),
    nonfinite=("nonfinite_alpha_corrected", "sum")).reset_index()
eff.to_csv(f"{OUT}/gradcampp_fix_effect.csv", index=False)
print(eff.round(3).to_string(index=False))
print(f"\nSaved to {OUT}/  — download the whole folder")
