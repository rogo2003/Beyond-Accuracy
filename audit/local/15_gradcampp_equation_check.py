"""15 — Check the Grad-CAM++ implementation against Eqs. (8) and (9) on the released D1 model (CPU, no GPU).

Runs locally on a copy of the D1 dataset (sartajbhuvaji/brain-tumor-classification-mri, folders
Training/ and Testing/). Uses the released checkpoint and pHash split in `Output files/`.

Checks
  A. The code in kaggle/09_gradcampp_fix.py and kaggle/03_xai_rerun.py equals Eqs. (8)-(9)
     (Chattopadhay et al., 2018, Eqs. 19-20) written out term by term in float64, on the real
     activations and gradients (first 3 images per class).
  B. The hooks capture A = output of backbone.blocks[-1] and g = dY^c/dA (compared with autograd.grad).
  C. 03 (fixed=True) and 09 (corrected) produce identical maps.
  D. The Kaggle outputs that depend only on Grad-CAM++ (constant-map flags under both formulas,
     first-revision-vs-corrected mask IoU and Spearman) reproduce on the same 200 D1 images.
  A gate first requires the D1 in-domain test accuracy to match the paper (92.94%).

Usage:  python audit/local/15_gradcampp_equation_check.py --data-root "D:/MRI/archive" [--out audit/results]
Runtime: about 5 minutes on a laptop CPU. Requires torch, timm, opencv-python, scipy, pandas.
"""
import argparse, glob, os, time
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True, action="append",
                help="folder containing Training/ and Testing/ of the D1 dataset (may be given more than once)")
ap.add_argument("--out", default="audit/results")
ap.add_argument("--k", type=int, default=50, help="correctly classified test images per class (paper: 50)")
a = ap.parse_args()
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
torch.set_num_threads(os.cpu_count() or 4)

# ── Model and preprocessing: the definitions of kaggle/00_load_models.py (before it reads /kaggle/input)
g = {}
exec(open(os.path.join(REPO, "audit/kaggle/00_load_models.py"), encoding="utf8").read().split("models = {ds: load_calibrated")[0], g)
BrainTumorModel, EVAL_TF, load_and_enhance, CLASSES, IMG_SIZE = (g[k] for k in
    ["BrainTumorModel", "EVAL_TF", "load_and_enhance", "CLASSES", "IMG_SIZE"])
ck = torch.load(os.path.join(REPO, "Output files/D1_final_model_v5_calibrated_pytorch.pth"), map_location="cpu", weights_only=False)
model = BrainTumorModel(4)
model.load_state_dict({k.removeprefix("module."): v for k, v in ck["model_state_dict"].items() if k != "n_averaged"}, strict=True)
model.eval()
for p in model.parameters(): p.requires_grad_(True)

# ── D1 test partition, mapped to the local dataset copy
split = pd.read_csv(os.path.join(REPO, "Output files/D1_phash_group_split.csv"), dtype={"phash": str})
te = split[split.split == "test"].reset_index(drop=True)
local = {}
for root in a.data_root:
    for f in glob.glob(os.path.join(root, "*", "*", "*")):
        parts = os.path.normpath(f).split(os.sep)
        local.setdefault((parts[-3], parts[-2], parts[-1]), f)
missing = [r.filename for r in te.itertuples() if (r.source_split, r.disk_class, r.filename) not in local]
assert not missing, f"{len(missing)} D1 test images not found under --data-root, e.g. {missing[:3]}"
te["local"] = [local[(r.source_split, r.disk_class, r.filename)] for r in te.itertuples()]
cls = CLASSES["D1"]; y = te.label.map({c: i for i, c in enumerate(cls)}).values
X = torch.stack([EVAL_TF(load_and_enhance(p)) for p in te.local])
with torch.no_grad():
    pred = torch.cat([model(X[i:i + 32]).argmax(1) for i in range(0, len(X), 32)]).numpy()
acc = (pred == y).mean()
print(f"[gate] D1 in-domain test accuracy {acc:.4f} (paper 0.9294)")
assert abs(acc - 0.9294) < 0.01, "model, preprocessing or images differ from the paper — stop"

# ── Grad-CAM++ exactly as released
def load_block(path, start, end, ns):
    t = open(os.path.join(REPO, path), encoding="utf8").read()
    exec(t[t.index(start):t.index(end)], ns); return ns
base = dict(torch=torch, F=torch.nn.functional, IMG_SIZE=IMG_SIZE, np=np)
gc09 = load_block("audit/kaggle/09_gradcampp_fix.py", "def alpha_reported", "def integrated_gradients",
                  dict(base, SEED=42, print=lambda *x, **k: None))["GradCAMpp"](model)
gc03 = load_block("audit/kaggle/03_xai_rerun.py", "class GradCAMpp", "def integrated_gradients", dict(base))["GradCAMpp"](model)

def eq8_eq9(A, G):
    """Eqs. (8)-(9) written out location by location (float64)."""
    A, G = A[0].double(), G[0].double(); C, H, W = A.shape
    L = torch.zeros(H, W, dtype=torch.float64)
    for k in range(C):
        S = A[k].sum()                                              # Σ_a Σ_b A^k_ab
        w = 0.0
        for i in range(H):
            for j in range(W):
                gij = G[k, i, j]
                alpha = gij ** 2 / (2 * gij ** 2 + S * gij ** 3 + 1e-7)   # Eq. (8)
                if not torch.isfinite(alpha): alpha = torch.tensor(0.0, dtype=torch.float64)
                w = w + alpha * max(float(gij), 0.0)                # Eq. (9): w^c_k = Σ_ij α ReLU(∂Y^c/∂A)
        L += w * A[k]
    L = torch.relu(L)                                               # Eq. (9): ReLU(Σ_k w^c_k A^k)
    up = torch.nn.functional.interpolate(L[None, None].float(), size=IMG_SIZE, mode="bilinear", align_corners=False).squeeze().numpy()
    return up / up.max() if up.max() > 0 else up

top_mask = lambda m: m >= np.percentile(m, 90)
def iou(p, q): u = (p | q).sum(); return float((p & q).sum() / u) if u else 0.0
def spear(p, q): return float(spearmanr(p.ravel(), q.ravel())[0]) if p.max() > 0 and q.max() > 0 else np.nan

kg = pd.read_csv(os.path.join(REPO, "New results/xai_gcpp_fix/gradcampp_fix_per_image.csv"))
kg = kg[kg.dataset == "D1"].set_index("test_row")
rows, eq_rows = [], []
t0 = time.time()
for ci, c in enumerate(cls):
    for k, j in enumerate(np.where((y == ci) & (pred == ci))[0][:a.k]):
        x = X[j:j + 1]
        (m_rep, _), (m_cor, _) = gc09(x, ci)
        A, G = gc09.acts.detach().clone(), gc09.grads.detach().clone()
        m03 = gc03(x, ci, fixed=True)
        if k < 3:
            acts = {}
            h = model.backbone.blocks[-1].register_forward_hook(lambda m, i, o: acts.setdefault("a", o))
            score = model(x)[0, ci]; h.remove()
            G_auto = torch.autograd.grad(score, acts["a"])[0]
            ref = eq8_eq9(A, G)
            eq_rows.append({"class": c, "test_row": int(j),
                            "max_diff_09_vs_eq": float(np.abs(m_cor - ref).max()),
                            "max_diff_03_vs_eq": float(np.abs(m03 - ref).max()),
                            "hook_grad_vs_autograd": float((G - G_auto).abs().max() / (G_auto.abs().max() + 1e-12)),
                            "hook_act_vs_forward": float((A - acts["a"].detach()).abs().max())})
        r = kg.loc[int(j)] if int(j) in kg.index else None
        rows.append({"class": c, "test_row": int(j), "in_kaggle_selection": r is not None and r["class"] == c,
                     "flat_first_revision": bool(m_rep.max() == 0), "flat_corrected": bool(m_cor.max() == 0),
                     "max_diff_03_vs_09": float(np.abs(m03 - m_cor).max()),
                     "mask_iou_rev_vs_cor": iou(top_mask(m_rep), top_mask(m_cor)), "spearman_rev_vs_cor": spear(m_rep, m_cor),
                     "kaggle_flat_first_revision": None if r is None else bool(r["flat_reported"]),
                     "kaggle_flat_corrected": None if r is None else bool(r["flat_corrected"]),
                     "kaggle_mask_iou_rev_vs_cor": None if r is None else r["gc_mask_iou_reported_vs_corrected"],
                     "kaggle_spearman_rev_vs_cor": None if r is None else r["gc_spearman_reported_vs_corrected"]})
print(f"Grad-CAM++ computed for {len(rows)} images in {(time.time() - t0) / 60:.1f} min")

E, R = pd.DataFrame(eq_rows), pd.DataFrame(rows)
os.makedirs(a.out, exist_ok=True)
E.to_csv(os.path.join(a.out, "gradcampp_equation_check_D1.csv"), index=False)
R.to_csv(os.path.join(a.out, "gradcampp_reproduction_D1.csv"), index=False)
pd.set_option("display.width", 200)
print("\n=== A/B. Code vs Eqs. (8)-(9) term by term; hooks vs autograd ===")
print(E.to_string(index=False, float_format=lambda v: f"{v:.2e}"))
print(f"\n=== C. 03 vs 09: max |difference| over {len(R)} images = {R.max_diff_03_vs_09.max():.2e}")
print("\n=== D. Reproduction of the Kaggle per-image output (D1) ===")
same_flat_rev = int((R.flat_first_revision == R.kaggle_flat_first_revision).sum())
same_flat_cor = int((R.flat_corrected == R.kaggle_flat_corrected).sum())
d_iou = (R.mask_iou_rev_vs_cor - R.kaggle_mask_iou_rev_vs_cor).abs()
print(f"same images as Kaggle: {int(R.in_kaggle_selection.sum())}/{len(R)}")
print(f"constant maps, first-revision formula: {same_flat_rev}/{len(R)} identical (local {int(R.flat_first_revision.sum())})")
print(f"constant maps, corrected formula:      {same_flat_cor}/{len(R)} identical (local {int(R.flat_corrected.sum())})")
print(f"mask IoU first revision vs corrected: mean {R.mask_iou_rev_vs_cor.mean():.4f} (Kaggle {R.kaggle_mask_iou_rev_vs_cor.mean():.4f}); "
      f"per-image |diff| median {d_iou.median():.4f}")
checks = {
    "code equals Eqs. (8)-(9) (max |diff| < 1e-4)": bool((E[["max_diff_09_vs_eq", "max_diff_03_vs_eq"]] < 1e-4).all().all()),
    "hooks equal autograd": bool((E.hook_grad_vs_autograd < 1e-6).all() and (E.hook_act_vs_forward == 0).all()),
    "03 and 09 identical": bool(R.max_diff_03_vs_09.max() == 0),
    "same 200 images as Kaggle": bool(R.in_kaggle_selection.all()),
    "constant-map flags reproduce": same_flat_rev == len(R) and same_flat_cor == len(R),
    "mask IoU reproduces (mean within 0.005)": abs(R.mask_iou_rev_vs_cor.mean() - R.kaggle_mask_iou_rev_vs_cor.mean()) < 0.005,
}
print("\n=== Checks ===")
for k, v in checks.items(): print(f"  {'PASS' if v else 'FAIL'}  {k}")
print("\nALL PASS" if all(checks.values()) else "\nSOME CHECKS FAILED")
