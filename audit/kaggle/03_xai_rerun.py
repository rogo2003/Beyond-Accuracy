# ═══════════════════════════════════════════════════════════════════════
# 03 — Re-run the XAI evaluation (Table 13, Fig. 7, Fig. 8) with fixes
#          (run after 00_load_models.py in the same session)
#
#  Fixes vs. the original notebook:
#   1. Grad-CAM++ channel weights use ReLU(∂y/∂A) (Chattopadhay et al., 2018);
#      the original omitted the ReLU, which can make the whole map negative → flat.
#   2. IoU is averaged over K correctly classified test images per class
#      (the original used ONE image per class, although Algorithm 6 says "each test image").
#   3. Deletion–insertion ranks each image by ITS OWN SHAP map
#      (the original reused one class-representative SHAP map for all 50 images).
#  A random-order deletion–insertion baseline is added for reference.
# ═══════════════════════════════════════════════════════════════════════
import json, time, shap
import torch.nn.functional as F
import matplotlib.pyplot as plt

DATASETS = ["D1", "D2", "D3", "D4"]      # trim this list to run fewer datasets
K        = 50                            # images per class (paper: K = 50)
TOP_K    = 0.10                          # top-10% masks
DI_STEPS = 100
SEED     = 42
OUT      = "/kaggle/working/xai_rerun"
os.makedirs(OUT, exist_ok=True)

CLASSES["D1"] = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASSES["D2"] = ["glioma", "meningioma", "no_tumor", "pituitary"]
PUBLISHED_IN_DOMAIN = {"D1": 0.9294, "D2": 0.9766, "D3": 0.9532, "D4": 0.9666}
# Published per-class Grad-CAM++ vs IG IoU (first test image of each class) — reproduction check
PUB_GI = {
 "D1": {"glioma": 0.0729, "meningioma": 0.0154, "no_tumor": 0.0577, "pituitary": 0.0411},
 "D2": {"glioma": 0.0701, "meningioma": 0.0193, "no_tumor": 0.0852, "pituitary": 0.1289},
 "D3": {"glioma": 0.0894, "meningioma": 0.0711, "pituitary": 0.1000},
 "D4": {"glioma": 0.0336, "meningioma": 0.1406, "no_tumor": 0.1000, "pituitary": 0.1000}}

def read_split_csv(ds):
    hits = glob.glob(f"/kaggle/input/**/{ds}_phash_group_split.csv", recursive=True)
    assert hits, f"{ds}_phash_group_split.csv not found"
    return pd.read_csv(hits[0], dtype={"phash": str})

class TempScaled(nn.Module):                 # calibrated model, as used for SHAP / deletion–insertion
    def __init__(self, m, T): super().__init__(); self.m, self.T = m, T
    def forward(self, x): return self.m(x) / self.T

class GradCAMpp:
    def __init__(self, model):
        self.model, self.layer = model, model.backbone.blocks[-1]   # same target as the notebook
        self.layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        self.layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))
    def __call__(self, x, c, fixed=True):
        self.model.zero_grad(); self.model(x)[0, c].backward()
        with torch.no_grad():
            g, a = self.grads, self.acts
            alpha = g.pow(2) / (2 * g.pow(2) + (a * g.pow(3)).sum((2, 3), keepdim=True) + 1e-7)
            w = (alpha * (F.relu(g) if fixed else g)).sum((2, 3), keepdim=True)
            cam = F.relu((w * a).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=IMG_SIZE, mode="bilinear", align_corners=False).squeeze().cpu().numpy()
            return cam / cam.max() if cam.max() > 0 else cam

def integrated_gradients(model, x, c, steps=DI_STEPS, chunk=34):
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

GREY = torch.tensor([0.0655, 0.1964, 0.4178], device=DEVICE).view(1, 3, 1, 1)
@torch.no_grad()
def deletion_insertion(cal_model, x, order, c, steps=DI_STEPS, bs=51):
    """Same protocol as the notebook (grey deletion, blurred-image insertion, softmax of calibrated model)."""
    H, W = x.shape[-2:]; n = H * W; step = n // steps
    img_np = x.squeeze().permute(1, 2, 0).cpu().numpy()
    blur = torch.from_numpy(cv2.GaussianBlur(img_np, (51, 51), 10)).permute(2, 0, 1)[None].float().to(DEVICE)
    rank = torch.empty(n, dtype=torch.long, device=DEVICE)
    rank[torch.as_tensor(order.copy(), device=DEVICE)] = torch.arange(n, device=DEVICE)
    d_out, i_out = [], []
    for s0 in range(0, steps + 1, bs):
        ks = torch.arange(s0, min(s0 + bs, steps + 1), device=DEVICE)
        m = (rank.view(1, 1, H, W) < (ks * step).clamp(max=n).view(-1, 1, 1, 1)).float()   # (b,1,H,W)
        d_img = x * (1 - m) + GREY * m
        i_img = blur * (1 - m) + x * m
        d_out.append(F.softmax(cal_model(d_img), 1)[:, c]); i_out.append(F.softmax(cal_model(i_img), 1)[:, c])
    d, i = torch.cat(d_out).cpu().numpy(), torch.cat(i_out).cpu().numpy()
    auc = lambda y: float(np.trapezoid(y, dx=1.0 / len(y)) if hasattr(np, "trapezoid") else np.trapz(y, dx=1.0 / len(y)))
    return d, i, auc(d), auc(i)

def boot_ci(a, B=2000, seed=SEED):
    a = np.asarray(a, float); r = np.random.default_rng(seed)
    m = a[r.integers(0, len(a), (B, len(a)))].mean(1); return np.percentile(m, [2.5, 97.5])

rows, gate_rows, summary = [], [], []
for ds in DATASETS:
    t0 = time.time()
    if ds not in models: models[ds] = load_calibrated(ds)
    raw, T = models[ds]; raw.eval()
    cal = TempScaled(raw, T).eval()
    for p_ in raw.parameters(): p_.requires_grad_(True)
    cls = CLASSES[ds]
    te = read_split_csv(ds); te = te[te.split == "test"].reset_index(drop=True)
    y = te.label.map({c: i for i, c in enumerate(cls)}).values
    X = torch.stack([EVAL_TF(load_and_enhance(p)) for p in te.path])          # CPU tensor (N,3,224,224)
    with torch.no_grad():
        pred = torch.cat([cal(X[i:i + 64].to(DEVICE)).argmax(1).cpu() for i in range(0, len(X), 64)]).numpy()
    acc = (pred == y).mean()
    print(f"\n[{ds}] in-domain acc {acc:.4f} (published {PUBLISHED_IN_DOMAIN[ds]:.4f})")
    assert abs(acc - PUBLISHED_IN_DOMAIN[ds]) < 0.01, "model/preprocessing mismatch — stop"

    gcam = GradCAMpp(raw)
    # ── Reproduction check: original Grad-CAM++ vs IG on the first test image of each class
    for ci, c in enumerate(cls):
        j = int(np.where(y == ci)[0][0]); x = X[j:j + 1].to(DEVICE)
        gi = iou(top_mask(gcam(x, ci, fixed=False)), top_mask(integrated_gradients(raw, x, ci)))
        gate_rows.append({"dataset": ds, "class": c, "published_GI": PUB_GI[ds][c], "reproduced_GI": round(gi, 4),
                          "match": abs(gi - PUB_GI[ds][c]) < 0.01})
    print(pd.DataFrame([g for g in gate_rows if g["dataset"] == ds]).to_string(index=False))

    rng = np.random.default_rng(SEED)
    bg = X[rng.choice(len(X), min(50, len(X)), replace=False)].to(DEVICE)
    explainer = shap.GradientExplainer(cal, bg)
    fig8 = {}
    for ci, c in enumerate(cls):
        idx = np.where((y == ci) & (pred == ci))[0][:K]
        xb = X[idx].to(DEVICE)
        sv, ranked = explainer.shap_values(xb, ranked_outputs=1, nsamples=200, rseed=SEED)
        to_np = lambda a: a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
        sv = to_np(sv[0] if isinstance(sv, list) else sv)                        # (n,3,H,W) or (n,3,H,W,1)
        if sv.ndim == 5: sv = sv[..., 0]
        assert (to_np(ranked).ravel() == ci).all(), "ranked output != true class"
        d_curves, i_curves, dr_curves, ir_curves = [], [], [], []
        for k, j in enumerate(idx):
            x = X[j:j + 1].to(DEVICE)
            sh = np.abs(sv[k]).mean(0); sh = sh / (sh.max() + 1e-8)
            gc_f, gc_o = gcam(x, ci, True), gcam(x, ci, False)
            ig = integrated_gradients(raw, x, ci)
            ms, mg, mo, mi = top_mask(sh), top_mask(gc_f), top_mask(gc_o), top_mask(ig)
            d, i, dA, iA = deletion_insertion(cal, x, np.argsort(sh.ravel())[::-1], ci)
            dr, ir, drA, irA = deletion_insertion(cal, x, rng.permutation(sh.size), ci)
            d_curves.append(d); i_curves.append(i); dr_curves.append(dr); ir_curves.append(ir)
            rows.append({"dataset": ds, "class": c, "test_row": int(j), "path": te.path[j],
                         "gc_fixed_flat": bool(gc_f.max() == 0), "gc_orig_flat": bool(gc_o.max() == 0),
                         "SG": iou(ms, mg), "SI": iou(ms, mi), "GI": iou(mg, mi),
                         "SG_orig": iou(ms, mo), "GI_orig": iou(mo, mi),
                         "del_auc": dA, "ins_auc": iA, "del_auc_random": drA, "ins_auc_random": irA})
            if k == 0: fig8[c] = (X[j], sh, gc_f, ig)
        r = pd.DataFrame([q for q in rows if q["dataset"] == ds and q["class"] == c])
        valid = r[~r.gc_fixed_flat]
        s = {"dataset": ds, "class": c, "n": len(r), "n_flat_fixed": int(r.gc_fixed_flat.sum()),
             "n_flat_orig": int(r.gc_orig_flat.sum())}
        for col in ["SG", "SI", "GI"]:
            v = (valid if col != "SI" else r)[col]
            s[f"{col}_mean"], s[f"{col}_sd"] = v.mean(), v.std(ddof=1)
        s["mean_IoU"] = np.mean([s["SG_mean"], s["SI_mean"], s["GI_mean"]])
        for col in ["del_auc", "ins_auc", "del_auc_random", "ins_auc_random"]:
            lo, hi = boot_ci(r[col]); s[col], s[col + "_lo"], s[col + "_hi"] = r[col].mean(), lo, hi
        s["_curves"] = (np.mean(d_curves, 0), np.mean(i_curves, 0), np.mean(dr_curves, 0), np.mean(ir_curves, 0))
        summary.append(s)
        print(f"  {c:11s} n={s['n']:2d} flat(orig/fixed)={s['n_flat_orig']}/{s['n_flat_fixed']} "
              f"IoU SG={s['SG_mean']:.3f} SI={s['SI_mean']:.3f} GI={s['GI_mean']:.3f} | "
              f"del {s['del_auc']:.3f} (rand {s['del_auc_random']:.3f}) ins {s['ins_auc']:.3f} (rand {s['ins_auc_random']:.3f})")

    # Fig. 7 replacement (per dataset): mean curves, SHAP order (solid) vs random order (dashed)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); xs = np.linspace(0, 1, DI_STEPS + 1)
    for s in [s for s in summary if s["dataset"] == ds]:
        d, i, dr, ir = s["_curves"]
        l, = ax[0].plot(xs, d, label=s["class"]); ax[0].plot(xs, dr, "--", color=l.get_color(), alpha=.6)
        ax[1].plot(xs, i, color=l.get_color(), label=s["class"]); ax[1].plot(xs, ir, "--", color=l.get_color(), alpha=.6)
    ax[0].set_title(f"{ds} deletion (solid: SHAP order, dashed: random)"); ax[1].set_title(f"{ds} insertion")
    for a_ in ax: a_.set_xlabel("Fraction of pixels"); a_.set_ylabel("Calibrated confidence"); a_.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/{ds}_deletion_insertion.png", dpi=200); plt.close()
    # Fig. 8 replacement (per dataset): first correctly classified image per class
    fig, ax = plt.subplots(len(fig8), 4, figsize=(12, 3 * len(fig8)), squeeze=False)
    for r_, (c, (img_t, sh, gc_f, ig)) in enumerate(fig8.items()):
        img = (img_t.permute(1, 2, 0).numpy() * np.array(NORM_STD) + np.array(NORM_MEAN)).clip(0, 1)
        for k_, (m_, cm, ttl) in enumerate([(img, None, "Input"), (sh, "bwr", "SHAP"), (gc_f, "jet", "Grad-CAM++ (fixed)"), (ig, "hot", "IG")]):
            ax[r_][k_].imshow(m_, cmap=cm); ax[r_][k_].set_title(f"{c} — {ttl}", fontsize=9); ax[r_][k_].axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/{ds}_xai_examples.png", dpi=200); plt.close()
    print(f"  [{ds}] done in {(time.time() - t0) / 60:.1f} min")

pd.DataFrame(rows).to_csv(f"{OUT}/xai_rerun_per_image.csv", index=False)
summ = pd.DataFrame([{k: v for k, v in s.items() if k != "_curves"} for s in summary])
summ.to_csv(f"{OUT}/xai_rerun_summary.csv", index=False)
gate = pd.DataFrame(gate_rows); gate.to_csv(f"{OUT}/xai_rerun_gate.csv", index=False)
print("\n=== REPRODUCTION CHECK (original Grad-CAM++ vs IG, first test image per class) ===")
print(gate.to_string(index=False)); print("GATE:", "PASS" if gate.match.all() else "PARTIAL — send me the table")
print("\n=== SUMMARY ===")
pd.set_option("display.width", 250)
print(summ[["dataset", "class", "n", "n_flat_orig", "n_flat_fixed", "SG_mean", "SI_mean", "GI_mean", "mean_IoU",
            "del_auc", "del_auc_random", "ins_auc", "ins_auc_random"]].round(3).to_string(index=False))
print(f"\nSaved to {OUT}/")
