# ═══════════════════════════════════════════════════════════════════════
# 08 — Fig. 1: representative images from D1–D4, replotted from the raw files
#
# Standalone (does not need 00 or a GPU). Kaggle inputs: the four raw datasets
# and the *_phash_group_split.csv files. Images are shown as released (no
# preprocessing); D3 therefore appears in its viridis pseudo-colour encoding
# (Section 3.1.1).
#
# Step 1  PREVIEW = True  → candidates_<DS>.png: 8 numbered candidates per class
#                           from each test partition.
# Step 2  put the chosen numbers in CHOICE and run with PREVIEW = False
#         → Figure_1_Dataset_Samples.pdf (vector text, images at native
#           resolution) and .png (600 dpi), 16.5 cm wide.
# ═══════════════════════════════════════════════════════════════════════
import glob, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = globals().get("OUT", "/kaggle/working")
PREVIEW = globals().get("PREVIEW", True)
# (dataset, class) → candidate number from the preview sheets (default 0)
CHOICE = globals().get("CHOICE", {})

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
LABEL = {"glioma": "Glioma", "meningioma": "Meningioma", "no_tumor": "No tumor", "pituitary": "Pituitary"}
NAMES = {"D1": "D1 (SARTAJ)", "D2": "D2 (BRISC 2025)", "D3": "D3 (Figshare)", "D4": "D4 (Nickparvar)"}
N_CAND = 8


def find_csv(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(f"./{name}")
    assert hits, f"{name} not found"
    return pd.read_csv(hits[0], dtype={"phash": str})


SPLITS = globals().get("SPLITS") or {ds: find_csv(f"{ds}_phash_group_split.csv") for ds in NAMES}


def candidates(ds, cls):
    df = SPLITS[ds]
    df = df[(df.split == "test") & (df.label == cls)].sort_values("path")
    df = df[~df.path.str.contains("-aug-", regex=False)]   # D4 ships augmented copies (Te-aug-*)
    return df.path.tolist()[:N_CAND]


def load_square(path):
    """Image as released (RGB), padded to a square with its own corner colour."""
    img = np.asarray(Image.open(path).convert("RGB"))
    h, w, _ = img.shape; s = max(h, w)
    out = np.empty((s, s, 3), np.uint8); out[:] = img[0, 0]
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return out


def preview(ds):
    present = [c for c in CLASSES if len(candidates(ds, c))]
    fig, axes = plt.subplots(len(present), N_CAND, figsize=(N_CAND * 1.6, len(present) * 1.8))
    for r, c in enumerate(present):
        for k, p in enumerate(candidates(ds, c)):
            ax = axes[r, k]; ax.imshow(load_square(p)); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{k}", fontsize=9)
            if k == 0: ax.set_ylabel(LABEL[c], fontsize=10)
    fig.suptitle(f"{NAMES[ds]} — test-partition candidates (number = CHOICE value)")
    fig.tight_layout(); fig.savefig(f"{OUT}/candidates_{ds}.png", dpi=110); plt.close(fig)
    print("saved", f"{OUT}/candidates_{ds}.png")


def figure():
    W = 6.5                                   # inches = 16.5 cm (full text width)
    fig, axes = plt.subplots(4, 4, figsize=(W, W * 1.02),
                             gridspec_kw=dict(wspace=0.04, hspace=0.06))
    used = []
    for r, ds in enumerate(NAMES):
        for c_i, cls in enumerate(CLASSES):
            ax = axes[r, c_i]; ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            cands = candidates(ds, cls)
            if not cands:                         # D3 has no no-tumor class
                ax.set_facecolor("white")
                ax.text(0.5, 0.5, "No no-tumor\nclass in D3", ha="center", va="center",
                        fontsize=8, color="0.35", transform=ax.transAxes)
                continue
            p = cands[CHOICE.get((ds, cls), 0)]
            used.append(dict(dataset=ds, cls=cls, path=p))
            ax.imshow(load_square(p), interpolation="none")
            if r == 0: ax.set_title(LABEL[cls], fontsize=9, fontweight="bold", pad=4)
            if c_i == 0: ax.set_ylabel(NAMES[ds], fontsize=9, fontweight="bold", labelpad=4)
    fig.savefig(f"{OUT}/Figure_1_Dataset_Samples.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(f"{OUT}/Figure_1_Dataset_Samples.png", bbox_inches="tight", dpi=600)
    pd.DataFrame(used).to_csv(f"{OUT}/Figure_1_selected_images.csv", index=False)
    print("saved Figure_1_Dataset_Samples.pdf/.png and Figure_1_selected_images.csv")


if PREVIEW:
    for ds in NAMES: preview(ds)
else:
    figure()
