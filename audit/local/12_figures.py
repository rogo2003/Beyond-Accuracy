"""12 — Figures 6, 7, 8, 9 and Supplementary Fig. S3 of the revised manuscript.

Runs locally (no GPU). Run 11_tables_and_stats.py first (it writes xai_tests.csv and table14b.csv).
Inputs:
  --results : table14_rerun_predictions.csv.gz, xai_rerun_summary.csv,
              D1–D4_xai_examples.png, D1–D4_deletion_insertion.png   (from the Kaggle cells)
  --tables  : table14b.csv, xai_tests.csv                             (from 11_tables_and_stats.py)
Fig. 6 also needs recal_reliability_diagrams.png, recal_temperatures.json (kaggle/05) and temperature_sensitivity.csv (kaggle/04).
Supplementary Fig. S2 is written by kaggle/02_d3_pair_diagnostic.py and Fig. S1 by kaggle/04.

Usage:  python audit/local/12_figures.py [--results .] [--tables audit/results] [--out audit/results/figures]
"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # validated categorical order
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})

def _font(size):
    for f in ["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(f): return ImageFont.truetype(f, size)
    return ImageFont.load_default()

def _clean(ax):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)

def fig7(results, tables, out):
    s = pd.read_csv(os.path.join(results, "xai_rerun_summary.csv"))
    t = pd.read_csv(os.path.join(tables, "xai_tests.csv"))
    s = s.merge(t[["dataset", "class", "del_p_holm", "ins_p_holm"]], on=["dataset", "class"])
    s["lab"] = s.dataset + " " + s["class"].str.replace("_", " ")
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    y = np.arange(len(s))[::-1]
    for ax, m, title, sig in [(axes[0], "del_auc", "(a) Deletion AUC (lower = more faithful)", "del_p_holm"),
                              (axes[1], "ins_auc", "(b) Insertion AUC (higher = more faithful)", "ins_p_holm")]:
        for off, pref, col, name in [(0.14, m, SERIES[0], "SHAP ordering"), (-0.14, m + "_random", SERIES[1], "Random ordering")]:
            ax.errorbar(s[pref], y + off, xerr=[s[pref] - s[pref + "_lo"], s[pref + "_hi"] - s[pref]], fmt="o",
                        color=col, ms=4.5, mec="white", mew=.8, elinewidth=1.2, capsize=0, label=name, zorder=3)
        for yi, p in zip(y, s[sig]):
            if p < 0.05: ax.text(1.02, yi, "*", transform=ax.get_yaxis_transform(), va="center", fontsize=11, color=INK)
        ax.set_title(title, loc="left", fontsize=10, color=INK); ax.set_xlim(0, 1)
        ax.set_xlabel("Area under curve (mean, 95% bootstrap CI)")
        ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); _clean(ax)
        for b in [3.5, 7.5, 10.5]: ax.axhline(len(s) - 1 - b, color=GRID, lw=.8)
    axes[0].set_yticks(y, s.lab); axes[0].legend(frameon=False, loc="lower right")
    fig.text(0.99, 0.005, "* SHAP ordering significantly better than random (one-sided Wilcoxon, Holm-corrected p < 0.05); "
             "K = 50 correctly classified test images per class", ha="right", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(out, "Figure_7_Deletion_Insertion.png"), dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(os.path.join(out, "Figure_7_Deletion_Insertion.pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)

def _stack(paths, labels, out_png, grid_cols, width, dpi):
    font = _font(40 if grid_cols == 2 else 44); lab_h, pad = 70, 30
    panels = []
    for p, l in zip(paths, labels):
        im = Image.open(p).convert("RGB"); im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
        c = Image.new("RGB", (width, im.height + lab_h), "white"); c.paste(im, (0, lab_h))
        ImageDraw.Draw(c).text((10, 12), l, fill=(11, 11, 11), font=font); panels.append(c)
    rows = [panels[i:i + grid_cols] for i in range(0, len(panels), grid_cols)]
    rh = [max(p.height for p in r) for r in rows]
    canvas = Image.new("RGB", (grid_cols * width + (grid_cols + 1) * pad, sum(rh) + (len(rows) + 1) * pad), "white")
    for ri, r in enumerate(rows):
        for ci, p in enumerate(r):
            canvas.paste(p, (pad + ci * (width + pad), pad + sum(rh[:ri]) + ri * pad))
    canvas.save(out_png, dpi=(dpi, dpi))

def fig8(results, out):
    names = ["(a) D1 — SARTAJ", "(b) D2 — BRISC-2025", "(c) D3 — Figshare (Kaggle viridis release)", "(d) D4 — Nickparvar"]
    _stack([os.path.join(results, f"D{i}_xai_examples.png") for i in range(1, 5)], names,
           os.path.join(out, "Figure_8_XAI_Examples.png"), grid_cols=2, width=1500, dpi=300)

def figS2(results, out):
    _stack([os.path.join(results, f"D{i}_deletion_insertion.png") for i in range(1, 5)],
           [f"({c}) D{i}" for i, c in zip(range(1, 5), "abcd")],
           os.path.join(out, "Supplementary_Fig_S3_Deletion_Insertion_Curves.png"), grid_cols=1, width=2400, dpi=200)

def fig6(results, out):
    """Fig. 6: (a) reliability diagrams (from 05_recalibrate.py) + (b, c) NLL / ECE vs temperature (from 04)."""
    import json
    rel = os.path.join(results, "recal_reliability_diagrams.png")
    tcsv = os.path.join(results, "temperature_sensitivity.csv")
    tjson = os.path.join(results, "recal_temperatures.json")
    if not all(os.path.exists(p) for p in (rel, tcsv, tjson)):
        print("  [fig6] skipped: needs recal_reliability_diagrams.png, temperature_sensitivity.csv, recal_temperatures.json")
        return
    temp = pd.read_csv(tcsv); T = json.load(open(tjson))
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.6))
    for ds, col in zip(["D1", "D2", "D3", "D4"], SERIES):
        for k, metric in enumerate(["nll", "ece"]):
            for part, ls in [("cal", "-"), ("test", "--")]:
                r = temp[(temp.dataset == ds) & (temp.split == part)]
                ax[k].plot(r["T"], r[metric], ls, color=col, lw=1.8, label=f"{ds} {'calibration' if part == 'cal' else 'test'}" if k == 0 else None)
            ax[k].axvline(T["converged_T"][ds], color=col, lw=1.0, alpha=0.8)
            ax[k].axvline(T["stored_T"][ds], color=col, lw=1.0, ls=":", alpha=0.9)
    ax[0].set_ylabel("Negative log-likelihood"); ax[1].set_ylabel("ECE (15 bins)")
    for a_ in ax:
        a_.set_xlabel("Temperature T"); a_.yaxis.grid(True, color=GRID, lw=0.8); a_.set_axisbelow(True); _clean(a_)
    ax[0].legend(frameon=False, ncol=2, fontsize=8)
    ax[0].set_title("(b) NLL vs T (solid: calibration, dashed: test)", loc="left", fontsize=11, color=INK)
    ax[1].set_title("(c) ECE vs T (vertical lines: converged T, solid; original T*, dotted)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    tmp = os.path.join(out, "_fig6_temperature_panels.png")
    fig.savefig(tmp, dpi=300, facecolor="white", bbox_inches="tight"); plt.close(fig)
    top = Image.open(rel).convert("RGB"); bot = Image.open(tmp).convert("RGB")
    W = top.width; bot = bot.resize((W, int(bot.height * W / bot.width)), Image.LANCZOS)
    lab_h, pad = 110, 40; font = _font(64)
    canvas = Image.new("RGB", (W, lab_h + top.height + pad + bot.height), "white")
    ImageDraw.Draw(canvas).text((20, 20), "(a) Reliability diagrams before (top) and after (bottom) temperature scaling", fill=(11, 11, 11), font=font)
    canvas.paste(top, (0, lab_h)); canvas.paste(bot, (0, lab_h + top.height + pad))
    canvas.save(os.path.join(out, "Figure_6_Calibration.png"), dpi=(300, 300))
    os.remove(tmp)

def fig9(results, tables, out):
    t = pd.read_csv(os.path.join(tables, "table14b.csv"))
    p = pd.read_csv(os.path.join(results, "table14_rerun_predictions.csv.gz"))
    p = p[p.pipeline == "harmonised"].copy(); p["ok"] = p.label == p.pred
    bins, labs = [-1, 0, 3, 6, 9, 12, 16, 64], ["0", "1–3", "4–6", "7–9", "10–12", "13–16", ">16"]
    p["bin"] = pd.cut(p.min_ham_seen, bins, labels=labs)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    x, w = np.arange(len(t)), 0.26
    for k, (col, name) in enumerate([("acc_original", "Published (original evaluation pipeline)"),
                                     ("acc_training", "Training pipeline"), ("acc_harmonised", "Harmonised pipeline")]):
        a.bar(x + (k - 1) * w, t[col], w - 0.03, color=SERIES[k], label=name, zorder=3)
        for xi, v in zip(x + (k - 1) * w, t[col]):
            a.text(xi, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=6.8, color=INK2, rotation=90)
    a.set_xticks(x, t.transfer); a.set_ylim(0, 112); a.set_yticks(range(0, 101, 20))
    a.set_ylabel("Accuracy on target test split (%)"); a.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    a.set_axisbelow(True); _clean(a)
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    a.set_title("(a) Zero-shot transfer accuracy under three input pipelines", loc="left", fontsize=10, color=INK)
    for i, pr in enumerate(t.transfer):
        s, tg = pr.split("→")
        d = p[(p.source == s) & (p.target == tg)].groupby("bin", observed=True).ok.agg(["mean", "size"])
        d = d[d["size"] >= 30]
        b.plot([labs.index(z) for z in d.index], d["mean"] * 100, "-o", color=SERIES[i], lw=2, ms=5,
               mec="white", mew=1, label=pr, zorder=3)
    b.set_xticks(range(len(labs)), labs)
    b.set_xlabel("pHash Hamming distance to nearest source image used in training (bits)")
    b.set_ylabel("Accuracy, full target dataset (%)"); b.set_ylim(84, 101.2)
    b.axvspan(-0.4, 2.5, color="#f1f0ec", zorder=0)
    b.text(1.05, 100.6, "near-duplicate (δ ≤ 6)", ha="center", fontsize=7.5, color=INK2)
    b.yaxis.grid(True, color=GRID, lw=0.8); b.set_axisbelow(True); _clean(b)
    b.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)
    b.set_title("(b) Harmonised accuracy vs. similarity to source training data", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "Figure_9_Cross_Dataset_Audit.png"), dpi=600, facecolor="white", bbox_inches="tight")
    fig.savefig(os.path.join(out, "Figure_9_Cross_Dataset_Audit.pdf"), facecolor="white", bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=".")
    ap.add_argument("--tables", default="audit/results")
    ap.add_argument("--out", default="audit/results/figures")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    fig6(a.results, a.out); fig7(a.results, a.tables, a.out); fig8(a.results, a.out); figS2(a.results, a.out); fig9(a.results, a.tables, a.out)
    print("Figures written to", a.out, ":", sorted(os.listdir(a.out)))

if __name__ == "__main__":
    main()
