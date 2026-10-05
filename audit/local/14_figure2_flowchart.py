"""Figure 2 (framework flowchart) rebuilt at printed size.

Times New Roman; 9 pt box text, 10 pt panel headings, 8 pt notes.
The interpretation notes are in the caption, not the figure.
Width = 482 pt (17 cm). One data unit = 1 pt, y grows downward.
Writes Figure_2_Framework.svg (text kept as text), .pdf and .png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({
    "font.family": "Times New Roman",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
W, H = 482.0, 590.0
FS, FS_HEAD, FS_NOTE = 9, 10, 8
LH = 10.6                      # line height for 9 pt
PAD = 4.5

BLUE, GREEN, PURPLE = "#24617d", "#41735b", "#66588a"
TXT, NOTE, ARR = "#172c39", "#52616b", "#52616b"
FILL_PART, FILL_DPA = "#f1f6f8", "#f5f9f6"

fig = plt.figure(figsize=(W / 72, H / 72))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")

texts_in_boxes = []            # (text artist, box x0, x1) for the overflow check


def h_of(n):
    return n * LH + 2 * PAD


def box(x0, y0, w, lines, color, fill="#ffffff", lw=0.8):
    h = h_of(len(lines))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=2.5",
                                fc=fill, ec=color, lw=lw, zorder=2))
    t = ax.text(x0 + w / 2, y0 + h / 2, "\n".join(lines), ha="center", va="center",
                fontsize=FS, color=TXT, linespacing=1.18, zorder=3)
    texts_in_boxes.append((t, x0, x0 + w))
    return dict(x0=x0, x1=x0 + w, y0=y0, y1=y0 + h, cx=x0 + w / 2, cy=y0 + h / 2)


def path(pts, color=ARR, dashed=False, head=True, lw=0.75):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys, color=color, lw=lw,
            ls=(0, (3.5, 2)) if dashed else "-", zorder=1, solid_capstyle="butt")
    if head:
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="-|>,head_length=3.6,head_width=1.8",
                                    color=color, lw=0, shrinkA=0, shrinkB=0, mutation_scale=1), zorder=4)


def heading(x, y, s):
    ax.text(x, y, s, fontsize=FS_HEAD, fontweight="bold", color=BLUE, va="center", ha="left")


def note(x, y, s, ha="left", color=NOTE, style="italic", fs=FS_NOTE):
    return ax.text(x, y, s, fontsize=fs, color=color, va="center", ha=ha, style=style)


# ---------------------------------------------------------------- Panel A
heading(4, 9, "A   DATA PREPARATION AND SEPARATE PARTITIONS")
a1 = box(4, 20, 222, ["D1–D4 public MRI benchmarks",
                      "Merge provider partitions within each dataset"], BLUE)
a2 = box(252, 20, 226, ["64-bit DCT pHash → distance ≤ 6 → union-find groups",
                        "Class-stratified group partitioning (seed 42)"], BLUE)
path([(a1["x1"], a1["cy"]), (a2["x0"], a2["cy"])], BLUE)
note(4, a1["y1"] + 8, "Nominal proportions refer to groups; patient-level independence is unverified.")

pw, pg, py = 110, 12, 76
parts = {}
for i, (name, pct) in enumerate([("TRAIN", "72%"), ("VALIDATION", "10%"),
                                 ("CALIBRATION", "8%"), ("TEST", "10%")]):
    parts[name] = box(4 + i * (pw + pg), py, pw, [name, f"Nominal {pct}"], BLUE, FILL_PART)
bus_y = 67
ax.plot([a2["cx"], a2["cx"]], [a2["y1"], bus_y], color=ARR, lw=0.75)
ax.plot([parts["TRAIN"]["cx"], parts["TEST"]["cx"]], [bus_y, bus_y], color=ARR, lw=0.75)
for p in parts.values():
    path([(p["cx"], bus_y), (p["cx"], p["y0"])])

# ---------------------------------------------------------------- Panel B
heading(84, 128, "B   MODEL TRAINING AND IMPLEMENTED DPA ARCHITECTURE")
LX, LW = 4, 134            # left column: training procedure
CX, CW = 152, 166          # centre column: architecture
RX, RW = 330, 128          # right column: monitoring + calibration
LANE_CAL, LANE_TEST = 467, 476
top = 141

l1 = box(LX, top, LW, ["Crop brain → bilinear resize", "→ 5 × 5 median filter → normalize",
                       "Training augmentation only"], BLUE)
l2 = box(LX, l1["y1"] + 13, LW, ["Warm-up: 15 epochs", "160 × 160; Adam 10⁻⁴",
                                 "Frozen backbone parameters"], BLUE)
l3 = box(LX, l2["y1"] + 13, LW, ["Fine-tuning: up to 100 epochs", "AdamW + batch-wise OneCycleLR",
                                 "224 × 224 from epoch 31", "SWA from epoch 76 + SWALR"], BLUE)
l4 = box(LX, l3["y1"] + 13, LW, ["Final SWA model", "Backbone + DPA + classifier",
                                 "Recompute BN statistics"], BLUE)
tx = 70                                   # TRAIN drop (inside the TRAIN box)
path([(tx, parts["TRAIN"]["y1"]), (tx, l1["y0"])], BLUE)
for a, b in [(l1, l2), (l2, l3), (l3, l4)]:
    path([(tx, a["y1"]), (tx, b["y0"])], BLUE)

c1 = box(CX, top + 5, CW, ["EfficientNetV2-B0", "ImageNet-pretrained feature extractor"], GREEN)
path([(l1["x1"], c1["cy"]), (c1["x0"], c1["cy"])], BLUE)
c2 = box(CX + 33, c1["y1"] + 14, CW - 66, ["Feature tensor F"], GREEN)
path([(c1["cx"], c1["y1"]), (c2["cx"], c2["y0"])], GREEN)

g0 = c2["y1"] + 9                          # DPA group
ix0, ix1 = CX + 14, CX + CW - 14
gap_w = (ix1 - ix0 - 12) / 2
gy = g0 + 11
gap = box(ix0, gy, gap_w, ["GAP → a"], GREEN)
gmp = box(ix1 - gap_w, gy, gap_w, ["GMP → m"], GREEN)
sls = box(ix0, gap["y1"] + 12, ix1 - ix0, ["Shared linear scores", "FP32 softmax over two pools"], GREEN)
vb = box(ix0, sls["y1"] + 15, ix1 - ix0, ["v = αₐ a + αₘ m"], GREEN)
g1 = vb["y1"] + 15
ax.add_patch(Rectangle((CX, g0), CW, g1 - g0, fc=FILL_DPA, ec=GREEN, lw=0.75,
                       ls=(0, (3.5, 2)), zorder=0.5))
ax.text(CX + CW - 5, g1 - 6.5, "Dual-Pooling Attention", fontsize=FS_NOTE, color=GREEN,
        style="italic", va="center", ha="right")
fork_y = g0 + 5
ax.plot([c2["cx"], c2["cx"]], [c2["y1"], fork_y], color=GREEN, lw=0.75)
ax.plot([gap["cx"], gmp["cx"]], [fork_y, fork_y], color=GREEN, lw=0.75)
path([(gap["cx"], fork_y), (gap["cx"], gap["y0"])], GREEN)
path([(gmp["cx"], fork_y), (gmp["cx"], gmp["y0"])], GREEN)
path([(gap["cx"], gap["y1"]), (gap["cx"], sls["y0"])], GREEN)
path([(gmp["cx"], gmp["y1"]), (gmp["cx"], sls["y0"])], GREEN)
path([(vb["cx"], sls["y1"]), (vb["cx"], vb["y0"])], GREEN)
note(vb["cx"] + 4, (sls["y1"] + vb["y0"]) / 2, "αₐ, αₘ", color=GREEN, style="normal")
bx0, bx1 = CX + 6, CX + CW - 6             # a and m bypass the score branch
path([(gap["x0"], gap["cy"]), (bx0, gap["cy"]), (bx0, vb["cy"]), (vb["x0"], vb["cy"])], GREEN)
path([(gmp["x1"], gmp["cy"]), (bx1, gmp["cy"]), (bx1, vb["cy"]), (vb["x1"], vb["cy"])], GREEN)
note(bx0 + 2.5, (sls["y1"] + vb["y0"]) / 2, "a", color=GREEN, style="italic")
note(bx1 - 2.5, (sls["y1"] + vb["y0"]) / 2, "m", color=GREEN, style="italic", ha="right")

c3 = box(CX, g1 + 13, CW, ["BN → Linear 512 → SiLU → Dropout 0.35",
                           "→ Linear 256 → SiLU", "→ Linear K → logits z"], GREEN)
path([(c1["cx"], g1), (c3["cx"], c3["y0"])], GREEN)
note(c3["cx"], c3["y1"] + 7.5, "K = 3 (D3); K = 4 (D1, D2, D4)", ha="center")

r1 = box(RX, c1["y0"], RW, ["Validation-loss monitoring", "Early stopping (patience 20)"], BLUE)
vx = parts["VALIDATION"]["cx"]
path([(vx, parts["VALIDATION"]["y1"]), (vx, 118), (440, 118), (440, r1["y0"])])

r3 = box(RX, c3["cy"] - h_of(3) / 2, RW, ["Freeze SWA model", "Fit T > 0 using calibration NLL",
                                          "Converged bounded fit"], PURPLE)
r2 = box(RX, r3["y0"] - 13 - h_of(3), RW, ["Calibration images", "Matched preprocessing",
                                           "No augmentation"], PURPLE)
r4 = box(RX, r3["y1"] + 13, RW, ["Calibrated predictor", "p = softmax(z / T)"], PURPLE)
cx_ = parts["CALIBRATION"]["cx"]
path([(cx_, parts["CALIBRATION"]["y1"]), (cx_, 112), (LANE_CAL, 112),
      (LANE_CAL, r2["cy"]), (r2["x1"], r2["cy"])], PURPLE)
path([(r2["cx"], r2["y1"]), (r3["cx"], r3["y0"])], PURPLE)
path([(r3["cx"], r3["y1"]), (r4["cx"], r4["y0"])], PURPLE)
path([(c3["x1"], c3["cy"]), (r3["x0"], r3["cy"])], GREEN)

# dashed control links: SWA weights -> backbone; validation monitoring -> fine-tuning
path([(l4["x1"], l4["cy"]), (142, l4["cy"]), (142, top - 2), (c1["cx"], top - 2),
      (c1["cx"], c1["y0"])], ARR, dashed=True)
dy = (c1["y1"] + c2["y0"]) / 2
path([(r1["x0"], r1["cy"]), (324, r1["cy"]), (324, dy), (147, dy), (147, l3["cy"]),
      (l3["x1"], l3["cy"])], ARR, dashed=True)

# ---------------------------------------------------------------- Panel C
yC = max(c3["y1"] + 16, r4["y1"]) + 14
heading(4, yC, "C   HELD-OUT EVALUATION AND COMPLEMENTARY AUDITS")
he = box(250, yC + 12, 208, ["Held-out test evaluation",
                             "Training-matched preprocessing; no augmentation"], PURPLE)
path([(r4["cx"], r4["y1"]), (r4["cx"], he["y0"])], PURPLE)
sx = parts["TEST"]["x1"] - 4
path([(LANE_TEST, parts["TEST"]["y1"]), (LANE_TEST, he["cy"]), (he["x1"], he["cy"])], PURPLE)
sep = box(4, yC + 12, 200, ["Separate training assessments",
                            "5-fold internal CV; cumulative ablation",
                            "7 classifier-trained baseline backbones",
                            "Same partitions; calibration split excluded"], BLUE)

bw = (W - 8 - 2 * 10) / 3
by = sep["y1"] + 18
ev = [box(4 + i * (bw + 10), by, bw, lines, PURPLE) for i, lines in enumerate([
    ["Classification + calibration", "Accuracy, F1, κ, OvR AUC",
     "ECE, NLL, Brier; risk–coverage", "Bootstrap CIs; McNemar–Holm"],
    ["Attribution assessment", "SHAP, IG, Grad-CAM++ (K = 50/class)",
     "Top-10% mask IoU vs chance", "Per-image deletion–insertion"],
    ["Provenance + transfer audit", "Six transfers among D2, D3, D4",
     "Original/training/harmonized inputs", "Near-duplicate exposure strata"]])]
bus2 = by - 8
ax.plot([he["cx"], he["cx"]], [he["y1"], bus2], color=PURPLE, lw=0.75)
ax.plot([ev[0]["cx"], ev[2]["cx"]], [bus2, bus2], color=PURPLE, lw=0.75)
for e in ev:
    path([(e["cx"], bus2), (e["cx"], e["y0"])], PURPLE)
path([(30, sep["y1"]), (30, ev[0]["y0"])], ARR, dashed=True)

# ---------------------------------------------------------------- legend
ly = ev[0]["y1"] + 13
ax.plot([4, 30], [ly, ly], color=ARR, lw=0.75)
note(35, ly, "Data flow", style="normal")
ax.plot([110, 136], [ly, ly], color=ARR, lw=0.75, ls=(0, (3.5, 2)))
note(141, ly, "Training control / model weights / comparison linkage", style="normal")
H_used = ly + 7

# ---------------------------------------------------------------- checks + save
fig.canvas.draw()
r = fig.canvas.get_renderer()
inv = ax.transData.inverted()
bad = 0
for t, x0, x1 in texts_in_boxes:
    bb = t.get_window_extent(r).transformed(inv)
    if bb.x0 < x0 + 2.5 or bb.x1 > x1 - 2.5:
        bad += 1
        print("OVERFLOW:", repr(t.get_text()[:40]), round(bb.x0, 1), round(bb.x1, 1), "box", x0, x1)
print(f"height used {H_used:.1f} pt of {H}; overflows: {bad}")

# trim the canvas to the content height
fig.set_size_inches(W / 72, H_used / 72)
ax.set_ylim(H_used, 0)
for ext in ("svg", "pdf"):
    fig.savefig(OUT / f"Figure_2_Framework.{ext}")
fig.savefig(OUT / "Figure_2_Framework.png", dpi=600)
print("saved", OUT, f"{W / 72 * 2.54:.2f} x {H_used / 72 * 2.54:.2f} cm")
