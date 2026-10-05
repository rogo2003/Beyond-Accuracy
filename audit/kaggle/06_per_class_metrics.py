# ═══════════════════════════════════════════════════════════════════════
# 06 — Per-class test metrics (Table 10), confusion matrices (Fig. 4) and
#      empirical ROC curves (Fig. 5) from the saved checkpoints
#      Run after 00 (uses `models`, `splits`, `predict_probs`, CLASSES);
#      probabilities use the converged temperatures.
# Outputs: per_class_metrics.csv, confusion_matrices.csv,
#          Figure_4_Confusion_matrices.png/.pdf, Figure_5_ROC_empirical.png/.pdf
# Built-in check: macro OvR AUC must reproduce Table 11/12 (D1 0.98961, ...).
# ═══════════════════════════════════════════════════════════════════════
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             precision_recall_fscore_support, accuracy_score)

CONVERGED_T = {"D1": 0.8752, "D2": 0.7446, "D3": 0.7351, "D4": 0.7368}
PUBLISHED_MACRO_AUC = {"D1": 0.98961, "D2": 0.99722, "D3": 0.99079, "D4": 0.99399}
NAMES = {"D1": "D1 (SARTAJ)", "D2": "D2 (BRISC)", "D3": "D3 (Figshare)", "D4": "D4 (Nickparvar)"}
LABEL = {"glioma": "Glioma", "meningioma": "Meningioma", "no_tumor": "No tumor",
         "pituitary": "Pituitary"}


def per_class_metrics(ds, model, paths, labels):
    cls = CLASSES[ds]
    y = np.array([cls.index(l) for l in labels])
    p = predict_probs(model, CONVERGED_T[ds], paths)
    yhat = p.argmax(1)
    cm = confusion_matrix(y, yhat, labels=range(len(cls)))
    prec, rec, f1, sup = precision_recall_fscore_support(y, yhat, labels=range(len(cls)),
                                                         zero_division=0)
    rows = []
    for k, c in enumerate(cls):
        fp = cm[:, k].sum() - cm[k, k]
        tn = cm.sum() - cm[k, :].sum() - fp
        rows.append(dict(dataset=ds, cls=c, precision=prec[k], recall=rec[k], f1=f1[k],
                         specificity=tn / (tn + fp), auc=roc_auc_score(y == k, p[:, k]),
                         support=int(sup[k])))
    df = pd.DataFrame(rows)
    acc, macro = accuracy_score(y, yhat), df.auc.mean()
    print(f"[{ds}] acc={acc:.4f}  macro OvR AUC={macro:.5f} "
          f"(published {PUBLISHED_MACRO_AUC[ds]:.5f})  "
          f"{'PASS' if abs(macro - PUBLISHED_MACRO_AUC[ds]) < 5e-5 else 'CHECK'}")
    return df, cm, y, p


def plot_roc(results, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (ds, (df, cm, y, p)) in zip(axes.ravel(), results.items()):
        for k, c in enumerate(CLASSES[ds]):
            fpr, tpr, _ = roc_curve(y == k, p[:, k])
            ax.step(fpr, tpr, where="post", lw=1.6,
                    label=f"{LABEL[c]} (AUC = {df.auc[k]:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey", label="Chance (AUC = 0.500)")
        ax.set(xlim=(-0.01, 1), ylim=(0, 1.01), xlabel="False positive rate (1 − specificity)",
               ylabel="True positive rate (sensitivity)",
               title=f"({'abcd'[list(results).index(ds)]}) {NAMES[ds]} — macro OvR AUC "
                     f"{df.auc.mean():.4f}")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/Figure_5_ROC_empirical.{ext}", dpi=600 if ext == "png" else None)


def plot_confusion(results, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (ds, (df, cm, y, p)) in zip(axes.ravel(), results.items()):
        names = [LABEL[c] for c in CLASSES[ds]]
        pct = cm / cm.sum(1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{cm[i, j]}\n({100 * pct[i, j]:.1f}%)", ha="center", va="center",
                        fontsize=9, color="white" if pct[i, j] > 0.5 else "black")
        ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names,
               yticklabels=names, xlabel="Predicted class", ylabel="True class",
               title=f"({'abcd'[list(results).index(ds)]}) {NAMES[ds]} — accuracy "
                     f"{100 * np.trace(cm) / cm.sum():.2f}%")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/Figure_4_Confusion_matrices.{ext}", dpi=600 if ext == "png" else None)


if __name__ == "__main__" or True:
    OUT = globals().get("OUT", "/kaggle/working")
    DATASETS = globals().get("DATASETS", ["D1", "D2", "D3", "D4"])
    results = {}
    for ds in DATASETS:
        te = splits[ds][splits[ds].split == "test"]
        model = models[ds][0]
        results[ds] = per_class_metrics(ds, model, te.path, te.label)
    pd.concat([r[0] for r in results.values()]).to_csv(f"{OUT}/per_class_metrics.csv", index=False)
    pd.concat([pd.DataFrame(r[1], index=CLASSES[ds], columns=CLASSES[ds]).assign(dataset=ds)
               for ds, r in results.items()]).to_csv(f"{OUT}/confusion_matrices.csv")
    if len(results) == 4:
        plot_roc(results, OUT)
        plot_confusion(results, OUT)
    print(pd.concat([r[0] for r in results.values()]).round(4).to_string(index=False))
