# ═══════════════════════════════════════════════════════════════════════
# 05 — Re-fit temperature scaling to convergence and recompute every
#      calibration-dependent in-domain result (Table 11, Table 12 Brier/AUC rows,
#      Fig. 6 reliability diagrams, selective prediction)
#      (run after 00_load_models.py; ~5 min on a T4)
#
# The original notebook's TemperatureScaler.set_temperature (T0 = 1.5, LBFGS lr = 0.01,
# one step of ≤ 100 iterations) stops before the NLL minimum (see 04b). Here T is fitted
# by bounded scalar minimisation of calibration-split NLL (converged).
#
# Checks: bootstrap CIs at the STORED T* must reproduce the published Table 12 values
# (same seed 42, B = 2000, same 8 metrics) before the converged-T values are reported.
#
# IMPORTANT: at the end this cell REPLACES the temperature in `models` with the
# converged T. Re-run 01_table14_transfer_audit.py and 03_xai_rerun.py afterwards so
# that Table 14b ECE/Brier and the deletion–insertion AUCs use the same calibration.
# Outputs → /kaggle/working/audit/recal_*.csv/.json/.png
# ═══════════════════════════════════════════════════════════════════════
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, cohen_kappa_score, confusion_matrix)

OUT = "/kaggle/working/audit"; os.makedirs(OUT, exist_ok=True)
ECE_BINS, B, SEED = 15, 2000, 42
PUB = {  # published Table 12 values at the stored T*: (point, ci_low, ci_high)
    "D1": {"Accuracy": (0.9294, 0.9029, 0.9559), "AUC": (0.9896, 0.9828, 0.9957), "Brier": (0.1174, 0.0841, 0.1528)},
    "D2": {"Accuracy": (0.9766, 0.9642, 0.9875), "AUC": (0.9971, 0.9937, 0.9994), "Brier": (0.0432, 0.0249, 0.0635)},
    "D3": {"Accuracy": (0.9532, 0.9264, 0.9766), "AUC": (0.9907, 0.9787, 0.9984), "Brier": (0.0712, 0.0377, 0.1085)},
    "D4": {"Accuracy": (0.9666, 0.9516, 0.9800), "AUC": (0.9940, 0.9897, 0.9975), "Brier": (0.0616, 0.0391, 0.0873)}}

@torch.no_grad()
def raw_logits(model, paths, bs=64):
    dl = DataLoader(PathDataset(paths), batch_size=bs, num_workers=2)
    return torch.cat([model(x.to(DEVICE)).float().cpu() for x in dl])

def ece_bins(y, prob):                          # notebook's compute_ece_and_reliability_diagram, without plotting
    conf, pred = prob.max(1), prob.argmax(1); acc_v = (pred == y).astype(float)
    e, bc, ba, bn = 0.0, [], [], []
    for lo, hi in zip(np.linspace(0, 1, ECE_BINS + 1)[:-1], np.linspace(0, 1, ECE_BINS + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            bc.append(conf[m].mean()); ba.append(acc_v[m].mean()); bn.append(m.sum())
            e += m.sum() / len(y) * abs(acc_v[m].mean() - conf[m].mean())
    return e, np.array(bc), np.array(ba), np.array(bn)

def specificity(y, p, k):
    cm = confusion_matrix(y, p, labels=list(range(k))); s = []
    for i in range(k):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp; tn = cm.sum() - tp - fp - fn
        s.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return float(np.mean(s))

def bootstrap(y, prob, k):                     # notebook's bootstrap_confidence_intervals_returning
    pred = prob.argmax(1); yoh = np.eye(k)[y]; n = len(y); rng = np.random.default_rng(SEED)
    def _m(yt, yp, ypr, yo):
        try: au = roc_auc_score(yo, ypr, multi_class="ovr", average="macro")
        except Exception: au = 0.5
        return (accuracy_score(yt, yp), precision_score(yt, yp, average="weighted", zero_division=0),
                recall_score(yt, yp, average="weighted", zero_division=0), f1_score(yt, yp, average="weighted", zero_division=0),
                au, cohen_kappa_score(yt, yp), float(np.mean(np.sum((ypr - yo) ** 2, 1))), specificity(yt, yp, k))
    boot = np.zeros((B, 8))
    for b in range(B):
        idx = rng.integers(0, n, n); boot[b] = _m(y[idx], pred[idx], prob[idx], yoh[idx])
    point = _m(y, pred, prob, yoh)
    names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC", "Kappa", "Brier", "Specificity"]
    return {nm: {"point": float(point[i]), "ci_low": float(np.percentile(boot[:, i], 2.5)),
                 "ci_high": float(np.percentile(boot[:, i], 97.5))} for i, nm in enumerate(names)}

def selective(y, prob):                         # notebook's selective_prediction_analysis
    conf = prob.max(1); correct = (prob.argmax(1) == y).astype(float); order = np.argsort(-conf); n = len(y)
    rows = []
    for c in [round(0.50 + 0.05 * i, 2) for i in range(11)]:
        k = max(1, int(round(c * n)))
        rows.append({"coverage": c, "accuracy": float(correct[order][:k].mean()), "threshold": float(conf[order][k - 1])})
    return rows

STORED_T = {ds: models[ds][1] for ds in ["D1", "D2", "D3", "D4"]}
t11, ci_rows, sel_rows, gate, conv_T, diag = [], [], [], [], {}, {}
for ds in ["D1", "D2", "D3", "D4"]:
    model = models[ds][0]; k = len(CLASSES[ds])
    z, y = {}, {}
    for part in ["cal", "test"]:
        sub = splits[ds][splits[ds].split == part]
        y[part] = sub.label.map({c: i for i, c in enumerate(CLASSES[ds])}).values
        z[part] = raw_logits(model, sub.path)
    yc = torch.as_tensor(y["cal"])
    T_new = float(minimize_scalar(lambda t: float(torch.nn.functional.cross_entropy(z["cal"] / t, yc)),
                                  bounds=(0.05, 10.0), method="bounded", options={"xatol": 1e-6}).x)
    conv_T[ds] = T_new
    probs = {name: torch.softmax(z["test"] / T, 1).numpy()
             for name, T in [("uncalibrated", 1.0), ("stored_T", STORED_T[ds]), ("converged_T", T_new)]}
    for name, T in [("uncalibrated", 1.0), ("stored_T", STORED_T[ds]), ("converged_T", T_new)]:
        p = probs[name]; e, bc, ba, bn = ece_bins(y["test"], p); diag[(ds, name)] = (bc, ba, bn, e)
        t11.append({"dataset": ds, "condition": name, "T": round(T, 4), "ece": e,
                    "brier": float(np.mean(np.sum((p - np.eye(k)[y["test"]]) ** 2, 1))),
                    "nll": float(torch.nn.functional.cross_entropy(z["test"] / T, torch.as_tensor(y["test"]))),
                    "auc_macro_ovr": roc_auc_score(np.eye(k)[y["test"]], p, multi_class="ovr", average="macro"),
                    "accuracy": float((p.argmax(1) == y["test"]).mean())})
    ci_stored = bootstrap(y["test"], probs["stored_T"], k); ci_new = bootstrap(y["test"], probs["converged_T"], k)
    for nm in ci_stored:
        ci_rows.append({"dataset": ds, "metric": nm,
                        **{f"stored_{q}": v for q, v in ci_stored[nm].items()}, **{f"converged_{q}": v for q, v in ci_new[nm].items()}})
    for nm, (pt, lo, hi) in PUB[ds].items():
        s = ci_stored[nm]
        gate.append({"dataset": ds, "metric": nm, "published": f"{pt:.4f} [{lo:.4f}, {hi:.4f}]",
                     "reproduced": f"{s['point']:.4f} [{s['ci_low']:.4f}, {s['ci_high']:.4f}]",
                     "match": max(abs(s["point"] - pt), abs(s["ci_low"] - lo), abs(s["ci_high"] - hi)) <= 0.0006})
    for r in selective(y["test"], probs["converged_T"]): sel_rows.append({"dataset": ds, **r})
    print(f"[{ds}] stored T*={STORED_T[ds]:.4f} → converged T={T_new:.4f}")

gate = pd.DataFrame(gate); print("\n=== Check: bootstrap at stored T* reproduces published Table 12 ===")
print(gate.to_string(index=False)); print("GATE:", "PASS" if gate.match.all() else "CHECK")
t11 = pd.DataFrame(t11); t11.to_csv(f"{OUT}/recal_table11.csv", index=False)
pd.DataFrame(ci_rows).to_csv(f"{OUT}/recal_bootstrap_ci.csv", index=False)
pd.DataFrame(sel_rows).to_csv(f"{OUT}/recal_selective_prediction.csv", index=False)
json.dump({"stored_T": STORED_T, "converged_T": conv_T}, open(f"{OUT}/recal_temperatures.json", "w"), indent=2)
pd.set_option("display.width", 200)
print("\n=== Table 11 (test split) ===\n" + t11.round(4).to_string(index=False))
ci = pd.DataFrame(ci_rows)
print("\n=== Table 12 rows that depend on T (converged) ===")
print(ci[ci.metric.isin(["AUC", "Brier"])][["dataset", "metric", "converged_point", "converged_ci_low", "converged_ci_high"]].round(4).to_string(index=False))
print("\n=== Selective prediction (converged T) ===\n" + pd.DataFrame(sel_rows).pivot(index="coverage", columns="dataset", values="accuracy").round(4).to_string())

# Fig. 6 replacement: reliability diagrams, uncalibrated vs converged T
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for c, ds in enumerate(["D1", "D2", "D3", "D4"]):
    for r, name in enumerate(["uncalibrated", "converged_T"]):
        bc, ba, bn, e = diag[(ds, name)]; a_ = ax[r][c]
        a_.bar(bc, ba, width=0.8 / ECE_BINS, color="#2a78d6", alpha=0.85, label="Accuracy")
        a_.bar(bc, bc - ba, bottom=ba, width=0.8 / ECE_BINS, color="#eb6834", alpha=0.45, label="Gap")
        a_.plot([0, 1], [0, 1], "k--", lw=1); a_.set_xlim(0, 1); a_.set_ylim(0, 1)
        T = 1.0 if name == "uncalibrated" else conv_T[ds]
        a_.set_title(f"{ds} — {'uncalibrated (T = 1)' if r == 0 else f'calibrated (T = {T:.3f})'}\nECE = {e:.4f}", fontsize=10)
        a_.set_xlabel("Confidence"); a_.set_ylabel("Accuracy")
ax[0][0].legend(frameon=False, loc="upper left")
plt.tight_layout(); plt.savefig(f"{OUT}/recal_reliability_diagrams.png", dpi=300); plt.close()

# Switch the loaded models to the converged temperature for any subsequent cell
for ds in conv_T: models[ds] = (models[ds][0], conv_T[ds])
print("\n`models` now carry the converged T. Re-run 01_table14_transfer_audit.py and 03_xai_rerun.py next.")
print(f"Saved to {OUT}/")
