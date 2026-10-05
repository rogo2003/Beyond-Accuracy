"""11 — Tables 13, 14b, 14c and the explainability statistics, from the Kaggle outputs.

Runs locally (no GPU). Inputs (written by the Kaggle cells):
  table14_rerun_results.csv              (01_table14_transfer_audit.py)
  xai_rerun_summary.csv, xai_rerun_per_image.csv   (03_xai_rerun.py)

Outputs (in --out): table14b.csv, table14c.csv, table13.csv, xai_tests.csv
and a check of every count quoted in the revised manuscript.

Usage:  python audit/local/11_tables_and_stats.py [--results .] [--out audit/results]
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

IN_DOMAIN = {"D2": 97.66, "D3": 95.32, "D4": 96.66}
ORDER = [("D2", "D4"), ("D4", "D2"), ("D2", "D3"), ("D4", "D3"), ("D3", "D2"), ("D3", "D4")]
CHANCE_IOU = 0.01 / 0.19      # expected IoU of two independent random 10% masks

def table14(res):
    g = lambda s, t, pl, sc, st, col: res[(res.source == s) & (res.target == t) & (res.pipeline == pl)
                                          & (res.scope == sc) & (res.stratum == st)][col].iloc[0]
    b, c = [], []
    for s, t in ORDER:
        n = int(g(s, t, "published", "test", "ALL", "n"))
        exp_n = int(g(s, t, "harmonised", "test", "exposed", "n"))
        b.append({"transfer": f"{s}→{t}", "n": n,
                  "acc_original": g(s, t, "published", "test", "ALL", "acc"),
                  "acc_training": g(s, t, "training", "test", "ALL", "acc"),
                  "acc_harmonised": g(s, t, "harmonised", "test", "ALL", "acc"),
                  "acc_harm_lo": g(s, t, "harmonised", "test", "ALL", "acc_lo"),
                  "acc_harm_hi": g(s, t, "harmonised", "test", "ALL", "acc_hi"),
                  **{k: g(s, t, "harmonised", "test", "ALL", k) for k in ["f1_weighted", "auc", "kappa", "brier", "ece"]},
                  "delta_acc": IN_DOMAIN[s] - g(s, t, "harmonised", "test", "ALL", "acc"),
                  "exposed_n": exp_n, "exposed_pct": 100 * exp_n / n})
        c.append({"transfer": f"{s}→{t}",
                  "exposed_n": int(g(s, t, "harmonised", "full", "exposed", "n")),
                  "exposed_acc": g(s, t, "harmonised", "full", "exposed", "acc"),
                  "clean_n": int(g(s, t, "harmonised", "full", "clean", "n")),
                  "clean_acc": g(s, t, "harmonised", "full", "clean", "acc"),
                  "clean_bal_acc": g(s, t, "harmonised", "full", "clean", "bal_acc")})
    return pd.DataFrame(b), pd.DataFrame(c)

def xai_tests(per):
    rows = []
    for (d, c), g in per.groupby(["dataset", "class"], sort=False):
        gv = g[~g.gc_fixed_flat]
        r = {"dataset": d, "class": c}
        for col, src in [("SG", gv), ("SI", g), ("GI", gv)]:
            r[col] = src[col].mean(); r[col + "_sd"] = src[col].std(ddof=1)
            r[col + "_p"] = wilcoxon(src[col] - CHANCE_IOU, alternative="greater").pvalue
        r["del_diff"] = (g.del_auc - g.del_auc_random).mean()
        r["ins_diff"] = (g.ins_auc - g.ins_auc_random).mean()
        r["del_p"] = wilcoxon(g.del_auc, g.del_auc_random, alternative="less").pvalue
        r["ins_p"] = wilcoxon(g.ins_auc, g.ins_auc_random, alternative="greater").pvalue
        r["n_flat_gradcam"] = int(g.gc_fixed_flat.sum())
        rows.append(r)
    t = pd.DataFrame(rows)
    for col in ["SG_p", "SI_p", "GI_p", "del_p", "ins_p"]:
        t[col + "_holm"] = multipletests(t[col], method="holm")[1]
    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=".")
    ap.add_argument("--out", default="audit/results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 220)

    res = pd.read_csv(os.path.join(a.results, "table14_rerun_results.csv"))
    t14b, t14c = table14(res)
    t14b.to_csv(os.path.join(a.out, "table14b.csv"), index=False); t14c.to_csv(os.path.join(a.out, "table14c.csv"), index=False)
    print("=== Table 14b ===\n" + t14b.round(2).to_string(index=False))
    print("\n=== Table 14c ===\n" + t14c.round(2).to_string(index=False))

    per = pd.read_csv(os.path.join(a.results, "xai_rerun_per_image.csv"))
    t = xai_tests(per); t.to_csv(os.path.join(a.out, "xai_tests.csv"), index=False)
    t13 = t[["dataset", "class", "SG", "SG_sd", "SI", "SI_sd", "GI", "GI_sd"]].copy()
    t13["mean"] = t13[["SG", "SI", "GI"]].mean(1)
    for col in ["SG", "SI", "GI"]:
        t13[col + "_above_chance"] = t[col + "_p_holm"] < 0.05
    t13.to_csv(os.path.join(a.out, "table13.csv"), index=False)
    overall = t13.groupby("dataset")[["SG", "SI", "GI"]].mean(); overall["mean"] = overall.mean(1)
    print("\n=== Table 13 (per class) ===\n" + t13.round(3).to_string(index=False))
    print("\n=== Table 13 (overall rows) ===\n" + overall.round(3).to_string())

    # Every count quoted in the revised manuscript
    checks = {
        "Original pipeline reproduces Table 14 (6/6)": all(abs(t14b.acc_original - [79.30, 63.55, 27.42, 31.77, 73.00, 69.25]) < 0.5),
        "Harmonised accuracy range 94.99–99.67": (round(t14b.acc_harmonised.min(), 2), round(t14b.acc_harmonised.max(), 2)) == (94.99, 99.67),
        "Exposed share of test sets 25.3–78.5%": (round(t14b.exposed_pct.min(), 1), round(t14b.exposed_pct.max(), 1)) == (25.3, 78.5),
        "Kappa 0.933–0.995, AUC >= 98.88": (round(t14b.kappa.min(), 3), round(t14b.kappa.max(), 3), round(t14b.auc.min(), 2)) == (0.933, 0.995, 98.88),
        "ΔAcc −3.92 to +2.67": (round(t14b.delta_acc.min(), 2), round(t14b.delta_acc.max(), 2)) == (-3.92, 2.67),
        "Exposed acc 98.78–99.50, clean acc 91.11–99.04": (round(t14c.exposed_acc.min(), 2), round(t14c.exposed_acc.max(), 2),
                                                          round(t14c.clean_acc.min(), 2), round(t14c.clean_acc.max(), 2)) == (98.78, 99.50, 91.11, 99.04),
        "SHAP–IG above chance in 15/15": int((t.SI_p_holm < 0.05).sum()) == 15,
        "SHAP–Grad-CAM++ above chance in 1/15": int((t.SG_p_holm < 0.05).sum()) == 1,
        "Grad-CAM++–IG above chance in 5/15": int((t.GI_p_holm < 0.05).sum()) == 5,
        "Insertion SHAP > random in 12/15": int((t.ins_p_holm < 0.05).sum()) == 12,
        "Deletion SHAP < random in 6/15": int((t.del_p_holm < 0.05).sum()) == 6,
        "Flat Grad-CAM++ maps: 10 of 750": int(t.n_flat_gradcam.sum()) == 10 and len(per) == 750,
        "Dataset mean IoU 0.091/0.085/0.096/0.075": [round(v, 3) for v in overall["mean"]] == [0.091, 0.085, 0.096, 0.075],
    }
    print("\n=== Checks against the revised manuscript ===")
    for k, v in checks.items(): print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("\nALL PASS" if all(checks.values()) else "\nSOME CHECKS FAILED")

if __name__ == "__main__":
    main()
