# ═══════════════════════════════════════════════════════════════════════
# 04b — Temperature scaling: diagnose the stored T* and re-fit to convergence
#       (run after 00_load_models.py and 04_temperature_and_duplicate_examples.py)
#
# 1. Replays the notebook's fit exactly (TemperatureScaler.set_temperature: T0 = 1.5,
#    LBFGS lr = 0.01, max_iter = 100, one step) on the calibration-split logits.
#    Check: must reproduce the stored T* (±0.005) → shows the fit stopped early.
# 2. Fits T to convergence (bounded scalar minimisation of calibration NLL).
# 3. Test-split accuracy, ECE (15 bins), NLL and Brier at T = 1, stored T*, converged T.
#    Check: ECE at T = 1 and at stored T* must reproduce the published values (±0.001).
# Output → /kaggle/working/audit/temperature_refit.csv
# ═══════════════════════════════════════════════════════════════════════
from scipy.optimize import minimize_scalar

PUB_ECE = {"D1": (0.0503, 0.0500), "D2": (0.0456, 0.0289), "D3": (0.0383, 0.0340), "D4": (0.0415, 0.0188)}  # (uncal, cal)

def notebook_fit(z, y):
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)
    def closure():
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(z / T.clamp(min=1e-4), y); loss.backward(); return loss
    opt.step(closure)
    return float(T.detach().clamp(min=1e-4))

def metrics(z, y, T):
    p = torch.softmax(z / T, 1).numpy(); yn = y.numpy()
    onehot = np.eye(p.shape[1])[yn]
    return {"acc": float((p.argmax(1) == yn).mean()), "ece": ece(p, yn),
            "nll": float(torch.nn.functional.cross_entropy(z / T, y)),
            "brier": float(np.mean(np.sum((p - onehot) ** 2, 1)))}

rows = []
for ds in ["D1", "D2", "D3", "D4"]:
    model, T_star = models[ds]
    z = {}; y = {}
    for part in ["cal", "test"]:
        sub = splits[ds][splits[ds].split == part]
        y[part] = torch.as_tensor(sub.label.map({c: i for i, c in enumerate(CLASSES[ds])}).values)
        z[part] = raw_logits(model, sub.path)
    T_replay = notebook_fit(z["cal"], y["cal"])
    T_conv = float(minimize_scalar(lambda t: float(torch.nn.functional.cross_entropy(z["cal"] / t, y["cal"])),
                                   bounds=(0.05, 10.0), method="bounded", options={"xatol": 1e-6}).x)
    for label, T in [("uncalibrated (T=1)", 1.0), ("stored T*", T_star), ("converged T", T_conv)]:
        m = metrics(z["test"], y["test"], T)
        rows.append({"dataset": ds, "setting": label, "T": round(T, 4), **{k: round(v, 4) for k, v in m.items()},
                     "cal_nll": round(float(torch.nn.functional.cross_entropy(z["cal"] / T, y["cal"])), 4)})
    print(f"[{ds}] stored T*={T_star:.4f}  notebook-replay T={T_replay:.4f} "
          f"({'REPRODUCED' if abs(T_replay - T_star) <= 0.005 else 'DIFFERENT'})  converged T={T_conv:.4f}")

res = pd.DataFrame(rows); res.to_csv(f"{OUT}/temperature_refit.csv", index=False)
print("\n=== Test-split metrics ===")
print(res.to_string(index=False))
print("\n=== Check: ECE reproduces the paper (uncalibrated, stored T*) ===")
for ds, (pu, pc) in PUB_ECE.items():
    eu = res[(res.dataset == ds) & (res.setting == "uncalibrated (T=1)")].ece.iloc[0]
    ec = res[(res.dataset == ds) & (res.setting == "stored T*")].ece.iloc[0]
    print(f"  {ds}: uncal {eu:.4f} vs {pu:.4f}, cal {ec:.4f} vs {pc:.4f} → "
          f"{'PASS' if abs(eu - pu) <= 0.001 and abs(ec - pc) <= 0.001 else 'CHECK'}")
