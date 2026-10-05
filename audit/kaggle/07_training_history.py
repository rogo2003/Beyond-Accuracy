# ═══════════════════════════════════════════════════════════════════════
# 07 — Re-run of the final-model training with per-epoch history logging
#      (Fig. 3, learning dynamics)
#
# Run after 00 (uses load_and_enhance, BrainTumorModel, CLASSES, splits,
# NORM_MEAN/NORM_STD, DEVICE). Kaggle settings: GPU T4, **Internet ON**
# (ImageNet weights for tf_efficientnetv2_b0.in1k are downloaded by timm).
#
# Replicates the notebook's train_model_pytorch() exactly (two-stage schedule,
# AdamW + batch-wise OneCycleLR, 160→224 px at fine-tuning index 30, SWA +
# SWALR from index 75, early stopping on validation loss with patience 20,
# update_bn on the training loader) and only adds logging. Training data are
# the same pHash group-split training/validation partitions.
#
# ~1 h per dataset on a T4. Set DATASETS to run a subset per session; each
# dataset's history is written to disk after every epoch, so a timed-out
# session keeps its progress.
#
# Outputs (/kaggle/working):
#   training_history_<DS>.csv   one row per epoch (stage, epoch, res, lr, losses, accs)
#   training_rerun_summary.csv  stop epoch, SWA snapshots, test accuracy vs released
#   Figure_3_Learning_curves.png/.pdf  (when all four histories exist)
# ═══════════════════════════════════════════════════════════════════════
import os, random, time, numpy as np, pandas as pd, torch, torch.nn as nn, timm
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch import amp
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

OUT = globals().get("OUT", "/kaggle/working")
DATASETS = globals().get("DATASETS", ["D1", "D2", "D3", "D4"])

# Notebook Config (unchanged)
BATCH, EPOCHS, WARMUP, PATIENCE = 32, 100, 15, 20
LR_HEAD, LR_FT, WD, DROPOUT, LS = 1e-4, 2e-5, 1e-4, 0.35, 0.05
RES_SCHEDULE, SWA_START = [(0, 160), (30, 224)], 75
RELEASED = {"D1": dict(acc=0.9294, snapshots=15), "D2": dict(acc=0.9766, snapshots=12),
            "D3": dict(acc=0.9532, snapshots=25), "D4": dict(acc=0.9666, snapshots=17)}


def seed_everything(seed=42):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def get_transforms(img_size, augment):
    if augment:
        return T.Compose([T.ToPILImage(), T.Resize((img_size, img_size)),
                          T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                          T.ColorJitter(brightness=0.1), T.RandomHorizontalFlip(p=0.0),
                          T.ToTensor(), T.Normalize(NORM_MEAN, NORM_STD)])
    return T.Compose([T.ToPILImage(), T.Resize((img_size, img_size)), T.ToTensor(),
                      T.Normalize(NORM_MEAN, NORM_STD)])


class ArrayDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform
    def __len__(self): return len(self.images)
    def __getitem__(self, i): return self.transform(self.images[i]), int(self.labels[i])


def load_split(ds, name):
    df = splits[ds][splits[ds].split == name]
    X = np.stack([load_and_enhance(p) for p in df.path]).astype(np.uint8)
    y = np.array([CLASSES[ds].index(l) for l in df.label])
    return X, y


def new_model(ds):
    m = BrainTumorModel(len(CLASSES[ds]), dropout=DROPOUT)
    try:
        m.backbone = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True,
                                       num_classes=0, global_pool="")
    except Exception as e:
        raise RuntimeError("Could not download ImageNet weights — turn Internet ON in the "
                           "Kaggle notebook settings.") from e
    return m.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    tot, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train:
                optimizer.zero_grad()
                with amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type != "cpu")):
                    out = model(x); loss = criterion(out, y)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                if scheduler is not None: scheduler.step()      # per batch (Stage 2)
            else:
                out = model(x); loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            correct += out.argmax(1).eq(y).sum().item(); tot += x.size(0)
    return loss_sum / tot, correct / tot


def train_with_history(ds):
    seed_everything(42)
    t0 = time.time()
    Xtr, ytr = load_split(ds, "train"); Xva, yva = load_split(ds, "val"); Xte, yte = load_split(ds, "test")
    print(f"[{ds}] train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}  (loaded in {time.time()-t0:.0f}s)")
    model = new_model(ds)
    crit = nn.CrossEntropyLoss(label_smoothing=LS)
    scaler = amp.GradScaler(enabled=(DEVICE.type != "cpu"))
    hist, csv = [], f"{OUT}/training_history_{ds}.csv"

    def log(**row):
        hist.append(row); pd.DataFrame(hist).to_csv(csv, index=False)

    # Stage 1: head warm-up, backbone frozen, 160 px, augmentation on, validation monitored only
    for p in model.backbone.parameters(): p.requires_grad = False
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    res = RES_SCHEDULE[0][1]
    tr_ds = ArrayDataset(Xtr, ytr, get_transforms(res, True))
    va_ds = ArrayDataset(Xva, yva, get_transforms(res, False))
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    va_dl = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=2)
    for e in range(WARMUP):
        tl, ta = run_epoch(model, tr_dl, crit, opt, scaler)
        vl, va = run_epoch(model, va_dl, crit)
        log(stage="warmup", epoch=e + 1, ft_index=None, res=res, lr=opt.param_groups[0]["lr"],
            train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va, swa=False, best=False)
        print(f"[{ds}] warm-up {e+1:>2}/{WARMUP}  loss {tl:.4f}/{vl:.4f}  acc {ta:.4f}/{va:.4f}")

    # Stage 2: full fine-tuning
    for p in model.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": LR_FT},
                             {"params": model.pool_head.parameters(), "lr": LR_FT * 5},
                             {"params": model.classifier.parameters(), "lr": LR_FT * 5}],
                            weight_decay=WD)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_FT, LR_FT * 5, LR_FT * 5],
                                                epochs=EPOCHS, steps_per_epoch=len(tr_dl),
                                                pct_start=0.1, div_factor=10.0, final_div_factor=100.0)
    swa_model, swa_sched = AveragedModel(model), SWALR(opt, swa_lr=LR_FT)
    best, counter, best_epoch, stop_epoch, ckpt = None, 0, None, EPOCHS, f"{OUT}/_best_{ds}.pth"
    for e in range(EPOCHS):
        for start, new_res in RES_SCHEDULE:
            if e == start and new_res != res:
                res = new_res
                tr_dl.dataset.transform = get_transforms(res, True)
                va_dl.dataset.transform = get_transforms(res, False)
        lr = opt.param_groups[0]["lr"]
        tl, ta = run_epoch(model, tr_dl, crit, opt, scaler, sched)
        vl, va = run_epoch(model, va_dl, crit)
        in_swa = e >= SWA_START
        if in_swa:
            swa_model.update_parameters(model); swa_sched.step()
        improved = best is None or -vl >= best          # notebook: ties count as improvement
        if improved:
            best, counter, best_epoch = -vl, 0, e + 1; torch.save(model.state_dict(), ckpt)
        else:
            counter += 1
        log(stage="finetune", epoch=e + 1, ft_index=e, res=res, lr=lr, train_loss=tl, train_acc=ta,
            val_loss=vl, val_acc=va, swa=in_swa, best=improved)
        print(f"[{ds}] fine-tune {e+1:>3}/{EPOCHS} res {res}  loss {tl:.4f}/{vl:.4f}  "
              f"acc {ta:.4f}/{va:.4f}{'  SWA' if in_swa else ''}{'  *' if improved else ''}")
        if counter >= PATIENCE:
            stop_epoch = e + 1; print(f"[{ds}] early stopping at fine-tuning epoch {stop_epoch}"); break

    update_bn(tr_dl, swa_model, device=DEVICE)
    te_dl = DataLoader(ArrayDataset(Xte, yte, get_transforms(224, False)), batch_size=64, num_workers=2)
    _, te_acc = run_epoch(swa_model, te_dl, crit)
    n_avg = int(swa_model.n_averaged.item())
    summary = dict(dataset=ds, stop_epoch=stop_epoch, best_val_epoch=best_epoch, swa_snapshots=n_avg,
                   released_snapshots=RELEASED[ds]["snapshots"], test_acc_rerun=te_acc,
                   test_acc_released=RELEASED[ds]["acc"], diff_pp=100 * (te_acc - RELEASED[ds]["acc"]),
                   minutes=(time.time() - t0) / 60)
    print(f"[{ds}] SUMMARY {summary}")
    os.remove(ckpt)
    return summary


def plot_fig3(out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = {"D1": "D1 (SARTAJ)", "D2": "D2 (BRISC)", "D3": "D3 (Figshare)", "D4": "D4 (Nickparvar)"}
    summ = pd.read_csv(f"{out}/training_rerun_summary.csv").drop_duplicates("dataset", keep="last").set_index("dataset")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, ds in zip(axes.ravel(), ["D1", "D2", "D3", "D4"]):
        h = pd.read_csv(f"{out}/training_history_{ds}.csv")
        x = np.arange(1, len(h) + 1)                       # continuous epoch count (warm-up first)
        nw = (h.stage == "warmup").sum()
        ax.axvspan(0.5, nw + 0.5, color="0.92", lw=0)
        sw = h.index[h.swa.astype(bool)]
        if len(sw): ax.axvspan(sw[0] + 0.5, len(h) + 0.5, color="#e8f0fb", lw=0)
        ax.axvline(nw + 30 + 0.5, color="0.4", ls=":", lw=1)
        ax.plot(x, h.train_loss, color="#1f4e9a", lw=1.4, label="Train loss")
        ax.plot(x, h.val_loss, color="#1f4e9a", lw=1.4, ls="--", label="Validation loss")
        ax.set_ylabel("Loss (cross-entropy, ε = 0.05)", color="#1f4e9a")
        ax2 = ax.twinx()
        ax2.plot(x, 100 * h.train_acc, color="#c0392b", lw=1.4, label="Train accuracy")
        ax2.plot(x, 100 * h.val_acc, color="#c0392b", lw=1.4, ls="--", label="Validation accuracy")
        ax2.set_ylabel("Accuracy (%)", color="#c0392b")
        s = summ.loc[ds]
        ax.set_title(f"({'abcd'[list(names).index(ds)]}) {names[ds]} — stopped at fine-tuning epoch "
                     f"{int(s.stop_epoch)}, {int(s.swa_snapshots)} SWA snapshots", fontsize=10)
        ax.set_xlabel("Epoch (15 warm-up epochs, then fine-tuning)")
        ax.set_xlim(0.5, nw + 100.5)
        if ds == "D1":
            l1, n1 = ax.get_legend_handles_labels(); l2, n2 = ax2.get_legend_handles_labels()
            ax.legend(l1 + l2, n1 + n2, loc="center right", fontsize=8)
    fig.text(0.5, 0.005, "Grey: head warm-up (160 px, frozen backbone). Dotted line: 160 → 224 px at the 31st "
             "fine-tuning epoch. Blue: SWA averaging window (from the 76th fine-tuning epoch).",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/Figure_3_Learning_curves.{ext}", dpi=600 if ext == "png" else None)
    print("saved Figure_3_Learning_curves")


summary_csv = f"{OUT}/training_rerun_summary.csv"
for ds in DATASETS:
    s = train_with_history(ds)
    prev = pd.read_csv(summary_csv) if os.path.exists(summary_csv) else pd.DataFrame()
    pd.concat([prev, pd.DataFrame([s])]).to_csv(summary_csv, index=False)

if all(os.path.exists(f"{OUT}/training_history_{d}.csv") for d in ["D1", "D2", "D3", "D4"]) \
        and os.path.exists(summary_csv):
    print(pd.read_csv(summary_csv).round(4).to_string(index=False))
    plot_fig3(OUT)
