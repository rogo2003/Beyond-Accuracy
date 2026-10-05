# ═══════════════════════════════════════════════════════════════════════
# 00 — Load calibrated D1–D4 checkpoints + in-domain sanity gate
#          (run first in every Kaggle session; 01–03 reuse its objects)
# Kaggle inputs: dataset with *_final_model_v5_calibrated_pytorch.pth and
# *_phash_group_split.csv, plus the raw D1–D4 datasets.
# ═══════════════════════════════════════════════════════════════════════
import os, glob, cv2, numpy as np, pandas as pd, torch, torch.nn as nn, timm
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM_MEAN, NORM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE, MEDIAN_K = (224, 224), 5

# Class order = sklearn LabelEncoder (alphabetical), as used in training
CLASSES = {"D1": ["glioma", "meningioma", "no_tumor", "pituitary"],
           "D2": ["glioma", "meningioma", "no_tumor", "pituitary"],
           "D3": ["glioma", "meningioma", "pituitary"],
           "D4": ["glioma", "meningioma", "no_tumor", "pituitary"]}

# ── Preprocessing: identical to the notebook's _load_and_enhance + eval transform
def crop_brain_contour(img):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    th = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    th = cv2.dilate(cv2.erode(th, None, iterations=2), None, iterations=2)
    cnts = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if not cnts: return img
    c = max(cnts, key=cv2.contourArea)
    l, r, t, b = c[:,:,0].min(), c[:,:,0].max(), c[:,:,1].min(), c[:,:,1].max()
    h, w, _ = img.shape
    return img[max(0,t-5):min(h,b+5), max(0,l-5):min(w,r+5)]

def load_and_enhance(p):
    img = cv2.imread(p)
    if img is None: raise FileNotFoundError(p)
    img = crop_brain_contour(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    return cv2.medianBlur(img, MEDIAN_K)

EVAL_TF = T.Compose([T.ToPILImage(), T.Resize(IMG_SIZE), T.ToTensor(),
                     T.Normalize(NORM_MEAN, NORM_STD)])

class PathDataset(Dataset):
    def __init__(self, paths): self.paths = list(paths)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return EVAL_TF(load_and_enhance(self.paths[i]))

# ── Model: identical architecture to the notebook
class DualPoolingAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gap, self.gmp = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        self.attn_fc = nn.Linear(c, 1, bias=True).float()
    def forward(self, x):
        cat = torch.cat([self.gap(x).view(x.size(0),1,-1),
                         self.gmp(x).view(x.size(0),1,-1)], dim=1)
        w = torch.softmax(self.attn_fc(cat.float()), dim=1).to(cat.dtype)
        return (cat * w).sum(dim=1)

class BrainTumorModel(nn.Module):
    def __init__(self, num_classes, dropout=0.35):
        super().__init__()
        # pretrained=False: all weights come from the checkpoint (no internet needed)
        self.backbone = timm.create_model('tf_efficientnetv2_b0.in1k', pretrained=False,
                                          num_classes=0, global_pool='')
        self.pool_head = DualPoolingAttention(1280)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1280), nn.Linear(1280, 512), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.classifier(self.pool_head(self.backbone(x)))

# ── Checkpoints are SWA AveragedModel state dicts
#    ('module.' prefix + an extra 'n_averaged' key) → strip before loading
def load_calibrated(ds):
    hits = glob.glob(f"/kaggle/input/**/{ds}_final_model_v5_calibrated_pytorch.pth", recursive=True)
    assert hits, f"{ds} checkpoint not found under /kaggle/input — is the dataset attached?"
    ck = torch.load(hits[0], map_location="cpu", weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in ck["model_state_dict"].items()
          if k != "n_averaged"}
    m = BrainTumorModel(len(CLASSES[ds]))
    m.load_state_dict(sd, strict=True)
    print(f"[{ds}] loaded {hits[0]}  T={ck['temperature']:.4f}")
    return m.to(DEVICE).eval(), float(ck["temperature"])

@torch.no_grad()
def predict_probs(model, temp, paths, bs=64):
    dl = DataLoader(PathDataset(paths), batch_size=bs, num_workers=2)
    out = [torch.softmax(model(x.to(DEVICE)) / temp, dim=1).cpu() for x in dl]
    return torch.cat(out).numpy()

models = {ds: load_calibrated(ds) for ds in ["D1", "D2", "D3", "D4"]}

# ── Load the pHash group-split CSVs (same splits the models were trained/tested on)
def find_csv(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) + glob.glob(f"./{name}")
    assert hits, f"{name} not found — upload it to the Kaggle dataset"
    return pd.read_csv(hits[0], dtype={"phash": str})
splits = {ds: find_csv(f"{ds}_phash_group_split.csv") for ds in ["D1", "D2", "D3", "D4"]}
for ds, df in splits.items():
    missing = [p for p in df.path.head(20) if not os.path.exists(p)]
    assert not missing, f"{ds} image paths don't resolve, e.g. {missing[0]} — attach the raw dataset"

# ── Sanity gate: each model on its OWN test split must match the published number
PUBLISHED_IN_DOMAIN = {"D1": 0.9294, "D2": 0.9766, "D3": 0.9532, "D4": 0.9666}
for ds in ["D1", "D2", "D3", "D4"]:
    te = splits[ds][splits[ds].split == "test"]
    y = te.label.map({c: i for i, c in enumerate(CLASSES[ds])}).values
    acc = (predict_probs(*models[ds], te.path).argmax(1) == y).mean()
    ok = abs(acc - PUBLISHED_IN_DOMAIN[ds]) < 0.01
    print(f"[{ds}] in-domain test acc = {acc:.4f} on n={len(te)} "
          f"(published {PUBLISHED_IN_DOMAIN[ds]:.4f}) {'PASS' if ok else 'FAIL'}")
    assert ok, "Model/preprocessing mismatch — do NOT proceed"
