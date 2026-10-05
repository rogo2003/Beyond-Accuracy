# Reproduction Guide

This document describes how to reproduce every table and figure in the manuscript from scratch.

---

## 1. Environment Setup

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate beyond-accuracy
```

### pip

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested on Python 3.10, CUDA 11.8, PyTorch 2.1. CPU-only runs are supported but the full canonical run will take 20–40× longer without a GPU.

---

## 2. Dataset Preparation

### On Kaggle (recommended)

Attach the relevant dataset to your notebook from the Kaggle dataset page. The loader will auto-detect the path. No manual path configuration is needed.

### Locally

Download and unzip the dataset, then update `Config.DATASET_DIR` in the notebook (or set a matching path in `DATASET_CONFIGS`). The loader supports three folder layouts:

```
Layout A (D1, D2, D4):          Layout C (D3):
  root/                            root/
    Training/                        1/
      glioma/                        2/
      meningioma/                    3/
      ...
    Testing/
      ...
```

---

## 3. Running the Notebook

### Step 1 — Set the run configuration

At the top of `beyond-accuracy.ipynb`, set exactly two variables:

```python
DATASET    = "D2"    # One of: "D1", "D2", "D3", "D4"
SMOKE_TEST = False   # False = full canonical run (~12h on Kaggle T4)
```

All other behaviour derives from these two settings. Outputs are written to `./outputs_pytorch/<DATASET>/` so all four dataset runs coexist without overwriting each other.

### Step 2 — Run all cells

Run all cells in order top to bottom. The pipeline executes the following stages automatically:

| Stage | Description | Key outputs |
|-------|-------------|-------------|
| 0 | Reproducibility seed | — |
| 1 | pHash screening & group split | `*_phash_group_split.csv`, `*_phash_dedup_report.json` |
| 2 | Image preprocessing | In-memory arrays |
| 3 | 5-fold cross-validation | CV accuracy mean ± std |
| 4 | Ablation study (5 configurations) | Ablation table |
| 5 | Final model training (SWA, progressive res) | `best_effnetv2_bt_pytorch.pth` |
| 6 | Temperature calibration on D_cal | Optimal T*, ECE before/after |
| 7 | Evaluation + bootstrap CIs | All Table 3–5 numbers |
| 8 | McNemar tests (Holm-Bonferroni) | p-values vs 7 baselines |
| 9 | XAI ensemble (SHAP, Grad-CAM++, IG) | `xai_ensemble_pytorch.pdf` |
| 10 | JSON summary | `experiment_summary.json` |

### Step 3 — Verify outputs

After the run completes, check `outputs_pytorch/<DATASET>/experiment_summary.json`. Every number reported in the manuscript is a key in this file. Cross-reference with the printed log in `experiment_results.txt`.

---

## 4. Expected Runtimes (Kaggle T4 GPU)

| Dataset | Canonical run | Smoke test |
|---------|--------------|------------|
| D1 | ~10–12 h | ~10 min |
| D2 | ~12–14 h | ~12 min |
| D3 | ~8–10 h  | ~8 min  |
| D4 | ~12–14 h | ~12 min |

Smoke test settings: 5 epochs, 2 CV folds, 100 bootstrap resamples, no baselines in full. Use `SMOKE_TEST = True` to verify the pipeline runs end-to-end before committing to the full run.

---

## 5. Verifying the pHash Leakage Check

After any run, open `outputs_pytorch/<DATASET>/<DATASET>_phash_dedup_report.json` and confirm:

```json
"leakage_check": "PASS: no pHash duplicate group crosses train/cal/val/test partitions"
```

If this field is missing or does not say PASS, the split integrity assertion in `_stratified_group_split_records` would have raised a `RuntimeError` and halted the run.

---

## 6. Loading a Saved Model for Inference

```python
import torch
from beyond_accuracy import BrainTumorModel, TemperatureScaler, Config

def strip_swa(sd):  # checkpoints are saved from torch.optim.swa_utils.AveragedModel
    return {k.removeprefix("module."): v for k, v in sd.items() if k != "n_averaged"}

# Load model weights only
model = BrainTumorModel(num_classes=Config.NUM_CLASSES).to("cuda")
model.load_state_dict(strip_swa(torch.load("outputs_pytorch/D2/final_model_v5_pytorch.pth")))
model.eval()

# Load calibrated model (SWA weights + optimal temperature T*).
# Without strip_swa, load_state_dict fails with "Missing key(s)".
checkpoint = torch.load("outputs_pytorch/D2/final_model_v5_calibrated_pytorch.pth", weights_only=False)
model.load_state_dict(strip_swa(checkpoint["model_state_dict"]), strict=True)
calibrated = TemperatureScaler(model).to("cuda")
calibrated.temperature.data.fill_(checkpoint["temperature"])
calibrated.eval()
```

> **Note:** `checkpoint["temperature"]` holds the originally fitted value (0.998 / 0.907 / 0.877 / 0.900 for D1–D4), which did not converge. The revised manuscript uses the converged temperatures 0.8752 / 0.7446 / 0.7351 / 0.7368; fill those in instead, or re-fit with `audit/kaggle/05_recalibrate.py`.

---

## 7. Reproducing Individual Tables

| Manuscript table | JSON key | Notes |
|-----------------|----------|-------|
| Table 2 (dataset splits) | `splits` | Also in `*_phash_group_split.csv` |
| Table 3 (main results) | `test_calibrated`, `bootstrap_ci_calibrated` | Point estimates + 95% CI |
| Table 4 (ablation) | `ablation` | 5 rows |
| Table 5 (baseline comparison) | `extra.baseline_results`, `mcnemar_holm` | Includes Holm-adjusted p |
| Table 6 (calibration) | `calibration` | ECE before/after, optimal T* |
| Table 7 (XAI agreement) | `xai.iou_per_class`, `xai.deletion_insertion_auc` | Original submission only (one image per class; see Section 9) |

In the **revised manuscript**, Table 10 (per-class metrics), Tables 11–12 (calibration), Table 14 (XAI agreement), Tables 15a–c (cross-dataset audit), Figs. 1–9 and Supplementary Tables S1–S2 and Figs. S1–S3 are produced by the scripts in [`audit/`](audit/README.md), not by the notebook JSON.

---

## 8. Common Issues

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `FileNotFoundError: No images found` | Dataset not attached / wrong path | Re-attach dataset on Kaggle or update `Config.DATASET_DIR` |
| `RuntimeError: Duplicate groups were split across partitions` | Hash collision edge case | Should never happen; if it does, file an issue with the dataset key |
| `thop.profile failed` | Model wrapper prevents FLOPs counting | The fallback estimate table activates automatically; no action needed |
| `CUDA out of memory` | Batch size too large | Reduce `Config.BATCH_SIZE` to 16 |
| Blank Grad-CAM++ maps | Missing ReLU on the gradients in the Grad-CAM++ channel weights (or hook on the wrong layer) | Use the corrected implementation in `audit/kaggle/03_xai_rerun.py`; the notebook targets `backbone.blocks[-1]`, check the `[GradCAM] Target:` print line |
| `Missing key(s) in state_dict` when loading a checkpoint | SWA `AveragedModel` keys (`module.` prefix, `n_averaged`) | See Section 6 |

---

## 9. Revision Audit (cross-dataset transfer and explainability)

The revised manuscript corrects three parts of the original analysis:

- the cross-dataset evaluation in `cross-gen.ipynb`, which omitted the training preprocessing and did not account for the viridis pseudo-colour encoding of the Kaggle D3 release;
- the explainability code in `beyond-accuracy.ipynb` (Grad-CAM++ gradient rectification, one-image IoU, and a shared SHAP map in deletion–insertion);
- the temperature fit in `TemperatureScaler.set_temperature`, whose single L-BFGS step does not converge (the revised results use a converged bounded minimisation).

The code for these corrections, with built-in checks that first reproduce the originally published numbers, is in [`audit/`](audit/README.md):

| Step | Script | Manuscript items |
|------|--------|------------------|
| GPU (Kaggle) | `audit/kaggle/00_load_models.py` → `01`, `02`, `03` | Tables 14, 15b–c; Figs. 7–9; Supplementary Figs. S2–S3; Supplementary Table S2 |
| GPU (Kaggle) | `audit/kaggle/04`, `04b`, `05` (then re-run `01`, `03`) | Converged temperature scaling: Tables 11–12, Fig. 6, selective prediction; Supplementary Fig. S1 |
| CPU (local) | `audit/local/13_phash_threshold_sensitivity.py` | Supplementary Table S1 (δ = 0–10) |
| CPU (local) | `audit/local/10_overlap_table14a.py` | Table 15a |
| CPU (local) | `audit/local/11_tables_and_stats.py` | Tables 14, 15b, 15c and all quoted counts (prints `ALL PASS`) |
| CPU (local) | `audit/local/12_figures.py` | Figs. 6, 7, 8, 9 and Supplementary Fig. S3 |
| GPU (Kaggle) | `audit/kaggle/00` → `06_per_class_metrics.py` | Table 10 (per-class specificity and AUC); Figs. 4–5 |
| GPU (Kaggle) | `audit/kaggle/00` → `07_training_history.py` | Fig. 3 (logged re-run of the final-model training) |
| CPU (Kaggle) | `audit/kaggle/08_figure1_dataset_examples.py` | Fig. 1 |
| CPU (local) | `audit/local/14_figure2_flowchart.py` | Fig. 2 |

Script and output file names containing `table14` keep the numbering of the original submission (now Tables 15a–c).
