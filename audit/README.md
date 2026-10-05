# Audit and re-analysis code (revised manuscript)

This folder contains the code behind the corrections made in the revised manuscript:

- the cross-dataset transfer audit (Section 3.10, Section 4.10, Tables 15a–c, Fig. 9, Supplementary Fig. S2, Supplementary Table S2);
- the corrected explainability evaluation (Section 3.7, Section 4.9, Table 14, Figs. 7–8, Supplementary Fig. S3);
- the corrected temperature scaling and the hyperparameter sensitivity analyses (Sections 3.5 and 4.7–4.7.1, Tables 11–12, Fig. 6, Supplementary Table S1, Supplementary Fig. S1);
- the per-class metrics, learning curves and figures regenerated in the final audit (Table 10, Figs. 1–5).

Script and output file names (e.g. `table14_rerun_*`, `10_overlap_table14a.py`) keep the table numbering of the original submission; the corresponding tables in the revised manuscript are Tables 15a–c (transfer audit) and Table 14 (IoU agreement).

Every GPU script first **reproduces the originally published numbers** and only then applies a correction, so each change can be traced.

## Contents

| Script | Where | Reproduces / produces | Built-in check |
|---|---|---|---|
| `kaggle/00_load_models.py` | Kaggle (GPU) | Loads the four calibrated checkpoints | In-domain test accuracy must match the paper: 92.94 / 97.66 / 95.32 / 96.66% |
| `kaggle/01_table14_transfer_audit.py` | Kaggle (GPU) | All six transfers under the original, training and harmonised pipelines, stratified by near-duplicate exposure → `table14_rerun_results.csv`, `table14_rerun_predictions.csv.gz` | The original pipeline must reproduce the six transfer accuracies published in Table 14 of the original submission (79.30, 63.55, 27.42, 31.77, 73.00, 69.25%) |
| `kaggle/02_d3_pair_diagnostic.py` | Kaggle (GPU) | D4 model on 300 near-identical D3/D4 pairs → `D3D4_pair_diagnostic.csv/.png` (Supplementary Fig. S2) | Expected: D4 copies 100.0%, D3 copies 43.3%, D3 rescaled 89.7% |
| `kaggle/03_xai_rerun.py` | Kaggle (GPU) | SHAP, corrected Grad-CAM++, IG, IoU and deletion–insertion on 50 correctly classified test images per class → `xai_rerun_*.csv`, `D*_xai_examples.png`, `D*_deletion_insertion.png` | The original Grad-CAM++ must reproduce the published Grad-CAM++–IG IoU for the first test image of every class (15/15) |
| `kaggle/04_temperature_and_duplicate_examples.py` | Kaggle (GPU) | NLL/ECE vs temperature T ∈ [0.5, 2.0] on calibration and test splits; example image pairs at δ = 0, 3, 6, 7, 8 | Minimum of calibration NLL must reproduce the stored T* (±0.02) |
| `kaggle/04b_temperature_refit.py` | Kaggle (GPU) | Replays the notebook's L-BFGS temperature fit, re-fits T to convergence, and reports test ECE/NLL/Brier at T = 1, stored T* and converged T | Replay must reproduce the stored T*; ECE at T = 1 and T* must reproduce the published values |
| `kaggle/05_recalibrate.py` | Kaggle (GPU) | Re-fits T to convergence; Table 11, Table 12 (Brier/AUC rows), Fig. 6 reliability diagrams, selective prediction. Then switches `models` to the converged T, so re-running `01` and `03` afterwards updates Table 15b ECE/Brier and the deletion–insertion AUCs | Bootstrap at the stored T* must reproduce published Table 12 values |
| `kaggle/06_per_class_metrics.py` | Kaggle (GPU) | Per-class precision, recall, specificity, F1 and one-vs-rest AUC (Table 10); confusion matrices (Fig. 4) and empirical ROC curves (Fig. 5) from the released checkpoints at the converged temperatures | Macro AUC must reproduce Tables 11–12 (0.98961 / 0.99722 / 0.99079 / 0.99399) |
| `kaggle/07_training_history.py` | Kaggle (GPU, Internet ON) | Re-run of the final-model training (identical code, seed and hyperparameters) with per-epoch logging → `training_history_D*.csv`, Fig. 3 | Reports re-run test accuracy and SWA snapshot count against the released checkpoints (92.94 / 97.66 / 95.32 / 96.66%; 15 / 12 / 25 / 17) |
| `kaggle/08_figure1_dataset_examples.py` | Kaggle (CPU) | Fig. 1: one test-partition image per class for D1–D4, as released (D3 in viridis), with the selected file list | — |
| `local/10_overlap_table14a.py` | Local (CPU) | Table 15a (pairwise near-duplicate overlap) from the `*_phash_group_split.csv` files | All 12 cells must match the manuscript |
| `local/11_tables_and_stats.py` | Local (CPU) | Tables 14, 15b, 15c; Wilcoxon/Holm tests for IoU and deletion–insertion | 13 checks against every count and range quoted in the text |
| `local/12_figures.py` | Local (CPU) | Figs. 6, 7, 8, 9 and Supplementary Fig. S3 | — |
| `local/13_phash_threshold_sensitivity.py` | Local (CPU) | Duplicate grouping for δ = 0…10 (groups, largest group, label-mixed groups) | δ = 6 must reproduce the notebook's group counts (1,920 / 3,861 / 1,593 / 3,659) |
| `local/14_figure2_flowchart.py` | Local (CPU) | Fig. 2 framework flowchart at printed size (17 cm wide; Times New Roman 9/10 pt) as SVG, PDF and PNG | Checks that every label fits inside its box |

## How to run

### 1. Kaggle (GPU, about 1–1.5 h in total on a T4)

Attach to one notebook:
- a dataset containing `D1–D4_final_model_v5_calibrated_pytorch.pth` and `D1–D4_phash_group_split.csv` (all in `Output files/` of this repository);
- the raw datasets: `sartajbhuvaji/brain-tumor-classification-mri`, `briscdataset/brisc2025`, `denizkavi1/brain-tumor`, `masoudnickparvar/brain-tumor-mri-dataset`.

Paste the scripts into cells and run them in order: `00` first, then `01`, `02`, `03`, `04` and `04b`; for the corrected calibration run `05` and then `01` and `03` again (each is independent of the others, but all reuse the objects created by `00`). Download the outputs from `/kaggle/working/`.

### 2. Local (CPU, a few minutes)

Put the Kaggle outputs in the repository root, then:

```bash
python audit/local/10_overlap_table14a.py      # Table 15a
python audit/local/11_tables_and_stats.py      # Tables 14, 15b, 15c + checks → "ALL PASS"
python audit/local/12_figures.py               # Figs. 6, 7, 8, 9, S3
python audit/local/13_phash_threshold_sensitivity.py   # pHash threshold sensitivity
python audit/local/14_figure2_flowchart.py audit/results   # Fig. 2 flowchart
```

Outputs go to `audit/results/`. Requires `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib` and `pillow`.

## What the audit found

**Cross-dataset transfer.** The original evaluation (`cross-gen.ipynb`) (1) omitted the brain-cropping and median-filtering steps used in training, and (2) did not account for the Kaggle release of D3 (`denizkavi1/brain-tumor`) being a four-channel **viridis pseudo-colour** rendering rather than grayscale MRI. After correcting both, the six transfers reach 94.99–99.67%, within a few points of in-domain accuracy. Between 25.3% and 78.5% of each transfer's test images are near-duplicates of images the source model was trained, validated or calibrated on, so these transfers are not external validation.

**Explainability.** The original notebook (1) computed Grad-CAM++ without the ReLU on the gradients in the channel weights, (2) computed the cross-method IoU (now Table 14) from one test image per class, and (3) ranked all 50 images per class by one class-representative SHAP map in the deletion–insertion analysis. After correction:
- SHAP–IG agreement exceeds chance in 15 of 15 class–dataset combinations, but Grad-CAM++ agrees with the other methods at close to chance level;
- insertion supports the SHAP rankings in 12 of 15 combinations, deletion in only 6 of 15.

**Calibration.** The notebook fitted the temperature with a single L-BFGS step (T₀ = 1.5, lr 0.01), which stopped before the NLL minimum; replaying it reproduces the stored T* exactly. The converged temperatures (0.875 / 0.745 / 0.735 / 0.737) lower test ECE on every dataset (0.0129–0.0288) without changing accuracy. `05_recalibrate.py` applies them; `01` and `03` were then re-run so that all calibration-dependent results use the same T.

**Hyperparameter sensitivity.** `13_phash_threshold_sensitivity.py` (δ = 0–10) and `04_temperature_and_duplicate_examples.py` (T ∈ [0.5, 2.0]; example pairs) provide Section 4.7.1, Fig. 6b–c, Supplementary Table S1 and Supplementary Fig. S1. The label-smoothing factor was not varied.

## Implementation notes

- **Checkpoints.** The `*_calibrated_pytorch.pth` files are `torch.optim.swa_utils.AveragedModel` state dicts. Strip the `module.` prefix and drop `n_averaged` before loading them into `BrainTumorModel` (done in `00_load_models.py`).
- **Class order.** Classes follow scikit-learn `LabelEncoder` (alphabetical) order: D3 = glioma, meningioma, pituitary; D1, D2 and D4 = glioma, meningioma, no_tumor, pituitary.
- **pHash values.** The 64-bit pHash values exceed the int64 range, so read the split CSVs with `dtype={"phash": str}`.
- **The original notebooks** (`beyond-accuracy.ipynb`, `cross-gen.ipynb`) are kept unchanged for provenance, except for a note at the top of each pointing here.
