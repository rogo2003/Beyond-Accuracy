# Beyond Accuracy: A Reliability-Oriented Evaluation Framework for Brain Tumor MRI Classification Using Calibration, Explainability, and Cross-Dataset Auditing


**Principal author:** Ibrahim Hayatu Hassan

---

## Experimental Overview

This repository contains the code for the paper *"Beyond Accuracy: A Reliability-Oriented Evaluation Framework for Brain Tumor MRI Classification Using Calibration, Explainability, and Cross-Dataset Auditing"*. The framework looks beyond accuracy-only evaluation by combining:

- an **EfficientNetV2-B0** backbone with a **Dual-Pooling Attention (DPA)** head;
- **perceptual-hash (pHash) duplicate screening** with group-level stratified partitioning, so that detected near-duplicate groups do not cross the train/calibration/validation/test partitions;
- **post-hoc temperature scaling** fitted on a dedicated calibration partition;
- **bootstrap 95% confidence intervals** (B = 2,000) and Holm-adjusted McNemar tests against seven baseline backbones;
- **explanation assessment** (SHAP, Grad-CAM++, Integrated Gradients): top-10% IoU agreement and deletion/insertion AUC;
- **selective prediction** (risk–coverage) analysis;
- a **cross-dataset transfer audit** of preprocessing, input encoding and near-duplicate overlap between benchmarks.

**Scope of the evaluation.** Patient identifiers are not available in the public benchmark releases, so partitioning operates at the image and pHash-group level. It does not guarantee patient-level separation, and absence of a detected pHash match does not certify independence. The benchmarks also share much of their source material: 25.3–78.5% of each transfer's target test images have a pHash match to a source-development image. The within-dataset and cross-dataset results are therefore **not** evidence of generalization to independent patients, institutions or clinical settings, and the transfer experiments are audits, not external validation.

## Where the reported numbers come from

- **Revised manuscript.** Table 10, Tables 11–12, Table 14, Tables 15a–c, Figs. 1–9 and the supplementary material are produced by the scripts in [`audit/`](audit/README.md), not by the notebook JSON. Each GPU script first reproduces the previously reported numbers before applying a correction.
- **Original notebook.** [`beyond-accuracy.ipynb`](beyond-accuracy.ipynb) trains the models and writes `experiment_summary.json` per dataset; it is kept unchanged for provenance. Its cross-dataset evaluation ([`cross-gen.ipynb`](cross-gen.ipynb)) and its explainability code contain errors that are documented and corrected in [`audit/README.md`](audit/README.md).

See [`REPRODUCE.md`](REPRODUCE.md) for the step-by-step reproduction guide.

---

## Repository Structure

```
beyond-accuracy/
├── beyond-accuracy.ipynb       # Original training/evaluation notebook (Kaggle, GPU)
├── cross-gen.ipynb             # Original cross-dataset notebook (kept for provenance)
├── audit/                      # Code behind the corrections in the revised manuscript
│   ├── kaggle/                 #   GPU scripts (00–09)
│   ├── local/                  #   CPU scripts: tables, statistics, figures (10–14)
│   └── results/                #   Tables and figures of the revised manuscript
├── Output files/               # Calibrated checkpoints and *_phash_group_split.csv per dataset
├── New results/, results/      # Kaggle outputs consumed by audit/local
├── requirements.txt, environment.yml
└── REPRODUCE.md                # Step-by-step reproduction guide
```

---

## Datasets

| Key | Name | Classes | Source |
|-----|------|---------|--------|
| `D1` | Kaggle SARTAJ | 4 | [Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) |
| `D2` | BRISC-2025 | 4 | [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025) |
| `D3` | Figshare (Cheng 2017), Kaggle release | 3 | [Kaggle](https://www.kaggle.com/datasets/denizkavi1/brain-tumor) |
| `D4` | Nickparvar (combination of D1, D3 and Br35H) | 4 | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |

The Kaggle release of D3 stores each slice as a four-channel viridis pseudo-colour PNG rather than a grayscale image; see Section 3.1.1 of the paper and [`audit/README.md`](audit/README.md).

---

## Quickstart

### Option A — Kaggle (recommended, free T4 GPU)

1. Upload `beyond-accuracy.ipynb` to [kaggle.com/code](https://www.kaggle.com/code).
2. Attach the desired dataset (see table above).
3. Set `DATASET = "D1"` (or D2/D3/D4) and `SMOKE_TEST = False`.
4. Set the accelerator to **GPU T4 x2** and run all cells.

To reproduce the revised manuscript's tables and figures from the released checkpoints (no retraining), follow [`audit/README.md`](audit/README.md).

### Option B — Local / Colab

```bash
git clone https://github.com/rogo2003/Beyond-Accuracy.git
cd Beyond-Accuracy
conda env create -f environment.yml
conda activate beyond-accuracy
jupyter notebook beyond-accuracy.ipynb
```

Set `DATASET` and point `Config.DATASET_DIR` to your local data path.

**Key configuration variables** (top of notebook):

```python
DATASET    = "D2"    # "D1" | "D2" | "D3" | "D4"
SMOKE_TEST = False   # True → ~10-min sanity run; False → full ~12h canonical run
```

---

## pHash Duplicate Screening

All images are pooled across provider-defined splits, hashed with a 64-bit DCT perceptual hash (H = 8), linked at Hamming distance δ ≤ 6 and grouped (union-find) before any partition is formed. Each group is assigned to a single partition, so **detected** exact and near-duplicate images do not appear in more than one of the train/calibration/validation/test subsets. Related slices of the same patient, or similar images missed at this threshold, can still fall into different partitions; threshold sensitivity (δ = 0–10) is reported in Supplementary Table S1.

Three audit files are written per dataset run:

| File | Contents |
|------|----------|
| `<D>_phash_group_split.csv` | Per-image partition assignment and group ID |
| `<D>_phash_candidate_pairs.csv` | All exact and near-duplicate pairs with Hamming distances |
| `<D>_phash_dedup_report.json` | Summary statistics and the check that no detected group crosses partitions |

---

## Figures

The figures of the revised manuscript are written to [`audit/results/figures/`](audit/results/figures/) by `audit/local/12_figures.py` (Figs. 6–9, Supplementary Fig. S3) and by the Kaggle scripts listed in [`audit/README.md`](audit/README.md) (Figs. 1–5, Supplementary Figs. S1–S2).

Fig. 8 and Table 14 use the Grad-CAM++ coefficients of Chattopadhay et al. (2018), corrected in the second revision (`audit/kaggle/09_gradcampp_fix.py`).

## Citation

If you use this code, please cite:

```bibtex
@article{hassan2026beyondaccuracy,
  title   = {Beyond Accuracy: A Reliability-Oriented Evaluation Framework for Brain Tumor MRI Classification Using Calibration, Explainability, and Cross-Dataset Auditing},
  author  = {Hassan, Ibrahim Hayatu and Alhussian, Hitham and Abdulkadir, Said Jadid and Eltahir, Majdy Mohamed Eltayeb and Adamu, Shamsuddeen and Garba, Aliyu and Mukhtar, Abubakar and Ibrahim, Aliyu Tetengi and Abdullahi, Mohammed},
  year    = {2026},
  note    = {Manuscript under review, BMC Medical Imaging}
}
```

---

## License

This code is released for academic reproducibility. See [LICENSE](LICENSE) for terms.
