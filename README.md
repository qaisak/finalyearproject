# Breast Cancer Risk Modelling - METABRIC Multi-Modal Benchmark

This repo contains the code used to model breast cancer outcomes on the METABRIC cohort (from cBioPortal), with external validation on TCGA-BRCA (Pan-Cancer Atlas 2018). There are two sides to it: a **survival side** (time-to-event models) and a **classification side** (predicting 5-year mortality from clinical and molecular features).



---

## Data

All input files live under `Data/`:

- `data_clinical_patient.txt` - patient-level clinical and outcome table.
- `data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt` - mRNA expression matrix as z-scores. Around 300 MB, tracked through Git LFS.
- `data_mutations.txt` - somatic mutation calls.
- `tcga_brca_pan_can_atlas_2018/` - TCGA-BRCA clinical data used only for external validation.

---

## Setup

Python 3.10 or newer. From the repo root:

```bash
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# macOS/Linux:  source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`xgboost` is included in `requirements.txt` because it is needed for the primary Trimodal classification benchmark. A few things are optional and only imported if present:

- `lightgbm` - enables LightGBM in the extended classification benchmark.
- `shap` - enables SHAP feature-importance plots in `extras_consolidated.py` and `classification_consolidated.py`.
- `torch`, `torchtuples`, `pycox` - enable DeepSurv survival models (~1 GB install). The survival script runs fine without them; DeepSurv rows are simply skipped.

To install DeepSurv dependencies:

```bash
pip install torch torchtuples pycox
```

---

## Scripts

Everything runs from the repo root. Figures and CSVs are written into `outputs/`, created on first run.

### `survival_consolidated.py` - main survival pipeline (C1, C2, C3, C4, C5 time-AUC)

Runs a head-to-head benchmark of six survival model families across four modality configurations (Clinical, Clin+Mut, mRNA, Trimodal):

- **CoxPH** with ridge penaliser tuned via 3-fold CV
- **Parametric AFT** - best of Weibull, Log-Logistic, and Log-Normal
- **RSF** (Random Survival Forest) and **GBSA** (Gradient Boosting Survival Analysis)
- **DeepSurv** (optional, requires torch + pycox)
- **CoxPH late fusion** and **DeepSurv late fusion** on Trimodal - both equal-weight and training-C-weighted variants

Also produces: nested 5-fold CV stability check, bootstrap ΔC pairwise significance, per-patient risk-score overlap (C4), subgroup bootstrap for CoxPH and RSF (C3), time-dependent AUC across 1/2/3/5/10-year horizons (C5), calibration, and TCGA-BRCA external validation.

```bash
python survival_consolidated.py
# Faster smoke test (lighter CV, skips subgroup bootstrap / TCGA / figures):
python survival_consolidated.py --fast
# Or: SURVIVAL_FAST=1 python survival_consolidated.py
```

### `classification_consolidated.py` - 5-year mortality classification (C1)

Benchmarks six classifiers (LogReg, RandomForest, HistGB, AdaBoost, SVC-RBF, XGBoost) across Clinical / Clin+Mut / Trimodal datasets, plus Stacking and Voting ensembles on Trimodal. Uses PR-AUC as the primary metric under class imbalance.

Also produces: DeLong + bootstrap significance for ROC-AUC contrasts (C1), learning-curve bias–variance audit, calibration curves, SHAP for clinical XGBoost (optional), and per-subgroup ROC (PAM50, ER status, age bands).

```bash
python classification_consolidated.py
```

### `extras_consolidated.py` - PAM50 vs PCA and SHAP modality decomposition

Answers "where is the signal coming from?" two ways:

- **PAM50 vs PCA-20 comparison**: RSF trained on clinical + PCA-20 mRNA vs clinical + PAM50 raw genes vs clinical + PAM50 pathway scores (luminal / proliferation / HER2 / basal).
- **Modality-decomposed SHAP on RSF Trimodal**: top-20 feature summary plus per-modality mean |SHAP| (total and per-feature normalised), showing whether mutation or mRNA features individually contribute as much as clinical features.

```bash
python extras_consolidated.py
```

### `clinical_utility.py` - DCA, power analysis, and CCA redundancy (C1 power, C5 DCA)

Standalone script with three analyses:

- **CCA redundancy** (C5 context): for each PAM50 subtype, computes the first canonical correlation between clinical features and mRNA-PCA20, then regresses this against the per-subtype RSF ΔC. Tests whether subtypes where clinical and molecular data are more redundant show smaller multi-modal gain (R² ≈ 0.108).
- **Decision Curve Analysis** (C5): converts RSF 5-year survival functions to mortality probabilities and plots net benefit vs threshold (0.05–0.60) for clinical-only vs trimodal RSF. ΔNB ≈ +0.029 at threshold 0.30.
- **Power analysis** (C1, C2): Hanley–McNeil power for ΔC = 0.04 at the observed n = 1,980 and event rate, confirming 83.5% power and providing the sample-size bound on the non-significant modality result.

```bash
python clinical_utility.py
```

### `expanded_external_validation.py` - TCGA subtype and sample-size extension checks

Extends the external validation analysis with two focused experiments:

- **Subtype-level TCGA validation**: evaluates clinical and trimodal RSF performance across harmonised PAM50 subtype groups in TCGA and reports subgroup C-index with uncertainty.
- **n-test sensitivity checks**: compares external C-index trends as test-set size is varied, to check how stable performance is under smaller external cohorts.

Also writes comparison figures and CSV summaries for subtype and sample-size analyses.

```bash
python expanded_external_validation.py
```

### `pam50_external_test.py` - TCGA PCA-20 vs PAM50 representation check

Standalone external test of mRNA representation choice in TCGA:

- Compares **clinical + PCA-20 mRNA** against **clinical + PAM50-derived mRNA summaries** under the same RSF evaluation setup.
- Reports whether the PAM50-style representation preserves external discrimination relative to the PCA baseline.

```bash
python pam50_external_test.py
```

---

## Outputs

All results land in `outputs/`. Key files:

| File | Description |
|---|---|
| `survival_results_primary.csv` | Nested 5-fold CV C-index for all model × modality combinations |
| `bootstrap_significance_surv.csv` | Held-out bootstrap ΔC with 95% CIs and p-values (C2) |
| `subgroup_bootstrap_significance.csv` | Per-PAM50-subtype CoxPH ΔC (C3) |
| `subgroup_bootstrap_rsf.csv` | Per-PAM50-subtype RSF ΔC (C3) |
| `feature_overlap_analysis.csv` | Per-patient Spearman ρ clinical vs trimodal risk scores (C4) |
| `survival_time_dependent_metrics.csv` | AUC at 1/2/3/5/10 years per model (C5) |
| `dca_results.csv` | Net benefit curves (C5) |
| `power_analysis.csv` | Hanley–McNeil power at observed n (C1, C2) |
| `classification_results_primary.csv` | Held-out ROC/PR/F1/Brier for all classifiers |
| `bootstrap_significance_clf.csv` | DeLong + bootstrap AUC contrasts for classifiers (C1) |
| `pam50_geneset_comparison.csv` | PAM50 vs PCA-20 RSF C-index comparison |
| `cca_redundancy_scores.csv` | Per-subtype CCA redundancy score vs RSF ΔC |
| `shap_rsf_top_features.csv` | Top-20 features by mean |SHAP| for RSF Trimodal |
| `tcga_external_validation.csv` | TCGA-BRCA external C-index for clinical RSF |
| `expanded_external_tcga_subtype_validation.csv` | Subtype-level TCGA external validation summary |
| `expanded_external_tcga_n_test_comparison.csv` | TCGA n-test sensitivity comparison table |
| `expanded_external_tcga_trimodal_comparison.csv` | Clinical vs trimodal TCGA external comparison |
| `pam50_tcga_representation_comparison.csv` | TCGA PCA-20 vs PAM50 representation comparison |
| `expanded_external_tcga_n_test_comparison.png` | Plot of TCGA external C-index under varying test-set sizes |
| `expanded_external_tcga_subtype_cindex.png` | Subtype-level TCGA C-index comparison figure |
| `expanded_external_tcga_trimodal_comparison.png` | Figure comparing clinical and trimodal external TCGA performance |
| `pam50_tcga_representation_comparison.png` | Figure comparing PCA-20 and PAM50 mRNA representations on TCGA |

---

## Git LFS

The mRNA expression matrix exceeds GitHub's 100 MB file limit and is tracked with Git LFS. After cloning:

```bash
git lfs install
git lfs pull
```

---

## License and attribution

METABRIC and TCGA data are subject to the original studies and cBioPortal's terms of use. If you reuse anything here, please cite the original datasets and publications.
