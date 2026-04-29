"""Clinical utility add-ons: CCA redundancy vs RSF delta, DCA on the 80/20 split,
and a quick power check. Loaders are inlined so this file runs standalone."""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

import os
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
HORIZON = 60  # 5-year horizon in months

# clinical column renames (same mapping as extras script)
CLINICAL_RENAME_MAP = {
    "AGE_AT_DIAGNOSIS": "age_at_diagnosis",
    "OS_MONTHS": "overall_survival_months",
    "LYMPH_NODES_EXAMINED_POSITIVE": "lymph_nodes_examined_positive",
    "NPI": "nottingham_prognostic_index",
    "CELLULARITY": "cellularity",
    "CHEMOTHERAPY": "chemotherapy",
    "ER_IHC": "er_status_measured_by_ihc",
    "HER2_SNP6": "her2_status_measured_by_snp6",
    "HORMONE_THERAPY": "hormone_therapy",
    "INFERRED_MENOPAUSAL_STATE": "inferred_menopausal_state",
    "INTCLUST": "integrative_cluster",
    "CLAUDIN_SUBTYPE": "pam50_+_claudin-low_subtype",
    "THREEGENE": "3-gene_classifier_subtype",
    "LATERALITY": "primary_tumor_laterality",
    "RADIO_THERAPY": "radio_therapy",
    "HISTOLOGICAL_SUBTYPE": "tumor_other_histologic_subtype",
    "BREAST_SURGERY": "type_of_breast_surgery",
    "COHORT": "cohort",
}

SURVIVAL_LEAK_COLS = {
    "overall_survival_months",
    "overall_survival",
    "death_from_cancer",
    "event",
    "os_months",
    "OS_MONTHS",
    "OS_STATUS",
    "RFS_MONTHS",
    "RFS_STATUS",
    "VITAL_STATUS",
}

CLASSIFICATION_LEAKAGE_COLS = {
    "overall_survival_months",
    "overall_survival",
    "death_from_cancer",
    "OS_STATUS_BIN",
    "OS_EVENT",
    "OS_STATUS",
    "VITAL_STATUS",
    "RFS_MONTHS",
    "RFS_STATUS",
    "SEX",
    "patient_id",
    "cancer_type",
    "cancer_type_detailed",
    "cohort",
    "integrative_cluster",
    "event",
    "os_months",
}

# RSF defaults aligned with survival run
RSF_PARAMS = {
    "n_estimators": 200,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
}

# loaders (inlined from extras)
def parse_os_status(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    if text in {"1", "1:DECEASED", "DECEASED"}:
        return 1.0
    if text in {"0", "0:LIVING", "LIVING"}:
        return 0.0
    return np.nan


def load_clinical_table(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    clinical_path = data_dir / "data_clinical_patient.txt"
    clin = pd.read_csv(clinical_path, sep="\t", comment="#", dtype=str)
    clin.columns = clin.columns.str.strip()
    for column in clin.columns:
        if clin[column].dtype == object:
            clin[column] = clin[column].str.strip()
    clin = clin.set_index("PATIENT_ID")
    clin.index = clin.index.str.upper()

    if "ER_IHC" in clin.columns:
        clin["ER_IHC"] = clin["ER_IHC"].replace({"Positve": "Positive"})

    clin = clin.rename(columns=CLINICAL_RENAME_MAP)
    clin["overall_survival"] = (
        clin["OS_STATUS"].apply(parse_os_status) if "OS_STATUS" in clin.columns else np.nan
    )
    clin["overall_survival_months"] = pd.to_numeric(
        clin["overall_survival_months"], errors="coerce"
    )
    clin["death_from_cancer"] = clin["overall_survival"].map(
        {1.0: "Died of Disease", 0.0: "Living"}
    )
    clin["event"] = clin["overall_survival"].astype(float)
    clin["os_months"] = clin["overall_survival_months"].astype(float)

    for column in [
        "age_at_diagnosis",
        "nottingham_prognostic_index",
        "lymph_nodes_examined_positive",
    ]:
        if column in clin.columns:
            clin[column] = pd.to_numeric(clin[column], errors="coerce")

    return clin


def load_mrna_matrix(data_dir: Path = DATA_DIR, prefix: str = "MRNA__") -> pd.DataFrame:
    mrna_path = data_dir / "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt"
    mrna_raw = pd.read_csv(mrna_path, sep="\t", low_memory=False)

    if "Hugo_Symbol" in mrna_raw.columns:
        mrna_raw = mrna_raw.drop(columns=["Entrez_Gene_Id"], errors="ignore")
        mrna_raw = mrna_raw.groupby("Hugo_Symbol").mean(numeric_only=True)
        mrna = mrna_raw.T
    else:
        mrna = mrna_raw.set_index(mrna_raw.columns[0]).T

    mrna.index = mrna.index.astype(str).str.strip().str.upper()
    mrna = mrna.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    mrna.columns = [f"{prefix}{column}" for column in mrna.columns]
    return mrna


def load_mutation_features(
    data_dir: Path = DATA_DIR,
    min_patient_count: int = 10,
    prefix: str = "MUT__",
) -> pd.DataFrame:
    mutation_path = data_dir / "data_mutations.txt"
    mut_raw = pd.read_csv(mutation_path, sep="\t", comment="#", dtype=str, low_memory=False)
    mut_raw.columns = mut_raw.columns.str.strip()

    silent = {"Silent", "synonymous_variant", "Synonymous", "Synonymous SNV"}
    mut_non_silent = mut_raw[~mut_raw["Variant_Classification"].isin(silent)].copy()
    mut_non_silent["pid"] = mut_non_silent["Tumor_Sample_Barcode"].str.strip().str.upper()

    gene_counts = mut_non_silent.groupby("Hugo_Symbol")["pid"].nunique()
    selected_genes = gene_counts[gene_counts >= min_patient_count].sort_values(ascending=False).index

    patient_gene = mut_non_silent.groupby(["pid", "Hugo_Symbol"]).size().unstack(fill_value=0)
    patient_gene = (patient_gene > 0).astype(np.int8)
    patient_gene = patient_gene[[gene for gene in selected_genes if gene in patient_gene.columns]]

    burden = mut_non_silent.groupby("pid").agg(
        mut_total=("Hugo_Symbol", "size"),
        mut_unique=("Hugo_Symbol", "nunique"),
    )
    burden["mut_log_burden"] = np.log1p(burden["mut_total"])

    mutation = patient_gene.join(
        burden[["mut_total", "mut_unique", "mut_log_burden"]], how="outer"
    ).fillna(0)
    mutation.columns = [f"{prefix}{column}" for column in mutation.columns]
    mutation.index = mutation.index.str.upper()
    return mutation


def get_survival_feature_columns(clin: pd.DataFrame) -> list[str]:
    return [column for column in clin.columns if column not in SURVIVAL_LEAK_COLS]


class TopVarianceSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 5000):
        self.k = k

    def fit(self, X, y=None):
        variances = np.nanvar(np.asarray(X), axis=0)
        keep = min(self.k, len(variances))
        self.selected_ = np.argsort(variances)[-keep:]
        self.selected_.sort()
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.selected_]


def encode_and_scale_clinical(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [column for column in X_train.columns if column not in cat_cols]

    X_train = X_train.copy()
    X_test = X_test.copy()
    for column in num_cols:
        X_train[column] = pd.to_numeric(X_train[column], errors="coerce")
        X_test[column] = pd.to_numeric(X_test[column], errors="coerce")

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

        encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        encoded_train = pd.DataFrame(
            encoder.fit_transform(X_train[cat_cols]),
            columns=encoder.get_feature_names_out(cat_cols),
            index=X_train.index,
        )
        encoded_test = pd.DataFrame(
            encoder.transform(X_test[cat_cols]),
            columns=encoder.get_feature_names_out(cat_cols),
            index=X_test.index,
        )
        X_train = pd.concat([X_train[num_cols], encoded_train], axis=1)
        X_test = pd.concat([X_test[num_cols], encoded_test], axis=1)

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train, X_test


def preprocess_mutation(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_np = scaler.transform(imputer.transform(X_test))
    return (
        pd.DataFrame(X_train_np, columns=X_train.columns, index=X_train.index),
        pd.DataFrame(X_test_np, columns=X_test.columns, index=X_test.index),
    )


def preprocess_mrna(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("topvar", TopVarianceSelector(k=5000)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, svd_solver="randomized", random_state=SEED)),
        ]
    )
    train_np = pipeline.fit_transform(X_train.values)
    test_np = pipeline.transform(X_test.values)
    columns = [f"MRNA_PC{i + 1}" for i in range(train_np.shape[1])]
    return (
        pd.DataFrame(train_np, columns=columns, index=X_train.index),
        pd.DataFrame(test_np, columns=columns, index=X_test.index),
    )


def make_surv_array(E: np.ndarray, T: np.ndarray):
    return Surv.from_arrays(event=E.astype(bool), time=T.astype(float))


# Helper: encode a single feature matrix (no train/test split needed for CCA)
def encode_clinical_single(X: pd.DataFrame) -> np.ndarray:
    """Encode categoricals, impute, and scale a clinical feature matrix in one shot."""
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    X = X.copy()
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    parts = []
    if num_cols:
        num_imp = SimpleImputer(strategy="median")
        parts.append(pd.DataFrame(
            num_imp.fit_transform(X[num_cols]),
            columns=num_cols,
            index=X.index,
        ))

    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        X[cat_cols] = cat_imp.fit_transform(X[cat_cols])
        enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        parts.append(pd.DataFrame(
            enc.fit_transform(X[cat_cols]),
            columns=enc.get_feature_names_out(cat_cols),
            index=X.index,
        ))

    X_enc = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)

    scaler = StandardScaler()
    return scaler.fit_transform(X_enc.values)


def compute_mrna_pca_single(X_mrna: pd.DataFrame, n_components: int = 20) -> np.ndarray:
    """Fit PCA pipeline on the given mRNA matrix and return transformed array."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("topvar", TopVarianceSelector(k=5000)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, svd_solver="randomized", random_state=SEED)),
    ])
    return pipe.fit_transform(X_mrna.values)


# ANALYSIS 1: CCA Redundancy
def run_cca_redundancy() -> None:
    print("\nCCA redundancy")

    # -- Load METABRIC data --------------------------------------------------
    print("  Loading clinical data...")
    clin = load_clinical_table()
    clin = clin.dropna(subset=["event", "os_months"]).copy()
    clin = clin[clin["os_months"] > 0].copy()

    print("  Loading mRNA matrix...")
    mrna = load_mrna_matrix()

    common_ids = sorted(set(clin.index) & set(mrna.index))
    print(f"  Patients with both modalities: {len(common_ids)}")

    clin_aligned = clin.loc[common_ids]
    mrna_aligned = mrna.loc[common_ids]

    # Clinical feature columns (no leakage)
    clin_feat_cols = get_survival_feature_columns(clin_aligned)
    # Also remove the subtype column itself to avoid using it as a feature
    clin_feat_cols = [c for c in clin_feat_cols if c != "pam50_+_claudin-low_subtype"]

    # Load RSF subgroup bootstrap results
    rsf_path = OUTPUT_DIR / "subgroup_bootstrap_rsf.csv"
    rsf_df = pd.read_csv(rsf_path)
    print(f"  RSF deltas loaded: {len(rsf_df)} subtypes")

    subtype_col = "pam50_+_claudin-low_subtype"
    subtypes = rsf_df["subtype"].tolist()

    redundancy_met = {}

    for subtype in subtypes:
        mask = clin_aligned[subtype_col].astype(str).str.strip() == subtype
        n_sub = mask.sum()
        print(f"  Subtype {subtype:15s}: n={n_sub}", end="")

        if n_sub < 30:
            print(" -- too few patients, skipping")
            continue

        X_clin_raw = clin_aligned.loc[mask, clin_feat_cols].copy()
        X_mrna_raw = mrna_aligned.loc[mask].copy()

        # Encode clinical
        X_clin_enc = encode_clinical_single(X_clin_raw)

        # mRNA PCA
        X_mrna_pca = compute_mrna_pca_single(X_mrna_raw, n_components=20)

        # Drop any rows with NaN in clinical encoding (should be none after imputation)
        valid = ~(np.isnan(X_clin_enc).any(axis=1) | np.isnan(X_mrna_pca).any(axis=1))
        if valid.sum() < 20:
            print(" -- not enough valid rows, skipping")
            continue

        X_c = X_clin_enc[valid]
        X_m = X_mrna_pca[valid]

        # Fit CCA with 1 component
        cca = CCA(n_components=1, max_iter=1000)
        try:
            cca.fit(X_c, X_m)
            U, V = cca.transform(X_c, X_m)
            # Pearson correlation of the first canonical variates
            r = float(np.corrcoef(U[:, 0], V[:, 0])[0, 1])
        except Exception as exc:
            print(f" -- CCA failed: {exc}")
            continue

        redundancy_met[subtype] = r
        print(f", CCA r={r:.4f}")

    # Merge with RSF deltas
    rows = []
    for _, row in rsf_df.iterrows():
        st = row["subtype"]
        rows.append({
            "Subtype": st,
            "Redundancy_METABRIC": redundancy_met.get(st, np.nan),
            "Delta_RSF": row["delta"],
        })

    result_df = pd.DataFrame(rows).dropna(subset=["Redundancy_METABRIC"])
    print(f"\n  Subtypes with valid CCA scores: {len(result_df)}")

    # -- Try TCGA -----------------------------------------------------------
    tcga_redundancy = {}
    tcga_cache_dir = DATA_DIR / "tcga_brca_pan_can_atlas_2018"
    harm_path = tcga_cache_dir / "tcga_brca_harmonised.csv"

    if harm_path.exists():
        print("\n  TCGA harmonised data found; attempting CCA on TCGA subtypes...")
        try:
            tcga_harm = pd.read_csv(harm_path, index_col=0)
            print(f"  TCGA patients: {len(tcga_harm)}")

            if "subtype" in tcga_harm.columns:
                # Map TCGA subtype names (BRCA_LumA -> LumA etc.)
                tcga_harm["subtype_short"] = (
                    tcga_harm["subtype"]
                    .str.replace("BRCA_", "", regex=False)
                    .str.strip()
                )
                # TCGA harmonised only has a small number of clinical features;
                # we'll use what's available for CCA_clin side
                feat_cols_tcga = [c for c in tcga_harm.columns
                                  if c not in {"os_months", "event", "subtype", "subtype_short"}]
                print(f"  TCGA feature columns: {feat_cols_tcga}")

                tcga_subtypes = tcga_harm["subtype_short"].dropna().unique()
                # Load TCGA raw for mRNA if we want, but harmonised has no mRNA.
                # We only have clinical features in the harmonised file.
                # CCA needs two views — we cannot do CCA without mRNA for TCGA.
                # Note to user: TCGA harmonised file has no mRNA data.
                print("  NOTE: TCGA harmonised data has no mRNA; CCA needs two views.")
                print("        Skipping TCGA CCA (insufficient data for two-view analysis).")
        except Exception as exc:
            print(f"  Warning: could not load TCGA data: {exc}")
    else:
        print("  TCGA harmonised data not found; skipping TCGA CCA.")

    # Add optional TCGA column (NaN if not available)
    result_df["Redundancy_TCGA"] = result_df["Subtype"].map(tcga_redundancy)

    # Save CSV
    out_csv = OUTPUT_DIR / "cca_redundancy_scores.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    # -- Scatter plot --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    x = result_df["Redundancy_METABRIC"].values
    y = result_df["Delta_RSF"].values
    labels = result_df["Subtype"].values

    ax.scatter(x, y, s=90, color="steelblue", zorder=3, label="METABRIC")

    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), textcoords="offset points",
                    xytext=(6, 3), fontsize=9)

    # Linear regression trendline
    if len(x) >= 2:
        slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
        r2 = r_val ** 2
        x_line = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
        ax.plot(x_line, slope * x_line + intercept, "r--", lw=1.5,
                label=f"Linear fit (R²={r2:.3f})")
        print(f"\n  Linear regression: slope={slope:.4f}, intercept={intercept:.4f}, "
              f"R²={r2:.4f}, p={p_val:.4f}")
    else:
        r2 = np.nan
        print("  Not enough points for regression.")

    # Overlay TCGA if available
    if tcga_redundancy:
        tcga_pts = result_df.dropna(subset=["Redundancy_TCGA"])
        ax.scatter(
            tcga_pts["Redundancy_TCGA"].values,
            tcga_pts["Delta_RSF"].values,
            s=90, marker="^", color="darkorange", zorder=3, label="TCGA"
        )

    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("CCA Redundancy Score (first canonical correlation)", fontsize=11)
    ax.set_ylabel("RSF Multi-modal C-index Delta", fontsize=11)
    ax.set_title("Clinical–mRNA Redundancy vs. Multi-modal Gain\nby PAM50 Subtype", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_png = OUTPUT_DIR / "cca_redundancy_vs_delta.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_png}")

    print(f"\n  CCA Redundancy scores (METABRIC):")
    for _, row in result_df.iterrows():
        print(f"    {row['Subtype']:15s}: r={row['Redundancy_METABRIC']:.4f}, "
              f"Delta_RSF={row['Delta_RSF']:.4f}")
    print(f"  R² (redundancy vs. delta) = {r2:.4f}")

    return result_df, r2


# ANALYSIS 2: Decision Curve Analysis
def compute_net_benefit(
    y_true: np.ndarray,
    prob_positive: np.ndarray,
    threshold: float,
    n: int,
) -> float:
    """Compute net benefit at a given threshold probability."""
    predicted_positive = prob_positive >= threshold
    tp = np.sum(predicted_positive & y_true)
    fp = np.sum(predicted_positive & ~y_true)
    nb = (tp / n) - (fp / n) * (threshold / (1.0 - threshold))
    return float(nb)


def run_dca() -> None:
    print("\nDecision curve (DCA)")

    # -- Load data -----------------------------------------------------------
    print("  Loading data for DCA...")
    clin = load_clinical_table()
    clin = clin.dropna(subset=["event", "os_months"]).copy()
    clin = clin[clin["os_months"] > 0].copy()

    mutation = load_mutation_features()
    mrna = load_mrna_matrix()

    clinical_columns = get_survival_feature_columns(clin)
    clin_features = clin[clinical_columns].copy()

    core_ids = sorted(set(clin.index) & set(mutation.index) & set(mrna.index))
    print(f"  Patients in trimodal intersection: {len(core_ids)}")

    T_all = clin.loc[core_ids, "os_months"].astype(float).values
    E_all = clin.loc[core_ids, "event"].astype(int).values

    X_clin_raw = clin_features.loc[core_ids].copy()
    X_tri_raw = pd.concat([
        clin_features.loc[core_ids],
        mutation.loc[core_ids],
        mrna.loc[core_ids],
    ], axis=1)

    # 80/20 stratified split (same as subgroup bootstrap)
    train_idx, test_idx = train_test_split(
        np.arange(len(core_ids)),
        test_size=0.2,
        stratify=E_all,
        random_state=SEED,
    )

    T_train, T_test = T_all[train_idx], T_all[test_idx]
    E_train, E_test = E_all[train_idx], E_all[test_idx]

    # -- Preprocess ----------------------------------------------------------
    print("  Preprocessing Clinical modality...")
    X_clin_train, X_clin_test = encode_and_scale_clinical(
        X_clin_raw.iloc[train_idx], X_clin_raw.iloc[test_idx]
    )

    print("  Preprocessing Trimodal modality...")
    clin_cols = [c for c in X_tri_raw.columns if not c.startswith(("MUT__", "MRNA__"))]
    mut_cols = [c for c in X_tri_raw.columns if c.startswith("MUT__")]
    mrna_cols = [c for c in X_tri_raw.columns if c.startswith("MRNA__")]

    parts_train, parts_test = [], []
    a, b = encode_and_scale_clinical(
        X_tri_raw[clin_cols].iloc[train_idx], X_tri_raw[clin_cols].iloc[test_idx]
    )
    parts_train.append(a)
    parts_test.append(b)
    a, b = preprocess_mutation(
        X_tri_raw[mut_cols].iloc[train_idx], X_tri_raw[mut_cols].iloc[test_idx]
    )
    parts_train.append(a)
    parts_test.append(b)
    a, b = preprocess_mrna(
        X_tri_raw[mrna_cols].iloc[train_idx], X_tri_raw[mrna_cols].iloc[test_idx],
        n_components=20,
    )
    parts_train.append(a)
    parts_test.append(b)
    X_tri_train = pd.concat(parts_train, axis=1)
    X_tri_test = pd.concat(parts_test, axis=1)

    # -- Fit RSF models and get survival functions ---------------------------
    y_train = make_surv_array(E_train, T_train)

    print("  Fitting RSF Clinical...")
    rsf_clin = RandomSurvivalForest(random_state=SEED, n_jobs=-1, **RSF_PARAMS)
    rsf_clin.fit(X_clin_train.values, y_train)

    print("  Fitting RSF Trimodal...")
    rsf_tri = RandomSurvivalForest(random_state=SEED, n_jobs=-1, **RSF_PARAMS)
    rsf_tri.fit(X_tri_train.values, y_train)

    # Get 5-year mortality probabilities = 1 - S(60 months)
    print("  Computing 5-year survival probabilities...")

    def get_5yr_mortality(rsf_model, X_test_arr):
        surv_fns = rsf_model.predict_survival_function(X_test_arr)
        probs = []
        for fn in surv_fns:
            # Evaluate at t=60 (or closest available time)
            times = fn.x
            vals = fn.y
            if HORIZON in times:
                s60 = vals[times == HORIZON][0]
            else:
                # Interpolate/extrapolate using closest time
                idx = np.searchsorted(times, HORIZON)
                if idx == 0:
                    s60 = vals[0]
                elif idx >= len(times):
                    s60 = vals[-1]
                else:
                    # Linear interpolation
                    t0, t1 = times[idx - 1], times[idx]
                    s0, s1 = vals[idx - 1], vals[idx]
                    s60 = s0 + (s1 - s0) * (HORIZON - t0) / (t1 - t0)
            probs.append(1.0 - float(s60))
        return np.array(probs)

    prob_clin = get_5yr_mortality(rsf_clin, X_clin_test.values)
    prob_tri = get_5yr_mortality(rsf_tri, X_tri_test.values)

    # 5-year outcome (event within 60 months)
    y_5yr = ((E_test == 1) & (T_test <= HORIZON)).astype(bool)
    n_test = len(y_5yr)
    prevalence = y_5yr.mean()
    print(f"  Test set: n={n_test}, 5-year events={y_5yr.sum()} (prevalence={prevalence:.3f})")

    # -- Decision curve computation ------------------------------------------
    thresholds = np.arange(0.05, 0.61, 0.01)

    nb_treat_all = []
    nb_treat_none = []
    nb_clin = []
    nb_tri = []

    for pt in thresholds:
        # Treat all
        ta = prevalence - (1.0 - prevalence) * pt / (1.0 - pt)
        nb_treat_all.append(ta)
        # Treat none
        nb_treat_none.append(0.0)
        # Models
        nb_clin.append(compute_net_benefit(y_5yr, prob_clin, pt, n_test))
        nb_tri.append(compute_net_benefit(y_5yr, prob_tri, pt, n_test))

    nb_treat_all = np.array(nb_treat_all)
    nb_treat_none = np.array(nb_treat_none)
    nb_clin = np.array(nb_clin)
    nb_tri = np.array(nb_tri)

    # -- Save CSV ------------------------------------------------------------
    dca_df = pd.DataFrame({
        "threshold": thresholds,
        "net_benefit_treat_all": nb_treat_all,
        "net_benefit_treat_none": nb_treat_none,
        "net_benefit_rsf_clinical": nb_clin,
        "net_benefit_rsf_trimodal": nb_tri,
    })
    out_csv = OUTPUT_DIR / "dca_results.csv"
    dca_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # -- Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(thresholds, nb_treat_all, "k--", lw=1.5, label="Treat All")
    ax.plot(thresholds, nb_treat_none, "k:", lw=1.5, label="Treat None")
    ax.plot(thresholds, nb_clin, "b-", lw=2, label="RSF Clinical")
    ax.plot(thresholds, nb_tri, "r-", lw=2, label="RSF Trimodal")

    ax.set_xlabel("Threshold Probability", fontsize=11)
    ax.set_ylabel("Net Benefit", fontsize=11)
    ax.set_title("Decision Curve Analysis\nRSF Clinical vs. Trimodal (5-year mortality)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    fig.tight_layout()

    out_png = OUTPUT_DIR / "dca_curves.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_png}")

    # Quick summary at a few thresholds
    for pt_show in [0.10, 0.20, 0.30, 0.40]:
        idx = np.argmin(np.abs(thresholds - pt_show))
        print(f"  pt={pt_show:.2f}: Treat-All={nb_treat_all[idx]:.4f}, "
              f"RSF-Clin={nb_clin[idx]:.4f}, RSF-Tri={nb_tri[idx]:.4f}")


# ANALYSIS 3: Power Analysis
def power_hanley_mcneil(
    delta: float,
    n: int,
    c_baseline: float,
    q: float,
    alpha: float = 0.05,
    r: float = 0.8,
) -> float:
    """
    Power to detect a C-index difference of `delta` using Hanley & McNeil approx.

    Parameters
    ----------
    delta      : observed C-index difference
    n          : total sample size
    c_baseline : C-index of baseline model
    q          : event rate (prevalence)
    alpha      : two-sided significance level
    r          : correlation between paired C-index estimates (typically ~0.8)
    """
    se1 = np.sqrt(c_baseline * (1 - c_baseline) / (n * q * (1 - q)))
    c2 = c_baseline + delta
    se2 = np.sqrt(c2 * (1 - c2) / (n * q * (1 - q)))
    se_delta = np.sqrt(se1**2 + se2**2 - 2 * r * se1 * se2)
    if se_delta == 0:
        return np.nan
    z = delta / se_delta
    z_alpha2 = norm.ppf(1 - alpha / 2)
    power = norm.cdf(z - z_alpha2) + norm.cdf(-z - z_alpha2)
    return float(power)


def n_for_power(
    target_power: float,
    delta: float,
    c_baseline: float,
    q: float,
    alpha: float = 0.05,
    r: float = 0.8,
    n_max: int = 100_000,
) -> int:
    """Binary search for n to achieve `target_power`."""
    lo, hi = 50, n_max
    while lo < hi - 1:
        mid = (lo + hi) // 2
        pw = power_hanley_mcneil(delta, mid, c_baseline, q, alpha, r)
        if pw < target_power:
            lo = mid
        else:
            hi = mid
    return hi


def run_power_analysis() -> None:
    print("\nPower analysis")

    # Load clinical data to compute actual event rate
    clin = load_clinical_table()
    clin = clin.dropna(subset=["event", "os_months"]).copy()
    clin = clin[clin["os_months"] > 0].copy()

    n_actual = len(clin)
    q = clin["event"].mean()  # overall event rate
    print(f"  N actual        : {n_actual}")
    print(f"  Event rate (q)  : {q:.4f}")

    # Parameters
    delta = 0.04
    c_baseline = 0.64   # approximate clinical-only C-index from bootstrap results
    alpha = 0.05
    r_corr = 0.8

    power_actual = power_hanley_mcneil(delta, n_actual, c_baseline, q, alpha, r_corr)
    print(f"  Delta           : {delta}")
    print(f"  C_baseline      : {c_baseline}")
    print(f"  Correlation (r) : {r_corr}")
    print(f"  Power at n={n_actual}: {power_actual:.4f} ({power_actual*100:.1f}%)")

    n_needed = n_for_power(0.80, delta, c_baseline, q, alpha, r_corr)
    print(f"  n needed for 80% power: {n_needed}")

    # Save CSV
    out_csv = OUTPUT_DIR / "power_analysis.csv"
    pd.DataFrame([{
        "n_actual": n_actual,
        "delta": delta,
        "c_baseline": c_baseline,
        "event_rate_q": round(q, 4),
        "correlation_r": r_corr,
        "alpha": alpha,
        "power_actual": round(power_actual, 4),
        "n_needed_80pct": n_needed,
    }]).to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    # Power curve
    n_range = np.arange(500, 10001, 50)
    powers = np.array([
        power_hanley_mcneil(delta, int(n), c_baseline, q, alpha, r_corr)
        for n in n_range
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_range, powers, "b-", lw=2, label=f"Power (Δ={delta}, q={q:.2f})")
    ax.axhline(0.80, color="green", ls="--", lw=1.5, label="80% power")
    ax.axvline(n_actual, color="red", ls="--", lw=1.5, label=f"Actual n={n_actual}")
    ax.axvline(n_needed, color="orange", ls=":", lw=1.5, label=f"n needed={n_needed}")

    ax.set_xlabel("Sample Size (n)", fontsize=11)
    ax.set_ylabel("Statistical Power", fontsize=11)
    ax.set_title(
        f"Power Curve for Detecting Δ={delta} in C-index\n"
        f"(Hanley & McNeil approximation, α={alpha}, r={r_corr})",
        fontsize=12,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_png = OUTPUT_DIR / "power_curve.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_png}")


# main

if __name__ == "__main__":
    print("Clinical utility")

    # Analysis 1
    cca_result, r2_val = run_cca_redundancy()

    # Analysis 2
    run_dca()

    # Analysis 3
    run_power_analysis()

    print("\nOutputs:")
    print(f"  CCA redundancy CSV  : {OUTPUT_DIR / 'cca_redundancy_scores.csv'}")
    print(f"  CCA plot            : {OUTPUT_DIR / 'cca_redundancy_vs_delta.png'}")
    print(f"  DCA results CSV     : {OUTPUT_DIR / 'dca_results.csv'}")
    print(f"  DCA plot            : {OUTPUT_DIR / 'dca_curves.png'}")
    print(f"  Power analysis CSV  : {OUTPUT_DIR / 'power_analysis.csv'}")
    print(f"  Power curve plot    : {OUTPUT_DIR / 'power_curve.png'}")
