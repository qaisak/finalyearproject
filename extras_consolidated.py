from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

try:
    import shap

    HAS_SHAP = True
except ImportError:
    shap = None  # type: ignore[assignment]
    HAS_SHAP = False

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# paths + io helpers
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SEED = 42
HORIZON = 60


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


def ensure_outputs_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def upsert_csv_rows(new_rows: pd.DataFrame, output_path: Path, key: str) -> Path:
    ensure_outputs_dir()
    if key not in new_rows.columns:
        raise ValueError(f"Expected column '{key}' in new_rows.")

    if output_path.exists():
        existing = pd.read_csv(output_path)
        if key in existing.columns:
            existing = existing[~existing[key].isin(new_rows[key])]
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows.copy()
    else:
        combined = new_rows.copy()

    combined.to_csv(output_path, index=False)
    return output_path


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


def build_five_year_labels(clin: pd.DataFrame, horizon: int = HORIZON) -> tuple[pd.DataFrame, pd.Series]:
    os_months = clin["overall_survival_months"]
    event = clin["overall_survival"].fillna(0).astype(int)
    mask = ((event == 1) & (os_months <= horizon)) | (os_months > horizon)
    y_5y = ((event == 1) & (os_months <= horizon)).astype(int)
    return clin.loc[mask].copy(), y_5y.loc[mask].copy()


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


def get_classification_feature_columns(clin: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidate = clin.drop(columns=[col for col in CLASSIFICATION_LEAKAGE_COLS if col in clin.columns])
    numeric_cols = candidate.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        column
        for column in candidate.columns
        if column not in numeric_cols and candidate[column].dtype in ("object", "category")
    ]
    return numeric_cols, categorical_cols


def align_frames(*frames: pd.DataFrame | pd.Series) -> list[pd.DataFrame | pd.Series]:
    shared_index: Iterable[str] | None = None
    for frame in frames:
        if shared_index is None:
            shared_index = frame.index
        else:
            shared_index = shared_index.intersection(frame.index)
    assert shared_index is not None
    return [frame.loc[shared_index].copy() for frame in frames]


# top variance selector
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


# survival preprocessing for shap
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


def get_survival_feature_columns(clin: pd.DataFrame) -> list[str]:
    return [column for column in clin.columns if column not in SURVIVAL_LEAK_COLS]


def encode_and_scale_clinical(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train, X_test


def preprocess_mutation(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _split_modality_columns(X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    clin_cols = [c for c in X.columns if not c.startswith(("MUT__", "MRNA__"))]
    mut_cols = [c for c in X.columns if c.startswith("MUT__")]
    mrna_cols = [c for c in X.columns if c.startswith("MRNA__")]
    return clin_cols, mut_cols, mrna_cols


def _preprocess_modality_split(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    pca_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clin_cols, mut_cols, mrna_cols = _split_modality_columns(X_train_raw)
    parts_train: list[pd.DataFrame] = []
    parts_test: list[pd.DataFrame] = []

    if clin_cols:
        a, b = encode_and_scale_clinical(X_train_raw[clin_cols], X_test_raw[clin_cols])
        parts_train.append(a)
        parts_test.append(b)
    if mut_cols:
        a, b = preprocess_mutation(X_train_raw[mut_cols], X_test_raw[mut_cols])
        parts_train.append(a)
        parts_test.append(b)
    if mrna_cols:
        a, b = preprocess_mrna(X_train_raw[mrna_cols], X_test_raw[mrna_cols], n_components=pca_components)
        parts_train.append(a)
        parts_test.append(b)

    return pd.concat(parts_train, axis=1), pd.concat(parts_test, axis=1)


def tune_rsf(X_train: pd.DataFrame, y_train, E_train: np.ndarray) -> dict[str, object]:
    param_grid = [
        {"n_estimators": 200, "min_samples_leaf": 10, "max_features": "sqrt"},
        {"n_estimators": 300, "min_samples_leaf": 10, "max_features": "sqrt"},
        {"n_estimators": 200, "min_samples_leaf": 15, "max_features": "sqrt"},
        {"n_estimators": 200, "min_samples_leaf": 10, "max_features": 0.5},
    ]
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_params = param_grid[0]
    best_score = -np.inf

    for params in param_grid:
        fold_scores = []
        for fit_idx, val_idx in splitter.split(X_train, E_train):
            model = RandomSurvivalForest(random_state=SEED, n_jobs=1, **params)
            model.fit(X_train.iloc[fit_idx].values, y_train[fit_idx])
            risk = model.predict(X_train.iloc[val_idx].values)
            score = concordance_index(
                y_train[val_idx]["time"],
                -risk,
                y_train[val_idx]["event"].astype(int),
            )
            fold_scores.append(score)
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    return best_params


# pam50 gene set vs pca
PAM50_GENES: list[str] = [
    "ESR1", "PGR", "ERBB2", "MKI67", "AURKA", "BIRC5", "CCNB1", "MYBL2",
    "GRB7", "ACTR3B", "ANLN", "BAG1", "BCL2", "BLVRA", "CCNE1", "CDC6",
    "CDC20", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "EXO1", "FGFR4",
    "FOXA1", "FOXC1", "GPR160", "KIF2C", "KRT14", "KRT17", "KRT5", "MAPT",
    "MDM2", "MELK", "MIA", "MMP11", "MLPH", "NAT1", "NDC80", "ORC6",
    "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
]

LUMINAL_GENES: list[str] = [
    "ESR1", "PGR", "FOXA1", "BCL2", "MAPT", "NAT1", "SLC39A6",
    "GPR160", "CXXC5", "MLPH",
]
PROLIFERATION_GENES: list[str] = [
    "MKI67", "AURKA", "BIRC5", "CCNB1", "MYBL2", "CEP55", "CCNE1",
    "CDC6", "CDC20", "UBE2C", "PTTG1", "RRM2", "EXO1", "ANLN",
    "NDC80", "ORC6", "KIF2C", "MELK", "CENPF", "TYMS", "UBE2T",
]
HER2_GENES: list[str] = ["ERBB2", "GRB7"]
BASAL_GENES: list[str] = [
    "KRT5", "KRT14", "KRT17", "CDH3", "EGFR", "FOXC1", "PHGDH", "SFRP1",
]

PATHWAY_GROUPS: dict[str, list[str]] = {
    "luminal_score": LUMINAL_GENES,
    "proliferation_score": PROLIFERATION_GENES,
    "her2_score": HER2_GENES,
    "basal_score": BASAL_GENES,
}


def _get_survival_clinical_columns(clin: pd.DataFrame) -> list[str]:
    """Mirror get_survival_feature_columns from the main pipeline (pam50 original)."""
    SURVIVAL_LEAK = {
        "overall_survival_months", "overall_survival", "death_from_cancer",
        "event", "os_months", "OS_MONTHS", "OS_STATUS", "RFS_MONTHS",
        "RFS_STATUS", "VITAL_STATUS",
    }
    drop = SURVIVAL_LEAK | {"patient_id", "cancer_type", "cancer_type_detailed",
                            "cohort", "integrative_cluster", "SEX",
                            "OS_STATUS_BIN", "OS_EVENT"}
    cols = [c for c in clin.columns if c not in drop]
    return cols


def load_pam50_features(mrna: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_cols = {col[6:]: col for col in mrna.columns if col.startswith("MRNA__")}

    found = [g for g in PAM50_GENES if g in gene_cols]
    missing = [g for g in PAM50_GENES if g not in gene_cols]

    print(f"PAM50 genes found : {len(found)}/{len(PAM50_GENES)}")
    if missing:
        print(f"  Missing genes    : {missing}")

    raw_cols = [gene_cols[g] for g in found]
    pam50_raw = mrna[raw_cols].copy()
    pam50_raw.columns = [f"PAM50__{g}" for g in found]

    pathway_rows: dict[str, pd.Series] = {}
    for score_name, gene_list in PATHWAY_GROUPS.items():
        avail = [gene_cols[g] for g in gene_list if g in gene_cols]
        found_in_group = [g for g in gene_list if g in gene_cols]
        missing_in_group = [g for g in gene_list if g not in gene_cols]
        if missing_in_group:
            print(f"  {score_name}: {len(found_in_group)}/{len(gene_list)} genes "
                  f"(missing: {missing_in_group})")
        else:
            print(f"  {score_name}: all {len(gene_list)} genes found")
        if avail:
            pathway_rows[score_name] = mrna[avail].mean(axis=1)
        else:
            pathway_rows[score_name] = pd.Series(np.nan, index=mrna.index)

    pam50_pw = pd.DataFrame(pathway_rows, index=mrna.index)
    return pam50_raw, pam50_pw


def preprocess_pca20(
    X_train: pd.DataFrame, X_test: pd.DataFrame, n_components: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("topvar", TopVarianceSelector(k=5000)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, svd_solver="randomized", random_state=SEED)),
    ])
    tr_np = pipe.fit_transform(X_train.values)
    te_np = pipe.transform(X_test.values)
    cols = [f"MRNA_PC{i+1}" for i in range(tr_np.shape[1])]
    return (
        pd.DataFrame(tr_np, columns=cols, index=X_train.index),
        pd.DataFrame(te_np, columns=cols, index=X_test.index),
    )


def preprocess_geneset(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    tr_np = scaler.fit_transform(imp.fit_transform(X_train))
    te_np = scaler.transform(imp.transform(X_test))
    return (
        pd.DataFrame(tr_np, columns=X_train.columns, index=X_train.index),
        pd.DataFrame(te_np, columns=X_train.columns, index=X_test.index),
    )


def tune_rsf_pam50(X_train: pd.DataFrame, y_train, E_train: np.ndarray) -> dict:
    """RSF tuning as in pam50_geneset_features.py (n_jobs=-1 in CV folds)."""
    param_grid = [
        {"n_estimators": 200, "min_samples_leaf": 10, "max_features": "sqrt"},
        {"n_estimators": 300, "min_samples_leaf": 10, "max_features": "sqrt"},
        {"n_estimators": 200, "min_samples_leaf": 15, "max_features": "sqrt"},
        {"n_estimators": 200, "min_samples_leaf": 10, "max_features": 0.5},
    ]
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_params = param_grid[0]
    best_score = -np.inf
    for params in param_grid:
        fold_scores = []
        for fit_idx, val_idx in splitter.split(X_train, E_train):
            m = RandomSurvivalForest(random_state=SEED, n_jobs=-1, **params)
            m.fit(X_train.iloc[fit_idx].values, y_train[fit_idx])
            risk = m.predict(X_train.iloc[val_idx].values)
            score = concordance_index(
                y_train[val_idx]["time"], -risk,
                y_train[val_idx]["event"].astype(int),
            )
            fold_scores.append(score)
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    return best_params


def fit_rsf_pam50(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    T_train: np.ndarray, E_train: np.ndarray,
    T_test: np.ndarray, E_test: np.ndarray,
) -> tuple[float, float]:
    y_train = make_surv_array(E_train, T_train)
    params = tune_rsf_pam50(X_train, y_train, E_train)
    model = RandomSurvivalForest(random_state=SEED, n_jobs=-1, **params)
    model.fit(X_train.values, y_train)
    preds_train = model.predict(X_train.values)
    preds_test = model.predict(X_test.values)
    c_test = concordance_index(T_test, -preds_test, E_test)
    c_train = concordance_index(T_train, -preds_train, E_train)
    return float(c_test), float(c_train)


def run_pam50_comparison() -> pd.DataFrame:
    ensure_outputs_dir()

    print("Loading clinical data ...")
    clin = load_clinical_table()
    clin = clin.dropna(subset=["event", "os_months"]).copy()
    clin = clin[clin["os_months"] > 0].copy()

    print("Loading mRNA data ...")
    mrna = load_mrna_matrix()

    core_ids = sorted(set(clin.index) & set(mrna.index))
    print(f"Patients with both clinical + mRNA: {len(core_ids)}")

    clin_aligned = clin.loc[core_ids]
    mrna_aligned = mrna.loc[core_ids]

    T = clin_aligned["os_months"].astype(float).values
    E = clin_aligned["event"].astype(int).values

    clinical_cols = _get_survival_clinical_columns(clin_aligned)
    clin_features = clin_aligned[clinical_cols].copy()

    print("\nPAM50 gene availability")
    pam50_raw, pam50_pw = load_pam50_features(mrna_aligned)

    train_idx, test_idx = train_test_split(
        np.arange(len(core_ids)),
        test_size=0.2,
        stratify=E,
        random_state=SEED,
    )
    T_train, T_test = T[train_idx], T[test_idx]
    E_train, E_test = E[train_idx], E[test_idx]

    print(f"\nTrain: {len(train_idx)} patients | Test: {len(test_idx)} patients")
    print(f"Train events: {E_train.sum()} | Test events: {E_test.sum()}")

    clin_tr = clin_features.iloc[train_idx]
    clin_te = clin_features.iloc[test_idx]
    mrna_tr = mrna_aligned.iloc[train_idx]
    mrna_te = mrna_aligned.iloc[test_idx]
    raw_tr = pam50_raw.iloc[train_idx]
    raw_te = pam50_raw.iloc[test_idx]
    pw_tr = pam50_pw.iloc[train_idx]
    pw_te = pam50_pw.iloc[test_idx]

    clin_tr_pp, clin_te_pp = encode_and_scale_clinical(clin_tr, clin_te)

    pca_tr, pca_te = preprocess_pca20(mrna_tr, mrna_te, n_components=20)
    raw_tr_pp, raw_te_pp = preprocess_geneset(raw_tr, raw_te)
    pw_tr_pp, pw_te_pp = preprocess_geneset(pw_tr, pw_te)

    conditions = {
        "Clinical + PCA-20": (
            pd.concat([clin_tr_pp, pca_tr], axis=1),
            pd.concat([clin_te_pp, pca_te], axis=1),
        ),
        "Clinical + PAM50 raw": (
            pd.concat([clin_tr_pp, raw_tr_pp], axis=1),
            pd.concat([clin_te_pp, raw_te_pp], axis=1),
        ),
        "Clinical + PAM50 pathway scores": (
            pd.concat([clin_tr_pp, pw_tr_pp], axis=1),
            pd.concat([clin_te_pp, pw_te_pp], axis=1),
        ),
    }

    rows = []
    for condition_name, (X_tr, X_te) in conditions.items():
        print(f"\nFitting RSF: {condition_name}  (features: {X_tr.shape[1]}) ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c_test, c_train = fit_rsf_pam50(X_tr, X_te, T_train, E_train, T_test, E_test)
        print(f"  Train C-index: {c_train:.4f}  |  Test C-index: {c_test:.4f}")
        rows.append({
            "representation": condition_name,
            "n_features": X_tr.shape[1],
            "train_c_index": round(c_train, 4),
            "test_c_index": round(c_test, 4),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_pam50_genes_found": pam50_raw.shape[1],
        })

    results_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "pam50_geneset_comparison.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print("\n" + results_df.to_string(index=False))
    return results_df


# shap on rsf
PCA_COMPONENTS = 20
N_SHAP_EXPLAIN = 50
N_BG_KMEANS = 10
RNG = np.random.default_rng(SEED)


def _prepare_trimodal_data():
    clin = load_clinical_table()
    clin = clin.dropna(subset=["event", "os_months"]).copy()
    clin = clin[clin["os_months"] > 0].copy()
    mutation = load_mutation_features()
    mrna = load_mrna_matrix()

    clinical_columns = get_survival_feature_columns(clin)
    clin_features = clin[clinical_columns].copy()

    core_ids = sorted(
        set(clin_features.index) & set(mutation.index) & set(mrna.index)
    )

    X = pd.concat(
        [clin_features.loc[core_ids], mutation.loc[core_ids], mrna.loc[core_ids]],
        axis=1,
    )
    T = clin.loc[core_ids, "os_months"].astype(float).values
    E = clin.loc[core_ids, "event"].astype(int).values
    return X, T, E


def run_shap_analysis():
    if not HAS_SHAP or shap is None:
        warnings.warn(
            "The shap library is not installed; skipping SHAP analysis.",
            UserWarning,
            stacklevel=2,
        )
        return

    ensure_outputs_dir()

    print("Loading and aligning modality data ...")
    X_full, T_full, E_full = _prepare_trimodal_data()
    n_samples = len(X_full)
    print(f"  Total patients: {n_samples}")

    train_idx, test_idx = train_test_split(
        np.arange(n_samples),
        test_size=0.2,
        stratify=E_full,
        random_state=SEED,
    )

    X_train_raw = X_full.iloc[train_idx].copy()
    X_test_raw = X_full.iloc[test_idx].copy()
    T_train, E_train = T_full[train_idx], E_full[train_idx]
    T_test, E_test = T_full[test_idx], E_full[test_idx]

    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    print("Preprocessing features ...")
    X_train, X_test = _preprocess_modality_split(X_train_raw, X_test_raw, PCA_COMPONENTS)
    feature_names = list(X_train.columns)
    print(f"  Feature count after preprocessing: {len(feature_names)}")

    SKIP_TUNING = True
    FIXED_PARAMS = {"n_estimators": 200, "min_samples_leaf": 10, "max_features": "sqrt"}

    y_train = make_surv_array(E_train, T_train)
    if SKIP_TUNING:
        params = FIXED_PARAMS
        print(f"  Using fixed params (confirmed best): {params}")
    else:
        print("Tuning RSF Trimodal...")
        params = tune_rsf(X_train, y_train, E_train)
        print(f"  Best params: {params}")

    print("Training RSF Trimodal...")
    rsf = RandomSurvivalForest(random_state=SEED, n_jobs=-1, **params)
    rsf.fit(X_train.values, y_train)

    preds_test = rsf.predict(X_test.values)
    c_idx = concordance_index(T_test, -preds_test, E_test)
    print(f"  Test C-index (80/20 split): {c_idx:.4f}")

    print("Computing SHAP values ...")

    shap_values = None
    used_kernel = False

    try:
        explainer = shap.TreeExplainer(rsf)
        shap_values = explainer.shap_values(X_test.values, check_additivity=False)
        print("  Used TreeExplainer successfully.")
    except Exception as tree_err:
        print(f"  TreeExplainer failed ({tree_err}); falling back to KernelSHAP ...")
        used_kernel = True

    if used_kernel or shap_values is None:
        n_explain = min(N_SHAP_EXPLAIN, len(X_test))
        ex_idx = RNG.choice(len(X_test), size=n_explain, replace=False)
        X_explain = X_test.values[ex_idx].astype(np.float64)

        print(f"  Building k-means background (k={N_BG_KMEANS}) ...")
        background = shap.kmeans(X_train.values.astype(np.float64), N_BG_KMEANS)

        predict_fn = lambda x: rsf.predict(x)  # noqa: E731
        explainer = shap.KernelExplainer(predict_fn, background)
        print(f"  Running KernelSHAP on {n_explain} test patients (nsamples=100) ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(X_explain, nsamples=100, silent=False)
        X_test = pd.DataFrame(X_explain, columns=X_test.columns)
        print(f"  KernelSHAP done ({n_explain} patients).")

    shap_values = np.array(shap_values)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top20_idx = np.argsort(mean_abs_shap)[::-1][:20]
    top20_names = [feature_names[i] for i in top20_idx]
    top20_values = mean_abs_shap[top20_idx]

    print("Saving shap_rsf_trimodal_summary.png ...")
    shap_top20 = shap_values[:, top20_idx]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_top20,
        features=X_test.values[:, top20_idx],
        feature_names=top20_names,
        show=False,
        max_display=20,
        plot_size=None,
    )
    plt.title("RSF Trimodal - SHAP Summary (Top 20 Features)", fontsize=13, pad=12)
    plt.tight_layout()
    out_summary = OUTPUT_DIR / "shap_rsf_trimodal_summary.png"
    plt.savefig(out_summary, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_summary}")

    print("Saving shap_rsf_modality_contribution.png ...")

    def _modality_label(name: str) -> str:
        if name.startswith("MUT__"):
            return "Mutation"
        if name.startswith("MRNA_PC"):
            return "mRNA PCA"
        return "Clinical"

    modality_labels = [_modality_label(f) for f in feature_names]

    modality_total: dict[str, float] = {}
    modality_mpf: dict[str, float] = {}
    modality_nfeat: dict[str, int] = {}
    for mod in ["Clinical", "Mutation", "mRNA PCA"]:
        feat_idx = [i for i, f in enumerate(feature_names) if modality_labels[i] == mod]
        if feat_idx:
            per_feat = float(np.abs(shap_values[:, feat_idx]).mean(axis=0).sum())
            modality_total[mod] = per_feat
            modality_mpf[mod] = float(np.abs(shap_values[:, feat_idx]).mean())
            modality_nfeat[mod] = len(feat_idx)
        else:
            modality_total[mod] = 0.0
            modality_mpf[mod] = 0.0
            modality_nfeat[mod] = 0

    mods = sorted(modality_total.keys(), key=lambda m: -modality_total[m])
    colors = {"Clinical": "#2196F3", "Mutation": "#FF5722", "mRNA PCA": "#4CAF50"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    total_vals = [modality_total[m] for m in mods]
    bar_colors = [colors[m] for m in mods]
    bars = ax.bar(mods, total_vals, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    for i, m in enumerate(mods):
        ax.text(i, total_vals[i] * 0.03, f"n={modality_nfeat[m]}",
                ha="center", va="bottom", fontsize=8, color="white", fontweight="bold")
    ax.set_title("Total mean |SHAP| by modality\n(sum over all features in group)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Sum of mean |SHAP|")
    ax.set_xlabel("Modality")

    ax2 = axes[1]
    mpf_vals = [modality_mpf[m] for m in mods]
    bars2 = ax2.bar(mods, mpf_vals, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax2.bar_label(bars2, fmt="%.4f", padding=3, fontsize=10)
    ax2.set_title("Mean |SHAP| per feature by modality\n(normalised for feature count)",
                  fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean |SHAP| per feature")
    ax2.set_xlabel("Modality")

    fig.suptitle("Does molecular data add NEW information beyond clinical?\n"
                 "RSF Trimodal SHAP modality decomposition",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_modality = OUTPUT_DIR / "shap_rsf_modality_contribution.png"
    plt.savefig(out_modality, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_modality}")

    print("Saving shap_rsf_top_features.csv ...")
    top20_df = pd.DataFrame({
        "rank": range(1, 21),
        "feature": top20_names,
        "modality": [_modality_label(n) for n in top20_names],
        "mean_abs_shap": top20_values,
    })
    out_csv = OUTPUT_DIR / "shap_rsf_top_features.csv"
    top20_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}")

    print("\nSHAP summary")
    print(f"Test C-index (80/20 split): {c_idx:.4f}")
    print("Modality contributions (total mean |SHAP|):")
    for mod, val in sorted(modality_total.items(), key=lambda kv: -kv[1]):
        print(f"  {mod:15s}: {val:.5f}")
    print("Top-10 features:")
    for _, row in top20_df.head(10).iterrows():
        print(f"  {row['rank']:2d}. [{row['modality']:10s}] {row['feature']}  ({row['mean_abs_shap']:.5f})")


# __main__
def main():
    ensure_outputs_dir()
    print("Extras")

    print("PAM50 vs PCA...")
    run_pam50_comparison()

    if HAS_SHAP:
        print("SHAP (RSF trimodal)...")
        run_shap_analysis()
    else:
        print("SHAP skipped (shap not installed)")
        warnings.warn(
            "Install the `shap` package to generate SHAP plots and shap_rsf_top_features.csv.",
            UserWarning,
            stacklevel=2,
        )

    print("done.")


if __name__ == "__main__":
    main()
