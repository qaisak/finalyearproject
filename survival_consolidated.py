from __future__ import annotations

import argparse
import json
import os
import urllib.request
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from lifelines import (
    CoxPHFitter,
    LogLogisticAFTFitter,
    LogNormalAFTFitter,
    WeibullAFTFitter,
)
from lifelines.utils import concordance_index

from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score
from sksurv.util import Surv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
    import torchtuples as tt  # noqa: F401
    from pycox.models import CoxPH as PycoxCoxPH  # noqa: F401

    HAS_DEEPSURV = True
except ImportError:
    HAS_DEEPSURV = False


# paths + io helpers
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SEED = 42
HORIZON = 60
# Set SURVIVAL_FAST=1 for quicker smoke tests (smaller outer CV, lighter DeepSurv).
SURVIVAL_FAST_ENV = os.environ.get("SURVIVAL_FAST", "").lower() in ("1", "true", "yes")


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

# survival shortlist body
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
TIME_HORIZONS_MONTHS = [12, 24, 36, 60, 120]


@dataclass
class FittedSurvivalModel:
    c_index: float
    train_c_index: float
    preds_test: np.ndarray
    preds_train: np.ndarray
    model: object | None = None


def compute_5y_metrics(T: np.ndarray, E: np.ndarray, risk: np.ndarray, higher_is_riskier: bool) -> tuple[float, float]:
    mask = (T > HORIZON) | (E == 1)
    y_bin = ((E[mask] == 1) & (T[mask] <= HORIZON)).astype(int)
    if len(np.unique(y_bin)) < 2:
        return np.nan, np.nan
    scores = risk[mask] if higher_is_riskier else -risk[mask]
    return roc_auc_score(y_bin, scores), average_precision_score(y_bin, scores)


# Preprocessing helpers
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
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns, index=X_test.index)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns, index=X_test.index)
    return X_train, X_test


def preprocess_mutation(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_np = scaler.transform(imputer.transform(X_test))
    return (
        pd.DataFrame(X_train_np, columns=X_train.columns, index=X_train.index),
        pd.DataFrame(X_test_np, columns=X_train.columns, index=X_test.index),
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


# CoxPH
def tune_cox_penalizer(
    X_train: pd.DataFrame, T_train: np.ndarray, E_train: np.ndarray, l1_ratio: float = 0.0
) -> float:
    candidates = [0.1, 1.0, 5.0]
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_penalty = candidates[0]
    best_score = -np.inf

    for penalty in candidates:
        fold_scores = []
        for fit_idx, val_idx in splitter.split(X_train, E_train):
            fold_train = X_train.iloc[fit_idx].copy()
            fold_val = X_train.iloc[val_idx].copy()
            fold_train["T"] = T_train[fit_idx]
            fold_train["E"] = E_train[fit_idx]
            try:
                model = CoxPHFitter(penalizer=penalty, l1_ratio=l1_ratio)
                model.fit(fold_train, duration_col="T", event_col="E", show_progress=False)
                risk = model.predict_partial_hazard(fold_val).values.ravel()
                score = concordance_index(T_train[val_idx], -risk, E_train[val_idx])
                fold_scores.append(score)
            except Exception:
                continue
        mean_score = np.mean(fold_scores) if fold_scores else 0.5
        if mean_score > best_score:
            best_score = mean_score
            best_penalty = penalty
    return best_penalty


def mean_val_c_index_cox(
    X_train: pd.DataFrame,
    T_train: np.ndarray,
    E_train: np.ndarray,
    l1_ratio: float = 0.0,
) -> float:
    """Mean validation C-index over 3 folds on the training set, using the tuned penaliser (for fusion weights)."""
    penalty = tune_cox_penalizer(X_train, T_train, E_train, l1_ratio=l1_ratio)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    fold_scores: list[float] = []
    for fit_idx, val_idx in splitter.split(X_train, E_train):
        fold_train = X_train.iloc[fit_idx].copy()
        fold_val = X_train.iloc[val_idx].copy()
        fold_train["T"] = T_train[fit_idx]
        fold_train["E"] = E_train[fit_idx]
        try:
            model = CoxPHFitter(penalizer=penalty, l1_ratio=l1_ratio)
            model.fit(fold_train, duration_col="T", event_col="E", show_progress=False)
            risk = model.predict_partial_hazard(fold_val).values.ravel()
            fold_scores.append(
                float(concordance_index(T_train[val_idx], -risk, E_train[val_idx]))
            )
        except Exception:
            continue
    return float(np.mean(fold_scores)) if fold_scores else 0.5


def fit_cox(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    T_train: np.ndarray,
    E_train: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    l1_ratio: float = 0.0,
) -> FittedSurvivalModel:
    penalty = tune_cox_penalizer(X_train, T_train, E_train, l1_ratio=l1_ratio)
    train_df = X_train.copy()
    train_df["T"] = T_train
    train_df["E"] = E_train

    model = CoxPHFitter(penalizer=penalty, l1_ratio=l1_ratio)
    model.fit(train_df, duration_col="T", event_col="E", show_progress=False)

    preds_test = model.predict_partial_hazard(X_test).values.ravel()
    preds_train = model.predict_partial_hazard(X_train).values.ravel()
    c_index = concordance_index(T_test, -preds_test, E_test)
    train_c_index = concordance_index(T_train, -preds_train, E_train)
    return FittedSurvivalModel(
        c_index=c_index,
        train_c_index=train_c_index,
        preds_test=preds_test,
        preds_train=preds_train,
        model=model,
    )


# Parametric AFT models (Weibull, Log-Logistic, Log-Normal)
def fit_aft(
    AFTClass,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    T_train: np.ndarray,
    E_train: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
) -> FittedSurvivalModel:
    """Fit a parametric AFT model with escalating penaliser. Returns NaN C-index on convergence failure."""
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_tr_num = X_train[num_cols].copy() if num_cols else X_train.copy()
    X_te_num = X_test[num_cols].copy() if num_cols else X_test.copy()

    df_tr = X_tr_num.copy()
    df_tr["T"] = T_train
    df_tr["E"] = E_train

    for pen in [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]:
        try:
            m = AFTClass(penalizer=pen, l1_ratio=0.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(df_tr, duration_col="T", event_col="E", show_progress=False)
            if hasattr(m, "params_"):
                break
        except Exception:
            continue
    else:
        return FittedSurvivalModel(
            c_index=np.nan,
            train_c_index=np.nan,
            preds_test=np.full(len(T_test), np.nan),
            preds_train=np.full(len(T_train), np.nan),
            model=None,
        )

    preds_test_raw = m.predict_median(X_te_num).values.ravel()
    preds_train_raw = m.predict_median(X_tr_num).values.ravel()
    preds_test_clean = np.where(np.isinf(preds_test_raw) | np.isnan(preds_test_raw), 9999, preds_test_raw)
    preds_train_clean = np.where(np.isinf(preds_train_raw) | np.isnan(preds_train_raw), 9999, preds_train_raw)

    c_index = concordance_index(T_test, preds_test_clean, E_test)
    train_c_index = concordance_index(T_train, preds_train_clean, E_train)
    return FittedSurvivalModel(
        c_index=c_index,
        train_c_index=train_c_index,
        preds_test=preds_test_clean,
        preds_train=preds_train_clean,
        model=m,
    )


def fit_best_aft(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    T_train: np.ndarray,
    E_train: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
) -> FittedSurvivalModel:
    """Pick the best-fitting parametric AFT (by training C-index) among Weibull / log-logistic / log-normal."""
    best: FittedSurvivalModel | None = None
    best_train = -np.inf
    for aft_cls in (WeibullAFTFitter, LogLogisticAFTFitter, LogNormalAFTFitter):
        fitted = fit_aft(aft_cls, X_train, X_test, T_train, E_train, T_test, E_test)
        if np.isnan(fitted.train_c_index):
            continue
        if fitted.train_c_index > best_train or (
            fitted.train_c_index == best_train
            and best is not None
            and not np.isnan(fitted.c_index)
            and (np.isnan(best.c_index) or fitted.c_index > best.c_index)
        ):
            best_train = float(fitted.train_c_index)
            best = fitted
    if best is None:
        return FittedSurvivalModel(
            c_index=np.nan,
            train_c_index=np.nan,
            preds_test=np.full(len(T_test), np.nan),
            preds_train=np.full(len(T_train), np.nan),
            model=None,
        )
    return best


# RSF
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


def fit_rsf(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    T_train: np.ndarray,
    E_train: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
) -> FittedSurvivalModel:
    y_train = make_surv_array(E_train, T_train)
    params = tune_rsf(X_train, y_train, E_train)
    model = RandomSurvivalForest(random_state=SEED, n_jobs=1, **params)
    model.fit(X_train.values, y_train)
    preds_train = model.predict(X_train.values)
    preds_test = model.predict(X_test.values)
    c_index = concordance_index(T_test, -preds_test, E_test)
    train_c_index = concordance_index(T_train, -preds_train, E_train)
    return FittedSurvivalModel(
        c_index=c_index,
        train_c_index=train_c_index,
        preds_test=preds_test,
        preds_train=preds_train,
        model=model,
    )


# GBSA
def tune_gbsa(X_train: pd.DataFrame, y_train, E_train: np.ndarray) -> dict[str, object]:
    param_grid = [
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "subsample": 1.0},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
    ]
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_params = param_grid[0]
    best_score = -np.inf

    for params in param_grid:
        fold_scores = []
        for fit_idx, val_idx in splitter.split(X_train, E_train):
            model = GradientBoostingSurvivalAnalysis(
                random_state=SEED,
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
            )
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


def fit_gbsa(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    T_train: np.ndarray,
    E_train: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
) -> FittedSurvivalModel:
    y_train = make_surv_array(E_train, T_train)
    params = tune_gbsa(X_train, y_train, E_train)
    model = GradientBoostingSurvivalAnalysis(random_state=SEED, **params)
    model.fit(X_train.values, y_train)
    preds_train = model.predict(X_train.values)
    preds_test = model.predict(X_test.values)
    c_index = concordance_index(T_test, -preds_test, E_test)
    train_c_index = concordance_index(T_train, -preds_train, E_train)
    return FittedSurvivalModel(
        c_index=c_index,
        train_c_index=train_c_index,
        preds_test=preds_test,
        preds_train=preds_train,
        model=model,
    )


# DeepSurv (pycox)
def _import_deepsurv():
    """Lazy import to avoid hard dependency on PyTorch."""
    if not HAS_DEEPSURV:
        return None
    import torch
    import torch.nn as nn
    import torchtuples as tt
    from pycox.models import CoxPH as PycoxCoxPH
    return torch, nn, tt, PycoxCoxPH


def _build_deepsurv_net(nn_mod, in_features: int, layers: list[int], dropout: float):
    modules = []
    prev = in_features
    for w in layers:
        modules += [nn_mod.Linear(prev, w), nn_mod.ReLU(), nn_mod.BatchNorm1d(w), nn_mod.Dropout(dropout)]
        prev = w
    modules.append(nn_mod.Linear(prev, 1))
    return nn_mod.Sequential(*modules)


def tune_deepsurv_architecture(
    X_train: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    *,
    fast: bool = False,
) -> dict:
    """Select DeepSurv architecture using inner train/val split (NOT the test set)."""
    configs = [
        {"layers": [128, 64], "dropout": 0.3, "lr": 0.001},
        {"layers": [256, 128], "dropout": 0.4, "lr": 0.0005},
        {"layers": [256, 128, 64], "dropout": 0.4, "lr": 0.0005},
        {"layers": [128, 64, 32], "dropout": 0.3, "lr": 0.001},
    ]
    _mods = _import_deepsurv()
    if _mods is None:
        return configs[0]
    if fast:
        return configs[0]
    torch, nn, tt, PycoxCoxPH = _mods

    n_val = max(1, int(0.15 * len(X_train)))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X_train))
    val_i, trn_i = perm[:n_val], perm[n_val:]

    X_tr_f = X_train[trn_i].astype(np.float32)
    T_tr_f = T_train[trn_i].astype(np.float32)
    E_tr_f = E_train[trn_i].astype(np.float32)
    X_val_f = X_train[val_i].astype(np.float32)
    T_val_f = T_train[val_i].astype(np.float32)
    E_val_f = E_train[val_i].astype(np.float32)

    best_c, best_cfg = 0.0, configs[0]
    for cfg in configs:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        net = _build_deepsurv_net(nn, X_tr_f.shape[1], cfg["layers"], cfg["dropout"])
        model = PycoxCoxPH(net, tt.optim.AdamWR(lr=cfg["lr"], decoupled_weight_decay=1e-4))
        model.optimizer.set_lr(cfg["lr"])
        try:
            model.fit(
                X_tr_f, (T_tr_f, E_tr_f),
                batch_size=128, epochs=100,
                callbacks=[tt.callbacks.EarlyStopping(patience=10)],
                val_data=tt.tuplefy(X_val_f, (T_val_f, E_val_f)),
                verbose=False,
            )
            preds_val = model.predict(X_val_f).ravel()
            mask = ~np.isnan(preds_val)
            if mask.sum() >= 10:
                c = concordance_index(T_val_f[mask], -preds_val[mask], E_val_f[mask])
            else:
                c = 0.5
        except Exception:
            c = 0.5
        if c > best_c:
            best_c, best_cfg = c, cfg
    return best_cfg


def fit_deepsurv(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    T_train: np.ndarray,
    E_train: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    cfg: dict | None = None,
    *,
    fast: bool = False,
) -> FittedSurvivalModel:
    _mods = _import_deepsurv()
    if _mods is None:
        return FittedSurvivalModel(
            c_index=np.nan,
            train_c_index=np.nan,
            preds_test=np.full(len(T_test), np.nan),
            preds_train=np.full(len(T_train), np.nan),
            model=None,
        )
    torch, nn, tt, PycoxCoxPH = _mods

    X_tr_f = X_train.values.astype(np.float32) if hasattr(X_train, "values") else X_train.astype(np.float32)
    X_te_f = X_test.values.astype(np.float32) if hasattr(X_test, "values") else X_test.astype(np.float32)
    T_tr_f = T_train.astype(np.float32)
    E_tr_f = E_train.astype(np.float32)
    T_te_f = T_test.astype(np.float32)
    E_te_f = E_test.astype(np.float32)

    if cfg is None:
        cfg = tune_deepsurv_architecture(X_tr_f, T_tr_f, E_tr_f, fast=fast)

    epochs = 40 if fast else 100
    patience = 5 if fast else 10
    seeds = [42] if fast else [42, 123, 7]
    all_preds_test = []
    all_preds_train = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = _build_deepsurv_net(nn, X_tr_f.shape[1], cfg["layers"], cfg["dropout"])
        model = PycoxCoxPH(net, tt.optim.AdamWR(lr=cfg["lr"], decoupled_weight_decay=1e-4))
        model.optimizer.set_lr(cfg["lr"])

        n_val = max(1, int(0.1 * len(X_tr_f)))
        perm = np.random.permutation(len(X_tr_f))
        val_i, trn_i = perm[:n_val], perm[n_val:]
        try:
            model.fit(
                X_tr_f[trn_i], (T_tr_f[trn_i], E_tr_f[trn_i]),
                batch_size=128,
                epochs=epochs,
                callbacks=[tt.callbacks.EarlyStopping(patience=patience)],
                val_data=tt.tuplefy(X_tr_f[val_i], (T_tr_f[val_i], E_tr_f[val_i])),
                verbose=False,
            )
            all_preds_test.append(model.predict(X_te_f).ravel())
            all_preds_train.append(model.predict(X_tr_f).ravel())
        except Exception:
            continue

    if not all_preds_test:
        return FittedSurvivalModel(
            c_index=np.nan,
            train_c_index=np.nan,
            preds_test=np.full(len(T_te_f), np.nan),
            preds_train=np.full(len(T_tr_f), np.nan),
            model=None,
        )

    avg_test = np.nanmean(all_preds_test, axis=0)
    avg_train = np.nanmean(all_preds_train, axis=0)
    mask_te = ~np.isnan(avg_test)
    mask_tr = ~np.isnan(avg_train)

    c_test = concordance_index(T_te_f[mask_te], -avg_test[mask_te], E_te_f[mask_te]) if mask_te.sum() >= 10 else np.nan
    c_train = concordance_index(T_tr_f[mask_tr], -avg_train[mask_tr], E_tr_f[mask_tr]) if mask_tr.sum() >= 10 else np.nan
    return FittedSurvivalModel(
        c_index=c_test,
        train_c_index=c_train,
        preds_test=avg_test,
        preds_train=avg_train,
        model=None,
    )


def fit_cox_late_fusion_trimodal(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    T_train: np.ndarray,
    E_train: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    pca_components: int,
    *,
    weighted: bool,
) -> FittedSurvivalModel:
    """Late fusion: combine modality-specific Cox partial hazards (log scale) on trimodal splits."""
    clin_cols, mut_cols, mrna_cols = _split_modality_columns(X_train_raw)
    if not (clin_cols and mut_cols and mrna_cols):
        return FittedSurvivalModel(
            c_index=np.nan,
            train_c_index=np.nan,
            preds_test=np.full(len(T_test), np.nan),
            preds_train=np.full(len(T_train), np.nan),
            model=None,
        )
    Xc_tr, Xc_te = encode_and_scale_clinical(X_train_raw[clin_cols], X_test_raw[clin_cols])
    Xm_tr, Xm_te = preprocess_mutation(X_train_raw[mut_cols], X_test_raw[mut_cols])
    Xr_tr, Xr_te = preprocess_mrna(
        X_train_raw[mrna_cols],
        X_test_raw[mrna_cols],
        n_components=pca_components,
    )
    cox_c = fit_cox(Xc_tr, Xc_te, T_train, E_train, T_test, E_test)
    cox_m = fit_cox(Xm_tr, Xm_te, T_train, E_train, T_test, E_test)
    cox_r = fit_cox(Xr_tr, Xr_te, T_train, E_train, T_test, E_test)

    def _log_h(z: np.ndarray) -> np.ndarray:
        return np.log(np.clip(z, 1e-12, None))

    lc_tr, lc_te = _log_h(cox_c.preds_train), _log_h(cox_c.preds_test)
    lm_tr, lm_te = _log_h(cox_m.preds_train), _log_h(cox_m.preds_test)
    lr_tr, lr_te = _log_h(cox_r.preds_train), _log_h(cox_r.preds_test)

    if weighted:
        wc = mean_val_c_index_cox(Xc_tr, T_train, E_train)
        wm = mean_val_c_index_cox(Xm_tr, T_train, E_train)
        wr = mean_val_c_index_cox(Xr_tr, T_train, E_train)
        ws = np.array([wc, wm, wr], dtype=float)
        if not np.all(np.isfinite(ws)) or float(ws.sum()) <= 0:
            ws = np.ones(3) / 3.0
        else:
            ws = ws / float(ws.sum())
        fused_tr = ws[0] * lc_tr + ws[1] * lm_tr + ws[2] * lr_tr
        fused_te = ws[0] * lc_te + ws[1] * lm_te + ws[2] * lr_te
    else:
        fused_tr = (lc_tr + lm_tr + lr_tr) / 3.0
        fused_te = (lc_te + lm_te + lr_te) / 3.0

    pred_tr = np.exp(fused_tr)
    pred_te = np.exp(fused_te)
    c_index = concordance_index(T_test, -pred_te, E_test)
    train_c_index = concordance_index(T_train, -pred_tr, E_train)
    return FittedSurvivalModel(
        c_index=c_index,
        train_c_index=train_c_index,
        preds_test=pred_te,
        preds_train=pred_tr,
        model=None,
    )


def fit_deepsurv_trimodal_fusion(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    T_train: np.ndarray,
    E_train: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    pca_components: int,
    *,
    weighted: bool,
    fast: bool,
) -> FittedSurvivalModel:
    """Equal- or training-C-weighted late fusion of modality-specific DeepSurv scores."""
    if not HAS_DEEPSURV:
        return FittedSurvivalModel(
            c_index=np.nan,
            train_c_index=np.nan,
            preds_test=np.full(len(T_test), np.nan),
            preds_train=np.full(len(T_train), np.nan),
            model=None,
        )
    clin_cols, mut_cols, mrna_cols = _split_modality_columns(X_train_raw)
    if not (clin_cols and mut_cols and mrna_cols):
        return FittedSurvivalModel(
            c_index=np.nan,
            train_c_index=np.nan,
            preds_test=np.full(len(T_test), np.nan),
            preds_train=np.full(len(T_train), np.nan),
            model=None,
        )
    Xc_tr, Xc_te = encode_and_scale_clinical(X_train_raw[clin_cols], X_test_raw[clin_cols])
    Xm_tr, Xm_te = preprocess_mutation(X_train_raw[mut_cols], X_test_raw[mut_cols])
    Xr_tr, Xr_te = preprocess_mrna(
        X_train_raw[mrna_cols],
        X_test_raw[mrna_cols],
        n_components=pca_components,
    )
    fc = fit_deepsurv(Xc_tr, Xc_te, T_train, E_train, T_test, E_test, fast=fast)
    fm = fit_deepsurv(Xm_tr, Xm_te, T_train, E_train, T_test, E_test, fast=fast)
    fr = fit_deepsurv(Xr_tr, Xr_te, T_train, E_train, T_test, E_test, fast=fast)
    if weighted:
        ws = np.array([fc.train_c_index, fm.train_c_index, fr.train_c_index], dtype=float)
        ws = np.nan_to_num(ws, nan=0.0, posinf=0.0, neginf=0.0)
        s = float(ws.sum())
        if s <= 0:
            ws = np.ones(3) / 3.0
        else:
            ws = ws / s
    else:
        ws = np.ones(3) / 3.0
    stack_te = np.vstack([fc.preds_test, fm.preds_test, fr.preds_test])
    stack_tr = np.vstack([fc.preds_train, fm.preds_train, fr.preds_train])
    fused_te = np.nansum(ws[:, None] * stack_te, axis=0)
    fused_tr = np.nansum(ws[:, None] * stack_tr, axis=0)
    mask_te = np.isfinite(fused_te)
    mask_tr = np.isfinite(fused_tr)
    c_test = (
        concordance_index(T_test[mask_te], -fused_te[mask_te], E_test[mask_te])
        if mask_te.sum() >= 10
        else np.nan
    )
    c_train = (
        concordance_index(T_train[mask_tr], -fused_tr[mask_tr], E_train[mask_tr])
        if mask_tr.sum() >= 10
        else np.nan
    )
    return FittedSurvivalModel(
        c_index=c_test,
        train_c_index=c_train,
        preds_test=fused_te,
        preds_train=fused_tr,
        model=None,
    )


def bootstrap_c_index_diff(
    T: np.ndarray,
    E: np.ndarray,
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    B: int = 2000,
    seed: int = SEED,
) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = len(T)
    c_a = concordance_index(T, risk_a, E)
    c_b = concordance_index(T, risk_b, E)
    observed_delta = c_a - c_b

    deltas = np.zeros(B)
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        try:
            c_boot_a = concordance_index(T[idx], risk_a[idx], E[idx])
            c_boot_b = concordance_index(T[idx], risk_b[idx], E[idx])
            deltas[b] = c_boot_a - c_boot_b
        except Exception:
            deltas[b] = np.nan

    deltas = deltas[~np.isnan(deltas)]
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    p_value = np.mean(deltas <= 0) if observed_delta > 0 else np.mean(deltas >= 0)
    p_value = min(2 * p_value, 1.0)
    return {
        "c_a": float(c_a),
        "c_b": float(c_b),
        "delta": float(observed_delta),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "B": int(len(deltas)),
    }


# Data preparation
def get_survival_feature_columns(clin: pd.DataFrame) -> list[str]:
    return [column for column in clin.columns if column not in SURVIVAL_LEAK_COLS]


def prepare_survival_modalities(pca_components: int) -> dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    clin = load_clinical_table()
    clin = clin.dropna(subset=["event", "os_months"]).copy()
    clin = clin[clin["os_months"] > 0].copy()
    mutation = load_mutation_features()
    mrna = load_mrna_matrix()

    clinical_columns = get_survival_feature_columns(clin)
    clin_features = clin[clinical_columns].copy()
    clinical_ids = set(clin_features.index)
    mutation_ids = set(mutation.index)
    mrna_ids = set(mrna.index)

    # Use the same patient set for every modality so unimodal C-indices are like-for-like
    # (intersection of patients with clinical + mutation + mRNA records).
    core_ids = sorted(clinical_ids & mutation_ids & mrna_ids)
    modality_ids = OrderedDict(
        [
            ("Clinical", core_ids),
            ("Mutation", core_ids),
            ("mRNA", core_ids),
            ("Clin+Mut", core_ids),
            ("Trimodal", core_ids),
        ]
    )

    prepared = {}
    for modality, ids in modality_ids.items():
        T = clin.loc[ids, "os_months"].astype(float).values
        E = clin.loc[ids, "event"].astype(int).values
        if modality == "Clinical":
            X = clin_features.loc[ids].copy()
        elif modality == "Mutation":
            X = mutation.loc[ids].copy()
        elif modality == "mRNA":
            X = mrna.loc[ids].copy()
        elif modality == "Clin+Mut":
            X = pd.concat([clin_features.loc[ids], mutation.loc[ids]], axis=1)
        else:
            X = pd.concat([clin_features.loc[ids], mutation.loc[ids], mrna.loc[ids]], axis=1)
        prepared[modality] = (X, T, E)
    return prepared


# Main evaluation loop
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


def _valid_eval_times(T_train: np.ndarray, T_test: np.ndarray) -> np.ndarray:
    train_pos = T_train[T_train > 0]
    test_pos = T_test[T_test > 0]
    if len(train_pos) == 0 or len(test_pos) == 0:
        return np.array([], dtype=float)
    lower = max(float(np.min(train_pos)), float(np.min(test_pos)))
    upper = min(float(np.max(T_train)), float(np.max(T_test)))
    times = [t for t in TIME_HORIZONS_MONTHS if lower < t < upper]
    return np.array(times, dtype=float)


def _survival_prob_matrix(
    model_name: str,
    fitted_model: object,
    X_test: pd.DataFrame,
    times: np.ndarray,
) -> np.ndarray | None:
    if fitted_model is None or len(times) == 0:
        return None
    try:
        if model_name == "CoxPH":
            surv_df = fitted_model.predict_survival_function(X_test, times=times)
            return surv_df.T.values
        if model_name in {"RSF", "GBSA"}:
            surv_fns = fitted_model.predict_survival_function(X_test.values, return_array=False)
            return np.vstack([fn(times) for fn in surv_fns])
    except Exception:
        return None
    return None


def _time_dependent_metrics(
    model_name: str,
    fitted: FittedSurvivalModel,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    T_train: np.ndarray,
    T_test: np.ndarray,
) -> dict[str, float]:
    times = _valid_eval_times(T_train=T_train, T_test=T_test)
    if len(times) == 0:
        return {"IBS": np.nan, **{f"AUC_{int(t // 12)}y": np.nan for t in TIME_HORIZONS_MONTHS}}

    auc_map = {f"AUC_{int(t // 12)}y": np.nan for t in TIME_HORIZONS_MONTHS}
    risk_for_auc = fitted.preds_test
    if model_name == "AFT_Best":
        risk_for_auc = -np.asarray(risk_for_auc, dtype=float)
    try:
        auc_values, _ = cumulative_dynamic_auc(y_train, y_test, risk_for_auc, times)
        for t, auc in zip(times, auc_values):
            auc_map[f"AUC_{int(t // 12)}y"] = float(auc)
    except Exception:
        pass

    ibs = np.nan
    surv_matrix = _survival_prob_matrix(
        model_name=model_name,
        fitted_model=fitted.model,
        X_test=X_test,
        times=times,
    )
    if surv_matrix is not None:
        try:
            ibs = float(integrated_brier_score(y_train, y_test, surv_matrix, times))
        except Exception:
            ibs = np.nan
    return {"IBS": ibs, **auc_map}


def _aggregate_primary_rows(fold_rows: list[dict[str, object]]) -> pd.DataFrame:
    fold_df = pd.DataFrame(fold_rows)
    numeric = ["Train_C_index", "Test_C_index", "ROC_AUC_5y", "PR_AUC_5y"]
    agg = fold_df.groupby(["Model", "Modality"])[numeric].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index().rename(
        columns={
            "Train_C_index_mean": "Train_C_index",
            "Train_C_index_std": "Train_C_index_std",
            "Test_C_index_mean": "Test_C_index",
            "Test_C_index_std": "Test_C_index_std",
            "ROC_AUC_5y_mean": "ROC_AUC_5y",
            "ROC_AUC_5y_std": "ROC_AUC_5y_std",
            "PR_AUC_5y_mean": "PR_AUC_5y",
            "PR_AUC_5y_std": "PR_AUC_5y_std",
        }
    )
    return agg


def _save_overlap_analysis(
    overlap_preds: dict[tuple[str, str], np.ndarray],
) -> None:
    rows: list[dict[str, object]] = []
    for model_name in ["CoxPH", "RSF", "GBSA"]:
        clinical = overlap_preds.get((model_name, "Clinical"))
        trimodal = overlap_preds.get((model_name, "Trimodal"))
        if clinical is None or trimodal is None:
            continue
        mask = ~np.isnan(clinical) & ~np.isnan(trimodal)
        if mask.sum() < 20:
            continue
        rho, p = spearmanr(clinical[mask], trimodal[mask])
        rows.append(
            {
                "Model": model_name,
                "N": int(mask.sum()),
                "Spearman_rho": float(rho),
                "p_value": float(p),
                "Interpretation": "High overlap suggests clinical variables already proxy molecular state.",
            }
        )
    overlap_path = OUTPUT_DIR / "feature_overlap_analysis.csv"
    pd.DataFrame(rows).to_csv(overlap_path, index=False)
    print(f"Saved {overlap_path}")


def evaluate_survival_shortlist(
    pca_components: int,
    selected_modalities: list[str] | None = None,
    *,
    n_outer_splits: int | None = None,
    survival_fast: bool = False,
) -> pd.DataFrame:
    modalities = prepare_survival_modalities(pca_components=pca_components)
    if selected_modalities:
        keep = [m for m in selected_modalities if m in modalities]
        modalities = {k: modalities[k] for k in keep}
        if len(modalities) == 0:
            raise ValueError(f"No valid modalities selected: {selected_modalities}")
    split_source = "Trimodal" if "Trimodal" in modalities else list(modalities.keys())[0]
    n_samples = len(modalities[split_source][0])
    n_sp = n_outer_splits if n_outer_splits is not None else (3 if survival_fast else 5)
    outer_folds = list(
        StratifiedKFold(n_splits=n_sp, shuffle=True, random_state=SEED).split(
            np.arange(n_samples),
            modalities[split_source][2],
        )
    )

    fold_rows: list[dict[str, object]] = []
    time_rows: list[dict[str, object]] = []
    overlap_preds: dict[tuple[str, str], np.ndarray] = {}
    for model_name in ["CoxPH", "RSF", "GBSA"]:
        overlap_preds[(model_name, "Clinical")] = np.full(n_samples, np.nan)
        overlap_preds[(model_name, "Trimodal")] = np.full(n_samples, np.nan)

    time_models = {"CoxPH", "RSF", "GBSA", "AFT_Best"}
    if HAS_DEEPSURV:
        time_models.add("DeepSurv")

    for modality, (X, T, E) in modalities.items():
        print(f"\n{modality} (nested {n_sp}-fold, n={len(X)})")
        for fold_idx, (train_idx, test_idx) in enumerate(outer_folds, start=1):
            X_train_raw = X.iloc[train_idx].copy()
            X_test_raw = X.iloc[test_idx].copy()
            T_train, E_train = T[train_idx], E[train_idx]
            T_test, E_test = T[test_idx], E[test_idx]
            X_train, X_test = _preprocess_modality_split(X_train_raw, X_test_raw, pca_components)
            y_train = make_surv_array(E_train, T_train)
            y_test = make_surv_array(E_test, T_test)

            model_specs: list[tuple[str, Callable[[], FittedSurvivalModel], bool]] = [
                ("CoxPH", lambda: fit_cox(X_train, X_test, T_train, E_train, T_test, E_test), True),
                ("RSF", lambda: fit_rsf(X_train, X_test, T_train, E_train, T_test, E_test), True),
                ("GBSA", lambda: fit_gbsa(X_train, X_test, T_train, E_train, T_test, E_test), True),
                (
                    "AFT_Best",
                    lambda: fit_best_aft(X_train, X_test, T_train, E_train, T_test, E_test),
                    False,
                ),
            ]
            if HAS_DEEPSURV:
                model_specs.append(
                    (
                        "DeepSurv",
                        lambda: fit_deepsurv(
                            X_train,
                            X_test,
                            T_train,
                            E_train,
                            T_test,
                            E_test,
                            fast=survival_fast,
                        ),
                        True,
                    )
                )

            print(f"  Fold {fold_idx}/{n_sp}")
            for model_name, fit_fn, higher_is_riskier in model_specs:
                fitted = fit_fn()
                roc_5y, pr_5y = compute_5y_metrics(
                    T_test,
                    E_test,
                    fitted.preds_test,
                    higher_is_riskier=higher_is_riskier,
                )
                fold_rows.append(
                    {
                        "Model": model_name,
                        "Modality": modality,
                        "Fold": fold_idx,
                        "Train_C_index": fitted.train_c_index,
                        "Test_C_index": fitted.c_index,
                        "ROC_AUC_5y": roc_5y,
                        "PR_AUC_5y": pr_5y,
                    }
                )
                if model_name in time_models:
                    tmetrics = _time_dependent_metrics(
                        model_name=model_name,
                        fitted=fitted,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        T_train=T_train,
                        T_test=T_test,
                    )
                    time_rows.append(
                        {
                            "Model": model_name,
                            "Modality": modality,
                            "Fold": fold_idx,
                            **tmetrics,
                        }
                    )
                if model_name in {"CoxPH", "RSF", "GBSA"} and modality in {"Clinical", "Trimodal"}:
                    overlap_preds[(model_name, modality)][test_idx] = fitted.preds_test
                c_out = "nan" if np.isnan(fitted.c_index) else f"{fitted.c_index:.4f}"
                print(f"    {model_name}: C={c_out}")

            if modality == "Trimodal":
                for fusion_model, weighted in (
                    ("CoxPH_LateFusion_Equal", False),
                    ("CoxPH_LateFusion_Weighted", True),
                ):
                    fitted = fit_cox_late_fusion_trimodal(
                        X_train_raw,
                        X_test_raw,
                        T_train,
                        E_train,
                        T_test,
                        E_test,
                        pca_components,
                        weighted=weighted,
                    )
                    roc_5y, pr_5y = compute_5y_metrics(
                        T_test, E_test, fitted.preds_test, higher_is_riskier=True
                    )
                    fold_rows.append(
                        {
                            "Model": fusion_model,
                            "Modality": modality,
                            "Fold": fold_idx,
                            "Train_C_index": fitted.train_c_index,
                            "Test_C_index": fitted.c_index,
                            "ROC_AUC_5y": roc_5y,
                            "PR_AUC_5y": pr_5y,
                        }
                    )
                    c_out = "nan" if np.isnan(fitted.c_index) else f"{fitted.c_index:.4f}"
                    print(f"    {fusion_model}: C={c_out}")

                if HAS_DEEPSURV:
                    for fusion_model, weighted in (
                        ("DeepSurv_Fusion_Equal", False),
                        ("DeepSurv_Fusion_Weighted", True),
                    ):
                        fitted = fit_deepsurv_trimodal_fusion(
                            X_train_raw,
                            X_test_raw,
                            T_train,
                            E_train,
                            T_test,
                            E_test,
                            pca_components,
                            weighted=weighted,
                            fast=survival_fast,
                        )
                        roc_5y, pr_5y = compute_5y_metrics(
                            T_test, E_test, fitted.preds_test, higher_is_riskier=True
                        )
                        fold_rows.append(
                            {
                                "Model": fusion_model,
                                "Modality": modality,
                                "Fold": fold_idx,
                                "Train_C_index": fitted.train_c_index,
                                "Test_C_index": fitted.c_index,
                                "ROC_AUC_5y": roc_5y,
                                "PR_AUC_5y": pr_5y,
                            }
                        )
                        c_out = "nan" if np.isnan(fitted.c_index) else f"{fitted.c_index:.4f}"
                        print(f"    {fusion_model}: C={c_out}")

    primary_df = _aggregate_primary_rows(fold_rows)
    primary_df["Evidence"] = f"Nested {n_sp}-fold CV"
    primary_df = primary_df.sort_values(["Test_C_index", "PR_AUC_5y"], ascending=False).reset_index(drop=True)

    time_df = pd.DataFrame(time_rows)
    if len(time_df):
        agg_cols = ["IBS", "AUC_1y", "AUC_2y", "AUC_3y", "AUC_5y", "AUC_10y"]
        time_summary = time_df.groupby(["Model", "Modality"])[agg_cols].agg(["mean", "std"])
        time_summary.columns = [f"{metric}_{stat}" for metric, stat in time_summary.columns]
        time_summary = time_summary.reset_index().rename(
            columns={
                "IBS_mean": "IBS",
                "IBS_std": "IBS_std",
                "AUC_1y_mean": "AUC_1y",
                "AUC_1y_std": "AUC_1y_std",
                "AUC_2y_mean": "AUC_2y",
                "AUC_2y_std": "AUC_2y_std",
                "AUC_3y_mean": "AUC_3y",
                "AUC_3y_std": "AUC_3y_std",
                "AUC_5y_mean": "AUC_5y",
                "AUC_5y_std": "AUC_5y_std",
                "AUC_10y_mean": "AUC_10y",
                "AUC_10y_std": "AUC_10y_std",
            }
        )
        top_keys = set(
            primary_df.sort_values("Test_C_index", ascending=False)
            .head(12)[["Model", "Modality"]]
            .itertuples(index=False, name=None)
        )
        time_top = time_summary[
            time_summary.apply(lambda r: (r["Model"], r["Modality"]) in top_keys, axis=1)
        ].copy()
        time_top.to_csv(OUTPUT_DIR / "survival_time_dependent_metrics.csv", index=False)
        print("Saved outputs/survival_time_dependent_metrics.csv")

    _save_overlap_analysis(overlap_preds)
    return primary_df


def run_survival_significance(
    pca_components: int = 20,
    *,
    survival_fast: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    print("\nSurvival significance")
    modalities = prepare_survival_modalities(pca_components=pca_components)
    B_boot = 400 if survival_fast else 2000

    results: dict[str, dict[str, object]] = {}
    for modality in ["Clinical", "Trimodal"]:
        X, T, E = modalities[modality]
        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=0.2,
            stratify=E,
            random_state=SEED,
        )
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        E_train, E_test = E[train_idx], E[test_idx]

        clin_cols = [c for c in X.columns if not c.startswith(("MUT__", "MRNA__"))]
        mut_cols = [c for c in X.columns if c.startswith("MUT__")]
        mrna_cols = [c for c in X.columns if c.startswith("MRNA__")]

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
            a, b = preprocess_mrna(
                X_train_raw[mrna_cols],
                X_test_raw[mrna_cols],
                n_components=pca_components,
            )
            parts_train.append(a)
            parts_test.append(b)

        X_train = pd.concat(parts_train, axis=1)
        X_test = pd.concat(parts_test, axis=1)

        cox = fit_cox(X_train, X_test, T_train, E_train, T_test, E_test)
        rsf = fit_rsf(X_train, X_test, T_train, E_train, T_test, E_test)
        gbsa = fit_gbsa(X_train, X_test, T_train, E_train, T_test, E_test)
        aft = fit_best_aft(X_train, X_test, T_train, E_train, T_test, E_test)

        results[f"CoxPH_{modality}"] = {
            "T_test": T_test,
            "E_test": E_test,
            "risk": -cox.preds_test,
            "c": cox.c_index,
        }
        results[f"RSF_{modality}"] = {
            "T_test": T_test,
            "E_test": E_test,
            "risk": -rsf.preds_test,
            "c": rsf.c_index,
            "rsf_preds_raw": rsf.preds_test,
            "T_test_raw": T_test,
            "E_test_raw": E_test,
        }
        results[f"GBSA_{modality}"] = {
            "T_test": T_test,
            "E_test": E_test,
            "risk": -gbsa.preds_test,
            "c": gbsa.c_index,
        }
        results[f"AFT_{modality}"] = {
            "T_test": T_test,
            "E_test": E_test,
            "risk": -np.asarray(aft.preds_test, dtype=float),
            "c": aft.c_index,
        }
        if HAS_DEEPSURV:
            ds = fit_deepsurv(
                X_train, X_test, T_train, E_train, T_test, E_test, fast=survival_fast
            )
            results[f"DeepSurv_{modality}"] = {
                "T_test": T_test,
                "E_test": E_test,
                "risk": -ds.preds_test,
                "c": ds.c_index,
            }

        if modality == "Trimodal":
            lf_eq = fit_cox_late_fusion_trimodal(
                X_train_raw,
                X_test_raw,
                T_train,
                E_train,
                T_test,
                E_test,
                pca_components,
                weighted=False,
            )
            lf_w = fit_cox_late_fusion_trimodal(
                X_train_raw,
                X_test_raw,
                T_train,
                E_train,
                T_test,
                E_test,
                pca_components,
                weighted=True,
            )
            results["CoxPH_LateFusion_Equal_Trimodal"] = {
                "T_test": T_test,
                "E_test": E_test,
                "risk": -lf_eq.preds_test,
                "c": lf_eq.c_index,
            }
            results["CoxPH_LateFusion_Weighted_Trimodal"] = {
                "T_test": T_test,
                "E_test": E_test,
                "risk": -lf_w.preds_test,
                "c": lf_w.c_index,
            }
            if HAS_DEEPSURV:
                dseq = fit_deepsurv_trimodal_fusion(
                    X_train_raw,
                    X_test_raw,
                    T_train,
                    E_train,
                    T_test,
                    E_test,
                    pca_components,
                    weighted=False,
                    fast=survival_fast,
                )
                dsw = fit_deepsurv_trimodal_fusion(
                    X_train_raw,
                    X_test_raw,
                    T_train,
                    E_train,
                    T_test,
                    E_test,
                    pca_components,
                    weighted=True,
                    fast=survival_fast,
                )
                results["DeepSurv_Fusion_Equal_Trimodal"] = {
                    "T_test": T_test,
                    "E_test": E_test,
                    "risk": -dseq.preds_test,
                    "c": dseq.c_index,
                }
                results["DeepSurv_Fusion_Weighted_Trimodal"] = {
                    "T_test": T_test,
                    "E_test": E_test,
                    "risk": -dsw.preds_test,
                    "c": dsw.c_index,
                }

        msg = (
            f"  {modality}: CoxPH C={cox.c_index:.4f}, RSF C={rsf.c_index:.4f}, "
            f"GBSA C={gbsa.c_index:.4f}, AFT C={aft.c_index:.4f}"
        )
        if HAS_DEEPSURV:
            msg += f", DeepSurv C={results[f'DeepSurv_{modality}']['c']:.4f}"
        print(msg, flush=True)

    comparisons = [
        ("RSF_Trimodal", "RSF_Clinical", "RSF Trimodal vs RSF Clinical (C-index)"),
        ("RSF_Trimodal", "GBSA_Trimodal", "RSF Trimodal vs GBSA Trimodal (C-index)"),
        ("RSF_Trimodal", "CoxPH_Clinical", "RSF Trimodal vs CoxPH Clinical (C-index)"),
        ("RSF_Trimodal", "CoxPH_Trimodal", "RSF Trimodal vs CoxPH Trimodal (C-index)"),
        ("RSF_Clinical", "CoxPH_Clinical", "RSF Clinical vs CoxPH Clinical (C-index)"),
        ("GBSA_Trimodal", "CoxPH_Clinical", "GBSA Trimodal vs CoxPH Clinical (C-index)"),
        ("RSF_Trimodal", "AFT_Trimodal", "RSF Trimodal vs AFT Trimodal (C-index)"),
        ("RSF_Trimodal", "CoxPH_LateFusion_Equal_Trimodal", "RSF Trimodal vs CoxPH late fusion equal (C-index)"),
    ]
    if HAS_DEEPSURV:
        comparisons.append(
            ("RSF_Trimodal", "DeepSurv_Trimodal", "RSF Trimodal vs DeepSurv Trimodal (C-index)")
        )
        comparisons.append(
            (
                "RSF_Trimodal",
                "DeepSurv_Fusion_Equal_Trimodal",
                "RSF Trimodal vs DeepSurv fusion equal (C-index)",
            )
        )

    rows: list[dict[str, object]] = []
    for model_a, model_b, label in comparisons:
        if model_a not in results or model_b not in results:
            continue
        comp = bootstrap_c_index_diff(
            results[model_a]["T_test"],
            results[model_a]["E_test"],
            results[model_a]["risk"],
            results[model_b]["risk"],
            B=B_boot,
        )
        print(f"  {label}: delta={comp['delta']:.4f}, p={comp['p_value']:.4f}", flush=True)
        rows.append(
            {
                "Comparison": label,
                "Metric_A": f"C={comp['c_a']:.4f}",
                "Metric_B": f"C={comp['c_b']:.4f}",
                "Delta": round(comp["delta"], 4),
                "CI_lower": round(comp["ci_lower"], 4),
                "CI_upper": round(comp["ci_upper"], 4),
                "p_value": round(comp["p_value"], 4),
                "B": comp["B"],
            }
        )

    return pd.DataFrame(rows), results


def save_survival_significance(
    significance_df: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    path = output_path if output_path is not None else OUTPUT_DIR / "bootstrap_significance_surv.csv"
    return upsert_csv_rows(significance_df, path, key="Comparison")


def run_rsf_calibration(
    survival_results: dict[str, dict[str, object]],
    output_path: Path | None = None,
) -> Path:
    print("\nRSF Trimodal calibration")
    result = survival_results["RSF_Trimodal"]
    T_test = result["T_test_raw"]
    E_test = result["E_test_raw"]
    risk = result["rsf_preds_raw"]

    mask_definite = (T_test > HORIZON) | (E_test == 1)
    y_5y = ((E_test[mask_definite] == 1) & (T_test[mask_definite] <= HORIZON)).astype(int)
    risk_definite = risk[mask_definite]

    risk_min = risk_definite.min()
    risk_max = risk_definite.max()
    prob_proxy = (risk_definite - risk_min) / (risk_max - risk_min + 1e-12)

    observed, predicted = calibration_curve(y_5y, prob_proxy, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(
        predicted,
        observed,
        "s-",
        color="#1976D2",
        linewidth=2,
        markersize=8,
        label=f"RSF Trimodal (n={len(y_5y)})",
    )
    ax.set_xlabel("Mean predicted risk (normalised)", fontsize=11)
    ax.set_ylabel("Observed 5-year mortality fraction", fontsize=11)
    ax.set_title("RSF Trimodal - 5-Year Calibration (held-out test set)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_path if output_path is not None else OUTPUT_DIR / "rsf_calibration.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    auc = roc_auc_score(y_5y, prob_proxy)
    print(f"  Saved {path}")
    print(f"  5-year binary AUC on definite subset: {auc:.4f}")
    print(f"  Calibration bins: predicted={np.round(predicted, 3)}")
    print(f"                    observed ={np.round(observed, 3)}")
    return path



# subgroup bootstrap Cox
def fit_cox_on_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    T_train: np.ndarray,
    E_train: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    l1_ratio: float = 0.0,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Fit CoxPH on training set, return C-indices and predictions.
    Returns: (train_c_index, test_c_index, test_predictions, train_predictions)
    """
    penalty = tune_cox_penalizer(X_train, T_train, E_train, l1_ratio=l1_ratio)
    train_df = X_train.copy()
    train_df["T"] = T_train
    train_df["E"] = E_train

    model = CoxPHFitter(penalizer=penalty, l1_ratio=l1_ratio)
    model.fit(train_df, duration_col="T", event_col="E", show_progress=False)

    preds_test = model.predict_partial_hazard(X_test).values.ravel()
    preds_train = model.predict_partial_hazard(X_train).values.ravel()
    c_index_test = concordance_index(T_test, -preds_test, E_test)
    c_index_train = concordance_index(T_train, -preds_train, E_train)
    return c_index_train, c_index_test, preds_test, preds_train


def bootstrap_c_index_diff_subgroup(
    T: np.ndarray,
    E: np.ndarray,
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    B: int = 1000,
    seed: int = SEED,
) -> dict[str, float]:
    """
    Paired bootstrap resampling on test-set subgroup patients.
    Returns dict with C-indices and significance metrics.
    """
    rng = np.random.RandomState(seed)
    n = len(T)

    if n < 2:
        return {
            "c_a": np.nan,
            "c_b": np.nan,
            "delta": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "B": 0,
        }

    c_a = concordance_index(T, -risk_a, E)
    c_b = concordance_index(T, -risk_b, E)
    observed_delta = c_a - c_b

    deltas = np.zeros(B)
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        try:
            c_boot_a = concordance_index(T[idx], -risk_a[idx], E[idx])
            c_boot_b = concordance_index(T[idx], -risk_b[idx], E[idx])
            deltas[b] = c_boot_a - c_boot_b
        except Exception:
            deltas[b] = np.nan

    deltas = deltas[~np.isnan(deltas)]

    if len(deltas) == 0:
        return {
            "c_a": float(c_a),
            "c_b": float(c_b),
            "delta": float(observed_delta),
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "B": 0,
        }

    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    p_value = np.mean(deltas <= 0) if observed_delta > 0 else np.mean(deltas >= 0)
    p_value = min(2 * p_value, 1.0)

    return {
        "c_a": float(c_a),
        "c_b": float(c_b),
        "delta": float(observed_delta),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "B": int(len(deltas)),
    }


def prepare_pam50_subgroup_modalities(pca_components: int) -> dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]]:
    """
    Load and prepare modalities with PAM50 subtype information.
    Returns dict mapping modality -> (X, T, E, pam50_subtype)
    """
    clin = load_clinical_table()
    clin = clin.dropna(subset=["event", "os_months"]).copy()
    clin = clin[clin["os_months"] > 0].copy()

    mutation = load_mutation_features()
    mrna = load_mrna_matrix()

    clinical_columns = get_survival_feature_columns(clin)
    clin_features = clin[clinical_columns].copy()
    clinical_ids = set(clin_features.index)
    mutation_ids = set(mutation.index)
    mrna_ids = set(mrna.index)

    # Use the same patient set for every modality (intersection)
    core_ids = sorted(clinical_ids & mutation_ids & mrna_ids)

    modality_ids = OrderedDict(
        [
            ("Clinical", core_ids),
            ("Trimodal", core_ids),
        ]
    )

    prepared = {}
    for modality, ids in modality_ids.items():
        T = clin.loc[ids, "os_months"].astype(float).values
        E = clin.loc[ids, "event"].astype(int).values
        pam50 = clin.loc[ids, "pam50_+_claudin-low_subtype"].astype(str).str.strip()

        if modality == "Clinical":
            X = clin_features.loc[ids].copy()
        else:  # Trimodal
            X = pd.concat([clin_features.loc[ids], mutation.loc[ids], mrna.loc[ids]], axis=1)

        prepared[modality] = (X, T, E, pam50)

    return prepared


def _run_subgroup_bootstrap_cox_inner(
    pca_components: int = 20,
    B: int = 1000,
) -> pd.DataFrame:
    """
    Run paired bootstrap significance tests for each PAM50 subtype.
    """
    print("\nSubgroup bootstrap (CoxPH)")
    modalities = prepare_pam50_subgroup_modalities(pca_components=pca_components)

    # Get the shared patients and do a single 80/20 train/test split
    X_clinical, T_all, E_all, pam50_all = modalities["Clinical"]

    # Do stratified train/test split on the full dataset
    train_idx, test_idx = train_test_split(
        np.arange(len(X_clinical)),
        test_size=0.2,
        stratify=E_all,
        random_state=SEED,
    )

    print(f"Train set size: {len(train_idx)}, Test set size: {len(test_idx)}")

    # Prepare data for both modalities
    results_by_modality = {}

    for modality in ["Clinical", "Trimodal"]:
        X_raw, T_all_mod, E_all_mod, pam50_all_mod = modalities[modality]

        X_train_raw = X_raw.iloc[train_idx].copy()
        X_test_raw = X_raw.iloc[test_idx].copy()
        T_train = T_all_mod[train_idx]
        E_train = E_all_mod[train_idx]
        T_test = T_all_mod[test_idx]
        E_test = E_all_mod[test_idx]
        pam50_test = pam50_all_mod.iloc[test_idx].values

        # Preprocess modality
        if modality == "Clinical":
            X_train, X_test = encode_and_scale_clinical(X_train_raw, X_test_raw)
        else:  # Trimodal
            clin_cols = [c for c in X_train_raw.columns if not c.startswith(("MUT__", "MRNA__"))]
            mut_cols = [c for c in X_train_raw.columns if c.startswith("MUT__")]
            mrna_cols = [c for c in X_train_raw.columns if c.startswith("MRNA__")]

            parts_train = []
            parts_test = []

            if clin_cols:
                a, b = encode_and_scale_clinical(X_train_raw[clin_cols], X_test_raw[clin_cols])
                parts_train.append(a)
                parts_test.append(b)
            if mut_cols:
                a, b = preprocess_mutation(X_train_raw[mut_cols], X_test_raw[mut_cols])
                parts_train.append(a)
                parts_test.append(b)
            if mrna_cols:
                a, b = preprocess_mrna(
                    X_train_raw[mrna_cols],
                    X_test_raw[mrna_cols],
                    n_components=pca_components,
                )
                parts_train.append(a)
                parts_test.append(b)

            X_train = pd.concat(parts_train, axis=1)
            X_test = pd.concat(parts_test, axis=1)

        # Fit CoxPH on full training set
        _, _, preds_test, _ = fit_cox_on_train(
            X_train, X_test, T_train, E_train, T_test, E_test
        )

        results_by_modality[modality] = {
            "preds_test": preds_test,
            "pam50_test": pam50_test,
            "T_test": T_test,
            "E_test": E_test,
        }

    # Now compute per-subtype comparisons
    clinical_preds = results_by_modality["Clinical"]["preds_test"]
    trimodal_preds = results_by_modality["Trimodal"]["preds_test"]
    pam50_test = results_by_modality["Clinical"]["pam50_test"]
    T_test = results_by_modality["Clinical"]["T_test"]
    E_test = results_by_modality["Clinical"]["E_test"]

    # Get unique subtypes
    unique_subtypes = pd.Series(pam50_test).dropna().unique()
    unique_subtypes = sorted([s for s in unique_subtypes if s.strip()])

    print(f"\nFound {len(unique_subtypes)} PAM50 subtypes: {unique_subtypes}")

    rows = []
    for subtype in unique_subtypes:
        mask = (pam50_test == subtype) | (pam50_test == subtype.strip())
        n_subtype = mask.sum()

        if n_subtype < 2:
            print(f"  {subtype}: n={n_subtype} (skipped, too small)")
            continue

        T_sub = T_test[mask]
        E_sub = E_test[mask]
        risk_clinical_sub = clinical_preds[mask]
        risk_trimodal_sub = trimodal_preds[mask]

        # Run bootstrap test
        result = bootstrap_c_index_diff_subgroup(
            T_sub, E_sub, risk_trimodal_sub, risk_clinical_sub, B=B, seed=SEED
        )

        rows.append({
            "subtype": subtype.strip(),
            "n_test": int(n_subtype),
            "c_clinical": round(result["c_b"], 4),
            "c_trimodal": round(result["c_a"], 4),
            "delta": round(result["delta"], 4),
            "ci_lower": round(result["ci_lower"], 4),
            "ci_upper": round(result["ci_upper"], 4),
            "p_value": round(result["p_value"], 4),
            "B": result["B"],
        })

        sig_str = "*" if result["p_value"] < 0.05 else ""
        print(f"  {subtype}: n={n_subtype}, "
              f"C_clin={result['c_b']:.4f}, C_trim={result['c_a']:.4f}, "
              f"delta={result['delta']:.4f}, p={result['p_value']:.4f}{sig_str}")

    results_df = pd.DataFrame(rows)
    return results_df


# subgroup bootstrap RSF
# Fixed RSF hyperparameters (matches subgroup_bootstrap_rsf.py)
RSF_PARAMS = {
    "n_estimators": 200,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
}


def fit_rsf_on_train(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    T_train: np.ndarray,
    E_train: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Fit RSF on training set with fixed hyperparameters, return C-indices and predictions.
    Returns: (train_c_index, test_c_index, test_risk_scores, train_risk_scores)

    RSF predict() returns higher = more risk. For concordance_index use -risk.
    """
    y_train = make_surv_array(E_train, T_train)

    model = RandomSurvivalForest(
        random_state=SEED,
        n_jobs=1,
        **RSF_PARAMS,
    )
    model.fit(X_train.values, y_train)

    preds_test = model.predict(X_test.values)
    preds_train = model.predict(X_train.values)

    # RSF: higher risk = shorter survival => use -risk for concordance_index
    c_index_test = concordance_index(T_test, -preds_test, E_test)
    c_index_train = concordance_index(T_train, -preds_train, E_train)

    return c_index_train, c_index_test, preds_test, preds_train

def _run_subgroup_bootstrap_rsf_inner(
    pca_components: int = 20,
    B: int = 1000,
) -> pd.DataFrame:
    """
    Run paired bootstrap significance tests for each PAM50 subtype using RSF.
    """
    print("\nSubgroup bootstrap (RSF)")
    print(f"RSF hyperparameters: {RSF_PARAMS}")
    modalities = prepare_pam50_subgroup_modalities(pca_components=pca_components)

    # Get the shared patients and do a single 80/20 stratified train/test split
    X_clinical, T_all, E_all, pam50_all = modalities["Clinical"]

    train_idx, test_idx = train_test_split(
        np.arange(len(X_clinical)),
        test_size=0.2,
        stratify=E_all,
        random_state=SEED,
    )

    print(f"Train set size: {len(train_idx)}, Test set size: {len(test_idx)}")

    # Prepare data for both modalities and fit RSF
    results_by_modality = {}

    for modality in ["Clinical", "Trimodal"]:
        print(f"\nFitting RSF {modality}...")
        X_raw, T_all_mod, E_all_mod, pam50_all_mod = modalities[modality]

        X_train_raw = X_raw.iloc[train_idx].copy()
        X_test_raw = X_raw.iloc[test_idx].copy()
        T_train = T_all_mod[train_idx]
        E_train = E_all_mod[train_idx]
        T_test = T_all_mod[test_idx]
        E_test = E_all_mod[test_idx]
        pam50_test = pam50_all_mod.iloc[test_idx].values

        # Preprocess modality (matches clean_survival_shortlist.py exactly)
        if modality == "Clinical":
            X_train, X_test = encode_and_scale_clinical(X_train_raw, X_test_raw)
        else:  # Trimodal
            clin_cols = [c for c in X_train_raw.columns if not c.startswith(("MUT__", "MRNA__"))]
            mut_cols = [c for c in X_train_raw.columns if c.startswith("MUT__")]
            mrna_cols = [c for c in X_train_raw.columns if c.startswith("MRNA__")]

            parts_train = []
            parts_test = []

            if clin_cols:
                a, b = encode_and_scale_clinical(X_train_raw[clin_cols], X_test_raw[clin_cols])
                parts_train.append(a)
                parts_test.append(b)
            if mut_cols:
                a, b = preprocess_mutation(X_train_raw[mut_cols], X_test_raw[mut_cols])
                parts_train.append(a)
                parts_test.append(b)
            if mrna_cols:
                a, b = preprocess_mrna(
                    X_train_raw[mrna_cols],
                    X_test_raw[mrna_cols],
                    n_components=pca_components,
                )
                parts_train.append(a)
                parts_test.append(b)

            X_train = pd.concat(parts_train, axis=1)
            X_test = pd.concat(parts_test, axis=1)

        # Fit RSF on full training set
        c_train, c_test, preds_test, _ = fit_rsf_on_train(
            X_train, X_test, T_train, E_train, T_test, E_test
        )

        print(f"  RSF {modality}: Train C-index={c_train:.4f}, Test C-index={c_test:.4f}")

        results_by_modality[modality] = {
            "preds_test": preds_test,
            "pam50_test": pam50_test,
            "T_test": T_test,
            "E_test": E_test,
        }

    # Compute per-subtype comparisons
    clinical_preds = results_by_modality["Clinical"]["preds_test"]
    trimodal_preds = results_by_modality["Trimodal"]["preds_test"]
    pam50_test = results_by_modality["Clinical"]["pam50_test"]
    T_test = results_by_modality["Clinical"]["T_test"]
    E_test = results_by_modality["Clinical"]["E_test"]

    # Get unique subtypes
    unique_subtypes = pd.Series(pam50_test).dropna().unique()
    unique_subtypes = sorted([s for s in unique_subtypes if s.strip()])

    print(f"\nFound {len(unique_subtypes)} PAM50 subtypes: {unique_subtypes}")
    print(f"\nRunning B={B} paired bootstrap per subtype...")

    rows = []
    for subtype in unique_subtypes:
        mask = (pam50_test == subtype) | (pam50_test == subtype.strip())
        n_subtype = mask.sum()

        if n_subtype < 2:
            print(f"  {subtype}: n={n_subtype} (skipped, too small)")
            continue

        T_sub = T_test[mask]
        E_sub = E_test[mask]
        risk_clinical_sub = clinical_preds[mask]
        risk_trimodal_sub = trimodal_preds[mask]

        # Run bootstrap test: trimodal (a) vs clinical (b), delta = C(trimodal) - C(clinical)
        result = bootstrap_c_index_diff_subgroup(
            T_sub, E_sub, risk_trimodal_sub, risk_clinical_sub, B=B, seed=SEED
        )

        rows.append({
            "subtype": subtype.strip(),
            "n_test": int(n_subtype),
            "c_clinical": round(result["c_b"], 4),
            "c_trimodal": round(result["c_a"], 4),
            "delta": round(result["delta"], 4),
            "ci_lower": round(result["ci_lower"], 4),
            "ci_upper": round(result["ci_upper"], 4),
            "p_value": round(result["p_value"], 4),
        })

        sig_str = "*" if result["p_value"] < 0.05 else ""
        print(f"  {subtype:20s}: n={n_subtype:3d}, "
              f"C_clin={result['c_b']:.4f}, C_rsf_tri={result['c_a']:.4f}, "
              f"delta={result['delta']:+.4f}, p={result['p_value']:.4f}{sig_str}")

    results_df = pd.DataFrame(rows)
    return results_df
# TCGA external validation
STUDY_ID = "brca_tcga_pan_can_atlas_2018"
TCGA_DATA_README = """TCGA-BRCA data for external validation (produced when running survival_consolidated.py).

Study: brca_tcga_pan_can_atlas_2018 (TCGA Pan-Cancer Atlas 2018) via cBioPortal REST API.
Endpoint (patient clinical, JSON): https://www.cbioportal.org/api/studies/brca_tcga_pan_can_atlas_2018/clinical-data?clinicalDataType=PATIENT&projection=DETAILED

Files:
- tcga_brca_clinical_raw.csv — wide table of all patient-level clinical attributes returned by the API (one row per patient).
- tcga_brca_harmonised.csv — subset mapped to METABRIC RSF features (age_at_diagnosis, lymph_nodes_examined_positive, subtype) plus os_months and event used in the pilot.

Regenerate: from the repository root, run `python survival_consolidated.py` (TCGA external validation runs as a step in the consolidated survival pipeline). If harmonised or raw CSVs already exist in this folder, the pipeline reuses them and avoids re-downloading.
"""


def save_tcga_downloads(
    save_dir,
    tcga_raw: pd.DataFrame,
    feat: pd.DataFrame,
    T: np.ndarray,
    E: np.ndarray,
) -> None:
    """Write raw API table and harmonised feature/survival table for offline reuse."""
    save_dir = save_dir.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    raw_path = save_dir / "tcga_brca_clinical_raw.csv"
    tcga_raw.to_csv(raw_path, encoding="utf-8")
    harmonised = feat.copy()
    harmonised["os_months"] = T
    harmonised["event"] = E
    harmonised.to_csv(save_dir / "tcga_brca_harmonised.csv", encoding="utf-8")
    readme = save_dir / "README.txt"
    readme.write_text(TCGA_DATA_README, encoding="utf-8")
    print(f"  Saved raw clinical: {raw_path}")
    print(f"  Saved harmonised:   {save_dir / 'tcga_brca_harmonised.csv'}")
    print(f"  Saved notes:        {readme}")


def download_tcga_clinical() -> pd.DataFrame:
    """Fetch TCGA-BRCA clinical data via cBioPortal API."""
    url = (
        f"https://www.cbioportal.org/api/studies/{STUDY_ID}/clinical-data"
        "?clinicalDataType=PATIENT&projection=DETAILED"
    )
    print(f"  Downloading from cBioPortal API ({STUDY_ID})...")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    data = urllib.request.urlopen(req, timeout=60).read()
    records = json.loads(data)

    patients: dict[str, dict] = defaultdict(dict)
    for r in records:
        patients[r["patientId"]][r["clinicalAttributeId"]] = r["value"]

    df = pd.DataFrame.from_dict(patients, orient="index")
    df.index.name = "PATIENT_ID"
    print(f"  Downloaded {len(df)} patients, {len(df.columns)} clinical attributes")
    return df


def harmonise_tcga(tcga: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Map TCGA-BRCA to a feature set compatible with METABRIC RSF Clinical."""
    tcga = tcga.copy()

    tcga["os_months"] = pd.to_numeric(tcga["OS_MONTHS"], errors="coerce")
    status = tcga["OS_STATUS"].astype(str).str.strip()
    tcga["event"] = status.map(lambda s: 1.0 if "DECEASED" in s.upper() else 0.0)

    tcga = tcga.dropna(subset=["os_months", "event"])
    tcga = tcga[tcga["os_months"] > 0]

    feat = pd.DataFrame(index=tcga.index)
    feat["age_at_diagnosis"] = pd.to_numeric(tcga["AGE"], errors="coerce")

    if "PATH_N_STAGE" in tcga.columns:
        n_stage = tcga["PATH_N_STAGE"].astype(str).str.upper()
        node_map = {
            "N0": 0, "N0 (I-)": 0, "N0 (I+)": 0, "N0 (MOL+)": 0,
            "N1": 2, "N1A": 1, "N1B": 1, "N1C": 2, "N1MI": 1,
            "N2": 5, "N2A": 5, "N3": 10, "N3A": 10, "N3B": 10, "N3C": 10,
            "NX": np.nan,
        }
        feat["lymph_nodes_examined_positive"] = n_stage.map(node_map)

    if "SUBTYPE" in tcga.columns:
        feat["subtype"] = tcga["SUBTYPE"].astype(str).str.strip()

    T = tcga["os_months"].values.astype(float)
    E = tcga["event"].values.astype(int)

    print(f"  Harmonised: {len(feat)} patients, {feat.shape[1]} features")
    print(f"  Events: {E.sum():.0f}/{len(E)} ({100*E.mean():.1f}%)")
    return feat, T, E


def prepare_metabric_clinical():
    """Prepare METABRIC clinical data with shared features."""
    clin = load_clinical_table()
    clin = clin.dropna(subset=["event", "os_months"]).copy()
    clin = clin[clin["os_months"] > 0].copy()

    T = clin["os_months"].values.astype(float)
    E = clin["event"].values.astype(int)

    feat = pd.DataFrame(index=clin.index)
    if "age_at_diagnosis" in clin.columns:
        feat["age_at_diagnosis"] = clin["age_at_diagnosis"]
    if "lymph_nodes_examined_positive" in clin.columns:
        feat["lymph_nodes_examined_positive"] = clin["lymph_nodes_examined_positive"]

    for col in ["pam50_+_claudin-low_subtype", "CLAUDIN_SUBTYPE"]:
        if col in clin.columns:
            feat["subtype"] = clin[col].astype(str).str.strip()
            break

    return feat, T, E


def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    parts_tr, parts_te = [], []
    if num_cols:
        num_imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        parts_tr.append(scaler.fit_transform(num_imp.fit_transform(X_train[num_cols])))
        parts_te.append(scaler.transform(num_imp.transform(X_test[num_cols])))

    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        parts_tr.append(enc.fit_transform(cat_imp.fit_transform(X_train[cat_cols])))
        parts_te.append(enc.transform(cat_imp.transform(X_test[cat_cols])))

    return np.hstack(parts_tr), np.hstack(parts_te)

def run_tcga_validation() -> None:
    ensure_outputs_dir()
    print("TCGA-BRCA external validation")
    print("loading METABRIC...")
    met_X, met_T, met_E = prepare_metabric_clinical()
    print(f"  METABRIC: {len(met_X)} patients, features: {list(met_X.columns)}")

    cache_dir = (DATA_DIR / "tcga_brca_pan_can_atlas_2018").resolve()
    harmonised_path = cache_dir / "tcga_brca_harmonised.csv"
    raw_path = cache_dir / "tcga_brca_clinical_raw.csv"
    if harmonised_path.exists():
        print(f"\nloading cached TCGA harmonised: {harmonised_path}")
        harmonised = pd.read_csv(harmonised_path, index_col=0)
        tcga_X = harmonised.drop(columns=["os_months", "event"], errors="ignore")
        tcga_T = harmonised["os_months"].values.astype(float)
        tcga_E = harmonised["event"].values.astype(int)
        print(f"  Loaded {len(tcga_X)} patients from cache.")
    elif raw_path.exists():
        print(f"\nloading cached TCGA raw: {raw_path}")
        tcga_raw = pd.read_csv(raw_path, index_col=0, encoding="utf-8")
        tcga_X, tcga_T, tcga_E = harmonise_tcga(tcga_raw)
        print(f"  Harmonised {len(tcga_X)} patients from cached raw.")
    else:
        print("\ndownloading TCGA-BRCA...")
        try:
            tcga_raw = download_tcga_clinical()
        except Exception as exc:
            print(f"  Download failed ({exc!r}); cannot proceed without cache or network.")
            return
        tcga_X, tcga_T, tcga_E = harmonise_tcga(tcga_raw)
        try:
            save_dir = cache_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            save_tcga_downloads(save_dir, tcga_raw, tcga_X, tcga_T, tcga_E)
        except Exception as exc:
            print(f"  (Optional) Could not save TCGA cache: {exc!r}")

    shared = sorted(set(met_X.columns) & set(tcga_X.columns))
    print(f"\n  Shared features: {shared}")
    met_X = met_X[shared]
    tcga_X = tcga_X[shared]

    print(f"\nRSF: train METABRIC n={len(met_X)}, test TCGA n={len(tcga_X)}")
    X_train_enc, X_test_enc = encode_features(met_X, tcga_X)
    y_train = Surv.from_arrays(event=met_E.astype(bool), time=met_T)

    rsf = RandomSurvivalForest(
        n_estimators=200, min_samples_leaf=15, max_features="sqrt",
        random_state=SEED, n_jobs=1,
    )
    rsf.fit(X_train_enc, y_train)
    risk_tcga = rsf.predict(X_test_enc)
    risk_met_train = rsf.predict(X_train_enc)
    c_internal = float(concordance_index(met_T, -risk_met_train, met_E))

    c_ext = concordance_index(tcga_T, -risk_tcga, tcga_E)

    rng = np.random.RandomState(SEED)
    n = len(tcga_T)
    boot_cs = []
    for _ in range(1000):
        idx = rng.randint(0, n, size=n)
        try:
            boot_cs.append(concordance_index(tcga_T[idx], -risk_tcga[idx], tcga_E[idx]))
        except Exception:
            continue
    ci_lo, ci_hi = np.percentile(boot_cs, 2.5), np.percentile(boot_cs, 97.5)

    print("\nresults:")
    print(f"  External C-index (TCGA-BRCA): {c_ext:.4f}")
    print(f"  95% Bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Internal C-index (METABRIC train resubstitution): {c_internal:.4f}")
    print(f"  Drop: {c_internal - c_ext:.4f}")

    results = pd.DataFrame([{
        "Dataset": "TCGA-BRCA (PanCancer Atlas)",
        "Model": "RSF Clinical",
        "Train_Set": f"METABRIC (N={len(met_X)})",
        "N_test": len(tcga_X),
        "Features": ", ".join(shared),
        "External_C_index": round(c_ext, 4),
        "CI_lower": round(ci_lo, 4),
        "CI_upper": round(ci_hi, 4),
        "Internal_C_index": round(c_internal, 4),
        "Delta": round(c_internal - c_ext, 4),
    }])
    out = OUTPUT_DIR / "tcga_external_validation.csv"
    results.to_csv(out, index=False)
    print(f"\n  Saved {out}")


# rsf figures + thesis export
def figures_rsf_clinical() -> None:
    """RSF clinical: Spearman importance + subgroup C-indices (matches thesis style)."""
    clin = load_clinical_table()
    clin = clin.dropna(subset=["event", "os_months"]).copy()
    clin = clin[clin["os_months"] > 0].copy()
    leak = {
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
    feat_cols = [c for c in clin.columns if c not in leak]
    X_df = clin[feat_cols].copy()
    T = clin["os_months"].astype(float).values
    E = clin["event"].astype(int).values
    pids = X_df.index
    idx_tr, idx_te = train_test_split(
        np.arange(len(pids)), test_size=0.2, stratify=E, random_state=SEED
    )
    X_train_df, X_test_df = X_df.iloc[idx_tr], X_df.iloc[idx_te]
    num_cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train_df.columns if c not in num_cols]

    def preprocess(X_tr, X_te):
        num_imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        Xn_tr = scaler.fit_transform(num_imp.fit_transform(X_tr[num_cols]))
        Xn_te = scaler.transform(num_imp.transform(X_te[num_cols]))
        if cat_cols:
            cat_imp = SimpleImputer(strategy="most_frequent")
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            Xc_tr = enc.fit_transform(cat_imp.fit_transform(X_tr[cat_cols]))
            Xc_te = enc.transform(cat_imp.transform(X_te[cat_cols]))
            return np.hstack([Xn_tr, Xc_tr]), np.hstack([Xn_te, Xc_te])
        return Xn_tr, Xn_te

    X_tr, X_te = preprocess(X_train_df, X_test_df)
    T_tr, T_te = T[idx_tr], T[idx_te]
    E_tr, E_te = E[idx_tr], E[idx_te]
    y_tr = Surv.from_arrays(event=E_tr.astype(bool), time=T_tr)
    feat_names = num_cols + cat_cols if cat_cols else num_cols

    rsf = RandomSurvivalForest(
        n_estimators=200, min_samples_leaf=15, max_features="sqrt", random_state=SEED, n_jobs=1
    )
    rsf.fit(X_tr, y_tr)
    risk_te = rsf.predict(X_te)
    c_idx = float(concordance_index_censored(E_te.astype(bool), T_te, risk_te)[0])

    correlations = []
    for i in range(X_te.shape[1]):
        rho, _ = spearmanr(X_te[:, i], risk_te)
        correlations.append(abs(rho) if rho == rho else 0.0)
    correlations = np.array(correlations)
    top_n = min(18, len(correlations))
    top_idx = np.argsort(correlations)[::-1][:top_n]
    clean = lambda n: str(n).replace("_", " ").title()
    names = [clean(feat_names[i]) for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(names[::-1], correlations[top_idx][::-1], color=plt.cm.Blues_r(np.linspace(0.3, 0.9, top_n)))
    ax.set_xlabel("|Spearman ρ| with predicted risk")
    ax.set_title(f"RSF clinical — importance proxy (test C-index = {c_idx:.4f})")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "rsf_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    test_pids = pids[idx_te]
    clin_test = clin.loc[test_pids]
    sub_rows = []

    def sub_c(mask, label):
        mask = np.asarray(mask)
        if mask.sum() < 15:
            return
        try:
            c = concordance_index_censored(
                E_te[mask].astype(bool), T_te[mask], risk_te[mask]
            )[0]
            sub_rows.append(
                {
                    "Subgroup": label,
                    "N": int(mask.sum()),
                    "Events": int(E_te[mask].sum()),
                    "C_index": round(float(c), 4),
                }
            )
        except Exception:
            return

    for col in ["ER_IHC", "er_status_measured_by_ihc"]:
        if col in clin_test.columns:
            er = clin_test[col].astype(str).str.strip().str.lower()
            sub_c(er == "positive", "ER Positive")
            sub_c(er == "negative", "ER Negative")
            break
    for col in ["pam50_+_claudin-low_subtype", "CLAUDIN_SUBTYPE"]:
        if col in clin_test.columns:
            pam = clin_test[col].astype(str).str.strip()
            for st in sorted(pam.unique()):
                if str(st).lower() in {"nan", "none", ""}:
                    continue
                sub_c((pam == st).values, str(st))
            break
    if "age_at_diagnosis" in clin_test.columns:
        age = pd.to_numeric(clin_test["age_at_diagnosis"], errors="coerce")
        sub_c((age < 50).fillna(False).values, "Age <50")
        sub_c(((age >= 50) & (age < 60)).fillna(False).values, "Age 50-60")
        sub_c(((age >= 60) & (age < 70)).fillna(False).values, "Age 60-70")
        sub_c((age >= 70).fillna(False).values, "Age 70+")

    pd.DataFrame(sub_rows).to_csv(OUTPUT_DIR / "rsf_subgroup_results_cursor.csv", index=False)
    print("Saved rsf_feature_importance.png and rsf_subgroup_results_cursor.csv")

def generate_subtype_comparison_figure() -> None:
    """Grouped bar chart: CoxPH/RSF clinical vs trimodal C-index by PAM50 subtype."""
    cox_path = OUTPUT_DIR / "subgroup_bootstrap_significance.csv"
    rsf_path = OUTPUT_DIR / "subgroup_bootstrap_rsf.csv"
    if not cox_path.exists() or not rsf_path.exists():
        print("generate_subtype_comparison_figure: missing subgroup CSVs; skip.")
        return

    cox = pd.read_csv(cox_path)
    rsf = pd.read_csv(rsf_path)

    def norm_subtype(s: str) -> str:
        t = str(s).strip().lower().replace("_", "-")
        mapping = {
            "luma": "LumA",
            "lumb": "LumB",
            "normal": "Normal",
            "claudin-low": "Claudin-low",
            "her2": "HER2",
            "basal": "Basal",
        }
        return mapping.get(t, str(s).strip())

    order = ["LumA", "LumB", "Normal", "Claudin-low", "HER2", "Basal"]
    cox["st"] = cox["subtype"].map(norm_subtype)
    rsf["st"] = rsf["subtype"].map(norm_subtype)

    x = np.arange(len(order))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))

    def row_for(df: pd.DataFrame, st: str) -> pd.Series | None:
        m = df["st"] == st
        if m.any():
            return df.loc[m].iloc[0]
        return None

    cox_c = []
    cox_t = []
    rsf_c = []
    rsf_t = []
    rsf_delta = []
    for st in order:
        cr = row_for(cox, st)
        rr = row_for(rsf, st)
        cox_c.append(float(cr["c_clinical"]) if cr is not None else np.nan)
        cox_t.append(float(cr["c_trimodal"]) if cr is not None else np.nan)
        rsf_c.append(float(rr["c_clinical"]) if rr is not None else np.nan)
        rsf_t.append(float(rr["c_trimodal"]) if rr is not None else np.nan)
        rsf_delta.append(float(rr["c_trimodal"]) - float(rr["c_clinical"]) if rr is not None else np.nan)

    ax.bar(x - 1.5 * width, cox_c, width, label="CoxPH Clinical", color="#3949AB")
    ax.bar(x - 0.5 * width, cox_t, width, label="CoxPH Trimodal", color="#5C6BC0")
    ax.bar(x + 0.5 * width, rsf_c, width, label="RSF Clinical", color="#C62828")
    ax.bar(x + 1.5 * width, rsf_t, width, label="RSF Trimodal", color="#E53935")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random (C=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("C-index (held-out test, by subtype)")
    ax.set_title("Clinical vs trimodal survival models across PAM50 subtypes")
    ax.set_ylim(0.35, 0.85)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    for i, st in enumerate(order):
        d = rsf_delta[i]
        if d == d:
            ax.annotate(
                f"RSF Δ={d:+.3f}",
                xy=(x[i], max(rsf_c[i], rsf_t[i], cox_c[i], cox_t[i], 0.5) + 0.02),
                ha="center",
                fontsize=8,
                color="#B71C1C",
            )

    plt.tight_layout()
    outp = OUTPUT_DIR / "subtype_cindex_comparison.png"
    fig.savefig(outp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outp}")


def run_subgroup_bootstrap_cox() -> None:
    ensure_outputs_dir()
    results_df = _run_subgroup_bootstrap_cox_inner(pca_components=20, B=1000)
    output_path = OUTPUT_DIR / "subgroup_bootstrap_significance.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    print("\nResults summary:")
    print(results_df.to_string(index=False))


def run_subgroup_bootstrap_rsf() -> None:
    ensure_outputs_dir()
    results_df = _run_subgroup_bootstrap_rsf_inner(pca_components=20, B=1000)
    output_path = OUTPUT_DIR / "subgroup_bootstrap_rsf.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")
    print("\nResults summary:")
    print(results_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="METABRIC survival pipeline (thesis-aligned models).")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Smaller nested CV, lighter bootstraps/DeepSurv; skips subgroup figures and TCGA.",
    )
    args = parser.parse_args()
    survival_fast = bool(args.fast or SURVIVAL_FAST_ENV)

    ensure_outputs_dir()
    print("Survival")
    if survival_fast:
        print("Mode: --fast (smaller CV / lighter DeepSurv; subgroup + TCGA skipped)")

    print("primary survival benchmark...")
    primary_df = evaluate_survival_shortlist(pca_components=20, survival_fast=survival_fast)
    primary_df.to_csv(OUTPUT_DIR / "survival_results_primary.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'survival_results_primary.csv'}")

    print("survival bootstrap significance...")
    significance_df, survival_details = run_survival_significance(
        pca_components=20,
        survival_fast=survival_fast,
    )
    save_survival_significance(significance_df)

    print("RSF calibration...")
    run_rsf_calibration(survival_details)

    if survival_fast:
        print("Skipping CoxPH/RSF subgroup bootstraps, TCGA, and figures (--fast).")
    else:
        print("CoxPH subgroup bootstrap...")
        run_subgroup_bootstrap_cox()

        print("RSF subgroup bootstrap...")
        run_subgroup_bootstrap_rsf()

        print("TCGA external validation...")
        run_tcga_validation()

        print("RSF clinical figures...")
        figures_rsf_clinical()

        print("subtype comparison...")
        generate_subtype_comparison_figure()

    print("done.")


if __name__ == "__main__":
    main()
