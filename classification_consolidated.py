from __future__ import annotations

# imports
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import calibration_curve

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import shap
except ImportError:
    shap = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# constants + data load
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


# feature prep
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


def load_classification_datasets() -> tuple[OrderedDict[str, pd.DataFrame], pd.Series, list[str], list[str]]:
    clin = load_clinical_table()
    clin, y_5y = build_five_year_labels(clin, horizon=HORIZON)
    mutation = load_mutation_features()
    mrna = load_mrna_matrix()

    clin, y_5y, mutation, mrna = align_frames(clin, y_5y, mutation, mrna)
    clinical_num_cols, clinical_cat_cols = get_classification_feature_columns(clin)
    clinical = clin[clinical_num_cols + clinical_cat_cols].copy()

    datasets = OrderedDict(
        [
            ("Clinical", clinical),
            ("Clin+Mut", pd.concat([clinical, mutation], axis=1)),
            ("mRNA", mrna.copy()),
            ("Trimodal", pd.concat([clinical, mutation, mrna], axis=1)),
        ]
    )
    return datasets, y_5y.astype(int), clinical_num_cols, clinical_cat_cols


def build_preprocessor(
    X: pd.DataFrame,
    clinical_num_cols: list[str],
    clinical_cat_cols: list[str],
    pca_components: int,
) -> ColumnTransformer:
    clinical_num = [column for column in clinical_num_cols if column in X.columns]
    clinical_cat = [column for column in clinical_cat_cols if column in X.columns]
    mutation_cols = [column for column in X.columns if column.startswith("MUT__")]
    mrna_cols = [column for column in X.columns if column.startswith("MRNA__")]

    transformers = []
    if clinical_num:
        transformers.append(
            (
                "clinical_num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                clinical_num,
            )
        )
    if clinical_cat:
        transformers.append(
            (
                "clinical_cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ohe",
                            OneHotEncoder(
                                drop="first",
                                sparse_output=False,
                                handle_unknown="ignore",
                            ),
                        ),
                    ]
                ),
                clinical_cat,
            )
        )
    if mutation_cols:
        transformers.append(
            (
                "mutation",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                mutation_cols,
            )
        )
    if mrna_cols:
        transformers.append(
            (
                "mrna",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("topvar", TopVarianceSelector(k=5000)),
                        ("scaler", StandardScaler()),
                        (
                            "pca",
                            PCA(
                                n_components=min(pca_components, max(2, len(mrna_cols) - 1)),
                                random_state=SEED,
                            ),
                        ),
                    ]
                ),
                mrna_cols,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


# model defs
def get_models(pos_weight: float) -> OrderedDict[str, object]:
    models: OrderedDict[str, object] = OrderedDict()
    models["LogReg"] = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=0.5,
        max_iter=5000,
        class_weight="balanced",
        random_state=SEED,
    )
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=600,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=1,
    )
    models["HistGB"] = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=SEED,
    )
    if xgb is not None:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            scale_pos_weight=pos_weight,
            eval_metric="aucpr",
            random_state=SEED,
            n_jobs=1,
            verbosity=0,
        )
    if lgb is not None:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            is_unbalance=True,
            random_state=SEED,
            n_jobs=1,
            verbose=-1,
        )
    models["AdaBoost"] = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=SEED),
        n_estimators=250,
        learning_rate=0.05,
        random_state=SEED,
    )
    models["SVC-RBF"] = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=SEED,
    )
    return models


def get_primary_models(pos_weight: float, dataset_name: str) -> OrderedDict[str, object]:
    all_models = get_models(pos_weight)
    keep = ["LogReg", "RandomForest", "HistGB", "AdaBoost", "SVC-RBF"]
    if dataset_name == "Trimodal" and "XGBoost" in all_models:
        keep.append("XGBoost")
    return OrderedDict((name, all_models[name]) for name in keep if name in all_models)


def get_trimodal_ensembles(pos_weight: float, primary_only: bool = True) -> OrderedDict[str, object]:
    base_models = [
        (
            "hgb",
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_depth=4,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=SEED,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=600,
                max_depth=12,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=1,
            ),
        ),
    ]
    if xgb is not None:
        base_models.append(
            (
                "xgb",
                xgb.XGBClassifier(
                    n_estimators=400,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=5.0,
                    scale_pos_weight=pos_weight,
                    eval_metric="aucpr",
                    random_state=SEED,
                    n_jobs=1,
                    verbosity=0,
                ),
            )
        )
    if lgb is not None and not primary_only:
        base_models.append(
            (
                "lgbm",
                lgb.LGBMClassifier(
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=5.0,
                    is_unbalance=True,
                    random_state=SEED,
                    n_jobs=1,
                    verbose=-1,
                ),
            )
        )

    ensembles: OrderedDict[str, object] = OrderedDict()
    ensembles["Stacking"] = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=SEED,
        ),
        cv=5,
        n_jobs=1,
        passthrough=False,
    )
    ensembles["Voting"] = VotingClassifier(estimators=base_models, voting="soft", n_jobs=1)
    return ensembles


# stats / tests
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n - 1 and sorted_x[k + 1] == sorted_x[k]:
            k += 1
        for j in range(i, k + 1):
            ranks[order[j]] = 0.5 * (i + k) + 1
        i = k + 1
    return ranks


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    aucs = np.zeros(k)
    v01 = np.zeros((k, n))
    v10 = np.zeros((k, m))

    for row in range(k):
        all_preds = np.concatenate([positive_examples[row], negative_examples[row]])
        ranks = _compute_midrank(all_preds)
        positive_ranks = ranks[:m]
        aucs[row] = (positive_ranks.sum() - m * (m + 1) / 2.0) / (m * n)

        ceiling = np.searchsorted(
            np.sort(negative_examples[row]), positive_examples[row], side="right"
        )
        v10[row] = ceiling / n

        ceiling2 = np.searchsorted(
            np.sort(positive_examples[row]), negative_examples[row], side="right"
        )
        v01[row] = 1.0 - ceiling2 / m

    s10 = np.cov(v10) if m > 1 else np.zeros((k, k))
    s01 = np.cov(v01) if n > 1 else np.zeros((k, k))
    if k == 1:
        s10 = np.atleast_2d(s10)
        s01 = np.atleast_2d(s01)
    return aucs, s10 / m + s01 / n


def delong_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    order = (-y_true).argsort()
    y_sorted = y_true[order]
    positives = int(y_sorted.sum())
    predictions = np.vstack([pred_a[order], pred_b[order]])
    aucs, cov = _fast_delong(predictions, positives)
    delta = aucs[0] - aucs[1]
    variance = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(max(variance, 1e-15))
    z_score = delta / se
    from scipy.stats import norm

    p_value = 2 * norm.sf(abs(z_score))
    return {
        "auc_1": float(aucs[0]),
        "auc_2": float(aucs[1]),
        "delta": float(delta),
        "se": float(se),
        "z": float(z_score),
        "p_value": float(p_value),
    }


def bootstrap_auc_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    B: int = 2000,
    seed: int = SEED,
) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    auc_a = roc_auc_score(y_true, pred_a)
    auc_b = roc_auc_score(y_true, pred_b)
    observed_delta = auc_a - auc_b

    deltas = np.zeros(B)
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2:
            deltas[b] = np.nan
            continue
        try:
            deltas[b] = roc_auc_score(y_boot, pred_a[idx]) - roc_auc_score(y_boot, pred_b[idx])
        except Exception:
            deltas[b] = np.nan

    deltas = deltas[~np.isnan(deltas)]
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    p_value = np.mean(deltas <= 0) if observed_delta > 0 else np.mean(deltas >= 0)
    p_value = min(2 * p_value, 1.0)
    return {
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "delta": float(observed_delta),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "B": int(len(deltas)),
    }


def run_classification_significance(pca_components: int = 50) -> pd.DataFrame:
    print("\nClassification significance")
    datasets, y, clinical_num_cols, clinical_cat_cols = load_classification_datasets()

    train_index, test_index = train_test_split(
        y.index,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )
    y_train = y.loc[train_index]
    y_test = y.loc[test_index]
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    trimodal_train = datasets["Trimodal"].loc[train_index]
    trimodal_test = datasets["Trimodal"].loc[test_index]
    trimodal_prep = build_preprocessor(
        trimodal_train,
        clinical_num_cols,
        clinical_cat_cols,
        pca_components,
    )
    voting_estimator = get_trimodal_ensembles(float(pos_weight), primary_only=True)["Voting"]
    stacking_estimator = get_trimodal_ensembles(float(pos_weight), primary_only=True)["Stacking"]
    voting_pipeline = Pipeline(
        [("preprocessor", trimodal_prep), ("model", clone(voting_estimator))]
    )
    stacking_pipeline = Pipeline(
        [("preprocessor", trimodal_prep), ("model", clone(stacking_estimator))]
    )
    voting_pipeline.fit(trimodal_train, y_train)
    stacking_pipeline.fit(trimodal_train, y_train)
    prob_trimodal_voting = voting_pipeline.predict_proba(trimodal_test)[:, 1]
    prob_trimodal_stacking = stacking_pipeline.predict_proba(trimodal_test)[:, 1]

    clinical_train = datasets["Clinical"].loc[train_index]
    clinical_test = datasets["Clinical"].loc[test_index]
    clinical_prep = build_preprocessor(
        clinical_train,
        clinical_num_cols,
        clinical_cat_cols,
        pca_components,
    )
    random_forest = get_primary_models(float(pos_weight), "Clinical")["RandomForest"]
    histgb = get_primary_models(float(pos_weight), "Clinical")["HistGB"]
    clinical_rf_pipeline = Pipeline(
        [("preprocessor", clinical_prep), ("model", clone(random_forest))]
    )
    clinical_hgb_pipeline = Pipeline(
        [("preprocessor", clinical_prep), ("model", clone(histgb))]
    )
    clinical_rf_pipeline.fit(clinical_train, y_train)
    clinical_hgb_pipeline.fit(clinical_train, y_train)
    prob_clinical_rf = clinical_rf_pipeline.predict_proba(clinical_test)[:, 1]
    prob_clinical_hgb = clinical_hgb_pipeline.predict_proba(clinical_test)[:, 1]

    y_array = y_test.values
    comparisons = [
        ("Trimodal Voting vs Clinical RandomForest (ROC-AUC)", prob_trimodal_voting, prob_clinical_rf),
        ("Trimodal Voting vs Clinical HistGB (ROC-AUC)", prob_trimodal_voting, prob_clinical_hgb),
        ("Trimodal Voting vs Trimodal Stacking (ROC-AUC)", prob_trimodal_voting, prob_trimodal_stacking),
    ]

    rows: list[dict[str, object]] = []
    for label, pred_a, pred_b in comparisons:
        delong = delong_test(y_array, pred_a, pred_b)
        bootstrap = bootstrap_auc_diff(y_array, pred_a, pred_b, B=2000)
        print(
            f"  {label}: AUC_A={delong['auc_1']:.4f}, AUC_B={delong['auc_2']:.4f}, "
            f"delta={delong['delta']:.4f}, DeLong p={delong['p_value']:.4f}, "
            f"bootstrap CI=[{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}], "
            f"p={bootstrap['p_value']:.4f}",
            flush=True,
        )
        rows.append(
            {
                "Comparison": label,
                "Metric_A": f"AUC={delong['auc_1']:.4f}",
                "Metric_B": f"AUC={delong['auc_2']:.4f}",
                "Delta": round(delong["delta"], 4),
                "DeLong_z": round(delong["z"], 3),
                "DeLong_p": round(delong["p_value"], 4),
                "Bootstrap_CI_lower": round(bootstrap["ci_lower"], 4),
                "Bootstrap_CI_upper": round(bootstrap["ci_upper"], 4),
                "Bootstrap_p": round(bootstrap["p_value"], 4),
            }
        )
    return pd.DataFrame(rows)


def save_classification_significance(
    significance_df: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    path = output_path if output_path is not None else OUTPUT_DIR / "bootstrap_significance_clf.csv"
    return upsert_csv_rows(significance_df, path, key="Comparison")


def evaluate_pipeline(
    dataset_name: str,
    stage: str,
    model_name: str,
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    clinical_num_cols: list[str],
    clinical_cat_cols: list[str],
    pca_components: int,
    cv_folds: int,
) -> dict[str, object]:
    preprocessor = build_preprocessor(X_train, clinical_num_cols, clinical_cat_cols, pca_components)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", clone(estimator)),
        ]
    )
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring="average_precision",
        n_jobs=1,
    )

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "Dataset": dataset_name,
        "Model": model_name,
        "Stage": stage,
        "CV_PR_AUC_mean": cv_scores.mean(),
        "CV_PR_AUC_std": cv_scores.std(),
        "Test_ROC": roc_auc_score(y_test, y_prob),
        "Test_PR": average_precision_score(y_test, y_prob),
        "Test_F1": f1_score(y_test, y_pred),
        "Test_Brier": brier_score_loss(y_test, y_prob),
        "Test_BalAcc": balanced_accuracy_score(y_test, y_pred),
    }


def evaluate_trimodal_rf_search(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    clinical_num_cols: list[str],
    clinical_cat_cols: list[str],
    pca_components: int,
    cv_folds: int,
) -> dict[str, object]:
    candidate_grid = [
        {"n_estimators": 500, "max_depth": 12, "min_samples_leaf": 5},
        {"n_estimators": 700, "max_depth": 14, "min_samples_leaf": 4},
        {"n_estimators": 900, "max_depth": 16, "min_samples_leaf": 3},
        {"n_estimators": 1000, "max_depth": 18, "min_samples_leaf": 2},
    ]

    preprocessor = build_preprocessor(X_train, clinical_num_cols, clinical_cat_cols, pca_components)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    best_params = candidate_grid[0]
    best_scores: np.ndarray | None = None
    best_mean = -np.inf

    for params in candidate_grid:
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_leaf=params["min_samples_leaf"],
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring="average_precision",
            n_jobs=1,
        )
        mean_score = float(scores.mean())
        print(
            "  [Trimodal RF tuning] "
            f"{params} -> CV PR-AUC {mean_score:.4f} +/- {scores.std():.4f}",
            flush=True,
        )
        if mean_score > best_mean:
            best_mean = mean_score
            best_params = params
            best_scores = scores

    assert best_scores is not None
    estimator = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=1,
    )
    result = evaluate_pipeline(
        dataset_name="Trimodal",
        stage="Tuned",
        model_name="RandomForest",
        estimator=estimator,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        clinical_num_cols=clinical_num_cols,
        clinical_cat_cols=clinical_cat_cols,
        pca_components=pca_components,
        cv_folds=cv_folds,
    )
    result["Model_Details"] = (
        f"n_estimators={best_params['n_estimators']}, "
        f"max_depth={best_params['max_depth']}, "
        f"min_samples_leaf={best_params['min_samples_leaf']}"
    )
    return result


def evaluate_svc_rbf_search(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    clinical_num_cols: list[str],
    clinical_cat_cols: list[str],
    pca_components: int,
    cv_folds: int,
) -> dict[str, object]:
    candidate_grid = [
        {"C": 0.1, "gamma": "scale"},
        {"C": 0.5, "gamma": "scale"},
        {"C": 1.0, "gamma": "scale"},
        {"C": 5.0, "gamma": "scale"},
        {"C": 1.0, "gamma": "auto"},
        {"C": 5.0, "gamma": "auto"},
    ]

    preprocessor = build_preprocessor(X_train, clinical_num_cols, clinical_cat_cols, pca_components)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    best_params = candidate_grid[0]
    best_scores: np.ndarray | None = None
    best_mean = -np.inf

    for params in candidate_grid:
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=params["C"],
                        gamma=params["gamma"],
                        probability=True,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        )
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=skf, scoring="average_precision", n_jobs=1,
        )
        mean_score = float(scores.mean())
        print(
            f"  [SVC-RBF tuning] C={params['C']}, gamma={params['gamma']} "
            f"-> CV PR-AUC {mean_score:.4f} +/- {scores.std():.4f}",
            flush=True,
        )
        if mean_score > best_mean:
            best_mean = mean_score
            best_params = params
            best_scores = scores

    final_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=best_params["C"],
                    gamma=best_params["gamma"],
                    probability=True,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]
    )
    final_pipeline.fit(X_train, y_train)
    y_prob = final_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    result = {
        "Dataset": "Clinical",
        "Model": "SVC-RBF",
        "Stage": "Tuned",
        "CV_PR_AUC_mean": float(best_scores.mean()),
        "CV_PR_AUC_std": float(best_scores.std()),
        "Test_ROC": roc_auc_score(y_test, y_prob),
        "Test_PR": average_precision_score(y_test, y_prob),
        "Test_F1": f1_score(y_test, y_pred),
        "Test_Brier": brier_score_loss(y_test, y_prob),
        "Test_BalAcc": balanced_accuracy_score(y_test, y_pred),
        "Model_Details": f"C={best_params['C']}, gamma={best_params['gamma']}",
    }
    return result


def run_clean_classification(
    cv_folds: int,
    pca_components: int,
    benchmark: str,
    selected_datasets: list[str] | None,
    output_path=None,
) -> pd.DataFrame:
    datasets, y, clinical_num_cols, clinical_cat_cols = load_classification_datasets()

    if benchmark == "primary":
        datasets = OrderedDict(
            (name, datasets[name])
            for name in ["Clinical", "Clin+Mut", "Trimodal"]
        )
    if selected_datasets:
        datasets = OrderedDict(
            (name, datasets[name])
            for name in selected_datasets
            if name in datasets
        )
    if not datasets:
        raise ValueError("No datasets selected for classification benchmark.")

    train_index, test_index = train_test_split(
        y.index,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )
    y_train = y.loc[train_index]
    y_test = y.loc[test_index]
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    rows: list[dict[str, object]] = []
    benchmark_start = time.time()

    for dataset_name, dataset in datasets.items():
        X_train = dataset.loc[train_index].copy()
        X_test = dataset.loc[test_index].copy()
        models = (
            get_primary_models(float(pos_weight), dataset_name)
            if benchmark == "primary"
            else get_models(float(pos_weight))
        )

        for model_name, estimator in models.items():
            print(
                f"[{time.strftime('%H:%M:%S')}] Running {dataset_name} / {model_name} ({benchmark})",
                flush=True,
            )
            result = evaluate_pipeline(
                dataset_name=dataset_name,
                stage="Baseline",
                model_name=model_name,
                estimator=estimator,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                clinical_num_cols=clinical_num_cols,
                clinical_cat_cols=clinical_cat_cols,
                pca_components=pca_components,
                cv_folds=cv_folds,
            )
            rows.append(result)
            if output_path is not None:
                pd.DataFrame(rows).to_csv(output_path, index=False)

    if "Clinical" in datasets and benchmark == "primary":
        clinical_train = datasets["Clinical"].loc[train_index].copy()
        clinical_test = datasets["Clinical"].loc[test_index].copy()
        print(f"[{time.strftime('%H:%M:%S')}] Running Clinical / SVC-RBF tuning", flush=True)
        svc_result = evaluate_svc_rbf_search(
            X_train=clinical_train,
            X_test=clinical_test,
            y_train=y_train,
            y_test=y_test,
            clinical_num_cols=clinical_num_cols,
            clinical_cat_cols=clinical_cat_cols,
            pca_components=pca_components,
            cv_folds=cv_folds,
        )
        rows.append(svc_result)
        if output_path is not None:
            pd.DataFrame(rows).to_csv(output_path, index=False)

    if "Trimodal" in datasets:
        trimodal_train = datasets["Trimodal"].loc[train_index].copy()
        trimodal_test = datasets["Trimodal"].loc[test_index].copy()
        if benchmark == "primary":
            tuned_result = evaluate_trimodal_rf_search(
                X_train=trimodal_train,
                X_test=trimodal_test,
                y_train=y_train,
                y_test=y_test,
                clinical_num_cols=clinical_num_cols,
                clinical_cat_cols=clinical_cat_cols,
                pca_components=pca_components,
                cv_folds=cv_folds,
            )
            rows.append(tuned_result)
            if output_path is not None:
                pd.DataFrame(rows).to_csv(output_path, index=False)

        for model_name, estimator in get_trimodal_ensembles(
            float(pos_weight),
            primary_only=(benchmark == "primary"),
        ).items():
            print(
                f"[{time.strftime('%H:%M:%S')}] Running Trimodal / {model_name} ({benchmark})",
                flush=True,
            )
            result = evaluate_pipeline(
                dataset_name="Trimodal",
                stage="Ensemble",
                model_name=model_name,
                estimator=estimator,
                X_train=trimodal_train,
                X_test=trimodal_test,
                y_train=y_train,
                y_test=y_test,
                clinical_num_cols=clinical_num_cols,
                clinical_cat_cols=clinical_cat_cols,
                pca_components=pca_components,
                cv_folds=cv_folds,
            )
            rows.append(result)
            if output_path is not None:
                pd.DataFrame(rows).to_csv(output_path, index=False)

    results = pd.DataFrame(rows)
    results["CV_PR_AUC"] = results.apply(
        lambda row: f"{row['CV_PR_AUC_mean']:.4f} +/- {row['CV_PR_AUC_std']:.4f}",
        axis=1,
    )
    keep_columns = [
        "Dataset",
        "Model",
        "Stage",
        "CV_PR_AUC",
        "Test_ROC",
        "Test_PR",
        "Test_F1",
        "Test_Brier",
        "Test_BalAcc",
    ]
    if "Model_Details" in results.columns:
        keep_columns.append("Model_Details")

    rounded = results[
        keep_columns
    ].copy()
    for column in ["Test_ROC", "Test_PR", "Test_F1", "Test_Brier", "Test_BalAcc"]:
        rounded[column] = rounded[column].round(4)
    # Primary sort: held-out ROC-AUC; PR-AUC as secondary tie-breaker.
    rounded = rounded.sort_values(["Test_ROC", "Test_PR"], ascending=False).reset_index(drop=True)
    print(
        f"\nCompleted {benchmark} classification benchmark in "
        f"{(time.time() - benchmark_start) / 60:.1f} minutes.",
        flush=True,
    )
    return rounded


# plots
def _extended_models(pos_weight: float) -> OrderedDict[str, object]:
    models = get_models(pos_weight)
    # SVC RBF with Platt scaling; can be slow on very wide trimodal — still tractable here.
    models["SVC-RBF"] = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=SEED,
    )
    return models


def _fit_all_models(
    datasets: OrderedDict,
    y: pd.Series,
    clinical_num_cols: list[str],
    clinical_cat_cols: list[str],
    pca_components: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray], np.ndarray]:
    """Returns metrics rows, (dataset, model) -> test probabilities, y_test array."""
    train_idx, test_idx = train_test_split(
        y.index,
        test_size=0.2,
        stratify=y,
        random_state=SEED,
    )
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx].values
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model_defs = _extended_models(pos_weight)

    rows: list[dict] = []
    probs: dict[tuple[str, str], np.ndarray] = {}

    for ds_name, X_full in datasets.items():
        X_train = X_full.loc[train_idx].copy()
        X_test = X_full.loc[test_idx].copy()
        preprocessor = build_preprocessor(
            X_train, clinical_num_cols, clinical_cat_cols, pca_components
        )

        for model_name, estimator in model_defs.items():
            t0 = time.time()
            pipe = Pipeline(
                [
                    ("preprocessor", clone(preprocessor)),
                    ("model", clone(estimator)),
                ]
            )
            try:
                pipe.fit(X_train, y_train)
                p = pipe.predict_proba(X_test)[:, 1]
            except Exception as exc:  # pragma: no cover - SVC / memory edge cases
                print(f"  [skip] {ds_name} / {model_name}: {exc}")
                continue

            probs[(ds_name, model_name)] = p
            rows.append(
                {
                    "Dataset": ds_name,
                    "Model": model_name,
                    "Test_ROC": roc_auc_score(y_test, p),
                    "Test_PR": average_precision_score(y_test, p),
                    "Test_Brier": brier_score_loss(y_test, p),
                    "fit_s": round(time.time() - t0, 1),
                }
            )
            print(
                f"  [{time.strftime('%H:%M:%S')}] {ds_name} / {model_name} "
                f"ROC={rows[-1]['Test_ROC']:.4f} ({rows[-1]['fit_s']}s)",
                flush=True,
            )

    return pd.DataFrame(rows), probs, y_test


def plot_roc_multi(
    datasets: OrderedDict,
    metrics: pd.DataFrame,
    probs: dict[tuple[str, str], np.ndarray],
    y_test: np.ndarray,
    out: Path,
) -> None:
    ds_names = list(datasets.keys())
    fig, axes = plt.subplots(1, len(ds_names), figsize=(5 * len(ds_names), 5))
    if len(ds_names) == 1:
        axes = np.array([axes])
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, ds_name in enumerate(ds_names):
        ax = axes[i]
        sub = metrics[metrics["Dataset"] == ds_name].sort_values("Test_ROC", ascending=False)
        for j, row in enumerate(sub.itertuples()):
            key = (ds_name, row.Model)
            if key not in probs:
                continue
            p = probs[key]
            fpr, tpr, _ = roc_curve(y_test, p)
            ax.plot(
                fpr,
                tpr,
                lw=1.8,
                color=cmap[j % 10],
                label=f"{row.Model} ({row.Test_ROC:.3f})",
            )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.35, lw=1)
        ax.set_title(ds_name)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

    fig.suptitle(
        "ROC curves — all baseline models (same train/test split, leakage-aware preprocessing)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_multi(
    datasets: OrderedDict,
    metrics: pd.DataFrame,
    probs: dict[tuple[str, str], np.ndarray],
    y_test: np.ndarray,
    out: Path,
) -> None:
    prevalence = float(y_test.mean())
    ds_names = list(datasets.keys())
    fig, axes = plt.subplots(1, len(ds_names), figsize=(5 * len(ds_names), 5))
    if len(ds_names) == 1:
        axes = np.array([axes])
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, ds_name in enumerate(ds_names):
        ax = axes[i]
        sub = metrics[metrics["Dataset"] == ds_name].sort_values("Test_PR", ascending=False)
        for j, row in enumerate(sub.itertuples()):
            key = (ds_name, row.Model)
            if key not in probs:
                continue
            p = probs[key]
            prec, rec, _ = precision_recall_curve(y_test, p)
            ax.plot(
                rec,
                prec,
                lw=1.8,
                color=cmap[j % 10],
                label=f"{row.Model} ({row.Test_PR:.3f})",
            )
        ax.axhline(prevalence, color="k", ls="--", alpha=0.35, lw=1, label=f"Prevalence {prevalence:.2f}")
        ax.set_title(ds_name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)

    fig.suptitle("Precision–recall curves — all baseline models", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bars(metrics: pd.DataFrame, out: Path) -> None:
    pivot_roc = metrics.pivot_table(index="Model", columns="Dataset", values="Test_ROC", aggfunc="max")
    pivot_brier = metrics.pivot_table(index="Model", columns="Dataset", values="Test_Brier", aggfunc="min")
    ds_names = list(pivot_roc.columns)
    x = np.arange(len(pivot_roc.index))
    width = min(0.22, 0.8 / max(len(ds_names), 1))
    colors = plt.cm.Set2(np.linspace(0, 1, len(ds_names)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for j, ds_name in enumerate(ds_names):
        if ds_name in pivot_roc.columns:
            ax1.bar(
                x + j * width - width * (len(ds_names) - 1) / 2,
                pivot_roc[ds_name],
                width * 0.92,
                label=ds_name,
                color=colors[j],
                edgecolor="black",
                linewidth=0.4,
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels(pivot_roc.index, rotation=35, ha="right")
    ax1.set_ylabel("ROC-AUC")
    ax1.set_xlabel("Model")
    ax1.set_title("ROC-AUC by model and dataset")
    ax1.legend(fontsize=8, loc="lower right")
    roc_min = float(pivot_roc.min().min())
    roc_max = float(pivot_roc.max().max())
    ax1.set_ylim(max(0.35, roc_min - 0.08), min(1.0, roc_max + 0.06))

    for j, ds_name in enumerate(ds_names):
        if ds_name in pivot_brier.columns:
            ax2.bar(
                x + j * width - width * (len(ds_names) - 1) / 2,
                pivot_brier[ds_name],
                width * 0.92,
                label=ds_name,
                color=colors[j],
                edgecolor="black",
                linewidth=0.4,
            )
    ax2.set_xticks(x)
    ax2.set_xticklabels(pivot_brier.index, rotation=35, ha="right")
    ax2.set_ylabel("Brier score (lower is better)")
    ax2.set_xlabel("Model")
    ax2.set_title("Brier score by model and dataset")
    ax2.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Model comparison — baselines incl. LogReg, RF, AdaBoost, SVC-RBF, boosting models",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_multi(
    datasets: OrderedDict,
    metrics: pd.DataFrame,
    probs: dict[tuple[str, str], np.ndarray],
    y_test: np.ndarray,
    out: Path,
    n_bins: int = 10,
) -> None:
    ds_names = list(datasets.keys())
    fig, axes = plt.subplots(1, len(ds_names), figsize=(5 * len(ds_names), 5))
    if len(ds_names) == 1:
        axes = np.array([axes])
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, ds_name in enumerate(ds_names):
        ax = axes[i]
        sub = metrics[metrics["Dataset"] == ds_name].sort_values("Model")
        for j, row in enumerate(sub.itertuples()):
            key = (ds_name, row.Model)
            if key not in probs:
                continue
            p = probs[key]
            try:
                prob_true, prob_pred = calibration_curve(
                    y_test, p, n_bins=n_bins, strategy="uniform"
                )
            except ValueError:
                continue
            ax.plot(
                prob_pred,
                prob_true,
                "s-",
                lw=1.4,
                markersize=4,
                color=cmap[j % 10],
                label=row.Model,
            )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.35, lw=1, label="Perfect")
        ax.set_title(ds_name)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend(loc="upper left", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle(
        f"Calibration curves ({n_bins} bins, uniform) — all fitted baselines incl. LR / RF / AdaBoost / SVC-RBF",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_survival_cindex_bars(surv_csv: Path, out: Path) -> None:
    if not surv_csv.is_file():
        print(f"  [skip] survival bars: missing {surv_csv}")
        return
    df = pd.read_csv(surv_csv)
    title_models = ", ".join(sorted(df["Model"].astype(str).unique()))
    df = df.sort_values("Test_C_index", ascending=True)
    labels = df["Model"].astype(str) + " / " + df["Modality"].astype(str)
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.22)))
    colours = []
    for m in df["Model"]:
        if m == "RSF":
            colours.append("#1565C0")
        elif m == "CoxPH":
            colours.append("#EF6C00")
        elif m == "Late Fusion":
            colours.append("#2E7D32")
        else:
            colours.append("#757575")
    ax.barh(labels, df["Test_C_index"], color=colours, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Held-out C-index")
    ax.axvline(0.5, color="k", ls="--", alpha=0.3)
    ax.set_title(
        f"Primary survival benchmark ({title_models}) — not comparable to classifiers"
    )
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def generate_thesis_classification_figures(
    pca_components: int = 50,
    out_dir: Path | None = None,
) -> Path:
    out_dir = out_dir or OUTPUT_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_outputs_dir()

    print("Loading datasets + fitting all baseline models (about 30-90 min on CPU)...")
    datasets, y, num_cols, cat_cols = load_classification_datasets()
    metrics, probs, y_test = _fit_all_models(
        datasets, y, num_cols, cat_cols, pca_components
    )
    metrics_path = out_dir / "thesis_figure_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved {metrics_path}")

    plot_roc_multi(datasets, metrics, probs, y_test, out_dir / "roc_curves.png")
    print(f"Saved {out_dir / 'roc_curves.png'}")
    plot_pr_multi(datasets, metrics, probs, y_test, out_dir / "pr_curves.png")
    print(f"Saved {out_dir / 'pr_curves.png'}")
    plot_comparison_bars(metrics, out_dir / "comparison_bars.png")
    print(f"Saved {out_dir / 'comparison_bars.png'}")
    plot_calibration_multi(datasets, metrics, probs, y_test, out_dir / "calibration_curves.png")
    print(f"Saved {out_dir / 'calibration_curves.png'}")

    surv = out_dir / "survival_results_primary.csv"
    if not surv.is_file():
        surv = OUTPUT_DIR / "survival_results_primary.csv"
    plot_survival_cindex_bars(surv, out_dir / "survival_primary_cindex_bars.png")

    return out_dir


def _clinical_xy_for_figures():
    clin = load_clinical_table()
    clin, y = build_five_year_labels(clin)
    num_cols, cat_cols = get_classification_feature_columns(clin)
    X = clin[num_cols + cat_cols].copy()
    return X, y, num_cols, cat_cols


def figures_learning_curves() -> None:
    """Learning curves on clinical-only data (train subset only; CV inside learning_curve)."""
    X, y, num_cols, cat_cols = _clinical_xy_for_figures()
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    num_imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xn_tr = scaler.fit_transform(num_imp.fit_transform(X_train[num_cols]))
    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xc_tr = enc.fit_transform(cat_imp.fit_transform(X_train[cat_cols]))
        X_proc = np.hstack([Xn_tr, Xc_tr])
    else:
        X_proc = Xn_tr

    models: OrderedDict[str, object] = OrderedDict(
        [
            (
                "Logistic Regression",
                __import__(
                    "sklearn.linear_model", fromlist=["LogisticRegression"]
                ).LogisticRegression(
                    C=0.1, solver="saga", max_iter=1000, class_weight="balanced", random_state=SEED
                ),
            ),
            (
                "Random Forest",
                __import__(
                    "sklearn.ensemble", fromlist=["RandomForestClassifier"]
                ).RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=1,
                ),
            ),
            (
                "HistGradientBoosting",
                __import__(
                    "sklearn.ensemble", fromlist=["HistGradientBoostingClassifier"]
                ).HistGradientBoostingClassifier(
                    max_iter=200, max_depth=4, learning_rate=0.05, random_state=SEED
                ),
            ),
        ]
    )
    if xgb is not None:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            random_state=SEED,
            eval_metric="auc",
            verbosity=0,
        )

    train_sizes = np.linspace(0.10, 1.0, 10)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    n_models = len(models)
    if n_models <= 2:
        nrows, ncols = 1, n_models
        figsize = (7 * n_models, 5)
    elif n_models == 3:
        nrows, ncols = 1, 3
        figsize = (18, 5)
    else:
        nrows, ncols = 2, 2
        figsize = (14, 10)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()
    lc_summary = []
    for ax, (name, model) in zip(axes_flat, models.items()):
        train_sz, train_scores, val_scores = learning_curve(
            model,
            X_proc,
            y_train.values,
            train_sizes=train_sizes,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1,
            shuffle=True,
            random_state=SEED,
        )
        tm, ts = train_scores.mean(axis=1), train_scores.std(axis=1)
        vm, vs = val_scores.mean(axis=1), val_scores.std(axis=1)
        gap = float(tm[-1] - vm[-1])
        lc_summary.append({"model": name, "train_auc_final": tm[-1], "cv_auc_final": vm[-1], "gap": gap})
        ax.plot(train_sz, tm, "o-", color="#2196F3", label="Training", linewidth=2, markersize=5)
        ax.fill_between(train_sz, tm - ts, tm + ts, alpha=0.15, color="#2196F3")
        ax.plot(train_sz, vm, "s-", color="#F44336", label="CV (5-fold)", linewidth=2, markersize=5)
        ax.fill_between(train_sz, vm - vs, vm + vs, alpha=0.15, color="#F44336")
        ax.set_title(f"{name}\nGap={gap:.3f}", fontsize=10)
        ax.set_xlabel("Training set size")
        ax.set_ylabel("ROC-AUC")
        ax.legend(fontsize=8)
        ax.set_ylim(0.45, 1.02)
        ax.grid(True, alpha=0.3)
    for j in range(len(models), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Learning curves (clinical features, 80% train partition only)", fontsize=12)
    plt.tight_layout()
    p = OUTPUT_DIR / "learning_curves.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(lc_summary).to_csv(OUTPUT_DIR / "learning_curve_summary.csv", index=False)
    print(f"Saved {p} and learning_curve_summary.csv")


def figures_shap_clinical_xgb() -> None:
    """SHAP bar plot for clinical XGBoost (optional — needs `shap` package)."""
    if shap is None:
        print("Package `shap` not installed; skipping shap_summary_clinical_xgb.png (pip install shap).")
        return
    if xgb is None:
        print("Package `xgboost` not installed; skipping shap_summary_clinical_xgb.png.")
        return
    X, y, num_cols, cat_cols = _clinical_xy_for_figures()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    num_imp = SimpleImputer(strategy="median")
    X_train_num = pd.DataFrame(
        num_imp.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index
    )
    X_test_num = pd.DataFrame(
        num_imp.transform(X_test[num_cols]), columns=num_cols, index=X_test.index
    )
    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = pd.DataFrame(
            enc.fit_transform(cat_imp.fit_transform(X_train[cat_cols])),
            columns=cat_cols,
            index=X_train.index,
        )
        X_test_cat = pd.DataFrame(
            enc.transform(cat_imp.transform(X_test[cat_cols])),
            columns=cat_cols,
            index=X_test.index,
        )
        X_train_proc = pd.concat([X_train_num, X_train_cat], axis=1)
        X_test_proc = pd.concat([X_test_num, X_test_cat], axis=1)
        feat_names = num_cols + cat_cols
    else:
        X_train_proc, X_test_proc = X_train_num, X_test_num
        feat_names = num_cols

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_proc)
    X_test_s = scaler.transform(X_test_proc)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=SEED,
        eval_metric="auc",
        verbosity=0,
    )
    xgb_clf.fit(X_train_s, y_train.values)
    explainer = shap.TreeExplainer(xgb_clf)
    shap_vals = explainer.shap_values(X_test_s)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap.summary_plot(
        shap_vals, X_test_s, feature_names=feat_names, show=False, plot_type="bar", max_display=20
    )
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary_clinical_xgb.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_summary_clinical_xgb.png")


def figures_clinical_xgb_subgroups() -> None:
    """Subgroup ROC/PR for clinical-only XGBoost (thesis-style interpretability)."""
    if xgb is None:
        print("Package `xgboost` not installed; skipping clf_subgroup_clinical_xgb outputs.")
        return
    X, y, num_cols, cat_cols = _clinical_xy_for_figures()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    num_imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(num_imp.fit_transform(X_train[num_cols]))
    X_test_num = scaler.transform(num_imp.transform(X_test[num_cols]))
    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train_cat = enc.fit_transform(cat_imp.fit_transform(X_train[cat_cols]))
        X_test_cat = enc.transform(cat_imp.transform(X_test[cat_cols]))
        X_tr = np.hstack([X_train_num, X_train_cat])
        X_te = np.hstack([X_test_num, X_test_cat])
    else:
        X_tr, X_te = X_train_num, X_test_num

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=SEED,
        eval_metric="auc",
        verbosity=0,
    )
    xgb_clf.fit(X_tr, y_train.values)
    probs = xgb_clf.predict_proba(X_te)[:, 1]
    rows = []
    clin_test = X_test.copy()
    y_test_arr = y_test.values

    def add_subgroup(mask: np.ndarray, label: str, category: str) -> None:
        mask = np.asarray(mask)
        if mask.sum() < 20:
            return
        y_sub = y_test_arr[mask]
        p_sub = probs[mask]
        if y_sub.sum() < 3 or (1 - y_sub).sum() < 3:
            return
        try:
            rows.append(
                {
                    "Category": category,
                    "Subgroup": label,
                    "N": int(mask.sum()),
                    "ROC_AUC": round(roc_auc_score(y_sub, p_sub), 4),
                    "PR_AUC": round(average_precision_score(y_sub, p_sub), 4),
                }
            )
        except ValueError:
            return

    for col in ["ER_STATUS", "ER_IHC"]:
        if col in clin_test.columns:
            er = clin_test[col].astype(str).str.strip().str.lower()
            add_subgroup(er == "positive", "ER Positive", "ER")
            add_subgroup(er == "negative", "ER Negative", "ER")
            break
    for col in ["CLAUDIN_SUBTYPE", "pam50_+_claudin-low_subtype"]:
        if col in clin_test.columns:
            pam = clin_test[col].astype(str).str.strip()
            for st in sorted(pam.unique()):
                if str(st).lower() in {"nan", "none", ""}:
                    continue
                add_subgroup((pam == st).values, str(st), "PAM50")
            break
    if "age_at_diagnosis" in clin_test.columns:
        age = pd.to_numeric(clin_test["age_at_diagnosis"], errors="coerce")
        add_subgroup((age < 50).fillna(False).values, "Age <50", "Age")
        add_subgroup(((age >= 50) & (age < 60)).fillna(False).values, "Age 50-60", "Age")
        add_subgroup(((age >= 60) & (age < 70)).fillna(False).values, "Age 60-70", "Age")
        add_subgroup((age >= 70).fillna(False).values, "Age 70+", "Age")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "clf_subgroup_clinical_xgb.csv", index=False)
    if len(df) == 0:
        print("Subgroup table empty (check column names in clinical data).")
        return
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.25)))
    ax.barh(df["Subgroup"], df["ROC_AUC"], color="#42A5F5")
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Clinical XGBoost — subgroup hold-out ROC-AUC")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "clf_subgroup_clinical_xgb.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved clf_subgroup_clinical_xgb.csv and clf_subgroup_clinical_xgb.png")


# __main__
def main() -> None:
    ensure_outputs_dir()
    print("Classification")

    # Step 1: Run primary benchmark
    print("primary benchmark...")
    output_path = OUTPUT_DIR / "classification_results_clean.csv"
    results = run_clean_classification(
        cv_folds=3,
        pca_components=50,
        benchmark="primary",
        selected_datasets=None,
        output_path=output_path,
    )
    results.to_csv(output_path, index=False)
    results.to_csv(OUTPUT_DIR / "classification_results_primary.csv", index=False)

    # Step 2: Run significance tests
    print("bootstrap significance...")
    significance_df = run_classification_significance(pca_components=50)
    sig_path = save_classification_significance(significance_df)
    print(f"Updated: {sig_path}")

    # Step 3: Generate thesis figures
    print("thesis figures...")
    generate_thesis_classification_figures(pca_components=50, out_dir=OUTPUT_DIR)

    # Step 4: Generate additional figures
    print("learning curves, SHAP, subgroup figures...")
    figures_learning_curves()
    figures_shap_clinical_xgb()
    figures_clinical_xgb_subgroups()

    print("done.")


if __name__ == "__main__":
    main()
