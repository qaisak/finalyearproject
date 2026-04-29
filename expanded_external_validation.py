

from __future__ import annotations

import json
from pathlib import Path
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from survival_consolidated import (
    DATA_DIR,
    OUTPUT_DIR,
    SEED,
    encode_and_scale_clinical,
    encode_features,
    ensure_outputs_dir,
    load_mrna_matrix,
    load_mutation_features,
    prepare_metabric_clinical,
    preprocess_mrna,
    preprocess_mutation,
)


TCGA_HARMONISED = DATA_DIR / "tcga_brca_pan_can_atlas_2018" / "tcga_brca_harmonised.csv"
TCGA_CACHE_DIR = DATA_DIR / "tcga_brca_pan_can_atlas_2018"
CBIO_BASE = "https://www.cbioportal.org/api"
TCGA_STUDY_ID = "brca_tcga_pan_can_atlas_2018"
TCGA_3WAY_SAMPLE_LIST = "brca_tcga_pan_can_atlas_2018_3way_complete"
TCGA_MRNA_PROFILE = "brca_tcga_pan_can_atlas_2018_rna_seq_v2_mrna_median_Zscores"
TCGA_MUTATION_PROFILE = "brca_tcga_pan_can_atlas_2018_mutations"
SUBTYPE_ORDER = ["Basal", "Her2", "LumA", "LumB", "Normal"]
TCGA_SUBTYPE_MAP = {
    "BRCA_Basal": "Basal",
    "BRCA_Her2": "Her2",
    "BRCA_LumA": "LumA",
    "BRCA_LumB": "LumB",
    "BRCA_Normal": "Normal",
}

# Experiment settings. Edit these here rather than passing command-line flags.
BOOTSTRAPS = 1000
MRNA_GENE_LIMIT = 1000
MRNA_PCA_COMPONENTS = 20
REFRESH_TCGA_CACHE = False
RUN_CLINICAL_ONLY_EXTENSION = True
RUN_TRIMODAL_EXTENSION = True


def normalise_tcga_subtype(value: object) -> object:
    """Convert TCGA subtype labels to METABRIC-style labels."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return TCGA_SUBTYPE_MAP.get(text, text.replace("BRCA_", ""))


def load_tcga_clinical() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
    if not TCGA_HARMONISED.exists():
        raise FileNotFoundError(
            f"Missing {TCGA_HARMONISED}. Run survival_consolidated.py once, or add the "
            "cached TCGA harmonised CSV before using this separate experiment."
        )

    tcga = pd.read_csv(TCGA_HARMONISED, index_col=0)
    required = {"os_months", "event"}
    missing = required - set(tcga.columns)
    if missing:
        raise ValueError(f"TCGA harmonised file is missing required columns: {sorted(missing)}")

    X = tcga.drop(columns=["os_months", "event"], errors="ignore").copy()
    if "subtype" in X.columns:
        X["subtype"] = X["subtype"].map(normalise_tcga_subtype)
        group_subtype = X["subtype"].fillna("Unknown").astype(str)
    else:
        group_subtype = pd.Series("Unknown", index=X.index, name="subtype")

    T = tcga["os_months"].to_numpy(dtype=float)
    E = tcga["event"].to_numpy(dtype=int)
    return X, T, E, group_subtype


def safe_c_index(T: np.ndarray, score: np.ndarray, E: np.ndarray) -> float:
    try:
        return float(concordance_index(T, score, E))
    except Exception:
        return float("nan")


def count_comparable_pairs(T: np.ndarray, E: np.ndarray) -> int:
    """Approximate Harrell comparable-pair count for interpretability."""
    comparable = 0
    n = len(T)
    for i in range(n):
        for j in range(i + 1, n):
            if T[i] < T[j] and E[i] == 1:
                comparable += 1
            elif T[j] < T[i] and E[j] == 1:
                comparable += 1
    return comparable


def cbioportal_get_json(path: str, *, timeout: int = 60) -> object:
    req = urllib.request.Request(
        f"{CBIO_BASE}{path}",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def cbioportal_post_json(path: str, payload: object, *, timeout: int = 180) -> object:
    # cBioPortal POST endpoints expect JSON payloads.
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{CBIO_BASE}{path}",
        data=body,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        text = response.read()
    return json.loads(text) if text else []


def chunks(values: list[object], size: int) -> list[list[object]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def lookup_entrez_ids(symbols: list[str], *, chunk_size: int = 500) -> dict[str, int]:
    """Resolve Hugo symbols to Entrez IDs through cBioPortal."""
    resolved: dict[str, int] = {}
    unique_symbols = sorted({str(symbol).strip() for symbol in symbols if str(symbol).strip()})
    for chunk in chunks(unique_symbols, chunk_size):
        records = cbioportal_post_json(
            "/genes/fetch?geneIdType=HUGO_GENE_SYMBOL",
            chunk,
            timeout=90,
        )
        for record in records:
            symbol = str(record.get("hugoGeneSymbol", "")).strip()
            entrez = record.get("entrezGeneId")
            if symbol and entrez is not None:
                resolved[symbol] = int(entrez)
    return resolved


def select_metabric_mrna_columns(mrna: pd.DataFrame, limit: int) -> list[str]:
    """Select a stable high-variance mRNA subset for cross-cohort PCA."""
    variances = pd.Series(np.nanvar(mrna.values, axis=0), index=mrna.columns)
    ordered = variances.sort_values(ascending=False).index.tolist()
    return ordered[: min(limit, len(ordered))]


def fetch_tcga_sample_ids() -> list[str]:
    return list(cbioportal_get_json(f"/sample-lists/{TCGA_3WAY_SAMPLE_LIST}/sample-ids"))


def fetch_tcga_mrna_matrix(
    metabric_mrna_columns: list[str],
    *,
    refresh_cache: bool = False,
    gene_chunk_size: int = 75,
) -> pd.DataFrame:
    """Fetch TCGA mRNA z-scores for selected METABRIC genes and cache as a matrix."""
    TCGA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TCGA_CACHE_DIR / f"tcga_3way_mrna_{len(metabric_mrna_columns)}genes_zscores.csv"
    if cache_path.exists() and not refresh_cache:
        return pd.read_csv(cache_path, index_col=0)

    symbols = [column.replace("MRNA__", "", 1) for column in metabric_mrna_columns]
    symbol_to_entrez = lookup_entrez_ids(symbols)
    entrez_to_symbol = {entrez: symbol for symbol, entrez in symbol_to_entrez.items()}
    entrez_ids = [symbol_to_entrez[symbol] for symbol in symbols if symbol in symbol_to_entrez]

    all_records: list[dict] = []
    for i, chunk in enumerate(chunks(entrez_ids, gene_chunk_size), start=1):
        print(f"  TCGA mRNA fetch chunk {i}/{int(np.ceil(len(entrez_ids) / gene_chunk_size))}")
        records = cbioportal_post_json(
            f"/molecular-profiles/{TCGA_MRNA_PROFILE}/molecular-data/fetch?projection=SUMMARY",
            {"sampleListId": TCGA_3WAY_SAMPLE_LIST, "entrezGeneIds": chunk},
            timeout=240,
        )
        all_records.extend(records)

    if not all_records:
        raise RuntimeError("No TCGA mRNA records were returned by cBioPortal.")

    long_df = pd.DataFrame(all_records)
    long_df["hugo_symbol"] = long_df["entrezGeneId"].map(entrez_to_symbol)
    long_df = long_df.dropna(subset=["patientId", "hugo_symbol", "value"])
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    matrix = long_df.pivot_table(
        index="patientId",
        columns="hugo_symbol",
        values="value",
        aggfunc="mean",
    )
    matrix.columns = [f"MRNA__{column}" for column in matrix.columns]
    matrix = matrix.reindex(columns=metabric_mrna_columns)
    matrix.index = matrix.index.astype(str).str.upper()
    matrix.to_csv(cache_path)
    return matrix


def fetch_tcga_mutation_features(
    metabric_mutation_columns: list[str],
    *,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    """Fetch TCGA binary mutation indicators for METABRIC recurrent mutation genes."""
    TCGA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TCGA_CACHE_DIR / "tcga_3way_mutation_recurrent_gene_features.csv"
    if cache_path.exists() and not refresh_cache:
        return pd.read_csv(cache_path, index_col=0)

    symbols = [column.replace("MUT__", "", 1) for column in metabric_mutation_columns]
    symbol_to_entrez = lookup_entrez_ids(symbols)
    entrez_to_symbol = {entrez: symbol for symbol, entrez in symbol_to_entrez.items()}
    entrez_ids = [symbol_to_entrez[symbol] for symbol in symbols if symbol in symbol_to_entrez]

    print(f"  TCGA mutation fetch for {len(entrez_ids)} recurrent METABRIC genes")
    records = cbioportal_post_json(
        f"/molecular-profiles/{TCGA_MUTATION_PROFILE}/mutations/fetch?projection=SUMMARY",
        {"sampleListId": TCGA_3WAY_SAMPLE_LIST, "entrezGeneIds": entrez_ids},
        timeout=240,
    )

    patient_ids = [
        sample_id.rsplit("-", 1)[0].upper()
        for sample_id in fetch_tcga_sample_ids()
    ]
    matrix = pd.DataFrame(0, index=sorted(set(patient_ids)), columns=metabric_mutation_columns)

    if records:
        long_df = pd.DataFrame(records)
        long_df["hugo_symbol"] = long_df["entrezGeneId"].map(entrez_to_symbol)
        long_df = long_df.dropna(subset=["patientId", "hugo_symbol"])
        long_df["patientId"] = long_df["patientId"].astype(str).str.upper()
        for patient_id, symbol in long_df[["patientId", "hugo_symbol"]].itertuples(index=False):
            column = f"MUT__{symbol}"
            if patient_id in matrix.index and column in matrix.columns:
                matrix.at[patient_id, column] = 1

    matrix.to_csv(cache_path)
    return matrix


def prepare_external_trimodal_frames(
    *,
    mrna_gene_limit: int,
    pca_components: int,
    refresh_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Build matched METABRIC train and TCGA test frames for clinical/trimodal RSF."""
    met_clin, met_T_all, met_E_all = prepare_metabric_clinical()
    met_mut_all = load_mutation_features()
    met_mrna_all = load_mrna_matrix()
    met_all_index = met_clin.index.copy()

    met_mrna_columns = select_metabric_mrna_columns(met_mrna_all, mrna_gene_limit)
    met_mutation_columns = [
        column
        for column in met_mut_all.columns
        if column.startswith("MUT__") and not column.startswith("MUT__mut_")
    ]

    tcga_clin, tcga_T_all, tcga_E_all, tcga_subtypes_all = load_tcga_clinical()
    tcga_all_index = tcga_clin.index.copy()
    tcga_mrna_all = fetch_tcga_mrna_matrix(
        met_mrna_columns,
        refresh_cache=refresh_cache,
    )
    tcga_mut_all = fetch_tcga_mutation_features(
        met_mutation_columns,
        refresh_cache=refresh_cache,
    )

    met_ids = sorted(set(met_clin.index) & set(met_mut_all.index) & set(met_mrna_all.index))
    tcga_ids = sorted(set(tcga_clin.index) & set(tcga_mut_all.index) & set(tcga_mrna_all.index))

    available_mrna_columns = [
        column
        for column in met_mrna_columns
        if column in tcga_mrna_all.columns and not tcga_mrna_all[column].isna().all()
    ]
    if len(available_mrna_columns) < pca_components:
        raise ValueError(
            f"Only {len(available_mrna_columns)} mRNA genes have TCGA values; "
            f"cannot compute {pca_components} PCA components."
        )

    met_clin = met_clin.loc[met_ids].copy()
    met_mut = met_mut_all.loc[met_ids, met_mutation_columns].copy()
    met_mrna = met_mrna_all.loc[met_ids, available_mrna_columns].copy()

    tcga_clin = tcga_clin.loc[tcga_ids, met_clin.columns].copy()
    tcga_mut = tcga_mut_all.reindex(index=tcga_ids, columns=met_mutation_columns, fill_value=0)
    tcga_mrna = tcga_mrna_all.reindex(index=tcga_ids, columns=available_mrna_columns)

    met_indexer = pd.Index(met_all_index).get_indexer(met_ids)
    tcga_indexer = pd.Index(tcga_all_index).get_indexer(tcga_ids)
    met_T = met_T_all[met_indexer]
    met_E = met_E_all[met_indexer]
    tcga_T = tcga_T_all[tcga_indexer]
    tcga_E = tcga_E_all[tcga_indexer]
    tcga_subtypes = tcga_subtypes_all.loc[tcga_ids]

    X_met_clinical = met_clin
    X_tcga_clinical = tcga_clin
    X_met_trimodal = pd.concat([met_clin, met_mut, met_mrna], axis=1)
    X_tcga_trimodal = pd.concat([tcga_clin, tcga_mut, tcga_mrna], axis=1)

    return (
        X_met_clinical,
        X_tcga_clinical,
        X_met_trimodal,
        X_tcga_trimodal,
        met_T,
        met_E,
        tcga_T,
        tcga_E,
        tcga_subtypes,
    )


def preprocess_external_frame(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    *,
    pca_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clin_cols = [c for c in X_train_raw.columns if not c.startswith(("MUT__", "MRNA__"))]
    mut_cols = [c for c in X_train_raw.columns if c.startswith("MUT__")]
    mrna_cols = [c for c in X_train_raw.columns if c.startswith("MRNA__")]

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


def fit_fixed_rsf(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    T_train: np.ndarray,
    E_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_train = Surv.from_arrays(event=E_train.astype(bool), time=T_train.astype(float))
    model = RandomSurvivalForest(
        n_estimators=300,
        min_samples_leaf=15,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=1,
    )
    model.fit(X_train.values, y_train)
    return model.predict(X_train.values), model.predict(X_test.values)


def bootstrap_c_index_ci(
    T: np.ndarray,
    score: np.ndarray,
    E: np.ndarray,
    *,
    B: int,
    seed: int,
) -> tuple[float, float, int]:
    rng = np.random.RandomState(seed)
    n = len(T)
    boot = []
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        c = safe_c_index(T[idx], score[idx], E[idx])
        if np.isfinite(c):
            boot.append(c)

    if not boot:
        return float("nan"), float("nan"), 0

    return (
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
        len(boot),
    )


def make_validation_row(
    *,
    label: str,
    mask: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    score: np.ndarray,
    B: int,
    seed: int,
) -> dict[str, object]:
    T_sub = T[mask]
    E_sub = E[mask]
    score_sub = score[mask]
    c_index = safe_c_index(T_sub, score_sub, E_sub)
    ci_lower, ci_upper, boot_n = bootstrap_c_index_ci(
        T_sub,
        score_sub,
        E_sub,
        B=B,
        seed=seed,
    )
    return {
        "subtype": label,
        "n_test": int(mask.sum()),
        "n_events": int(E_sub.sum()),
        "event_rate": round(float(E_sub.mean()), 4) if len(E_sub) else np.nan,
        "comparable_pairs": count_comparable_pairs(T_sub, E_sub),
        "c_index": round(c_index, 4) if np.isfinite(c_index) else np.nan,
        "ci_lower": round(ci_lower, 4) if np.isfinite(ci_lower) else np.nan,
        "ci_upper": round(ci_upper, 4) if np.isfinite(ci_upper) else np.nan,
        "bootstrap_replicates": boot_n,
    }


def bootstrap_c_index_diff(
    T: np.ndarray,
    E: np.ndarray,
    clinical_score: np.ndarray,
    trimodal_score: np.ndarray,
    *,
    B: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.RandomState(seed)
    n = len(T)
    c_clinical = safe_c_index(T, clinical_score, E)
    c_trimodal = safe_c_index(T, trimodal_score, E)
    observed_delta = c_trimodal - c_clinical

    deltas = []
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        c_boot_clin = safe_c_index(T[idx], clinical_score[idx], E[idx])
        c_boot_tri = safe_c_index(T[idx], trimodal_score[idx], E[idx])
        if np.isfinite(c_boot_clin) and np.isfinite(c_boot_tri):
            deltas.append(c_boot_tri - c_boot_clin)

    if deltas:
        arr = np.asarray(deltas)
        ci_lower = float(np.percentile(arr, 2.5))
        ci_upper = float(np.percentile(arr, 97.5))
        p_one_side = float(np.mean(arr <= 0)) if observed_delta > 0 else float(np.mean(arr >= 0))
        p_value = min(2.0 * p_one_side, 1.0)
    else:
        ci_lower = ci_upper = p_value = float("nan")

    return {
        "c_clinical": c_clinical,
        "c_trimodal": c_trimodal,
        "delta": observed_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "bootstrap_replicates": len(deltas),
    }


def make_trimodal_comparison_row(
    *,
    label: str,
    mask: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    clinical_score: np.ndarray,
    trimodal_score: np.ndarray,
    B: int,
    seed: int,
) -> dict[str, object]:
    T_sub = T[mask]
    E_sub = E[mask]
    clinical_sub = clinical_score[mask]
    trimodal_sub = trimodal_score[mask]
    stats = bootstrap_c_index_diff(
        T_sub,
        E_sub,
        clinical_sub,
        trimodal_sub,
        B=B,
        seed=seed,
    )
    return {
        "subtype": label,
        "n_test": int(mask.sum()),
        "n_events": int(E_sub.sum()),
        "event_rate": round(float(E_sub.mean()), 4) if len(E_sub) else np.nan,
        "comparable_pairs": count_comparable_pairs(T_sub, E_sub),
        "c_clinical": round(float(stats["c_clinical"]), 4)
        if np.isfinite(float(stats["c_clinical"]))
        else np.nan,
        "c_trimodal": round(float(stats["c_trimodal"]), 4)
        if np.isfinite(float(stats["c_trimodal"]))
        else np.nan,
        "delta_trimodal_minus_clinical": round(float(stats["delta"]), 4)
        if np.isfinite(float(stats["delta"]))
        else np.nan,
        "delta_ci_lower": round(float(stats["ci_lower"]), 4)
        if np.isfinite(float(stats["ci_lower"]))
        else np.nan,
        "delta_ci_upper": round(float(stats["ci_upper"]), 4)
        if np.isfinite(float(stats["ci_upper"]))
        else np.nan,
        "p_value": round(float(stats["p_value"]), 4)
        if np.isfinite(float(stats["p_value"]))
        else np.nan,
        "bootstrap_replicates": int(stats["bootstrap_replicates"]),
    }


def load_internal_subgroup_n() -> dict[str, int]:
    path = OUTPUT_DIR / "subgroup_bootstrap_rsf.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if not {"subtype", "n_test"}.issubset(df.columns):
        return {}
    return dict(zip(df["subtype"].astype(str), df["n_test"].astype(int)))


def plot_n_test_comparison(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df[df["subtype"].isin(SUBTYPE_ORDER + ["claudin-low"])].copy()
    x = np.arange(len(plot_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(
        x - width / 2,
        plot_df["internal_metabric_holdout_n_test"].fillna(0),
        width,
        label="Internal METABRIC held-out",
        color="#607D8B",
    )
    ax.bar(
        x + width / 2,
        plot_df["external_tcga_n_test"].fillna(0),
        width,
        label="External TCGA-BRCA",
        color="#C62828",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["subtype"], rotation=20, ha="right")
    ax.set_ylabel("Subtype test patients")
    ax.set_title("Subtype test size: internal hold-out vs external TCGA")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_c_index(validation_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = validation_df[validation_df["subtype"].isin(["All_TCGA", "Known_PAM50"] + SUBTYPE_ORDER)]
    plot_df = plot_df.copy()
    x = np.arange(len(plot_df))
    c = plot_df["c_index"].to_numpy(dtype=float)
    lo = plot_df["ci_lower"].to_numpy(dtype=float)
    hi = plot_df["ci_upper"].to_numpy(dtype=float)
    yerr = np.vstack([c - lo, hi - c])

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x, c, color="#3949AB", alpha=0.88)
    ax.errorbar(x, c, yerr=yerr, fmt="none", ecolor="#202020", capsize=4, linewidth=1)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_ylim(0.35, 0.85)
    ax.set_ylabel("External C-index")
    ax.set_title("METABRIC-trained clinical RSF on TCGA-BRCA")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["subtype"], rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.25)

    for i, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(
            i,
            min(float(row.ci_upper) + 0.025, 0.84),
            f"n={int(row.n_test)}\ne={int(row.n_events)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trimodal_comparison(comparison_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = comparison_df[
        comparison_df["subtype"].isin(["All_3way_TCGA", "Known_PAM50"] + SUBTYPE_ORDER)
    ].copy()
    x = np.arange(len(plot_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.bar(
        x - width / 2,
        plot_df["c_clinical"],
        width,
        label="Clinical",
        color="#607D8B",
    )
    ax.bar(
        x + width / 2,
        plot_df["c_trimodal"],
        width,
        label="Trimodal",
        color="#C62828",
    )
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_ylim(0.35, 0.85)
    ax.set_ylabel("External C-index")
    ax.set_title("External TCGA 3-way validation: clinical vs trimodal RSF")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["subtype"], rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    for i, row in enumerate(plot_df.itertuples(index=False)):
        y = max(float(row.c_clinical), float(row.c_trimodal))
        ax.text(
            i,
            min(y + 0.025, 0.84),
            f"n={int(row.n_test)}\ne={int(row.n_events)}\nD={float(row.delta_trimodal_minus_clinical):+.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_expanded_external_validation(B: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_outputs_dir()

    met_X, met_T, met_E = prepare_metabric_clinical()
    tcga_X, tcga_T, tcga_E, tcga_subtypes = load_tcga_clinical()

    shared = sorted(set(met_X.columns) & set(tcga_X.columns))
    met_X = met_X[shared]
    tcga_X = tcga_X[shared]

    X_train_enc, X_test_enc = encode_features(met_X, tcga_X)
    y_train = Surv.from_arrays(event=met_E.astype(bool), time=met_T)

    rsf = RandomSurvivalForest(
        n_estimators=300,
        min_samples_leaf=15,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=1,
    )
    rsf.fit(X_train_enc, y_train)

    tcga_rsf_prediction = rsf.predict(X_test_enc)
    met_rsf_prediction = rsf.predict(X_train_enc)
    tcga_score_for_cindex = -tcga_rsf_prediction
    met_score_for_cindex = -met_rsf_prediction

    validation_rows = [
        make_validation_row(
            label="All_TCGA",
            mask=np.ones(len(tcga_T), dtype=bool),
            T=tcga_T,
            E=tcga_E,
            score=tcga_score_for_cindex,
            B=B,
            seed=SEED,
        ),
        make_validation_row(
            label="Known_PAM50",
            mask=(tcga_subtypes != "Unknown").to_numpy(),
            T=tcga_T,
            E=tcga_E,
            score=tcga_score_for_cindex,
            B=B,
            seed=SEED + 1,
        ),
    ]

    for idx, subtype in enumerate(SUBTYPE_ORDER):
        validation_rows.append(
            make_validation_row(
                label=subtype,
                mask=(tcga_subtypes == subtype).to_numpy(),
                T=tcga_T,
                E=tcga_E,
                score=tcga_score_for_cindex,
                B=B,
                seed=SEED + 10 + idx,
            )
        )

    if (tcga_subtypes == "Unknown").any():
        validation_rows.append(
            make_validation_row(
                label="Unknown",
                mask=(tcga_subtypes == "Unknown").to_numpy(),
                T=tcga_T,
                E=tcga_E,
                score=tcga_score_for_cindex,
                B=B,
                seed=SEED + 99,
            )
        )

    validation_df = pd.DataFrame(validation_rows)
    validation_df.insert(0, "dataset", "TCGA-BRCA PanCancer Atlas")
    validation_df.insert(1, "model", "RSF Clinical")
    validation_df.insert(2, "train_set", "METABRIC")
    validation_df.insert(3, "train_n", len(met_X))
    validation_df.insert(4, "train_events", int(met_E.sum()))
    validation_df.insert(5, "shared_features", ", ".join(shared))
    validation_df["internal_train_c_index_resubstitution"] = round(
        safe_c_index(met_T, met_score_for_cindex, met_E), 4
    )
    validation_df["note"] = (
        "Clinical-only external validation; TCGA subtype labels mapped from BRCA_* "
        "to METABRIC-style subtype names."
    )

    internal_n = load_internal_subgroup_n()
    tcga_counts = tcga_subtypes.value_counts().to_dict()
    tcga_events = {
        subtype: int(tcga_E[(tcga_subtypes == subtype).to_numpy()].sum())
        for subtype in set(tcga_subtypes)
    }
    comparison_rows = []
    for subtype in SUBTYPE_ORDER + ["claudin-low"]:
        external_n = int(tcga_counts.get(subtype, 0))
        internal = internal_n.get(subtype, np.nan)
        comparison_rows.append(
            {
                "subtype": subtype,
                "internal_metabric_holdout_n_test": internal,
                "external_tcga_n_test": external_n,
                "external_tcga_events": int(tcga_events.get(subtype, 0)),
                "n_gain_vs_internal": external_n - internal if pd.notna(internal) else np.nan,
                "comment": (
                    "No TCGA claudin-low label in cached harmonised data."
                    if subtype == "claudin-low"
                    else "Independent TCGA external subtype test size."
                ),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)

    validation_path = OUTPUT_DIR / "expanded_external_tcga_subtype_validation.csv"
    comparison_path = OUTPUT_DIR / "expanded_external_tcga_n_test_comparison.csv"
    validation_df.to_csv(validation_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)

    plot_n_test_comparison(
        comparison_df,
        OUTPUT_DIR / "expanded_external_tcga_n_test_comparison.png",
    )
    plot_c_index(
        validation_df,
        OUTPUT_DIR / "expanded_external_tcga_subtype_cindex.png",
    )

    print("\nExpanded external validation complete.")
    print(f"Saved {validation_path}")
    print(f"Saved {comparison_path}")
    print(f"Saved {OUTPUT_DIR / 'expanded_external_tcga_n_test_comparison.png'}")
    print(f"Saved {OUTPUT_DIR / 'expanded_external_tcga_subtype_cindex.png'}")
    print("\nExternal validation summary:")
    print(
        validation_df[
            [
                "subtype",
                "n_test",
                "n_events",
                "comparable_pairs",
                "c_index",
                "ci_lower",
                "ci_upper",
            ]
        ].to_string(index=False)
    )
    print("\nTest-size comparison:")
    print(comparison_df.to_string(index=False))
    return validation_df, comparison_df


def run_trimodal_external_validation(
    *,
    B: int,
    mrna_gene_limit: int,
    pca_components: int,
    refresh_cache: bool,
) -> pd.DataFrame:
    ensure_outputs_dir()
    print("\nPreparing external trimodal TCGA validation...")
    (
        X_met_clinical_raw,
        X_tcga_clinical_raw,
        X_met_trimodal_raw,
        X_tcga_trimodal_raw,
        met_T,
        met_E,
        tcga_T,
        tcga_E,
        tcga_subtypes,
    ) = prepare_external_trimodal_frames(
        mrna_gene_limit=mrna_gene_limit,
        pca_components=pca_components,
        refresh_cache=refresh_cache,
    )

    print(
        f"  Train METABRIC core n={len(X_met_trimodal_raw)}, events={int(met_E.sum())}; "
        f"test TCGA 3-way n={len(X_tcga_trimodal_raw)}, events={int(tcga_E.sum())}"
    )
    print(
        "  Features: "
        f"{X_met_clinical_raw.shape[1]} clinical, "
        f"{sum(c.startswith('MUT__') for c in X_met_trimodal_raw.columns)} mutation genes, "
        f"{sum(c.startswith('MRNA__') for c in X_met_trimodal_raw.columns)} mRNA genes -> "
        f"{pca_components} mRNA PCs"
    )

    X_met_clinical, X_tcga_clinical = preprocess_external_frame(
        X_met_clinical_raw,
        X_tcga_clinical_raw,
        pca_components=pca_components,
    )
    X_met_trimodal, X_tcga_trimodal = preprocess_external_frame(
        X_met_trimodal_raw,
        X_tcga_trimodal_raw,
        pca_components=pca_components,
    )

    print("  Fitting clinical RSF...")
    clinical_train_pred, clinical_test_pred = fit_fixed_rsf(
        X_met_clinical,
        X_tcga_clinical,
        met_T,
        met_E,
    )
    print("  Fitting trimodal RSF...")
    trimodal_train_pred, trimodal_test_pred = fit_fixed_rsf(
        X_met_trimodal,
        X_tcga_trimodal,
        met_T,
        met_E,
    )

    clinical_test_score = -clinical_test_pred
    trimodal_test_score = -trimodal_test_pred
    clinical_train_score = -clinical_train_pred
    trimodal_train_score = -trimodal_train_pred

    rows = [
        make_trimodal_comparison_row(
            label="All_3way_TCGA",
            mask=np.ones(len(tcga_T), dtype=bool),
            T=tcga_T,
            E=tcga_E,
            clinical_score=clinical_test_score,
            trimodal_score=trimodal_test_score,
            B=B,
            seed=SEED,
        ),
        make_trimodal_comparison_row(
            label="Known_PAM50",
            mask=(tcga_subtypes != "Unknown").to_numpy(),
            T=tcga_T,
            E=tcga_E,
            clinical_score=clinical_test_score,
            trimodal_score=trimodal_test_score,
            B=B,
            seed=SEED + 1,
        ),
    ]

    for idx, subtype in enumerate(SUBTYPE_ORDER):
        rows.append(
            make_trimodal_comparison_row(
                label=subtype,
                mask=(tcga_subtypes == subtype).to_numpy(),
                T=tcga_T,
                E=tcga_E,
                clinical_score=clinical_test_score,
                trimodal_score=trimodal_test_score,
                B=B,
                seed=SEED + 10 + idx,
            )
        )

    if (tcga_subtypes == "Unknown").any():
        rows.append(
            make_trimodal_comparison_row(
                label="Unknown",
                mask=(tcga_subtypes == "Unknown").to_numpy(),
                T=tcga_T,
                E=tcga_E,
                clinical_score=clinical_test_score,
                trimodal_score=trimodal_test_score,
                B=B,
                seed=SEED + 99,
            )
        )

    comparison_df = pd.DataFrame(rows)
    comparison_df.insert(0, "dataset", "TCGA-BRCA PanCancer Atlas 3-way complete")
    comparison_df.insert(1, "model", "RSF fixed-parameter external validation")
    comparison_df.insert(2, "train_set", "METABRIC")
    comparison_df.insert(3, "train_n", len(X_met_trimodal_raw))
    comparison_df.insert(4, "train_events", int(met_E.sum()))
    comparison_df.insert(5, "clinical_features", ", ".join(X_met_clinical_raw.columns))
    comparison_df.insert(6, "mutation_features", sum(c.startswith("MUT__") for c in X_met_trimodal_raw.columns))
    comparison_df.insert(7, "mrna_genes_requested", mrna_gene_limit)
    comparison_df.insert(8, "mrna_genes_used", sum(c.startswith("MRNA__") for c in X_met_trimodal_raw.columns))
    comparison_df.insert(9, "mrna_pca_components", pca_components)
    comparison_df["internal_train_c_clinical"] = round(
        safe_c_index(met_T, clinical_train_score, met_E),
        4,
    )
    comparison_df["internal_train_c_trimodal"] = round(
        safe_c_index(met_T, trimodal_train_score, met_E),
        4,
    )
    comparison_df["note"] = (
        "Clinical and trimodal models are evaluated on the same TCGA 3-way-complete "
        "patients. Mutation features are binary recurrent METABRIC genes; mRNA uses "
        "TCGA RNA-seq z-scores for selected METABRIC genes and PCA fitted on METABRIC."
    )

    out_path = OUTPUT_DIR / "expanded_external_tcga_trimodal_comparison.csv"
    comparison_df.to_csv(out_path, index=False)
    fig_path = OUTPUT_DIR / "expanded_external_tcga_trimodal_comparison.png"
    plot_trimodal_comparison(comparison_df, fig_path)

    print("\nExternal trimodal comparison summary:")
    print(
        comparison_df[
            [
                "subtype",
                "n_test",
                "n_events",
                "comparable_pairs",
                "c_clinical",
                "c_trimodal",
                "delta_trimodal_minus_clinical",
                "delta_ci_lower",
                "delta_ci_upper",
                "p_value",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved {out_path}")
    print(f"Saved {fig_path}")
    return comparison_df


def main() -> None:
    if RUN_CLINICAL_ONLY_EXTENSION:
        run_expanded_external_validation(B=BOOTSTRAPS)
    if RUN_TRIMODAL_EXTENSION:
        run_trimodal_external_validation(
            B=BOOTSTRAPS,
            mrna_gene_limit=MRNA_GENE_LIMIT,
            pca_components=MRNA_PCA_COMPONENTS,
            refresh_cache=REFRESH_TCGA_CACHE,
        )


if __name__ == "__main__":
    main()
