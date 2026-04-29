

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from extras_consolidated import PAM50_GENES, PATHWAY_GROUPS
from expanded_external_validation import (
    TCGA_CACHE_DIR,
    bootstrap_c_index_ci,
    bootstrap_c_index_diff,
    count_comparable_pairs,
    fetch_tcga_mrna_matrix,
    load_tcga_clinical,
    safe_c_index,
    select_metabric_mrna_columns,
)
from survival_consolidated import (
    OUTPUT_DIR,
    SEED,
    encode_and_scale_clinical,
    ensure_outputs_dir,
    load_mrna_matrix,
    prepare_metabric_clinical,
    preprocess_mrna,
)


# Experiment settings. Keep these fixed so the CSVs are reproducible.
MRNA_GENE_LIMIT = 1000
MRNA_PCA_COMPONENTS = 20
BOOTSTRAPS = 1000
RSF_TREES = 200
REFRESH_TCGA_CACHE = False


def load_pam50_representations(mrna: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    gene_cols = {col.replace("MRNA__", "", 1): col for col in mrna.columns if col.startswith("MRNA__")}
    found = [gene for gene in PAM50_GENES if gene in gene_cols and not mrna[gene_cols[gene]].isna().all()]
    raw = mrna[[gene_cols[gene] for gene in found]].copy()
    raw.columns = [f"PAM50__{gene}" for gene in found]

    pathway_rows: dict[str, pd.Series] = {}
    for score_name, genes in PATHWAY_GROUPS.items():
        available = [gene_cols[gene] for gene in genes if gene in gene_cols and not mrna[gene_cols[gene]].isna().all()]
        if available:
            pathway_rows[score_name] = mrna[available].mean(axis=1)
        else:
            pathway_rows[score_name] = pd.Series(np.nan, index=mrna.index)
    pathways = pd.DataFrame(pathway_rows, index=mrna.index)
    return raw, pathways, found


def preprocess_dense_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_np = scaler.fit_transform(imputer.fit_transform(X_train))
    test_np = scaler.transform(imputer.transform(X_test))
    return (
        pd.DataFrame(train_np, columns=X_train.columns, index=X_train.index),
        pd.DataFrame(test_np, columns=X_train.columns, index=X_test.index),
    )


def fit_external_rsf(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    T_train: np.ndarray,
    E_train: np.ndarray,
    *,
    n_estimators: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_train = Surv.from_arrays(event=E_train.astype(bool), time=T_train.astype(float))
    model = RandomSurvivalForest(
        n_estimators=n_estimators,
        min_samples_leaf=15,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train.values, y_train)
    return model.predict(X_train.values), model.predict(X_test.values)


def prepare_frames(
    *,
    mrna_gene_limit: int,
    pca_components: int,
    refresh_cache: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    met_clin, met_T_all, met_E_all = prepare_metabric_clinical()
    met_mrna_all = load_mrna_matrix()
    met_index_all = met_clin.index.copy()

    tcga_clin, tcga_T_all, tcga_E_all, tcga_subtypes_all = load_tcga_clinical()
    tcga_index_all = tcga_clin.index.copy()

    pca_mrna_columns = select_metabric_mrna_columns(met_mrna_all, mrna_gene_limit)
    tcga_pca_mrna = fetch_tcga_mrna_matrix(
        pca_mrna_columns,
        refresh_cache=refresh_cache,
    )
    available_pca_columns = [
        col for col in pca_mrna_columns if col in tcga_pca_mrna.columns and not tcga_pca_mrna[col].isna().all()
    ]
    if len(available_pca_columns) < pca_components:
        raise ValueError(
            f"Only {len(available_pca_columns)} broad mRNA genes are TCGA-available; "
            f"cannot make {pca_components} PCA components."
        )

    met_pam50_columns = [f"MRNA__{gene}" for gene in PAM50_GENES if f"MRNA__{gene}" in met_mrna_all.columns]
    tcga_pam50_mrna = fetch_tcga_mrna_matrix(
        met_pam50_columns,
        refresh_cache=refresh_cache,
    )
    available_pam50_columns = [
        col for col in met_pam50_columns if col in tcga_pam50_mrna.columns and not tcga_pam50_mrna[col].isna().all()
    ]

    met_ids = sorted(set(met_clin.index) & set(met_mrna_all.index))
    tcga_ids = sorted(set(tcga_clin.index) & set(tcga_pca_mrna.index) & set(tcga_pam50_mrna.index))

    met_indexer = pd.Index(met_index_all).get_indexer(met_ids)
    tcga_indexer = pd.Index(tcga_index_all).get_indexer(tcga_ids)
    met_T = met_T_all[met_indexer]
    met_E = met_E_all[met_indexer]
    tcga_T = tcga_T_all[tcga_indexer]
    tcga_E = tcga_E_all[tcga_indexer]

    met_clin = met_clin.loc[met_ids].copy()
    tcga_clin = tcga_clin.loc[tcga_ids, met_clin.columns].copy()

    met_pca_mrna = met_mrna_all.loc[met_ids, available_pca_columns].copy()
    tcga_pca_mrna = tcga_pca_mrna.loc[tcga_ids, available_pca_columns].copy()

    met_pam50_mrna = met_mrna_all.loc[met_ids, available_pam50_columns].copy()
    tcga_pam50_mrna = tcga_pam50_mrna.loc[tcga_ids, available_pam50_columns].copy()

    met_pam50_raw, met_pam50_pathway, pam50_found = load_pam50_representations(met_pam50_mrna)
    tcga_pam50_raw, tcga_pam50_pathway, tcga_pam50_found = load_pam50_representations(tcga_pam50_mrna)

    shared_pam50_raw = sorted(set(met_pam50_raw.columns) & set(tcga_pam50_raw.columns))
    met_pam50_raw = met_pam50_raw[shared_pam50_raw]
    tcga_pam50_raw = tcga_pam50_raw[shared_pam50_raw]

    shared_pathways = sorted(set(met_pam50_pathway.columns) & set(tcga_pam50_pathway.columns))
    met_pam50_pathway = met_pam50_pathway[shared_pathways]
    tcga_pam50_pathway = tcga_pam50_pathway[shared_pathways]

    clin_train, clin_test = encode_and_scale_clinical(met_clin, tcga_clin)
    pca_train, pca_test = preprocess_mrna(met_pca_mrna, tcga_pca_mrna, n_components=pca_components)
    raw_train, raw_test = preprocess_dense_features(met_pam50_raw, tcga_pam50_raw)
    pathway_train, pathway_test = preprocess_dense_features(met_pam50_pathway, tcga_pam50_pathway)

    frames = {
        "Clinical": (clin_train, clin_test),
        "Clinical + PCA-20": (
            pd.concat([clin_train, pca_train], axis=1),
            pd.concat([clin_test, pca_test], axis=1),
        ),
        "Clinical + PAM50 raw": (
            pd.concat([clin_train, raw_train], axis=1),
            pd.concat([clin_test, raw_test], axis=1),
        ),
        "Clinical + PAM50 pathway scores": (
            pd.concat([clin_train, pathway_train], axis=1),
            pd.concat([clin_test, pathway_test], axis=1),
        ),
    }

    meta = {
        "n_train": len(met_ids),
        "n_train_events": int(met_E.sum()),
        "n_test": len(tcga_ids),
        "n_test_events": int(tcga_E.sum()),
        "n_pca_genes_requested": mrna_gene_limit,
        "n_pca_genes_used": len(available_pca_columns),
        "n_pca_components": pca_components,
        "n_pam50_metabric_available": len(pam50_found),
        "n_pam50_tcga_available": len(tcga_pam50_found),
        "n_pam50_shared_raw": len(shared_pam50_raw),
        "n_pam50_pathway_scores": len(shared_pathways),
        "tcga_known_pam50_n": int((tcga_subtypes_all.loc[tcga_ids] != "Unknown").sum()),
        "tcga_cache_dir": str(TCGA_CACHE_DIR),
    }
    return met_clin, tcga_clin, frames, met_T, met_E, tcga_T, tcga_E, meta


def run_pam50_external_test(
    *,
    mrna_gene_limit: int,
    pca_components: int,
    B: int,
    n_estimators: int,
    refresh_cache: bool,
) -> pd.DataFrame:
    ensure_outputs_dir()
    print("\nPreparing PAM50-on-TCGA representation experiment...", flush=True)
    _, _, frames, met_T, met_E, tcga_T, tcga_E, meta = prepare_frames(
        mrna_gene_limit=mrna_gene_limit,
        pca_components=pca_components,
        refresh_cache=refresh_cache,
    )
    print(
        f"  Train METABRIC clinical+mRNA n={meta['n_train']}, events={meta['n_train_events']}; "
        f"test TCGA 3-way n={meta['n_test']}, events={meta['n_test_events']}"
    )
    print(
        f"  PCA genes used: {meta['n_pca_genes_used']}/{meta['n_pca_genes_requested']}; "
        f"PAM50 raw genes shared: {meta['n_pam50_shared_raw']}; "
        f"pathway scores: {meta['n_pam50_pathway_scores']}"
    )

    predictions: dict[str, dict[str, np.ndarray | float]] = {}
    for label, (X_train, X_test) in frames.items():
        print(
            f"  Fitting RSF: {label} ({X_train.shape[1]} features, {n_estimators} trees)",
            flush=True,
        )
        train_pred, test_pred = fit_external_rsf(
            X_train,
            X_test,
            met_T,
            met_E,
            n_estimators=n_estimators,
        )
        train_score = -train_pred
        test_score = -test_pred
        predictions[label] = {
            "train_score": train_score,
            "test_score": test_score,
            "train_c": safe_c_index(met_T, train_score, met_E),
            "test_c": safe_c_index(tcga_T, test_score, tcga_E),
            "n_features": X_train.shape[1],
        }

    rows = []
    clinical_score = predictions["Clinical"]["test_score"]
    pca_score = predictions["Clinical + PCA-20"]["test_score"]
    for idx, label in enumerate(frames.keys()):
        test_score = predictions[label]["test_score"]
        ci_lower, ci_upper, boot_n = bootstrap_c_index_ci(
            tcga_T,
            test_score,
            tcga_E,
            B=B,
            seed=SEED + idx,
        )
        if label == "Clinical":
            delta_clin = {
                "delta": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_value": np.nan,
                "bootstrap_replicates": boot_n,
            }
        else:
            delta_clin = bootstrap_c_index_diff(
                tcga_T,
                tcga_E,
                clinical_score,
                test_score,
                B=B,
                seed=SEED + 20 + idx,
            )

        if label == "Clinical + PCA-20":
            delta_pca = {
                "delta": 0.0,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "p_value": np.nan,
                "bootstrap_replicates": boot_n,
            }
        elif label == "Clinical":
            delta_pca = bootstrap_c_index_diff(
                tcga_T,
                tcga_E,
                pca_score,
                test_score,
                B=B,
                seed=SEED + 40 + idx,
            )
        else:
            delta_pca = bootstrap_c_index_diff(
                tcga_T,
                tcga_E,
                pca_score,
                test_score,
                B=B,
                seed=SEED + 40 + idx,
            )

        rows.append(
            {
                "representation": label,
                "n_features": int(predictions[label]["n_features"]),
                "train_c_index": round(float(predictions[label]["train_c"]), 4),
                "tcga_c_index": round(float(predictions[label]["test_c"]), 4),
                "tcga_ci_lower": round(float(ci_lower), 4),
                "tcga_ci_upper": round(float(ci_upper), 4),
                "delta_vs_clinical": round(float(delta_clin["delta"]), 4)
                if np.isfinite(float(delta_clin["delta"]))
                else np.nan,
                "delta_vs_clinical_ci_lower": round(float(delta_clin["ci_lower"]), 4)
                if np.isfinite(float(delta_clin["ci_lower"]))
                else np.nan,
                "delta_vs_clinical_ci_upper": round(float(delta_clin["ci_upper"]), 4)
                if np.isfinite(float(delta_clin["ci_upper"]))
                else np.nan,
                "delta_vs_clinical_p": round(float(delta_clin["p_value"]), 4)
                if np.isfinite(float(delta_clin["p_value"]))
                else np.nan,
                "delta_vs_pca20": round(float(delta_pca["delta"]), 4)
                if np.isfinite(float(delta_pca["delta"]))
                else np.nan,
                "delta_vs_pca20_ci_lower": round(float(delta_pca["ci_lower"]), 4)
                if np.isfinite(float(delta_pca["ci_lower"]))
                else np.nan,
                "delta_vs_pca20_ci_upper": round(float(delta_pca["ci_upper"]), 4)
                if np.isfinite(float(delta_pca["ci_upper"]))
                else np.nan,
                "delta_vs_pca20_p": round(float(delta_pca["p_value"]), 4)
                if np.isfinite(float(delta_pca["p_value"]))
                else np.nan,
                "bootstrap_replicates": int(boot_n),
            }
        )

    results = pd.DataFrame(rows)
    for key, value in meta.items():
        results[key] = value
    results["rsf_n_estimators"] = n_estimators
    results["comparable_pairs"] = count_comparable_pairs(tcga_T, tcga_E)
    results["note"] = (
        "Fixed-parameter RSF trained on METABRIC clinical+mRNA and evaluated on "
        "TCGA-BRCA 3-way-complete patients. PCA uses high-variance METABRIC genes "
        "available in TCGA; PAM50 uses shared PAM50 genes and four mean z-score "
        "pathway summaries."
    )

    out_csv = OUTPUT_DIR / "pam50_tcga_representation_comparison.csv"
    results.to_csv(out_csv, index=False)
    out_png = OUTPUT_DIR / "pam50_tcga_representation_comparison.png"
    plot_results(results, out_png)

    print("\nPAM50-on-TCGA representation summary:")
    print(
        results[
            [
                "representation",
                "n_features",
                "tcga_c_index",
                "tcga_ci_lower",
                "tcga_ci_upper",
                "delta_vs_clinical",
                "delta_vs_pca20",
                "delta_vs_pca20_p",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved {out_csv}")
    print(f"Saved {out_png}")
    return results


def plot_results(results: pd.DataFrame, out_path: Path) -> None:
    plot_df = results.copy()
    x = np.arange(len(plot_df))
    c = plot_df["tcga_c_index"].to_numpy(dtype=float)
    lo = plot_df["tcga_ci_lower"].to_numpy(dtype=float)
    hi = plot_df["tcga_ci_upper"].to_numpy(dtype=float)
    yerr = np.vstack([c - lo, hi - c])

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    colors = ["#607D8B", "#3949AB", "#C62828", "#EF6C00"]
    ax.bar(x, c, color=colors[: len(plot_df)], alpha=0.9)
    ax.errorbar(x, c, yerr=yerr, fmt="none", ecolor="#222222", capsize=4, linewidth=1)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_ylim(0.35, 0.85)
    ax.set_ylabel("External TCGA C-index")
    ax.set_title("TCGA expanded test: PCA-20 vs PAM50 mRNA representations")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["representation"], rotation=18, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    for i, row in enumerate(plot_df.itertuples(index=False)):
        if row.representation == "Clinical":
            label = "baseline"
        else:
            label = f"Dclin={float(row.delta_vs_clinical):+.3f}"
        ax.text(i, min(float(row.tcga_ci_upper) + 0.02, 0.84), label, ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    run_pam50_external_test(
        mrna_gene_limit=MRNA_GENE_LIMIT,
        pca_components=MRNA_PCA_COMPONENTS,
        B=BOOTSTRAPS,
        n_estimators=RSF_TREES,
        refresh_cache=REFRESH_TCGA_CACHE,
    )


if __name__ == "__main__":
    main()
