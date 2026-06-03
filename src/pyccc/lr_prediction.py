from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .density import build_predicted_lr_table, estimate_lr_density_prior
from .embeddings import ESMC_300M_MODEL_NAME, embed_proteins_esmc
from .pair_features import make_lr_pair_features
from .roles import predict_protein_roles
from .sequence import load_cds_translations, load_protein_fasta, match_expression_genes


DBFREE_STACK_NAME = "esmc300m_lgbm_role_classifiers_lgbm_pair_ranker_clade_density_v0"
DEFAULT_DBFREE_ROLE_MODEL = "models/universal_esmc300m_lgbm_role_classifiers_v0"
DEFAULT_DBFREE_PAIR_MODEL = "models/universal_esmc300m_lgbm_pair_ranker_v0"


def generate_lr_candidates_dbfree(
    adata,
    proteins: pd.DataFrame,
    roles: pd.DataFrame,
    *,
    embeddings: pd.DataFrame | None = None,
    gene_id_key: str | None = None,
    expression_min_fraction: float = 0.02,
    ligand_role_min: float = 0.30,
    receptor_role_min: float = 0.30,
    max_ligands: int = 3000,
    max_receptors: int = 3000,
    max_candidate_pairs: int = 5_000_000,
    nearest_neighbor_pairs: int = 0,
    nearest_neighbors_per_ligand: int = 10,
    ligand_candidates: str | Path | Sequence[str] | None = None,
    receptor_candidates: str | Path | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate role-, expression-, and optional embedding-neighbor LR pairs."""

    if max_candidate_pairs < 1:
        raise ValueError("`max_candidate_pairs` must be at least 1.")
    if nearest_neighbor_pairs < 0:
        raise ValueError("`nearest_neighbor_pairs` must be non-negative.")
    if nearest_neighbors_per_ligand < 1:
        raise ValueError("`nearest_neighbors_per_ligand` must be at least 1.")
    if nearest_neighbor_pairs and embeddings is None:
        raise ValueError("`embeddings` is required when `nearest_neighbor_pairs` is greater than 0.")
    expressed = _expressed_gene_fractions(adata, gene_id_key=gene_id_key)
    expressed = expressed[expressed["expression_fraction"] >= expression_min_fraction]
    role_frame = roles.merge(proteins[["gene_id", "protein_id"]], on=["gene_id", "protein_id"], how="inner")
    role_frame = role_frame.merge(expressed, on="gene_id", how="inner")
    if ligand_candidates is not None:
        ligands = role_frame[role_frame["gene_id"].isin(_candidate_list(ligand_candidates))].copy()
    else:
        ligands = role_frame[role_frame["ligand_like_score"].astype(float) >= ligand_role_min].copy()
    if receptor_candidates is not None:
        receptors = role_frame[role_frame["gene_id"].isin(_candidate_list(receptor_candidates))].copy()
    else:
        receptors = role_frame[role_frame["receptor_like_score"].astype(float) >= receptor_role_min].copy()
    ligands = ligands.sort_values(["ligand_like_score", "expression_fraction"], ascending=False).head(max_ligands)
    receptors = receptors.sort_values(["receptor_like_score", "expression_fraction"], ascending=False).head(max_receptors)
    if ligands.empty or receptors.empty:
        raise ValueError("No ligand or receptor candidates remain after role/expression filtering.")
    neighbor_ligands = ligands.copy()
    neighbor_receptors = receptors.copy()
    total = len(ligands) * len(receptors)
    neighbor_budget = min(int(nearest_neighbor_pairs), int(max_candidate_pairs))
    cross_product_budget = max(0, int(max_candidate_pairs) - neighbor_budget)
    if total > cross_product_budget:
        keep_receptors = (
            max(1, cross_product_budget // max(len(ligands), 1))
            if cross_product_budget > 0
            else 0
        )
        receptors = receptors.head(keep_receptors)
    rows = []
    if cross_product_budget > 0:
        for lig in ligands.itertuples(index=False):
            for rec in receptors.itertuples(index=False):
                if str(lig.gene_id) == str(rec.gene_id):
                    continue
                rows.append(_candidate_row(lig, rec, strategy="role_expression_cross_product"))
    if neighbor_budget > 0:
        rows.extend(
            _embedding_nearest_neighbor_candidate_rows(
                neighbor_ligands,
                neighbor_receptors,
                embeddings,
                top_pairs=neighbor_budget,
                neighbors_per_ligand=nearest_neighbors_per_ligand,
            )
        )
    if not rows:
        raise ValueError("Candidate generation produced no non-self LR pairs.")
    return _deduplicate_candidate_rows(rows).head(max_candidate_pairs)


def train_lr_link_predictor(
    training_table,
    embeddings: pd.DataFrame,
    *,
    model: str = "lightgbm",
    feature_encoder: str = "pca128_absdiff_hadamard_v1",
    validation_splits: Sequence[str] = ("leave_species_out", "leave_resource_out", "leave_family_out"),
    negative_strategy: str = "pu_degree_matched",
    output_dir: str | Path,
    negative_ratio: int = 5,
    easy_negative_fraction: float = 0.05,
    excluded_homology_radius: str = "family_pair",
    negative_repeats: int = 1,
    calibration_method: str | None = "isotonic",
    density_groupby: str = "clade",
    max_reference_pairs: int = 20000,
    random_state: int = 0,
) -> dict[str, object]:
    """Train a pairwise LR ranker and write a model card."""

    from joblib import dump

    if negative_repeats < 1:
        raise ValueError("`negative_repeats` must be at least 1.")
    interactions = getattr(training_table, "interactions", training_table)
    pairs = _training_pairs(
        interactions,
        embeddings=embeddings,
        negative_ratio=negative_ratio,
        negative_strategy=negative_strategy,
        easy_negative_fraction=easy_negative_fraction,
        excluded_homology_radius=excluded_homology_radius,
        random_state=random_state,
    )
    y = pairs["label"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Training requires at least one positive and one pseudo-negative pair.")
    train_idx, test_idx = _random_train_test_indices(y, random_state=random_state)
    random_metrics = _evaluate_pair_split(
        pairs,
        embeddings,
        y,
        train_idx,
        test_idx,
        model=model,
        feature_encoder=feature_encoder,
        random_state=random_state,
    )
    metrics = {
        "pr_auc": float(random_metrics["pr_auc"]),
        "roc_auc": float(random_metrics["roc_auc"]),
        "top_k_precision": random_metrics.get("top_k_precision", {}),
        "top_k_recall": random_metrics.get("top_k_recall", {}),
        "top_k_enrichment": random_metrics.get("top_k_enrichment", {}),
        "baseline_pr_auc": random_metrics.get("baseline_pr_auc", {}),
        "baseline_top_k_precision": random_metrics.get("baseline_top_k_precision", {}),
        "baseline_top_k_recall": random_metrics.get("baseline_top_k_recall", {}),
        "baseline_top_k_enrichment": random_metrics.get("baseline_top_k_enrichment", {}),
        "baseline_delta_pr_auc": random_metrics.get("baseline_delta_pr_auc", {}),
        "baseline_delta_top_k_precision": random_metrics.get("baseline_delta_top_k_precision", {}),
        "baseline_delta_top_k_recall": random_metrics.get("baseline_delta_top_k_recall", {}),
        "baseline_delta_top_k_enrichment": random_metrics.get("baseline_delta_top_k_enrichment", {}),
        "n_pairs": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "negative_strategy": negative_strategy,
        "negative_sampling": _negative_sampling_summary(pairs),
    }
    training_metadata = _training_metadata(interactions, embeddings, pairs)
    validation_report = _validation_report(
        pairs,
        embeddings,
        y,
        requested_splits=validation_splits,
        model=model,
        feature_encoder=feature_encoder,
        random_state=random_state,
    )
    repeat_report = _negative_repeat_validation_report(
        interactions,
        embeddings,
        requested_splits=validation_splits,
        model=model,
        feature_encoder=feature_encoder,
        negative_ratio=negative_ratio,
        negative_strategy=negative_strategy,
        easy_negative_fraction=easy_negative_fraction,
        excluded_homology_radius=excluded_homology_radius,
        n_repeats=negative_repeats,
        random_state=random_state,
    )
    features = make_lr_pair_features(pairs, embeddings, encoder=feature_encoder, fit_pca=True)
    clf = _fit_pair_model(model, features.X, y, random_state=random_state)
    reference = _reference_annotation_payload(pairs, features.X, max_reference_pairs=max_reference_pairs, random_state=random_state)
    calibration = _fit_calibrator_from_split(
        pairs,
        embeddings,
        y,
        train_idx,
        test_idx,
        model=model,
        feature_encoder=feature_encoder,
        method=calibration_method,
        random_state=random_state,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": clf,
            "pca_model": features.pca_model,
            "feature_encoder": feature_encoder,
            "feature_names": features.feature_names,
            "calibrator": calibration["calibrator"],
            "reference_pairs": reference["pairs"],
            "reference_features": reference["features"],
        },
        output / "lr_link_model.joblib",
    )
    density_prior = _write_density_prior(interactions, output, groupby=density_groupby)
    card = {
        "model_name": output.name,
        "model_stack": DBFREE_STACK_NAME if model == "lightgbm" else "fixture_or_baseline_pair_ranker",
        "model_type": model,
        "feature_encoder": feature_encoder,
        "density_prior_groupby": density_groupby,
        "density_prior_path": str(output / "density_prior.tsv") if density_prior is not None else "",
        "final_model_training": "all_pairs_after_validation",
        "validation_feature_encoder_fit": "train_split_only",
        "calibration_method": calibration["method"],
        "calibration_metrics": calibration["metrics"],
        "validation_splits": list(validation_splits),
        "metrics": metrics,
        "training_resources": training_metadata["training_resources"],
        "species_included": training_metadata["species_included"],
        "clades_included": training_metadata["clades_included"],
        "positive_labels_by_species": training_metadata["positive_labels_by_species"],
        "pseudo_negatives_by_species": training_metadata["pseudo_negatives_by_species"],
        "embedding_model": training_metadata["embedding_model"],
        "pair_model_params": _pair_model_params(model, random_state=random_state),
        "validation_report": validation_report,
        "negative_strategy": negative_strategy,
        "negative_ratio": negative_ratio,
        "easy_negative_fraction": easy_negative_fraction,
        "excluded_homology_radius": excluded_homology_radius,
        "negative_repeats": negative_repeats,
        "negative_repeat_report": repeat_report,
        "negative_sampling": _negative_sampling_summary(pairs),
        "output_dir": str(output),
    }
    (output / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (output / "model_card.md").write_text(_model_card_markdown(card), encoding="utf-8")
    return card


def evaluate_lr_model_quality_gates(
    model_card: dict[str, object] | str | Path,
    *,
    required_splits: Sequence[str] = ("leave_species_out",),
    required_baselines: Sequence[str] = ("degree_prior", "embedding_cosine", "expression_only", "role_only", "density_matched_random", "random"),
    min_pr_auc_delta: float = 0.0,
    top_k: str | None = None,
    min_top_k_delta: float | None = None,
) -> pd.DataFrame:
    """Evaluate whether a model card clears baseline comparison gates."""

    card = _load_model_card(model_card)
    rows = []
    report = card.get("validation_report", {})
    if min_top_k_delta is not None and top_k is None:
        top_k = "top_100"
    for split in required_splits:
        item = report.get(split)
        if not isinstance(item, dict):
            rows.append(_gate_row(split, "", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, "missing split"))
            continue
        if item.get("status") != "ok":
            rows.append(
                _gate_row(
                    split,
                    "",
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    False,
                    str(item.get("reason", "split not usable")),
                )
            )
            continue
        metric = _split_metric_item(item)
        model_pr = metric.get("mean_pr_auc", metric.get("pr_auc", np.nan))
        baselines = metric.get("mean_baseline_pr_auc", metric.get("baseline_pr_auc", {}))
        model_top_k = _metric_top_k(metric, top_k) if top_k else np.nan
        baseline_top_k = metric.get("mean_baseline_top_k_precision", metric.get("baseline_top_k_precision", {}))
        for baseline in required_baselines:
            if baseline not in baselines:
                rows.append(
                    _gate_row(
                        split,
                        baseline,
                        model_pr,
                        np.nan,
                        np.nan,
                        model_top_k,
                        np.nan,
                        np.nan,
                        False,
                        "missing baseline",
                    )
                )
                continue
            base_pr = float(baselines[baseline])
            delta = float(model_pr) - base_pr
            base_top_k = _baseline_top_k(baseline_top_k, baseline, top_k) if top_k else np.nan
            top_delta = float(model_top_k) - float(base_top_k) if top_k and pd.notna(model_top_k) and pd.notna(base_top_k) else np.nan
            pr_pass = bool(pd.notna(delta) and delta >= min_pr_auc_delta)
            top_pass = True
            reasons = []
            if not pr_pass:
                reasons.append("PR-AUC below threshold")
            if min_top_k_delta is not None:
                top_pass = bool(pd.notna(top_delta) and top_delta >= min_top_k_delta)
                if not top_pass:
                    reasons.append(f"{top_k} precision below threshold" if top_k else "top-K precision missing")
            passed = pr_pass and top_pass
            rows.append(
                _gate_row(
                    split,
                    baseline,
                    model_pr,
                    base_pr,
                    delta,
                    model_top_k,
                    base_top_k,
                    top_delta,
                    passed,
                    "ok" if passed else "; ".join(reasons),
                )
            )
    out = pd.DataFrame(rows)
    out.attrs["passed"] = bool(out["passed"].all()) if not out.empty else False
    out.attrs["min_pr_auc_delta"] = float(min_pr_auc_delta)
    out.attrs["top_k"] = top_k
    out.attrs["min_top_k_delta"] = min_top_k_delta
    return out


def score_lr_candidates(
    candidate_pairs: pd.DataFrame,
    embeddings: pd.DataFrame,
    *,
    model: str | Path = "heuristic",
    batch_size: int = 200000,
) -> pd.DataFrame:
    """Score candidate LR pairs with a trained model or deterministic heuristic."""

    pairs = candidate_pairs.copy()
    if str(model) == "heuristic":
        out = pairs.copy()
        out["model_score"] = _heuristic_pair_scores(out, embeddings)
        out["calibrated_probability"] = out["model_score"]
        out["model_name"] = "heuristic_fixture"
        out["warning"] = _append_warning(out.get("warning", pd.Series([""] * len(out))), ["heuristic_pair_ranker"])
        return out.sort_values("model_score", ascending=False).reset_index(drop=True)

    from joblib import load

    payload = load(Path(model) / "lr_link_model.joblib")
    rows = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs.iloc[start : start + batch_size].copy()
        features = make_lr_pair_features(chunk, embeddings, encoder=payload["feature_encoder"], pca_model=payload.get("pca_model"))
        chunk["model_score"] = _predict_scores(payload["model"], features.X)
        chunk["calibrated_probability"] = _apply_calibrator(payload.get("calibrator"), chunk["model_score"].to_numpy(dtype=float))
        chunk = _annotate_nearest_reference(chunk, features.X, payload)
        rows.append(chunk)
    return pd.concat(rows, ignore_index=True).sort_values("model_score", ascending=False).reset_index(drop=True)


def predict_lr_dbfree(
    adata,
    *,
    cds_fasta: str | Path | None = None,
    protein_fasta: str | Path | None = None,
    gene_id_key: str | None = None,
    species_name: str = "target_species",
    species_hint: str = "unknown",
    model: str | Path = DEFAULT_DBFREE_PAIR_MODEL,
    role_model: str | Path | None = DEFAULT_DBFREE_ROLE_MODEL,
    density_prior: pd.DataFrame | float | str = "auto",
    min_score: float = 0.50,
    max_pairs: int = 50000,
    max_pairs_per_ligand: int = 200,
    max_pairs_per_receptor: int = 200,
    allow_low_score_density_fill: bool = False,
    cache_dir: str | Path | None = None,
    embedding_backend: str = "auto",
    embedding_model_name: str = ESMC_300M_MODEL_NAME,
    embedding_model_revision: str | None = None,
    allow_fixture_models: bool = False,
    ligand_candidates: str | Path | Sequence[str] | None = None,
    receptor_candidates: str | Path | Sequence[str] | None = None,
    **candidate_kwargs,
):
    """Predict a target-species candidate LR table, then return CellChatDB."""

    if (cds_fasta is None) == (protein_fasta is None):
        raise ValueError("Provide exactly one of `cds_fasta` or `protein_fasta`.")
    role_model_bypassed = _role_model_bypassed(role_model, ligand_candidates, receptor_candidates)
    _validate_dbfree_prediction_stack(
        model=model,
        role_model=role_model,
        role_model_bypassed=role_model_bypassed,
        density_prior=density_prior,
        embedding_backend=embedding_backend,
        embedding_model_name=embedding_model_name,
        allow_fixture_models=allow_fixture_models,
    )
    proteins = load_cds_translations(cds_fasta) if cds_fasta is not None else load_protein_fasta(protein_fasta)
    gene_match = match_expression_genes(adata, proteins, gene_id_key=gene_id_key)
    gene_match_summary = _gene_match_summary(gene_match)
    emb = embed_proteins_esmc(
        proteins,
        model_name=embedding_model_name,
        model_revision=embedding_model_revision,
        cache_dir=Path(cache_dir) / "esmc" if cache_dir is not None else None,
        backend=embedding_backend,
    )
    roles = (
        _explicit_candidate_roles(proteins, ligand_candidates, receptor_candidates)
        if role_model_bypassed
        else predict_protein_roles(proteins, emb, model=role_model)
    )
    candidates = generate_lr_candidates_dbfree(
        adata,
        proteins,
        roles,
        embeddings=emb,
        gene_id_key=gene_id_key,
        ligand_candidates=ligand_candidates,
        receptor_candidates=receptor_candidates,
        **candidate_kwargs,
    )
    scores = score_lr_candidates(candidates, emb, model=model)
    resolved_density_prior = _auto_density_prior(density_prior, model)
    pair_metadata = _pair_model_metadata(model)
    resolved_embedding_backend = "hash" if embedding_backend == "hash" or str(embedding_model_name) in {"hash", "fake"} else "esmc"
    db = build_predicted_lr_table(
        scores,
        roles=roles,
        density_prior=resolved_density_prior,
        species_hint=species_hint,
        min_score=min_score,
        max_pairs=max_pairs,
        max_pairs_per_ligand=max_pairs_per_ligand,
        max_pairs_per_receptor=max_pairs_per_receptor,
        allow_low_score_density_fill=allow_low_score_density_fill,
        model_name=pair_metadata["model_name"],
        model_version=pair_metadata["model_version"],
        model_revision=embedding_model_revision or pair_metadata["model_revision"],
        feature_encoder=pair_metadata["feature_encoder"],
        name=f"dbfree_predicted_{species_name}",
    )
    db.metadata["proteins"] = proteins
    db.metadata["gene_match"] = gene_match
    db.metadata["gene_match_summary"] = gene_match_summary
    db.metadata["embeddings"] = emb
    db.metadata["dbfree_model_stack"] = {
        "stack_name": DBFREE_STACK_NAME,
        "embedding_model_name": embedding_model_name,
        "embedding_model_revision": embedding_model_revision or "",
        "embedding_backend": resolved_embedding_backend,
        "role_model": "explicit_candidates" if role_model_bypassed else str(role_model),
        "role_model_bypassed": bool(role_model_bypassed),
        "pair_model": str(model),
        "pair_model_name": pair_metadata["model_name"],
        "density_prior": "auto_from_pair_model" if isinstance(density_prior, str) and density_prior == "auto" else "user_supplied",
        "density_groupby": "clade",
        "allow_fixture_models": bool(allow_fixture_models),
        **gene_match_summary,
    }
    summary = db.metadata.get("prediction_summary")
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        summary["model_stack"] = DBFREE_STACK_NAME
        summary["embedding_model_name"] = embedding_model_name
        summary["embedding_model_revision"] = embedding_model_revision or ""
        summary["embedding_backend"] = db.metadata["dbfree_model_stack"]["embedding_backend"]
        summary["role_model"] = "explicit_candidates" if role_model_bypassed else str(role_model)
        summary["role_model_bypassed"] = bool(role_model_bypassed)
        summary["pair_model"] = str(model)
        summary["pair_model_name"] = pair_metadata["model_name"]
        summary["density_groupby"] = "clade"
        summary["allow_fixture_models"] = bool(allow_fixture_models)
        for key, value in gene_match_summary.items():
            summary[key] = value
    extra_warnings = []
    if str(role_model) == "heuristic":
        extra_warnings.append("heuristic_role_model")
    if role_model_bypassed:
        extra_warnings.append("explicit_candidate_role_bypass")
    if resolved_embedding_backend == "hash":
        extra_warnings.append("hash_embedding_backend")
    if gene_match_summary["n_expression_only_genes"] > 0:
        extra_warnings.append("unmatched_expression_genes")
    if gene_match_summary["n_protein_only_genes"] > 0:
        extra_warnings.append("unmatched_protein_genes")
    if extra_warnings:
        _add_prediction_warnings(db, extra_warnings)
    return db


def _role_model_bypassed(
    role_model: str | Path | None,
    ligand_candidates: str | Path | Sequence[str] | None,
    receptor_candidates: str | Path | Sequence[str] | None,
) -> bool:
    if role_model is not None and str(role_model).lower() not in {"none", "bypass", "explicit_candidates"}:
        return False
    if ligand_candidates is None or receptor_candidates is None:
        raise ValueError("Bypassing the role model requires both `ligand_candidates` and `receptor_candidates`.")
    return True


def _explicit_candidate_roles(
    proteins: pd.DataFrame,
    ligand_candidates: str | Path | Sequence[str] | None,
    receptor_candidates: str | Path | Sequence[str] | None,
) -> pd.DataFrame:
    ligands = _candidate_list(ligand_candidates or [])
    receptors = _candidate_list(receptor_candidates or [])
    out = proteins[["gene_id", "protein_id"]].copy()
    out["ligand_like_score"] = out["gene_id"].astype(str).isin(ligands).astype(float)
    out["receptor_like_score"] = out["gene_id"].astype(str).isin(receptors).astype(float)
    out["secreted_like_score"] = out["ligand_like_score"]
    out["membrane_like_score"] = out["receptor_like_score"]
    out["ecm_like_score"] = 0.0
    out["out_of_domain_score"] = 0.0
    return out


def _gene_match_summary(gene_match: pd.DataFrame) -> dict[str, int]:
    in_expression = gene_match["in_expression"].astype(bool)
    in_proteins = gene_match["in_proteins"].astype(bool)
    return {
        "n_expression_genes": int(in_expression.sum()),
        "n_protein_genes": int(in_proteins.sum()),
        "n_matched_genes": int((in_expression & in_proteins).sum()),
        "n_expression_only_genes": int((in_expression & ~in_proteins).sum()),
        "n_protein_only_genes": int((~in_expression & in_proteins).sum()),
    }


def _write_density_prior(interactions: pd.DataFrame, output: Path, *, groupby: str) -> pd.DataFrame | None:
    density_interactions = interactions.copy()
    if "is_positive_label" in density_interactions.columns:
        density_interactions = density_interactions[density_interactions["is_positive_label"].astype(bool)].copy()
    try:
        density = estimate_lr_density_prior(density_interactions, groupby=groupby)
    except ValueError:
        return None
    density.to_csv(output / "density_prior.tsv", sep="\t", index=False)
    return density


def _auto_density_prior(density_prior: pd.DataFrame | float | str, model: str | Path) -> pd.DataFrame | float | str:
    if not (isinstance(density_prior, str) and density_prior == "auto"):
        return density_prior
    if str(model) == "heuristic":
        return density_prior
    path = Path(model)
    table_path = path / "density_prior.tsv"
    if table_path.exists():
        return pd.read_csv(table_path, sep="\t")
    return density_prior


def _validate_dbfree_prediction_stack(
    *,
    model: str | Path,
    role_model: str | Path | None,
    role_model_bypassed: bool,
    density_prior: pd.DataFrame | float | str,
    embedding_backend: str,
    embedding_model_name: str,
    allow_fixture_models: bool,
) -> None:
    if allow_fixture_models:
        return

    problems = []
    missing_paths = []
    model_name_lower = str(embedding_model_name).lower()
    if embedding_backend == "hash" or str(embedding_model_name) in {"hash", "fake"}:
        problems.append("production DB-free prediction requires ESMC-300M embeddings, not the hash fixture backend")
    if not ("esmc" in model_name_lower and "300m" in model_name_lower):
        problems.append(f"production DB-free prediction expects an ESMC-300M model name, got `{embedding_model_name}`")
    if role_model_bypassed:
        pass
    elif str(role_model) == "heuristic":
        problems.append("production DB-free prediction requires trained LightGBM protein role classifiers")
    else:
        role_path = Path(role_model)
        if not (role_path / "role_model.joblib").exists():
            missing_paths.append(f"LightGBM protein role classifiers: {role_path / 'role_model.joblib'}")
    if str(model) == "heuristic":
        problems.append("production DB-free prediction requires a trained LightGBM pair ranker")
    else:
        pair_path = Path(model)
        if not (pair_path / "lr_link_model.joblib").exists():
            missing_paths.append(f"LightGBM pair ranker: {pair_path / 'lr_link_model.joblib'}")
        if isinstance(density_prior, str) and density_prior == "auto" and not (pair_path / "density_prior.tsv").exists():
            missing_paths.append(f"clade-aware density prior: {pair_path / 'density_prior.tsv'}")
    if isinstance(density_prior, (float, int)):
        problems.append("production DB-free prediction requires a clade-aware density table or `density_prior='auto'`, not a scalar prior")
    elif isinstance(density_prior, str) and density_prior != "auto":
        problems.append("production DB-free prediction requires a clade-aware density table or `density_prior='auto'`, not a scalar string prior")
    elif isinstance(density_prior, pd.DataFrame) and not ({"clade", "species_hint"} & set(density_prior.columns)):
        problems.append("production DB-free prediction density table must contain `clade` or `species_hint`")

    if problems:
        raise ValueError("; ".join(problems) + ". Pass `allow_fixture_models=True` only for tests or dry runs.")
    if missing_paths:
        raise FileNotFoundError(
            "Production DB-free prediction needs the ESMC-300M embedding + LightGBM role classifiers + LightGBM pair ranker + clade-aware density prior stack files. Missing: "
            + "; ".join(missing_paths)
            + ". Train them first or pass explicit paths."
        )


def _pair_model_metadata(model: str | Path) -> dict[str, str]:
    if str(model) == "heuristic":
        return {
            "model_name": "heuristic_fixture",
            "model_version": "0",
            "model_revision": "",
            "feature_encoder": "heuristic",
        }
    path = Path(model)
    card_path = path / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
    embedding_model = card.get("embedding_model", {}) if isinstance(card.get("embedding_model"), dict) else {}
    return {
        "model_name": str(card.get("model_name") or path.name or DEFAULT_DBFREE_PAIR_MODEL),
        "model_version": str(card.get("model_version") or card.get("version") or "0"),
        "model_revision": _first_metadata_value(embedding_model.get("model_revision")),
        "feature_encoder": str(card.get("feature_encoder") or "pca128_absdiff_hadamard_v1"),
    }


def _first_metadata_value(value) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if value is None:
        return ""
    return str(value)


def _reference_annotation_payload(
    pairs: pd.DataFrame,
    X: np.ndarray,
    *,
    max_reference_pairs: int,
    random_state: int,
) -> dict[str, object]:
    positives = pairs[pairs["label"].astype(int) == 1].copy()
    if positives.empty:
        return {"pairs": pd.DataFrame(), "features": np.empty((0, X.shape[1]), dtype=np.float32)}
    if len(positives) > max_reference_pairs:
        positives = positives.sample(n=max_reference_pairs, random_state=random_state)
    idx = positives.index.to_numpy(dtype=int)
    cols = [col for col in ("ligand_gene", "receptor_gene", "species", "resource", "pathway", "annotation") if col in positives.columns]
    return {
        "pairs": positives[cols].reset_index(drop=True),
        "features": np.asarray(X[idx], dtype=np.float32),
    }


def _annotate_nearest_reference(chunk: pd.DataFrame, X: np.ndarray, payload: dict[str, object]) -> pd.DataFrame:
    ref_pairs = payload.get("reference_pairs")
    ref_features = payload.get("reference_features")
    if not isinstance(ref_pairs, pd.DataFrame) or ref_pairs.empty or ref_features is None or len(ref_features) == 0:
        out = chunk.copy()
        out["warning"] = _append_warning(out.get("warning", pd.Series([""] * len(out))), ["nearest_reference_unavailable"])
        return out
    ref_X = np.asarray(ref_features, dtype=np.float32)
    nearest_idx, nearest_dist = _nearest_reference_indices(np.asarray(X, dtype=np.float32), ref_X)
    out = chunk.copy()
    nearest = ref_pairs.iloc[nearest_idx].reset_index(drop=True)
    out["nearest_reference_ligand"] = nearest.get("ligand_gene", pd.Series([""] * len(out))).astype(str).to_numpy()
    out["nearest_reference_receptor"] = nearest.get("receptor_gene", pd.Series([""] * len(out))).astype(str).to_numpy()
    out["nearest_reference_lr"] = out["nearest_reference_ligand"].astype(str) + "->" + out["nearest_reference_receptor"].astype(str)
    out["nearest_reference_species"] = nearest.get("species", pd.Series([""] * len(out))).astype(str).to_numpy()
    out["nearest_reference_resource"] = nearest.get("resource", pd.Series([""] * len(out))).astype(str).to_numpy()
    out["nearest_reference_pathway"] = nearest.get("pathway", pd.Series([""] * len(out))).astype(str).to_numpy()
    out["nearest_reference_distance"] = nearest_dist.astype(float)
    return out


def _nearest_reference_indices(X: np.ndarray, ref_X: np.ndarray, *, block_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    ref_norm = np.sum(ref_X * ref_X, axis=1)
    best_idx = np.zeros(X.shape[0], dtype=int)
    best_dist = np.full(X.shape[0], np.inf, dtype=np.float64)
    for start in range(0, X.shape[0], block_size):
        stop = min(start + block_size, X.shape[0])
        block = X[start:stop]
        dist = np.sum(block * block, axis=1)[:, None] + ref_norm[None, :] - 2.0 * block @ ref_X.T
        dist = np.maximum(dist, 0.0)
        idx = np.argmin(dist, axis=1)
        best_idx[start:stop] = idx
        best_dist[start:stop] = np.sqrt(dist[np.arange(stop - start), idx])
    return best_idx, best_dist


def _add_prediction_warnings(db, warnings_: Sequence[str]) -> None:
    db.interactions["warning"] = _append_warning(db.interactions.get("warning", pd.Series([""] * db.interactions.shape[0])), warnings_)
    summary = db.metadata.get("prediction_summary")
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        summary["warning"] = _append_warning(summary.get("warning", pd.Series([""] * len(summary))), warnings_)


def _append_warning(values: pd.Series, warnings_: Sequence[str]) -> pd.Series:
    clean = [str(item) for item in warnings_ if str(item)]
    if not clean:
        return values.fillna("").astype(str)

    def combine(value) -> str:
        parts = [part for part in str(value or "").split(";") if part]
        for warning in clean:
            if warning not in parts:
                parts.append(warning)
        return ";".join(parts)

    return values.fillna("").map(combine)


def _expressed_gene_fractions(adata, *, gene_id_key: str | None) -> pd.DataFrame:
    genes = pd.Index(adata.var_names.astype(str)) if gene_id_key is None else pd.Index(adata.var[gene_id_key].astype(str))
    x = adata.X
    if sparse.issparse(x):
        frac = np.asarray((x > 0).mean(axis=0)).ravel()
    else:
        frac = np.asarray(x > 0).mean(axis=0)
    return pd.DataFrame({"gene_id": genes.astype(str), "expression_fraction": frac.astype(float)})


def _candidate_row(lig, rec, *, strategy: str, embedding_cosine: float | None = None) -> dict[str, object]:
    row = {
        "ligand_gene": str(lig.gene_id),
        "receptor_gene": str(rec.gene_id),
        "ligand_role_score": float(lig.ligand_like_score),
        "receptor_role_score": float(rec.receptor_like_score),
        "ligand_secreted_like_score": float(getattr(lig, "secreted_like_score", lig.ligand_like_score)),
        "ligand_membrane_like_score": float(getattr(lig, "membrane_like_score", 0.0)),
        "receptor_secreted_like_score": float(getattr(rec, "secreted_like_score", 0.0)),
        "receptor_membrane_like_score": float(getattr(rec, "membrane_like_score", rec.receptor_like_score)),
        "ligand_expression_fraction": float(lig.expression_fraction),
        "receptor_expression_fraction": float(rec.expression_fraction),
        "candidate_strategy": strategy,
    }
    if embedding_cosine is not None:
        row["embedding_cosine"] = float(embedding_cosine)
    return row


def _embedding_nearest_neighbor_candidate_rows(
    ligands: pd.DataFrame,
    receptors: pd.DataFrame,
    embeddings: pd.DataFrame | None,
    *,
    top_pairs: int,
    neighbors_per_ligand: int,
    block_size: int = 256,
) -> list[dict[str, object]]:
    if embeddings is None or top_pairs <= 0:
        return []
    lookup = {
        str(row.gene_id): np.asarray(row.embedding, dtype=np.float32)
        for row in embeddings.itertuples(index=False)
    }
    ligands = ligands[ligands["gene_id"].astype(str).isin(lookup)].copy()
    receptors = receptors[receptors["gene_id"].astype(str).isin(lookup)].copy()
    if ligands.empty or receptors.empty:
        raise ValueError("Embedding nearest-neighbor candidate generation found no ligand/receptor embeddings.")
    receptor_vectors = np.vstack([lookup[str(gene)] for gene in receptors["gene_id"].astype(str)]).astype(np.float32)
    receptor_vectors = _l2_normalize(receptor_vectors)
    candidate_rows: list[tuple[float, dict[str, object]]] = []
    k = min(int(neighbors_per_ligand), len(receptors))
    for start in range(0, len(ligands), block_size):
        block = ligands.iloc[start : start + block_size]
        ligand_vectors = np.vstack([lookup[str(gene)] for gene in block["gene_id"].astype(str)]).astype(np.float32)
        ligand_vectors = _l2_normalize(ligand_vectors)
        cosine = ligand_vectors @ receptor_vectors.T
        top_idx = np.argpartition(-cosine, kth=np.arange(k), axis=1)[:, :k]
        for i, lig in enumerate(block.itertuples(index=False)):
            ordered = top_idx[i][np.argsort(-cosine[i, top_idx[i]])]
            for j in ordered:
                rec = receptors.iloc[int(j)]
                if str(lig.gene_id) == str(rec.gene_id):
                    continue
                row = _candidate_row(
                    lig,
                    rec,
                    strategy="embedding_nearest_neighbor",
                    embedding_cosine=float(cosine[i, j]),
                )
                candidate_rows.append((float(cosine[i, j]), row))
    candidate_rows.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in candidate_rows[:top_pairs]]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, denom, out=np.zeros_like(x, dtype=np.float32), where=denom > 0)


def _deduplicate_candidate_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    by_pair: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["ligand_gene"]), str(row["receptor_gene"]))
        if key not in by_pair:
            by_pair[key] = row.copy()
            continue
        existing = by_pair[key]
        strategies = [part for part in str(existing.get("candidate_strategy", "")).split(";") if part]
        strategy = str(row.get("candidate_strategy", ""))
        if strategy and strategy not in strategies:
            strategies.append(strategy)
            existing["candidate_strategy"] = ";".join(strategies)
        if "embedding_cosine" in row and pd.isna(existing.get("embedding_cosine", np.nan)):
            existing["embedding_cosine"] = row["embedding_cosine"]
    return pd.DataFrame(by_pair.values()).reset_index(drop=True)


def _candidate_list(value: str | Path | Sequence[str]) -> set[str]:
    if isinstance(value, (str, Path)) and Path(value).exists():
        return {line.strip() for line in Path(value).read_text(encoding="utf-8").splitlines() if line.strip()}
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def _load_model_card(model_card: dict[str, object] | str | Path) -> dict[str, object]:
    if isinstance(model_card, dict):
        return model_card
    path = Path(model_card)
    if path.is_dir():
        path = path / "model_card.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _split_metric_item(item: dict[str, object]) -> dict[str, object]:
    if "summary" in item and isinstance(item["summary"], dict):
        return item["summary"]
    return item


def _metric_top_k(metric: dict[str, object], top_k: str | None) -> float:
    if top_k is None:
        return np.nan
    values = metric.get("mean_top_k_precision", metric.get("top_k_precision", {}))
    if not isinstance(values, dict) or top_k not in values:
        return np.nan
    return float(values[top_k])


def _baseline_top_k(values: object, baseline: str, top_k: str | None) -> float:
    if top_k is None or not isinstance(values, dict):
        return np.nan
    item = values.get(baseline, {})
    if not isinstance(item, dict) or top_k not in item:
        return np.nan
    return float(item[top_k])


def _gate_row(
    split: str,
    baseline: str,
    model_pr_auc,
    baseline_pr_auc,
    delta_pr_auc,
    model_top_k_precision,
    baseline_top_k_precision,
    delta_top_k_precision,
    passed: bool,
    reason: str,
) -> dict[str, object]:
    return {
        "split": split,
        "baseline": baseline,
        "model_pr_auc": float(model_pr_auc) if pd.notna(model_pr_auc) else np.nan,
        "baseline_pr_auc": float(baseline_pr_auc) if pd.notna(baseline_pr_auc) else np.nan,
        "delta_pr_auc": float(delta_pr_auc) if pd.notna(delta_pr_auc) else np.nan,
        "model_top_k_precision": float(model_top_k_precision) if pd.notna(model_top_k_precision) else np.nan,
        "baseline_top_k_precision": float(baseline_top_k_precision) if pd.notna(baseline_top_k_precision) else np.nan,
        "delta_top_k_precision": float(delta_top_k_precision) if pd.notna(delta_top_k_precision) else np.nan,
        "passed": bool(passed),
        "reason": reason,
    }


def _training_pairs(
    interactions: pd.DataFrame,
    *,
    embeddings: pd.DataFrame | None = None,
    negative_ratio: int,
    negative_strategy: str,
    easy_negative_fraction: float,
    excluded_homology_radius: str,
    random_state: int,
) -> pd.DataFrame:
    if negative_ratio < 1:
        raise ValueError("`negative_ratio` must be at least 1.")
    if not 0.0 <= easy_negative_fraction < 1.0:
        raise ValueError("`easy_negative_fraction` must be in [0, 1).")
    if negative_strategy != "pu_degree_matched":
        raise ValueError("Only `negative_strategy='pu_degree_matched'` is supported.")
    if excluded_homology_radius not in {"family_pair", "none", ""}:
        raise ValueError("`excluded_homology_radius` must be one of: family_pair, none.")
    if "is_positive_label" in interactions.columns:
        positives = interactions[interactions["is_positive_label"].astype(bool)].copy()
    else:
        positives = interactions.copy()
    positives = _positive_pair_metadata(positives)
    positives["label"] = 1
    for col in ("negative_strategy", "negative_seed", "degree_matching", "excluded_homology_radius", "positive_resource_blacklist_for_fold", "easy_negative"):
        positives[col] = ""
    rng = np.random.default_rng(random_state)
    negatives = []
    positive_set_by_species = {
        species: set(zip(sub["ligand_gene"].astype(str), sub["receptor_gene"].astype(str)))
        for species, sub in positives.groupby("species", sort=False)
    }
    positive_family_pairs_by_species = {
        species: _positive_family_pairs(sub)
        for species, sub in positives.groupby("species", sort=False)
    }
    easy_pool = _easy_negative_gene_pool(positives, embeddings)
    for (species, resource), sub in positives.groupby(["species", "resource"], sort=False):
        species_sub = positives[positives["species"].astype(str) == str(species)]
        ligands, ligand_p = _degree_pool(species_sub["ligand_gene"])
        receptors, receptor_p = _degree_pool(species_sub["receptor_gene"])
        positive_set = positive_set_by_species[str(species)]
        positive_family_pairs = positive_family_pairs_by_species[str(species)]
        positive_resource_blacklist = _positive_resource_blacklist(species_sub)
        target = len(sub) * negative_ratio
        easy_target = (
            min(int(np.ceil(target * easy_negative_fraction)), max(target - 1, 0))
            if easy_pool.get(str(species))
            else 0
        )
        hard_target = target - easy_target
        species_negatives = []
        tries = 0
        while len(species_negatives) < hard_target and tries < hard_target * 40 + 100:
            tries += 1
            lig = str(rng.choice(ligands, p=ligand_p))
            rec = str(rng.choice(receptors, p=receptor_p))
            if lig == rec or (lig, rec) in positive_set:
                continue
            ligand_family = _family_for_gene(species_sub, "ligand", lig)
            receptor_family = _family_for_gene(species_sub, "receptor", rec)
            if _is_homolog_near_positive(
                ligand_family,
                receptor_family,
                positive_family_pairs,
                excluded_homology_radius=excluded_homology_radius,
            ):
                continue
            species_negatives.append(
                _negative_pair_row(
                    species=species,
                    resource=resource,
                    sub=sub,
                    ligand_gene=lig,
                    receptor_gene=rec,
                    ligand_family=ligand_family,
                    receptor_family=receptor_family,
                    ligand_role_score=_role_score_for_gene(species_sub, "ligand", lig),
                    receptor_role_score=_role_score_for_gene(species_sub, "receptor", rec),
                    ligand_expression_fraction=_expression_fraction_for_gene(species_sub, "ligand", lig),
                    receptor_expression_fraction=_expression_fraction_for_gene(species_sub, "receptor", rec),
                    negative_strategy=negative_strategy,
                    random_state=random_state,
                    excluded_homology_radius=excluded_homology_radius,
                    positive_resource_blacklist_for_fold=positive_resource_blacklist,
                    easy_negative=False,
                )
            )
        species_negatives.extend(
            _sample_easy_negative_rows(
                easy_pool,
                n=easy_target,
                rng=rng,
                species=species,
                resource=resource,
                sub=sub,
                positive_set=positive_set,
                positive_resource_blacklist_for_fold=positive_resource_blacklist,
                negative_strategy=negative_strategy,
                random_state=random_state,
                excluded_homology_radius=excluded_homology_radius,
            )
        )
        negatives.extend(species_negatives)
    pairs = pd.concat([positives, pd.DataFrame(negatives)], ignore_index=True)
    return pairs.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _positive_pair_metadata(interactions: pd.DataFrame) -> pd.DataFrame:
    out = interactions.copy()
    for col, default in {"species": "unknown_species", "resource": "unknown_resource"}.items():
        if col not in out.columns:
            out[col] = default
    if "clade" not in out.columns:
        out["clade"] = ""
    for col in ("ligand_role_score", "receptor_role_score", "ligand_expression_fraction", "receptor_expression_fraction"):
        if col not in out.columns:
            out[col] = ""
    for col in ("pathway", "annotation"):
        if col not in out.columns:
            out[col] = ""
    family_aliases = {
        "ligand_family": ("ligand_family", "ligand_protein_family", "ligand_homology_cluster", "homology_cluster"),
        "receptor_family": ("receptor_family", "receptor_protein_family", "receptor_homology_cluster", "homology_cluster"),
    }
    for canonical, aliases in family_aliases.items():
        if canonical not in out.columns:
            for alias in aliases:
                if alias in out.columns:
                    out[canonical] = out[alias]
                    break
        if canonical not in out.columns:
            out[canonical] = ""
    cols = [
        "species",
        "clade",
        "resource",
        "ligand_gene",
        "receptor_gene",
        "ligand_family",
        "receptor_family",
        "ligand_role_score",
        "receptor_role_score",
        "ligand_expression_fraction",
        "receptor_expression_fraction",
        "pathway",
        "annotation",
    ]
    out = out[cols].drop_duplicates().copy()
    for col in cols:
        out[col] = out[col].fillna("").astype(str)
    return out


def _positive_family_pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    pairs = set()
    for row in frame.itertuples(index=False):
        ligand_family = str(getattr(row, "ligand_family", "") or "")
        receptor_family = str(getattr(row, "receptor_family", "") or "")
        if ligand_family and receptor_family:
            pairs.add((ligand_family, receptor_family))
    return pairs


def _is_homolog_near_positive(
    ligand_family: str,
    receptor_family: str,
    positive_family_pairs: set[tuple[str, str]],
    *,
    excluded_homology_radius: str,
) -> bool:
    if excluded_homology_radius in {"", "none"}:
        return False
    if not ligand_family or not receptor_family:
        return False
    return (str(ligand_family), str(receptor_family)) in positive_family_pairs


def _negative_pair_row(
    *,
    species,
    resource,
    sub: pd.DataFrame,
    ligand_gene: str,
    receptor_gene: str,
    ligand_family: str,
    receptor_family: str,
    ligand_role_score,
    receptor_role_score,
    ligand_expression_fraction,
    receptor_expression_fraction,
    negative_strategy: str,
    random_state: int,
    excluded_homology_radius: str,
    positive_resource_blacklist_for_fold: str,
    easy_negative: bool,
) -> dict[str, object]:
    return {
        "species": str(species),
        "clade": str(sub["clade"].iloc[0]) if "clade" in sub.columns else "",
        "resource": str(resource),
        "ligand_gene": str(ligand_gene),
        "receptor_gene": str(receptor_gene),
        "ligand_family": str(ligand_family),
        "receptor_family": str(receptor_family),
        "ligand_role_score": ligand_role_score,
        "receptor_role_score": receptor_role_score,
        "ligand_expression_fraction": ligand_expression_fraction,
        "receptor_expression_fraction": receptor_expression_fraction,
        "pathway": "",
        "annotation": "",
        "label": 0,
        "negative_strategy": negative_strategy,
        "negative_seed": int(random_state),
        "degree_matching": not easy_negative,
        "excluded_homology_radius": excluded_homology_radius if excluded_homology_radius else "none",
        "positive_resource_blacklist_for_fold": str(positive_resource_blacklist_for_fold),
        "easy_negative": bool(easy_negative),
    }


def _easy_negative_gene_pool(positives: pd.DataFrame, embeddings: pd.DataFrame | None) -> dict[str, list[str]]:
    if embeddings is None or "species" not in embeddings.columns:
        return {}
    positive_genes = {
        str(species): set(sub["ligand_gene"].astype(str)) | set(sub["receptor_gene"].astype(str))
        for species, sub in positives.groupby("species", sort=False)
    }
    pool: dict[str, list[str]] = {}
    for species, sub in embeddings.groupby("species", sort=False):
        genes = sorted(set(sub["gene_id"].astype(str)) - positive_genes.get(str(species), set()))
        if genes:
            pool[str(species)] = genes
    return pool


def _sample_easy_negative_rows(
    easy_pool: dict[str, list[str]],
    *,
    n: int,
    rng: np.random.Generator,
    species,
    resource,
    sub: pd.DataFrame,
    positive_set: set[tuple[str, str]],
    positive_resource_blacklist_for_fold: str,
    negative_strategy: str,
    random_state: int,
    excluded_homology_radius: str,
) -> list[dict[str, object]]:
    genes = easy_pool.get(str(species), [])
    if n <= 0 or len(genes) < 2:
        return []
    rows = []
    tries = 0
    while len(rows) < n and tries < n * 20 + 100:
        tries += 1
        lig, rec = rng.choice(genes, size=2, replace=False)
        lig = str(lig)
        rec = str(rec)
        if (lig, rec) in positive_set:
            continue
        rows.append(
            _negative_pair_row(
                species=species,
                resource=resource,
                sub=sub,
                ligand_gene=lig,
                receptor_gene=rec,
                ligand_family="",
                receptor_family="",
                ligand_role_score=0.0,
                receptor_role_score=0.0,
                ligand_expression_fraction=0.0,
                receptor_expression_fraction=0.0,
                negative_strategy=negative_strategy,
                random_state=random_state,
                excluded_homology_radius=excluded_homology_radius,
                positive_resource_blacklist_for_fold=positive_resource_blacklist_for_fold,
                easy_negative=True,
            )
        )
    return rows


def _negative_sampling_summary(pairs: pd.DataFrame) -> dict[str, object]:
    negatives = pairs[pairs["label"].astype(int) == 0].copy()
    positives = pairs[pairs["label"].astype(int) == 1].copy()
    if negatives.empty:
        return {
            "n_positive": int(len(positives)),
            "n_negative": 0,
            "negative_by_species": {},
            "easy_negative_count": 0,
            "degree_matching": False,
            "excluded_homology_radius": "",
            "positive_resource_blacklist_for_fold": "",
        }
    return {
        "n_positive": int(len(positives)),
        "n_negative": int(len(negatives)),
        "negative_by_species": {str(k): int(v) for k, v in negatives["species"].astype(str).value_counts().sort_index().items()},
        "positive_by_species": {str(k): int(v) for k, v in positives["species"].astype(str).value_counts().sort_index().items()},
        "easy_negative_count": int(pd.Series(negatives.get("easy_negative", False)).astype(bool).sum()),
        "degree_matching": bool(pd.Series(negatives.get("degree_matching", False)).astype(bool).any()),
        "excluded_homology_radius": ";".join(sorted(set(negatives.get("excluded_homology_radius", pd.Series(dtype=str)).astype(str)))),
        "positive_resource_blacklist_for_fold": _join_semicolon_unique(negatives.get("positive_resource_blacklist_for_fold", pd.Series(dtype=str))),
        "negative_strategy": ";".join(sorted(set(negatives.get("negative_strategy", pd.Series(dtype=str)).astype(str)))),
    }


def _training_metadata(interactions: pd.DataFrame, embeddings: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, object]:
    positives = pairs[pairs["label"].astype(int) == 1].copy()
    negatives = pairs[pairs["label"].astype(int) == 0].copy()
    return {
        "training_resources": _sorted_strings(interactions.get("resource", pd.Series(dtype=str))),
        "species_included": _sorted_strings(interactions.get("species", pd.Series(dtype=str))),
        "clades_included": _sorted_strings(interactions.get("clade", pd.Series(dtype=str))),
        "positive_labels_by_species": _count_by(positives, "species"),
        "pseudo_negatives_by_species": _count_by(negatives, "species"),
        "embedding_model": {
            "model_name": _sorted_strings(embeddings.get("model_name", pd.Series(dtype=str))),
            "model_revision": _sorted_strings(embeddings.get("model_revision", pd.Series(dtype=str))),
            "embedding_backend": _sorted_strings(embeddings.get("embedding_backend", pd.Series(dtype=str))),
            "pooling": _sorted_strings(embeddings.get("pooling", pd.Series(dtype=str))),
            "n_embeddings": int(len(embeddings)),
        },
    }


def _sorted_strings(values: pd.Series) -> list[str]:
    if values.empty:
        return []
    parts = []
    for value in values.dropna().astype(str):
        parts.extend(_split_semicolon_parts(value))
    return sorted(set(parts))


def _count_by(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[column].astype(str).value_counts().sort_index().items()}


def _degree_pool(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    counts = values.astype(str).value_counts()
    genes = counts.index.to_numpy(dtype=str)
    weights = counts.to_numpy(dtype=float)
    weights = weights / weights.sum()
    return genes, weights


def _family_for_gene(frame: pd.DataFrame, side: str, gene: str) -> str:
    gene_col = f"{side}_gene"
    family_col = f"{side}_family"
    if family_col not in frame.columns:
        return ""
    sub = frame[frame[gene_col].astype(str) == str(gene)]
    if sub.empty:
        return ""
    values = [str(value) for value in sub[family_col].astype(str) if str(value)]
    return values[0] if values else ""


def _role_score_for_gene(frame: pd.DataFrame, side: str, gene: str) -> str:
    gene_col = f"{side}_gene"
    score_col = f"{side}_role_score"
    if score_col not in frame.columns:
        return ""
    sub = frame[frame[gene_col].astype(str) == str(gene)]
    if sub.empty:
        return ""
    values = pd.to_numeric(sub[score_col], errors="coerce").dropna()
    return "" if values.empty else str(float(values.iloc[0]))


def _expression_fraction_for_gene(frame: pd.DataFrame, side: str, gene: str) -> str:
    gene_col = f"{side}_gene"
    expr_col = f"{side}_expression_fraction"
    if expr_col not in frame.columns:
        return ""
    sub = frame[frame[gene_col].astype(str) == str(gene)]
    if sub.empty:
        return ""
    values = pd.to_numeric(sub[expr_col], errors="coerce").dropna()
    return "" if values.empty else str(float(values.iloc[0]))


def _positive_resource_blacklist(frame: pd.DataFrame) -> str:
    if "resource" not in frame.columns:
        return ""
    return _join_semicolon_unique(frame["resource"])


def _join_semicolon_unique(values: pd.Series) -> str:
    parts = []
    for value in values.dropna().astype(str):
        parts.extend(_split_semicolon_parts(value))
    return ";".join(sorted(set(parts)))


def _split_semicolon_parts(value: object) -> list[str]:
    return [part.strip() for part in str(value).replace("|", ";").split(";") if part.strip()]


def _random_train_test_indices(y: np.ndarray, *, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    n_classes = len(np.unique(y))
    test_size = max(n_classes, int(np.ceil(len(y) * 0.25)))
    if len(y) - test_size < n_classes:
        test_size = n_classes
    return train_test_split(idx, test_size=test_size, random_state=random_state, stratify=y)


def _validation_report(
    pairs: pd.DataFrame,
    embeddings: pd.DataFrame,
    y: np.ndarray,
    *,
    requested_splits: Sequence[str],
    model: str,
    feature_encoder: str,
    random_state: int,
) -> dict[str, object]:
    report: dict[str, object] = {}
    train_idx, test_idx = _random_train_test_indices(y, random_state=random_state)
    report["random_stratified"] = _evaluate_pair_split(pairs, embeddings, y, train_idx, test_idx, model=model, feature_encoder=feature_encoder, random_state=random_state)
    for split in requested_splits:
        if split == "leave_species_out":
            report[split] = _leave_one_group_report(pairs, embeddings, y, group_col="species", model=model, feature_encoder=feature_encoder, random_state=random_state)
        elif split == "leave_resource_out":
            report[split] = _leave_one_group_report(pairs, embeddings, y, group_col="resource", model=model, feature_encoder=feature_encoder, random_state=random_state)
        elif split == "leave_family_out":
            report[split] = _leave_family_report(pairs, embeddings, y, model=model, feature_encoder=feature_encoder, random_state=random_state)
        elif split == "leave_clade_out":
            if "clade" in pairs.columns:
                report[split] = _leave_one_group_report(pairs, embeddings, y, group_col="clade", model=model, feature_encoder=feature_encoder, random_state=random_state)
            else:
                report[split] = {"status": "skipped", "reason": "Pair table has no `clade` column."}
        else:
            report[split] = {"status": "skipped", "reason": f"Unknown validation split `{split}`."}
    return report


def _negative_repeat_validation_report(
    interactions: pd.DataFrame,
    embeddings: pd.DataFrame,
    *,
    requested_splits: Sequence[str],
    model: str,
    feature_encoder: str,
    negative_ratio: int,
    negative_strategy: str,
    easy_negative_fraction: float,
    excluded_homology_radius: str,
    n_repeats: int,
    random_state: int,
) -> dict[str, object]:
    if n_repeats <= 1:
        return {"status": "skipped", "reason": "negative_repeats <= 1", "n_repeats": int(n_repeats)}
    rows = []
    seeds = [int(random_state + i) for i in range(n_repeats)]
    for repeat, seed in enumerate(seeds):
        try:
            pairs = _training_pairs(
                interactions,
                embeddings=embeddings,
                negative_ratio=negative_ratio,
                negative_strategy=negative_strategy,
                easy_negative_fraction=easy_negative_fraction,
                excluded_homology_radius=excluded_homology_radius,
                random_state=seed,
            )
            y = pairs["label"].astype(int).to_numpy()
            if len(np.unique(y)) < 2:
                raise ValueError("Sampled training pairs have fewer than two classes.")
            report = _validation_report(
                pairs,
                embeddings,
                y,
                requested_splits=requested_splits,
                model=model,
                feature_encoder=feature_encoder,
                random_state=seed,
            )
        except Exception as exc:  # pragma: no cover - defensive metadata path
            rows.append(
                {
                    "repeat": int(repeat),
                    "negative_seed": int(seed),
                    "split": "all",
                    "status": "skipped",
                    "reason": str(exc),
                }
            )
            continue
        rows.extend(_repeat_metric_rows(report, repeat=repeat, seed=seed))
    summary = _summarize_repeat_metric_rows(rows)
    return {
        "status": "ok" if any(item.get("n_usable_repeats", 0) for item in summary.values()) else "skipped",
        "n_repeats": int(n_repeats),
        "negative_seeds": seeds,
        "per_repeat": rows,
        "summary": summary,
    }


def _repeat_metric_rows(report: dict[str, object], *, repeat: int, seed: int) -> list[dict[str, object]]:
    rows = []
    for split, item in report.items():
        if not isinstance(item, dict):
            continue
        row: dict[str, object] = {
            "repeat": int(repeat),
            "negative_seed": int(seed),
            "split": str(split),
            "status": str(item.get("status", "skipped")),
        }
        metric = _split_metric_item(item)
        if row["status"] == "ok" and isinstance(metric, dict):
            row["pr_auc"] = _coerce_float(metric.get("mean_pr_auc", metric.get("pr_auc")))
            row["roc_auc"] = _coerce_float(metric.get("mean_roc_auc", metric.get("roc_auc")))
            top_k = metric.get("mean_top_k_precision", metric.get("top_k_precision", {}))
            if isinstance(top_k, dict):
                for key, value in top_k.items():
                    row[f"precision_{key}"] = _coerce_float(value)
            top_k_recall = metric.get("mean_top_k_recall", metric.get("top_k_recall", {}))
            if isinstance(top_k_recall, dict):
                for key, value in top_k_recall.items():
                    row[f"recall_{key}"] = _coerce_float(value)
            top_k_enrichment = metric.get("mean_top_k_enrichment", metric.get("top_k_enrichment", {}))
            if isinstance(top_k_enrichment, dict):
                for key, value in top_k_enrichment.items():
                    row[f"enrichment_{key}"] = _coerce_float(value)
        else:
            row["reason"] = str(item.get("reason", "no usable folds"))
        rows.append(row)
    return rows


def _summarize_repeat_metric_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    summary: dict[str, object] = {}
    for split, sub in frame.groupby("split", sort=False):
        usable = sub[sub["status"].astype(str) == "ok"].copy()
        item: dict[str, object] = {
            "n_repeats": int(len(sub)),
            "n_usable_repeats": int(len(usable)),
        }
        skipped = sub[sub["status"].astype(str) != "ok"]
        if not skipped.empty and "reason" in skipped.columns:
            item["skipped_reasons"] = sorted({str(reason) for reason in skipped["reason"].dropna().astype(str)})
        if not usable.empty:
            item.update(
                {
                    "pr_auc_mean": _mean_float(usable["pr_auc"]),
                    "pr_auc_std": _std_float(usable["pr_auc"]),
                    "pr_auc_variance": _var_float(usable["pr_auc"]),
                    "roc_auc_mean": _mean_float(usable["roc_auc"]),
                    "roc_auc_std": _std_float(usable["roc_auc"]),
                    "roc_auc_variance": _var_float(usable["roc_auc"]),
                }
            )
            precision_cols = [col for col in usable.columns if col.startswith("precision_top_")]
            if precision_cols:
                item["top_k_precision_mean"] = {
                    col.removeprefix("precision_"): _mean_float(usable[col])
                    for col in precision_cols
                }
                item["top_k_precision_std"] = {
                    col.removeprefix("precision_"): _std_float(usable[col])
                    for col in precision_cols
                }
            recall_cols = [col for col in usable.columns if col.startswith("recall_top_")]
            if recall_cols:
                item["top_k_recall_mean"] = {
                    col.removeprefix("recall_"): _mean_float(usable[col])
                    for col in recall_cols
                }
                item["top_k_recall_std"] = {
                    col.removeprefix("recall_"): _std_float(usable[col])
                    for col in recall_cols
                }
            enrichment_cols = [col for col in usable.columns if col.startswith("enrichment_top_")]
            if enrichment_cols:
                item["top_k_enrichment_mean"] = {
                    col.removeprefix("enrichment_"): _mean_float(usable[col])
                    for col in enrichment_cols
                }
                item["top_k_enrichment_std"] = {
                    col.removeprefix("enrichment_"): _std_float(usable[col])
                    for col in enrichment_cols
                }
        summary[str(split)] = item
    return summary


def _coerce_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _mean_float(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def _std_float(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _var_float(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.var(ddof=1)) if len(values) > 1 else 0.0


def _leave_one_group_report(
    pairs: pd.DataFrame,
    embeddings: pd.DataFrame,
    y: np.ndarray,
    *,
    group_col: str,
    model: str,
    feature_encoder: str,
    random_state: int,
) -> dict[str, object]:
    if group_col not in pairs.columns:
        return {"status": "skipped", "reason": f"Pair table has no `{group_col}` column."}
    folds = []
    groups = _membership_groups(pairs[group_col]) if group_col == "resource" else sorted(set(pairs[group_col].astype(str)))
    values = pairs[group_col].astype(str)
    for group in groups:
        if group_col == "resource":
            membership = values.map(lambda value: group in _split_semicolon_parts(value)).to_numpy(dtype=bool)
        else:
            membership = values.to_numpy() == str(group)
        test_idx = np.flatnonzero(membership)
        train_idx = np.flatnonzero(~membership)
        fold = _evaluate_pair_split(pairs, embeddings, y, train_idx, test_idx, model=model, feature_encoder=feature_encoder, random_state=random_state)
        fold["held_out"] = str(group)
        if group_col == "resource":
            fold["positive_resource_blacklist_for_fold"] = str(group)
        folds.append(fold)
    usable = [fold for fold in folds if fold["status"] == "ok"]
    return {"status": "ok" if usable else "skipped", "folds": folds, "summary": _fold_summary(usable)}


def _membership_groups(values: pd.Series) -> list[str]:
    groups = []
    for value in values.dropna().astype(str):
        groups.extend(_split_semicolon_parts(value))
    return sorted(set(groups))


def _leave_family_report(
    pairs: pd.DataFrame,
    embeddings: pd.DataFrame,
    y: np.ndarray,
    *,
    model: str,
    feature_encoder: str,
    random_state: int,
) -> dict[str, object]:
    if "ligand_family" not in pairs.columns or "receptor_family" not in pairs.columns:
        return {"status": "skipped", "reason": "Pair table has no ligand/receptor family columns."}
    ligand_family = pairs["ligand_family"].fillna("").astype(str)
    receptor_family = pairs["receptor_family"].fillna("").astype(str)
    family = pd.Series(
        [sorted({f"ligand:{lig}", f"receptor:{rec}"} - {"ligand:", "receptor:"}) for lig, rec in zip(ligand_family, receptor_family, strict=True)],
        index=pairs.index,
    )
    valid = family.map(bool)
    if not valid.any():
        return {"status": "skipped", "reason": "No non-empty family labels are available."}
    folds = []
    groups = sorted({item for values in family[valid] for item in values})
    for group in groups:
        mask = family.map(lambda values: group in values).to_numpy(dtype=bool)
        test_idx = np.flatnonzero(mask)
        train_idx = np.flatnonzero(~mask)
        fold = _evaluate_pair_split(pairs, embeddings, y, train_idx, test_idx, model=model, feature_encoder=feature_encoder, random_state=random_state)
        fold["held_out"] = str(group)
        folds.append(fold)
    usable = [fold for fold in folds if fold["status"] == "ok"]
    return {"status": "ok" if usable else "skipped", "folds": folds, "summary": _fold_summary(usable)}


def _evaluate_pair_split(
    pairs: pd.DataFrame,
    embeddings: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    model: str,
    feature_encoder: str,
    random_state: int,
) -> dict[str, object]:
    if len(train_idx) == 0 or len(test_idx) == 0:
        return {"status": "skipped", "reason": "Empty train or test split.", "n_train": int(len(train_idx)), "n_test": int(len(test_idx))}
    y_train = y[train_idx]
    y_test = y[test_idx]
    if len(np.unique(y_train)) < 2:
        return {"status": "skipped", "reason": "Training split has fewer than two classes.", "n_train": int(len(train_idx)), "n_test": int(len(test_idx))}
    if len(np.unique(y_test)) < 2:
        return {"status": "skipped", "reason": "Test split has fewer than two classes.", "n_train": int(len(train_idx)), "n_test": int(len(test_idx))}
    train_features = make_lr_pair_features(pairs.iloc[train_idx], embeddings, encoder=feature_encoder, fit_pca=True)
    test_features = make_lr_pair_features(pairs.iloc[test_idx], embeddings, encoder=feature_encoder, pca_model=train_features.pca_model)
    clf = _fit_pair_model(model, train_features.X, y_train, random_state=random_state)
    scores = _predict_scores(clf, test_features.X)
    metrics = _classification_metrics(y_test, scores)
    baselines = _baseline_metrics(pairs.iloc[train_idx], pairs.iloc[test_idx], test_features, y_test, random_state=random_state)
    return {
        "status": "ok",
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_positive": int(y_train.sum()),
        "n_test_positive": int(y_test.sum()),
        "pr_auc": metrics["pr_auc"],
        "roc_auc": metrics["roc_auc"],
        "top_k_precision": metrics["top_k_precision"],
        "top_k_recall": metrics["top_k_recall"],
        "top_k_enrichment": metrics["top_k_enrichment"],
        "baseline_pr_auc": {name: value["pr_auc"] for name, value in baselines.items()},
        "baseline_top_k_precision": {name: value["top_k_precision"] for name, value in baselines.items()},
        "baseline_top_k_recall": {name: value["top_k_recall"] for name, value in baselines.items()},
        "baseline_top_k_enrichment": {name: value["top_k_enrichment"] for name, value in baselines.items()},
        "baseline_delta_pr_auc": {name: metrics["pr_auc"] - value["pr_auc"] for name, value in baselines.items()},
        "baseline_delta_top_k_precision": {
            name: {
                k: metrics["top_k_precision"][k] - baseline_metrics["top_k_precision"][k]
                for k in metrics["top_k_precision"]
            }
            for name, baseline_metrics in baselines.items()
        },
        "baseline_delta_top_k_recall": {
            name: {
                k: metrics["top_k_recall"][k] - baseline_metrics["top_k_recall"][k]
                for k in metrics["top_k_recall"]
            }
            for name, baseline_metrics in baselines.items()
        },
        "baseline_delta_top_k_enrichment": {
            name: {
                k: metrics["top_k_enrichment"][k] - baseline_metrics["top_k_enrichment"][k]
                for k in metrics["top_k_enrichment"]
            }
            for name, baseline_metrics in baselines.items()
        },
        "family_failure_cases": _family_failure_cases(pairs.iloc[test_idx], y_test, scores),
        "feature_encoder_fit": "train_split_only",
    }


def _classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    ranking = _top_k_ranking_metrics(y_true, scores, ks=(100, 500, 1000, 5000))
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": _safe_roc_auc(y_true, scores, roc_auc_score),
        **ranking,
    }


def _baseline_metrics(
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    test_features,
    y_test: np.ndarray,
    *,
    random_state: int,
) -> dict[str, dict[str, object]]:
    scores = {
        "degree_prior": _degree_prior_scores(train_pairs, test_pairs),
        "embedding_cosine": _feature_scores(test_features, "cosine"),
        "family_pair_transfer": _family_pair_transfer_scores(train_pairs, test_pairs),
        "expression_only": _expression_only_scores(test_pairs),
        "role_only": _role_only_scores(test_pairs),
        "density_matched_random": _density_matched_random_scores(train_pairs, test_pairs, random_state=random_state),
        "random": np.random.default_rng(random_state).random(len(test_pairs)),
    }
    return {name: _classification_metrics(y_test, values) for name, values in scores.items()}


def _degree_prior_scores(train_pairs: pd.DataFrame, test_pairs: pd.DataFrame) -> np.ndarray:
    positives = train_pairs[train_pairs["label"].astype(int) == 1]
    ligand_counts = positives["ligand_gene"].astype(str).value_counts()
    receptor_counts = positives["receptor_gene"].astype(str).value_counts()
    scores = []
    for row in test_pairs.itertuples(index=False):
        scores.append(float(ligand_counts.get(str(row.ligand_gene), 0) + 1) * float(receptor_counts.get(str(row.receptor_gene), 0) + 1))
    values = np.asarray(scores, dtype=float)
    return values / values.max() if values.size and values.max() > 0 else values


def _feature_scores(features, feature_name: str) -> np.ndarray:
    if feature_name not in features.feature_names:
        return np.zeros(len(features.pairs), dtype=float)
    values = features.X[:, features.feature_names.index(feature_name)].astype(float)
    if feature_name == "cosine":
        values = (values + 1.0) / 2.0
    return values


def _family_pair_transfer_scores(train_pairs: pd.DataFrame, test_pairs: pd.DataFrame) -> np.ndarray:
    required = {"ligand_family", "receptor_family", "label"}
    if not required.issubset(train_pairs.columns) or not {"ligand_family", "receptor_family"}.issubset(test_pairs.columns):
        return np.zeros(len(test_pairs), dtype=float)
    positives = train_pairs[train_pairs["label"].astype(int) == 1]
    positive_family_pairs = {
        (str(row.ligand_family), str(row.receptor_family))
        for row in positives.itertuples(index=False)
        if str(row.ligand_family) and str(row.receptor_family)
    }
    if not positive_family_pairs:
        return np.zeros(len(test_pairs), dtype=float)
    return np.asarray(
        [
            1.0 if (str(row.ligand_family), str(row.receptor_family)) in positive_family_pairs else 0.0
            for row in test_pairs.itertuples(index=False)
        ],
        dtype=float,
    )


def _role_only_scores(test_pairs: pd.DataFrame) -> np.ndarray:
    ligand = pd.to_numeric(test_pairs.get("ligand_role_score", pd.Series([0.5] * len(test_pairs))), errors="coerce").fillna(0.5).to_numpy(dtype=float)
    receptor = pd.to_numeric(test_pairs.get("receptor_role_score", pd.Series([0.5] * len(test_pairs))), errors="coerce").fillna(0.5).to_numpy(dtype=float)
    return (ligand + receptor) / 2.0


def _expression_only_scores(test_pairs: pd.DataFrame) -> np.ndarray:
    ligand = pd.to_numeric(test_pairs.get("ligand_expression_fraction", pd.Series([0.5] * len(test_pairs))), errors="coerce").fillna(0.5).to_numpy(dtype=float)
    receptor = pd.to_numeric(test_pairs.get("receptor_expression_fraction", pd.Series([0.5] * len(test_pairs))), errors="coerce").fillna(0.5).to_numpy(dtype=float)
    values = ligand * receptor
    return np.clip(values, 0.0, 1.0)


def _density_matched_random_scores(train_pairs: pd.DataFrame, test_pairs: pd.DataFrame, *, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = len(test_pairs)
    if n == 0:
        return np.asarray([], dtype=float)
    labels = pd.to_numeric(train_pairs.get("label", pd.Series(dtype=float)), errors="coerce").dropna()
    density = float(labels.mean()) if not labels.empty else 0.5
    n_high = int(np.clip(round(density * n), 1, n))
    scores = rng.random(n) * 0.1
    high = rng.choice(n, size=n_high, replace=False)
    scores[high] = 0.9 + rng.random(n_high) * 0.1
    return scores


def _family_failure_cases(
    test_pairs: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    max_cases: int = 10,
) -> list[dict[str, object]]:
    required = {"ligand_family", "receptor_family"}
    if not required.issubset(test_pairs.columns):
        return []
    frame = pd.DataFrame(
        {
            "ligand_family": test_pairs["ligand_family"].fillna("").astype(str).to_numpy(),
            "receptor_family": test_pairs["receptor_family"].fillna("").astype(str).to_numpy(),
            "label": np.asarray(y_true, dtype=int),
            "score": np.asarray(scores, dtype=float),
        }
    )
    frame = frame[(frame["ligand_family"] != "") | (frame["receptor_family"] != "")].copy()
    if frame.empty:
        return []
    rows = []
    for (ligand_family, receptor_family), sub in frame.groupby(["ligand_family", "receptor_family"], sort=False):
        positives = sub[sub["label"] == 1]
        pseudo_negatives = sub[sub["label"] == 0]
        max_negative = _maybe_float(pseudo_negatives["score"].max()) if not pseudo_negatives.empty else None
        min_positive = _maybe_float(positives["score"].min()) if not positives.empty else None
        failure_score = max(
            max_negative if max_negative is not None else 0.0,
            1.0 - min_positive if min_positive is not None else 0.0,
        )
        rows.append(
            {
                "ligand_family": str(ligand_family),
                "receptor_family": str(receptor_family),
                "n_pairs": int(len(sub)),
                "n_positive": int((sub["label"] == 1).sum()),
                "n_pseudo_negative": int((sub["label"] == 0).sum()),
                "mean_score": float(sub["score"].mean()),
                "max_pseudo_negative_score": max_negative,
                "min_positive_score": min_positive,
                "high_score_pseudo_negative_count": int((pseudo_negatives["score"] >= 0.5).sum()) if not pseudo_negatives.empty else 0,
                "low_score_positive_count": int((positives["score"] < 0.5).sum()) if not positives.empty else 0,
                "failure_score": float(failure_score),
            }
        )
    rows.sort(key=lambda item: (float(item["failure_score"]), int(item["n_pairs"])), reverse=True)
    return rows[: max(int(max_cases), 0)]


def _maybe_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _fit_calibrator_from_split(
    pairs: pd.DataFrame,
    embeddings: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    calib_idx: np.ndarray,
    *,
    model: str,
    feature_encoder: str,
    method: str | None,
    random_state: int,
) -> dict[str, object]:
    if method is None or method == "none":
        return {"method": "none", "calibrator": None, "metrics": {}}
    y_train = y[train_idx]
    y_calib = y[calib_idx]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_calib)) < 2:
        return {"method": "skipped", "calibrator": None, "metrics": {"reason": "Calibration split has fewer than two classes."}}
    train_features = make_lr_pair_features(pairs.iloc[train_idx], embeddings, encoder=feature_encoder, fit_pca=True)
    calib_features = make_lr_pair_features(pairs.iloc[calib_idx], embeddings, encoder=feature_encoder, pca_model=train_features.pca_model)
    clf = _fit_pair_model(model, train_features.X, y_train, random_state=random_state)
    raw = _predict_scores(clf, calib_features.X)
    calibrator = _fit_calibrator(raw, y_calib, method=method, random_state=random_state)
    calibrated = _apply_calibrator(calibrator, raw)
    metrics = {
        "brier_score": _brier_score(y_calib, calibrated),
        "ece_10bin": _expected_calibration_error(y_calib, calibrated, n_bins=10),
        "n_calibration": int(len(y_calib)),
        "n_calibration_positive": int(y_calib.sum()),
    }
    return {"method": method, "calibrator": calibrator, "metrics": metrics}


def _fit_calibrator(raw_scores: np.ndarray, y: np.ndarray, *, method: str, random_state: int):
    raw_scores = np.asarray(raw_scores, dtype=float)
    y = np.asarray(y, dtype=int)
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_scores, y)
        return calibrator
    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        calibrator = LogisticRegression(random_state=random_state)
        calibrator.fit(raw_scores.reshape(-1, 1), y)
        return calibrator
    raise ValueError("`calibration_method` must be one of: isotonic, sigmoid, none.")


def _apply_calibrator(calibrator, raw_scores: np.ndarray) -> np.ndarray:
    raw_scores = np.asarray(raw_scores, dtype=float)
    if calibrator is None:
        return raw_scores
    if hasattr(calibrator, "predict_proba"):
        return np.asarray(calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1], dtype=float)
    return np.asarray(calibrator.predict(raw_scores), dtype=float)


def _brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    return float(np.mean((probabilities - y_true) ** 2))


def _expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, *, n_bins: int) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (probabilities >= lo) & (probabilities < hi if hi < 1.0 else probabilities <= hi)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def _top_k_ranking_metrics(y_true: np.ndarray, scores: np.ndarray, *, ks: Sequence[int]) -> dict[str, dict[str, float]]:
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    n_positive = float(y_true.sum())
    prevalence = float(y_true.mean()) if len(y_true) else float("nan")
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    enrichment: dict[str, float] = {}
    for k in ks:
        key = f"top_{k}"
        kk = min(int(k), len(order))
        if kk == 0:
            precision[key] = float("nan")
            recall[key] = float("nan")
            enrichment[key] = float("nan")
            continue
        selected = y_true[order[:kk]]
        precision[key] = float(selected.mean())
        recall[key] = float(selected.sum() / n_positive) if n_positive > 0 else float("nan")
        enrichment[key] = float(precision[key] / prevalence) if prevalence > 0 else float("nan")
    return {
        "top_k_precision": precision,
        "top_k_recall": recall,
        "top_k_enrichment": enrichment,
    }


def _fold_summary(folds: Sequence[dict[str, object]]) -> dict[str, object]:
    if not folds:
        return {"n_usable_folds": 0}
    baseline_names = sorted({name for fold in folds for name in fold.get("baseline_pr_auc", {})})
    top_k_names = sorted({name for fold in folds for name in fold.get("top_k_precision", {})})
    baseline_mean = {
        name: float(np.mean([float(fold["baseline_pr_auc"][name]) for fold in folds if name in fold.get("baseline_pr_auc", {})]))
        for name in baseline_names
    }
    delta_mean = {
        name: float(np.mean([float(fold["baseline_delta_pr_auc"][name]) for fold in folds if name in fold.get("baseline_delta_pr_auc", {})]))
        for name in baseline_names
    }
    top_k_mean = _mean_top_k_metric(folds, "top_k_precision", top_k_names)
    top_k_recall_mean = _mean_top_k_metric(folds, "top_k_recall", top_k_names)
    top_k_enrichment_mean = _mean_top_k_metric(folds, "top_k_enrichment", top_k_names)
    baseline_top_k_mean = _mean_baseline_top_k_metric(folds, "baseline_top_k_precision", baseline_names, top_k_names)
    baseline_top_k_delta_mean = _mean_baseline_top_k_metric(folds, "baseline_delta_top_k_precision", baseline_names, top_k_names)
    baseline_top_k_recall_mean = _mean_baseline_top_k_metric(folds, "baseline_top_k_recall", baseline_names, top_k_names)
    baseline_top_k_recall_delta_mean = _mean_baseline_top_k_metric(folds, "baseline_delta_top_k_recall", baseline_names, top_k_names)
    baseline_top_k_enrichment_mean = _mean_baseline_top_k_metric(folds, "baseline_top_k_enrichment", baseline_names, top_k_names)
    baseline_top_k_enrichment_delta_mean = _mean_baseline_top_k_metric(folds, "baseline_delta_top_k_enrichment", baseline_names, top_k_names)
    out = {
        "n_usable_folds": len(folds),
        "mean_pr_auc": float(np.mean([float(fold["pr_auc"]) for fold in folds])),
        "mean_roc_auc": float(np.mean([float(fold["roc_auc"]) for fold in folds])),
    }
    if top_k_mean:
        out["mean_top_k_precision"] = top_k_mean
        out["mean_top_k_recall"] = top_k_recall_mean
        out["mean_top_k_enrichment"] = top_k_enrichment_mean
    if baseline_mean:
        out["mean_baseline_pr_auc"] = baseline_mean
        out["mean_baseline_delta_pr_auc"] = delta_mean
        out["mean_baseline_top_k_precision"] = baseline_top_k_mean
        out["mean_baseline_delta_top_k_precision"] = baseline_top_k_delta_mean
        out["mean_baseline_top_k_recall"] = baseline_top_k_recall_mean
        out["mean_baseline_delta_top_k_recall"] = baseline_top_k_recall_delta_mean
        out["mean_baseline_top_k_enrichment"] = baseline_top_k_enrichment_mean
        out["mean_baseline_delta_top_k_enrichment"] = baseline_top_k_enrichment_delta_mean
    return out


def _mean_top_k_metric(folds: Sequence[dict[str, object]], metric: str, top_k_names: Sequence[str]) -> dict[str, float]:
    out = {}
    for key in top_k_names:
        values = [
            float(fold[metric][key])
            for fold in folds
            if isinstance(fold.get(metric), dict) and key in fold[metric]
        ]
        if values:
            out[key] = float(np.mean(values))
    return out


def _mean_baseline_top_k_metric(
    folds: Sequence[dict[str, object]],
    metric: str,
    baseline_names: Sequence[str],
    top_k_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    out = {}
    for baseline in baseline_names:
        values_by_k = {}
        for key in top_k_names:
            values = [
                float(fold[metric][baseline][key])
                for fold in folds
                if isinstance(fold.get(metric), dict)
                and baseline in fold[metric]
                and key in fold[metric][baseline]
            ]
            if values:
                values_by_k[key] = float(np.mean(values))
        if values_by_k:
            out[baseline] = values_by_k
    return out


def _fit_pair_model(model: str, X: np.ndarray, y: np.ndarray, *, random_state: int):
    params = _pair_model_params(model, random_state=random_state)
    if model == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install LR prediction support with `pyccc[predict]` to train LightGBM models.") from exc
        clf = lgb.LGBMClassifier(**params)
    elif model == "sklearn":
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(**params)
    else:
        raise ValueError("`model` must be one of: lightgbm, sklearn.")
    clf.fit(X, y)
    return clf


def _pair_model_params(model: str, *, random_state: int) -> dict[str, object]:
    if model == "lightgbm":
        return {
            "objective": "binary",
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": int(random_state),
            "verbose": -1,
        }
    if model == "sklearn":
        return {"random_state": int(random_state), "max_iter": 100}
    raise ValueError("`model` must be one of: lightgbm, sklearn.")


def _predict_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        return np.asarray(model.predict(X), dtype=float)


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray, scorer) -> float:
    try:
        return float(scorer(y_true, scores))
    except ValueError:
        return float("nan")


def _heuristic_pair_scores(pairs: pd.DataFrame, embeddings: pd.DataFrame) -> np.ndarray:
    features = make_lr_pair_features(pairs, embeddings)
    cosine_idx = features.feature_names.index("cosine")
    cosine = features.X[:, cosine_idx]
    ligand_role = pd.to_numeric(pairs.get("ligand_role_score", pd.Series([0.5] * len(pairs))), errors="coerce").fillna(0.5).to_numpy()
    receptor_role = pd.to_numeric(pairs.get("receptor_role_score", pd.Series([0.5] * len(pairs))), errors="coerce").fillna(0.5).to_numpy()
    score = 0.35 * ((cosine + 1.0) / 2.0) + 0.325 * ligand_role + 0.325 * receptor_role
    return np.clip(score, 0.0, 1.0)


def _model_card_markdown(card: dict[str, object]) -> str:
    lines = [
        "# pyccc LR Link Predictor Model Card",
        "",
        f"- Model type: `{card['model_type']}`",
        f"- Feature encoder: `{card['feature_encoder']}`",
        f"- Final model training: `{card['final_model_training']}`",
        f"- Validation feature encoder fit: `{card['validation_feature_encoder_fit']}`",
        f"- Calibration method: `{card['calibration_method']}`",
        f"- Negative strategy: `{card['negative_strategy']}`",
        f"- Negative ratio: `{card['negative_ratio']}`",
        f"- Easy negative fraction: `{card.get('easy_negative_fraction', 0.0)}`",
        f"- Excluded homology radius: `{card.get('excluded_homology_radius', '')}`",
        f"- Negative repeats: `{card.get('negative_repeats', 1)}`",
        f"- Validation splits requested: {', '.join(card['validation_splits'])}",
        f"- Random split PR-AUC: {card['metrics']['pr_auc']:.4f}",
        f"- Species included: {', '.join(card.get('species_included', [])) or 'not recorded'}",
        f"- Training resources: {', '.join(card.get('training_resources', [])) or 'not recorded'}",
        "",
        "## Validation Summary",
        "",
    ]
    for name, item in card.get("validation_report", {}).items():
        if not isinstance(item, dict):
            continue
        if item.get("status") == "ok" and "summary" in item:
            summary = item["summary"]
            baseline = summary.get("mean_baseline_pr_auc", {})
            strongest = max(baseline.items(), key=lambda kv: kv[1]) if baseline else None
            suffix = f", strongest baseline `{strongest[0]}` {float(strongest[1]):.4f}" if strongest else ""
            rank_suffix = _model_card_rank_suffix(summary)
            lines.append(f"- `{name}`: {summary.get('n_usable_folds', 0)} usable folds, mean PR-AUC {float(summary.get('mean_pr_auc', float('nan'))):.4f}{suffix}{rank_suffix}")
        elif item.get("status") == "ok":
            baseline = item.get("baseline_pr_auc", {})
            strongest = max(baseline.items(), key=lambda kv: kv[1]) if baseline else None
            suffix = f", strongest baseline `{strongest[0]}` {float(strongest[1]):.4f}" if strongest else ""
            rank_suffix = _model_card_rank_suffix(item)
            lines.append(f"- `{name}`: PR-AUC {float(item.get('pr_auc', float('nan'))):.4f}{suffix}{rank_suffix}")
        else:
            lines.append(f"- `{name}`: skipped ({item.get('reason', 'no usable folds')})")
    sampling = card.get("negative_sampling", {})
    if isinstance(sampling, dict):
        lines.extend(
            [
                "",
                "## Negative Sampling",
                "",
                f"- Positives: {sampling.get('n_positive', 0)}",
                f"- Pseudo-negatives: {sampling.get('n_negative', 0)}",
                f"- Easy pseudo-negatives: {sampling.get('easy_negative_count', 0)}",
                f"- Degree matching: {sampling.get('degree_matching', False)}",
                f"- Homology exclusion: `{sampling.get('excluded_homology_radius', '')}`",
                f"- Positive resource blacklist for folds: `{sampling.get('positive_resource_blacklist_for_fold', '')}`",
            ]
        )
    repeat_report = card.get("negative_repeat_report", {})
    if isinstance(repeat_report, dict) and repeat_report.get("status") == "ok":
        lines.extend(["", "## Negative Sampling Repeat Variance", ""])
        for split, item in repeat_report.get("summary", {}).items():
            if not isinstance(item, dict) or item.get("n_usable_repeats", 0) == 0:
                continue
            lines.append(
                "- `{split}`: {n} usable repeats, PR-AUC {mean:.4f} +/- {std:.4f}".format(
                    split=split,
                    n=item.get("n_usable_repeats", 0),
                    mean=float(item.get("pr_auc_mean", float("nan"))),
                    std=float(item.get("pr_auc_std", float("nan"))),
                )
            )
    embedding_model = card.get("embedding_model", {})
    if isinstance(embedding_model, dict):
        lines.extend(
            [
                "",
                "## Model Stack",
                "",
                f"- Embedding model: {', '.join(embedding_model.get('model_name', [])) or 'not recorded'}",
                f"- Embedding revision: {', '.join(embedding_model.get('model_revision', [])) or 'not recorded'}",
                f"- Embedding backend: {', '.join(embedding_model.get('embedding_backend', [])) or 'not recorded'}",
                f"- Embedding pooling: {', '.join(embedding_model.get('pooling', [])) or 'not recorded'}",
                f"- Embeddings: {embedding_model.get('n_embeddings', 0)}",
                f"- Pair model parameters: `{json.dumps(card.get('pair_model_params', {}), sort_keys=True)}`",
            ]
        )
    if card.get("calibration_metrics"):
        lines.extend(
            [
                "",
                "## Calibration",
                "",
                f"- Brier score: {float(card['calibration_metrics'].get('brier_score', float('nan'))):.4f}",
                f"- ECE 10-bin: {float(card['calibration_metrics'].get('ece_10bin', float('nan'))):.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Intended Use",
            "",
            "This model predicts candidate ligand-receptor pairs from protein embeddings for downstream pyccc scoring.",
            "",
            "## Caveat",
            "",
            "Predicted pairs are computational candidates and are not biochemical validation.",
        ]
    )
    return "\n".join(lines)


def _model_card_rank_suffix(metric: dict[str, object]) -> str:
    enrichment = metric.get("mean_top_k_enrichment", metric.get("top_k_enrichment", {}))
    recall = metric.get("mean_top_k_recall", metric.get("top_k_recall", {}))
    if not isinstance(enrichment, dict) or not isinstance(recall, dict):
        return ""
    enrichment_top100 = enrichment.get("top_100")
    recall_top100 = recall.get("top_100")
    if enrichment_top100 is None or recall_top100 is None:
        return ""
    return ", top-100 enrichment {enrichment:.2f}x, top-100 recall {recall:.4f}".format(
        enrichment=float(enrichment_top100),
        recall=float(recall_top100),
    )
