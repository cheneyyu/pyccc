from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .density import build_predicted_lr_table, estimate_lr_density_prior
from .embeddings import embed_proteins_esmc
from .pair_features import make_lr_pair_features
from .roles import predict_protein_roles
from .sequence import load_cds_translations, load_protein_fasta


def generate_lr_candidates_dbfree(
    adata,
    proteins: pd.DataFrame,
    roles: pd.DataFrame,
    *,
    gene_id_key: str | None = None,
    expression_min_fraction: float = 0.02,
    ligand_role_min: float = 0.30,
    receptor_role_min: float = 0.30,
    max_ligands: int = 3000,
    max_receptors: int = 3000,
    max_candidate_pairs: int = 5_000_000,
    ligand_candidates: str | Path | Sequence[str] | None = None,
    receptor_candidates: str | Path | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Generate role- and expression-filtered candidate LR pairs."""

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
    total = len(ligands) * len(receptors)
    if total > max_candidate_pairs:
        keep_receptors = max(1, max_candidate_pairs // max(len(ligands), 1))
        receptors = receptors.head(keep_receptors)
        total = len(ligands) * len(receptors)
    rows = []
    for lig in ligands.itertuples(index=False):
        for rec in receptors.itertuples(index=False):
            if str(lig.gene_id) == str(rec.gene_id):
                continue
            rows.append(
                {
                    "ligand_gene": str(lig.gene_id),
                    "receptor_gene": str(rec.gene_id),
                    "ligand_role_score": float(lig.ligand_like_score),
                    "receptor_role_score": float(rec.receptor_like_score),
                    "ligand_expression_fraction": float(lig.expression_fraction),
                    "receptor_expression_fraction": float(rec.expression_fraction),
                    "candidate_strategy": "role_expression_cross_product",
                }
            )
    if not rows:
        raise ValueError("Candidate generation produced no non-self LR pairs.")
    return pd.DataFrame(rows).head(max_candidate_pairs)


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
    calibration_method: str | None = "isotonic",
    density_groupby: str = "clade",
    random_state: int = 0,
) -> dict[str, object]:
    """Train a pairwise LR link predictor and write a model card."""

    from joblib import dump

    interactions = getattr(training_table, "interactions", training_table)
    pairs = _training_pairs(interactions, negative_ratio=negative_ratio, random_state=random_state)
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
        "baseline_pr_auc": random_metrics.get("baseline_pr_auc", {}),
        "baseline_top_k_precision": random_metrics.get("baseline_top_k_precision", {}),
        "baseline_delta_pr_auc": random_metrics.get("baseline_delta_pr_auc", {}),
        "baseline_delta_top_k_precision": random_metrics.get("baseline_delta_top_k_precision", {}),
        "n_pairs": int(len(y)),
        "n_positive": int(y.sum()),
        "negative_strategy": negative_strategy,
    }
    validation_report = _validation_report(
        pairs,
        embeddings,
        y,
        requested_splits=validation_splits,
        model=model,
        feature_encoder=feature_encoder,
        random_state=random_state,
    )
    features = make_lr_pair_features(pairs, embeddings, encoder=feature_encoder, fit_pca=True)
    clf = _fit_pair_model(model, features.X, y, random_state=random_state)
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
    dump({"model": clf, "pca_model": features.pca_model, "feature_encoder": feature_encoder, "feature_names": features.feature_names, "calibrator": calibration["calibrator"]}, output / "lr_link_model.joblib")
    density_prior = _write_density_prior(interactions, output, groupby=density_groupby)
    card = {
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
        "validation_report": validation_report,
        "negative_strategy": negative_strategy,
        "negative_ratio": negative_ratio,
        "output_dir": str(output),
    }
    (output / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (output / "model_card.md").write_text(_model_card_markdown(card), encoding="utf-8")
    return card


def evaluate_lr_model_quality_gates(
    model_card: dict[str, object] | str | Path,
    *,
    required_splits: Sequence[str] = ("leave_species_out",),
    required_baselines: Sequence[str] = ("degree_prior", "embedding_cosine", "role_only", "random"),
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
        return out.sort_values("model_score", ascending=False).reset_index(drop=True)

    from joblib import load

    payload = load(Path(model) / "lr_link_model.joblib")
    rows = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs.iloc[start : start + batch_size].copy()
        features = make_lr_pair_features(chunk, embeddings, encoder=payload["feature_encoder"], pca_model=payload.get("pca_model"))
        chunk["model_score"] = _predict_scores(payload["model"], features.X)
        chunk["calibrated_probability"] = _apply_calibrator(payload.get("calibrator"), chunk["model_score"].to_numpy(dtype=float))
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
    model: str | Path = "heuristic",
    role_model: str | Path = "heuristic",
    density_prior: pd.DataFrame | float | str = "auto",
    max_pairs: int = 50000,
    cache_dir: str | Path | None = None,
    embedding_backend: str = "auto",
    ligand_candidates: str | Path | Sequence[str] | None = None,
    receptor_candidates: str | Path | Sequence[str] | None = None,
    **candidate_kwargs,
):
    """Predict a target-species candidate LR table, then return CellChatDB."""

    if (cds_fasta is None) == (protein_fasta is None):
        raise ValueError("Provide exactly one of `cds_fasta` or `protein_fasta`.")
    proteins = load_cds_translations(cds_fasta) if cds_fasta is not None else load_protein_fasta(protein_fasta)
    emb = embed_proteins_esmc(proteins, cache_dir=Path(cache_dir) / "esmc" if cache_dir is not None else None, backend=embedding_backend)
    roles = predict_protein_roles(proteins, emb, model=role_model)
    candidates = generate_lr_candidates_dbfree(
        adata,
        proteins,
        roles,
        gene_id_key=gene_id_key,
        ligand_candidates=ligand_candidates,
        receptor_candidates=receptor_candidates,
        **candidate_kwargs,
    )
    scores = score_lr_candidates(candidates, emb, model=model)
    resolved_density_prior = _auto_density_prior(density_prior, model)
    db = build_predicted_lr_table(
        scores,
        roles=roles,
        density_prior=resolved_density_prior,
        species_hint=species_hint,
        max_pairs=max_pairs,
        model_name=str(model),
        model_revision="",
        name=f"dbfree_predicted_{species_name}",
    )
    db.metadata["proteins"] = proteins
    db.metadata["embeddings"] = emb
    return db


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


def _expressed_gene_fractions(adata, *, gene_id_key: str | None) -> pd.DataFrame:
    genes = pd.Index(adata.var_names.astype(str)) if gene_id_key is None else pd.Index(adata.var[gene_id_key].astype(str))
    x = adata.X
    if sparse.issparse(x):
        frac = np.asarray((x > 0).mean(axis=0)).ravel()
    else:
        frac = np.asarray(x > 0).mean(axis=0)
    return pd.DataFrame({"gene_id": genes.astype(str), "expression_fraction": frac.astype(float)})


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


def _training_pairs(interactions: pd.DataFrame, *, negative_ratio: int, random_state: int) -> pd.DataFrame:
    if "is_positive_label" in interactions.columns:
        positives = interactions[interactions["is_positive_label"].astype(bool)].copy()
    else:
        positives = interactions.copy()
    positives = _positive_pair_metadata(positives)
    positives["label"] = 1
    rng = np.random.default_rng(random_state)
    negatives = []
    positive_set_by_species = {
        species: set(zip(sub["ligand_gene"].astype(str), sub["receptor_gene"].astype(str)))
        for species, sub in positives.groupby("species", sort=False)
    }
    for (species, resource), sub in positives.groupby(["species", "resource"], sort=False):
        species_sub = positives[positives["species"].astype(str) == str(species)]
        ligands, ligand_p = _degree_pool(species_sub["ligand_gene"])
        receptors, receptor_p = _degree_pool(species_sub["receptor_gene"])
        positive_set = positive_set_by_species[str(species)]
        target = len(sub) * negative_ratio
        species_negatives = []
        tries = 0
        while len(species_negatives) < target and tries < target * 20 + 100:
            tries += 1
            lig = str(rng.choice(ligands, p=ligand_p))
            rec = str(rng.choice(receptors, p=receptor_p))
            if lig == rec or (lig, rec) in positive_set:
                continue
            species_negatives.append(
                {
                    "species": str(species),
                    "clade": str(sub["clade"].iloc[0]) if "clade" in sub.columns else "",
                    "resource": str(resource),
                    "ligand_gene": lig,
                    "receptor_gene": rec,
                    "ligand_family": _family_for_gene(species_sub, "ligand", lig),
                    "receptor_family": _family_for_gene(species_sub, "receptor", rec),
                    "ligand_role_score": _role_score_for_gene(species_sub, "ligand", lig),
                    "receptor_role_score": _role_score_for_gene(species_sub, "receptor", rec),
                    "label": 0,
                    "negative_strategy": "pu_degree_matched",
                    "negative_seed": int(random_state),
                }
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
    for col in ("ligand_role_score", "receptor_role_score"):
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
    cols = ["species", "clade", "resource", "ligand_gene", "receptor_gene", "ligand_family", "receptor_family", "ligand_role_score", "receptor_role_score"]
    out = out[cols].drop_duplicates().copy()
    for col in cols:
        out[col] = out[col].fillna("").astype(str)
    return out


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
    for group in sorted(set(pairs[group_col].astype(str))):
        test_idx = np.flatnonzero(pairs[group_col].astype(str).to_numpy() == str(group))
        train_idx = np.flatnonzero(pairs[group_col].astype(str).to_numpy() != str(group))
        fold = _evaluate_pair_split(pairs, embeddings, y, train_idx, test_idx, model=model, feature_encoder=feature_encoder, random_state=random_state)
        fold["held_out"] = str(group)
        folds.append(fold)
    usable = [fold for fold in folds if fold["status"] == "ok"]
    return {"status": "ok" if usable else "skipped", "folds": folds, "summary": _fold_summary(usable)}


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
        "baseline_pr_auc": {name: value["pr_auc"] for name, value in baselines.items()},
        "baseline_top_k_precision": {name: value["top_k_precision"] for name, value in baselines.items()},
        "baseline_delta_pr_auc": {name: metrics["pr_auc"] - value["pr_auc"] for name, value in baselines.items()},
        "baseline_delta_top_k_precision": {
            name: {
                k: metrics["top_k_precision"][k] - baseline_metrics["top_k_precision"][k]
                for k in metrics["top_k_precision"]
            }
            for name, baseline_metrics in baselines.items()
        },
        "feature_encoder_fit": "train_split_only",
    }


def _classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": _safe_roc_auc(y_true, scores, roc_auc_score),
        "top_k_precision": _top_k_precision(y_true, scores, ks=(100, 500, 1000, 5000)),
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
        "role_only": _role_only_scores(test_pairs),
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


def _role_only_scores(test_pairs: pd.DataFrame) -> np.ndarray:
    ligand = pd.to_numeric(test_pairs.get("ligand_role_score", pd.Series([0.5] * len(test_pairs))), errors="coerce").fillna(0.5).to_numpy(dtype=float)
    receptor = pd.to_numeric(test_pairs.get("receptor_role_score", pd.Series([0.5] * len(test_pairs))), errors="coerce").fillna(0.5).to_numpy(dtype=float)
    return (ligand + receptor) / 2.0


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


def _top_k_precision(y_true: np.ndarray, scores: np.ndarray, *, ks: Sequence[int]) -> dict[str, float]:
    order = np.argsort(-scores)
    return {f"top_{k}": float(y_true[order[: min(k, len(order))]].mean()) if len(order) else float("nan") for k in ks}


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
    top_k_mean = {
        name: float(np.mean([float(fold["top_k_precision"][name]) for fold in folds if name in fold.get("top_k_precision", {})]))
        for name in top_k_names
    }
    baseline_top_k_mean = {
        baseline: {
            k: float(
                np.mean(
                    [
                        float(fold["baseline_top_k_precision"][baseline][k])
                        for fold in folds
                        if baseline in fold.get("baseline_top_k_precision", {})
                        and k in fold["baseline_top_k_precision"][baseline]
                    ]
                )
            )
            for k in top_k_names
        }
        for baseline in baseline_names
    }
    baseline_top_k_delta_mean = {
        baseline: {
            k: float(
                np.mean(
                    [
                        float(fold["baseline_delta_top_k_precision"][baseline][k])
                        for fold in folds
                        if baseline in fold.get("baseline_delta_top_k_precision", {})
                        and k in fold["baseline_delta_top_k_precision"][baseline]
                    ]
                )
            )
            for k in top_k_names
        }
        for baseline in baseline_names
    }
    out = {
        "n_usable_folds": len(folds),
        "mean_pr_auc": float(np.mean([float(fold["pr_auc"]) for fold in folds])),
        "mean_roc_auc": float(np.mean([float(fold["roc_auc"]) for fold in folds])),
    }
    if top_k_mean:
        out["mean_top_k_precision"] = top_k_mean
    if baseline_mean:
        out["mean_baseline_pr_auc"] = baseline_mean
        out["mean_baseline_delta_pr_auc"] = delta_mean
        out["mean_baseline_top_k_precision"] = baseline_top_k_mean
        out["mean_baseline_delta_top_k_precision"] = baseline_top_k_delta_mean
    return out


def _fit_pair_model(model: str, X: np.ndarray, y: np.ndarray, *, random_state: int):
    if model == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install LR prediction support with `pyccc[predict]` to train LightGBM models.") from exc
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
    elif model == "sklearn":
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(random_state=random_state, max_iter=100)
    else:
        raise ValueError("`model` must be one of: lightgbm, sklearn.")
    clf.fit(X, y)
    return clf


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
        f"- Validation splits requested: {', '.join(card['validation_splits'])}",
        f"- Random split PR-AUC: {card['metrics']['pr_auc']:.4f}",
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
            lines.append(f"- `{name}`: {summary.get('n_usable_folds', 0)} usable folds, mean PR-AUC {float(summary.get('mean_pr_auc', float('nan'))):.4f}{suffix}")
        elif item.get("status") == "ok":
            baseline = item.get("baseline_pr_auc", {})
            strongest = max(baseline.items(), key=lambda kv: kv[1]) if baseline else None
            suffix = f", strongest baseline `{strongest[0]}` {float(strongest[1]):.4f}" if strongest else ""
            lines.append(f"- `{name}`: PR-AUC {float(item.get('pr_auc', float('nan'))):.4f}{suffix}")
        else:
            lines.append(f"- `{name}`: skipped ({item.get('reason', 'no usable folds')})")
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
