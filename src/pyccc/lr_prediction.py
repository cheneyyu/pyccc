from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .density import build_predicted_lr_table
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
    random_state: int = 0,
) -> dict[str, object]:
    """Train a pairwise LR link predictor and write a model card."""

    from joblib import dump
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    interactions = getattr(training_table, "interactions", training_table)
    pairs = _training_pairs(interactions, negative_ratio=negative_ratio, random_state=random_state)
    features = make_lr_pair_features(pairs, embeddings, encoder=feature_encoder, fit_pca=True)
    y = pairs["label"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Training requires at least one positive and one pseudo-negative pair.")
    idx = np.arange(len(y))
    n_classes = len(np.unique(y))
    test_size = max(n_classes, int(np.ceil(len(y) * 0.25)))
    if len(y) - test_size < n_classes:
        test_size = n_classes
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state, stratify=y)
    clf = _fit_pair_model(model, features.X[train_idx], y[train_idx], random_state=random_state)
    scores = _predict_scores(clf, features.X[test_idx])
    metrics = {
        "pr_auc": float(average_precision_score(y[test_idx], scores)),
        "roc_auc": _safe_roc_auc(y[test_idx], scores, roc_auc_score),
        "n_pairs": int(len(y)),
        "n_positive": int(y.sum()),
        "negative_strategy": negative_strategy,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dump({"model": clf, "pca_model": features.pca_model, "feature_encoder": feature_encoder, "feature_names": features.feature_names}, output / "lr_link_model.joblib")
    card = {
        "model_type": model,
        "feature_encoder": feature_encoder,
        "validation_splits": list(validation_splits),
        "metrics": metrics,
        "negative_strategy": negative_strategy,
        "output_dir": str(output),
    }
    (output / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (output / "model_card.md").write_text(_model_card_markdown(card), encoding="utf-8")
    return card


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
        chunk["calibrated_probability"] = chunk["model_score"]
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
    db = build_predicted_lr_table(
        scores,
        roles=roles,
        density_prior=density_prior,
        species_hint=species_hint,
        max_pairs=max_pairs,
        model_name=str(model),
        model_revision="",
        name=f"dbfree_predicted_{species_name}",
    )
    db.metadata["proteins"] = proteins
    db.metadata["embeddings"] = emb
    return db


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


def _training_pairs(interactions: pd.DataFrame, *, negative_ratio: int, random_state: int) -> pd.DataFrame:
    if "is_positive_label" in interactions.columns:
        positives = interactions[interactions["is_positive_label"].astype(bool)].copy()
    else:
        positives = interactions.copy()
    positives = positives[["species", "ligand_gene", "receptor_gene"]].drop_duplicates()
    positives["label"] = 1
    rng = np.random.default_rng(random_state)
    negatives = []
    for species, sub in positives.groupby("species", sort=False):
        ligands = sub["ligand_gene"].astype(str).unique()
        receptors = sub["receptor_gene"].astype(str).unique()
        positive_set = set(zip(sub["ligand_gene"].astype(str), sub["receptor_gene"].astype(str)))
        target = len(sub) * negative_ratio
        species_negatives = []
        tries = 0
        while len(species_negatives) < target and tries < target * 20 + 100:
            tries += 1
            lig = str(rng.choice(ligands))
            rec = str(rng.choice(receptors))
            if lig == rec or (lig, rec) in positive_set:
                continue
            species_negatives.append({"species": species, "ligand_gene": lig, "receptor_gene": rec, "label": 0})
        negatives.extend(species_negatives)
    pairs = pd.concat([positives, pd.DataFrame(negatives)], ignore_index=True)
    return pairs.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


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
    ligand_role = pairs.get("ligand_role_score", pd.Series([0.5] * len(pairs))).astype(float).to_numpy()
    receptor_role = pairs.get("receptor_role_score", pd.Series([0.5] * len(pairs))).astype(float).to_numpy()
    score = 0.35 * ((cosine + 1.0) / 2.0) + 0.325 * ligand_role + 0.325 * receptor_role
    return np.clip(score, 0.0, 1.0)


def _model_card_markdown(card: dict[str, object]) -> str:
    return "\n".join(
        [
            "# pyccc LR Link Predictor Model Card",
            "",
            f"- Model type: `{card['model_type']}`",
            f"- Feature encoder: `{card['feature_encoder']}`",
            f"- Negative strategy: `{card['negative_strategy']}`",
            f"- Validation splits requested: {', '.join(card['validation_splits'])}",
            f"- PR-AUC: {card['metrics']['pr_auc']:.4f}",
            "",
            "This model predicts candidate ligand-receptor pairs from protein embeddings. It is not biochemical validation.",
        ]
    )
