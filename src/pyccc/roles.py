from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROLE_COLUMNS = ("ligand_like_score", "receptor_like_score", "secreted_like_score", "membrane_like_score", "ecm_like_score", "out_of_domain_score")


def train_protein_role_classifier(
    training_table,
    embeddings: pd.DataFrame,
    *,
    roles: Sequence[str] = ("ligand_like", "receptor_like", "secreted_like", "membrane_like"),
    model: str = "lightgbm",
    output_dir: str | Path,
    random_state: int = 0,
) -> dict[str, object]:
    """Train one-vs-rest protein role classifiers from fixture or real labels."""

    from joblib import dump

    interactions = getattr(training_table, "interactions", training_table)
    proteins = _role_training_labels(interactions, embeddings, roles)
    X = _embedding_matrix(proteins)
    models = {}
    metrics = {}
    for role in roles:
        y = proteins[role].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            models[role] = {"constant": float(y[0]) if len(y) else 0.0}
            metrics[role] = {"n_positive": int(y.sum()), "constant": True}
        else:
            clf = _fit_role_model(model, X, y, random_state=random_state)
            models[role] = clf
            metrics[role] = {"n_positive": int(y.sum()), "constant": False}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dump({"models": models, "roles": list(roles), "model": model, "embedding_genes": proteins["gene_id"].astype(str).tolist()}, output / "role_model.joblib")
    card = {"model_type": "protein_role_classifier", "classifier": model, "roles": list(roles), "metrics": metrics, "output_dir": str(output)}
    (output / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


def predict_protein_roles(
    proteins: pd.DataFrame,
    embeddings: pd.DataFrame | None = None,
    *,
    model: str | Path | dict[str, object] = "heuristic",
) -> pd.DataFrame:
    """Predict ligand/receptor-like role scores for proteins."""

    base = proteins[["gene_id", "protein_id", "protein_sequence"]].copy()
    if str(model) == "heuristic":
        return _heuristic_roles(base)

    from joblib import load

    if embeddings is None:
        raise ValueError("`embeddings` is required for a trained role model.")
    payload = model if isinstance(model, dict) else load(Path(model) / "role_model.joblib")
    X = _embedding_matrix(embeddings)
    out = proteins[["gene_id", "protein_id"]].copy()
    for role in payload["roles"]:
        score_col = f"{role}_score"
        clf = payload["models"][role]
        if isinstance(clf, dict) and "constant" in clf:
            out[score_col] = float(clf["constant"])
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                out[score_col] = clf.predict_proba(X)[:, 1]
    for col in ROLE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    return out[["gene_id", "protein_id", *ROLE_COLUMNS]]


def _role_training_labels(interactions: pd.DataFrame, embeddings: pd.DataFrame, roles: Sequence[str]) -> pd.DataFrame:
    genes = embeddings[["gene_id", "embedding"]].copy()
    ligand_genes = set(interactions.get("ligand_gene", pd.Series(dtype=str)).astype(str))
    receptor_genes = set(interactions.get("receptor_gene", pd.Series(dtype=str)).astype(str))
    genes["ligand_like"] = genes["gene_id"].astype(str).isin(ligand_genes)
    genes["receptor_like"] = genes["gene_id"].astype(str).isin(receptor_genes)
    genes["secreted_like"] = genes["ligand_like"]
    genes["membrane_like"] = genes["receptor_like"]
    genes["ecm_like"] = False
    genes["out_of_domain"] = False
    for role in roles:
        if role not in genes.columns:
            genes[role] = False
    return genes


def _embedding_matrix(frame: pd.DataFrame) -> np.ndarray:
    if "embedding" not in frame.columns:
        raise ValueError("Embedding table is missing `embedding`.")
    return np.vstack([np.asarray(value, dtype=np.float32) for value in frame["embedding"]])


def _fit_role_model(model: str, X: np.ndarray, y: np.ndarray, *, random_state: int):
    if model == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install DB-free prediction support with `pyccc[predict]` to train LightGBM role classifiers.") from exc
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
    elif model == "sklearn":
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    else:
        raise ValueError("`model` must be one of: lightgbm, sklearn.")
    clf.fit(X, y)
    return clf


def _heuristic_roles(proteins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in proteins.itertuples(index=False):
        seq = str(row.protein_sequence).upper()
        length = len(seq)
        hydrophobic = sum(aa in "AILMFWYV" for aa in seq) / max(length, 1)
        cysteine = seq.count("C") / max(length, 1)
        ligand = float(np.clip(0.75 - (length / 1600.0) + cysteine * 2.0, 0.05, 0.95))
        receptor = float(np.clip((length / 1200.0) + hydrophobic * 0.8, 0.05, 0.95))
        secreted = float(np.clip(ligand + cysteine, 0.05, 0.95))
        membrane = float(np.clip(receptor + hydrophobic * 0.4, 0.05, 0.95))
        rows.append(
            {
                "gene_id": str(row.gene_id),
                "protein_id": str(row.protein_id),
                "ligand_like_score": ligand,
                "receptor_like_score": receptor,
                "secreted_like_score": secreted,
                "membrane_like_score": membrane,
                "ecm_like_score": min(0.95, cysteine * 3.0),
                "out_of_domain_score": 0.0,
            }
        )
    return pd.DataFrame(rows)
