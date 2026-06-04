from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .model_resources import resolve_dbfree_model_path


ROLE_COLUMNS = ("ligand_like_score", "receptor_like_score", "secreted_like_score", "membrane_like_score", "ecm_like_score", "out_of_domain_score")


def train_protein_role_classifier(
    training_table,
    embeddings: pd.DataFrame,
    *,
    roles: Sequence[str] = ("ligand_like", "receptor_like", "secreted_like", "membrane_like"),
    model: str = "lightgbm",
    output_dir: str | Path,
    random_state: int = 0,
    validation_fraction: float = 0.25,
) -> dict[str, object]:
    """Train one-vs-rest protein role classifiers from fixture or real labels."""

    from joblib import dump

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("`validation_fraction` must be between 0 and 1.")
    interactions = getattr(training_table, "interactions", training_table)
    proteins = _role_training_labels(interactions, embeddings, roles)
    X = _embedding_matrix(proteins)
    models = {}
    metrics = {}
    for role in roles:
        y = proteins[role].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            models[role] = {"constant": float(y[0]) if len(y) else 0.0}
            metrics[role] = _constant_role_metrics(y)
        else:
            validation = _evaluate_role_holdout(
                model,
                X,
                y,
                validation_fraction=validation_fraction,
                random_state=random_state,
            )
            clf = _fit_role_model(model, X, y, random_state=random_state)
            models[role] = clf
            metrics[role] = {
                "n_positive": int(y.sum()),
                "n_negative": int(len(y) - y.sum()),
                "prevalence": float(y.mean()) if len(y) else 0.0,
                "constant": False,
                **validation,
            }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dump({"models": models, "roles": list(roles), "model": model, "embedding_genes": proteins["gene_id"].astype(str).tolist()}, output / "role_model.joblib")
    card = {
        "model_name": output.name,
        "model_stack": "esmc300m_lgbm_role_classifiers_v0" if model == "lightgbm" else "fixture_or_baseline_role_classifier",
        "model_type": "protein_role_classifier",
        "classifier": model,
        "roles": list(roles),
        "n_training_proteins": int(len(proteins)),
        "embedding_model": _embedding_metadata(embeddings),
        "validation_method": "stratified_holdout",
        "validation_fraction": float(validation_fraction),
        "metrics": metrics,
        "output_dir": str(output),
    }
    (output / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    (output / "model_card.md").write_text(_role_model_card_markdown(card), encoding="utf-8")
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
    payload = model if isinstance(model, dict) else load(resolve_dbfree_model_path(model, expected_file="role_model.joblib") / "role_model.joblib")
    X = _embedding_matrix(embeddings, genes=proteins["gene_id"].astype(str))
    out = proteins[["gene_id", "protein_id"]].copy()
    for role in payload["roles"]:
        score_col = f"{role}_score"
        clf = payload["models"][role]
        if isinstance(clf, dict) and "constant" in clf:
            out[score_col] = float(clf["constant"])
        else:
            out[score_col] = _role_model_scores(clf, X)
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


def _embedding_matrix(frame: pd.DataFrame, genes: Sequence[str] | None = None) -> np.ndarray:
    if "embedding" not in frame.columns:
        raise ValueError("Embedding table is missing `embedding`.")
    if genes is None:
        return np.vstack([np.asarray(value, dtype=np.float32) for value in frame["embedding"]])
    lookup = {str(row.gene_id): np.asarray(row.embedding, dtype=np.float32) for row in frame.itertuples(index=False)}
    missing = [str(gene) for gene in genes if str(gene) not in lookup]
    if missing:
        shown = ", ".join(missing[:5])
        raise ValueError(f"Missing embeddings for {len(missing)} proteins: {shown}")
    return np.vstack([lookup[str(gene)] for gene in genes]).astype(np.float32)


def _constant_role_metrics(y: np.ndarray) -> dict[str, object]:
    return {
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "prevalence": float(y.mean()) if len(y) else 0.0,
        "constant": True,
        "validation_status": "skipped",
        "validation_reason": "constant labels",
    }


def _embedding_metadata(embeddings: pd.DataFrame) -> dict[str, object]:
    return {
        "model_name": _sorted_strings(embeddings.get("model_name", pd.Series(dtype=str))),
        "model_revision": _sorted_strings(embeddings.get("model_revision", pd.Series(dtype=str))),
        "embedding_backend": _sorted_strings(embeddings.get("embedding_backend", pd.Series(dtype=str))),
        "pooling": _sorted_strings(embeddings.get("pooling", pd.Series(dtype=str))),
        "n_embeddings": int(len(embeddings)),
    }


def _sorted_strings(values: pd.Series) -> list[str]:
    if values.empty:
        return []
    return sorted({str(value) for value in values.dropna().astype(str) if str(value)})


def _evaluate_role_holdout(
    model: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    validation_fraction: float,
    random_state: int,
) -> dict[str, object]:
    class_counts = np.bincount(y.astype(int), minlength=2)
    if len(y) < 4 or class_counts.min() < 2:
        return {
            "validation_status": "skipped",
            "validation_reason": "need at least two positive and two negative proteins",
        }

    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    test_count = max(2, int(np.ceil(len(y) * validation_fraction)))
    test_count = min(test_count, len(y) - 2)
    if test_count < 2 or len(y) - test_count < 2:
        return {
            "validation_status": "skipped",
            "validation_reason": "stratified holdout would be too small",
        }
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_count,
        stratify=y,
        random_state=random_state,
    )
    clf = _fit_role_model(model, X[train_idx], y[train_idx], random_state=random_state)
    scores = _role_model_scores(clf, X[test_idx])
    y_test = y[test_idx].astype(int)
    pr_auc = float(average_precision_score(y_test, scores))
    roc_auc = float(roc_auc_score(y_test, scores)) if len(np.unique(y_test)) == 2 else np.nan
    top_k = {f"top_{k}": _top_k_precision(y_test, scores, k) for k in (10, 100, 500)}
    baseline = float(y_test.mean())
    return {
        "validation_status": "ok",
        "validation_n_train": int(len(train_idx)),
        "validation_n_test": int(len(test_idx)),
        "validation_pr_auc": pr_auc,
        "validation_roc_auc": roc_auc,
        "validation_baseline_pr_auc": baseline,
        "validation_delta_pr_auc": float(pr_auc - baseline),
        "validation_top_k_precision": top_k,
    }


def _role_model_scores(clf, X: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        return clf.predict_proba(X)[:, 1].astype(float)


def _top_k_precision(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(y) == 0:
        return np.nan
    kk = min(int(k), len(y))
    order = np.argsort(-scores)[:kk]
    return float(y[order].mean())


def _role_model_card_markdown(card: dict[str, object]) -> str:
    metrics = card.get("metrics", {})
    lines = [
        "# pyccc Protein Role Classifier",
        "",
        f"- classifier: `{card.get('classifier', '')}`",
        f"- training proteins: {card.get('n_training_proteins', 0)}",
        f"- validation method: {card.get('validation_method', '')}",
    ]
    embedding_model = card.get("embedding_model", {})
    if isinstance(embedding_model, dict):
        lines.extend(
            [
                f"- embedding model: {', '.join(embedding_model.get('model_name', [])) or 'not recorded'}",
                f"- embedding backend: {', '.join(embedding_model.get('embedding_backend', [])) or 'not recorded'}",
                f"- embedding pooling: {', '.join(embedding_model.get('pooling', [])) or 'not recorded'}",
            ]
        )
    lines.extend(
        [
            "",
            "## Role Metrics",
            "",
            "| role | positives | negatives | PR-AUC | baseline PR-AUC | delta | status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if isinstance(metrics, dict):
        for role, item in metrics.items():
            if not isinstance(item, dict):
                continue
            lines.append(
                "| {role} | {pos} | {neg} | {pr} | {base} | {delta} | {status} |".format(
                    role=role,
                    pos=item.get("n_positive", 0),
                    neg=item.get("n_negative", 0),
                    pr=_markdown_float(item.get("validation_pr_auc")),
                    base=_markdown_float(item.get("validation_baseline_pr_auc")),
                    delta=_markdown_float(item.get("validation_delta_pr_auc")),
                    status=item.get("validation_status", "skipped"),
                )
            )
    lines.extend(
        [
            "",
            "Predicted roles are sequence-derived candidate annotations used to reduce the LR search space.",
            "They should not be interpreted as experimentally validated secretion or membrane-localization evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_float(value) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


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
