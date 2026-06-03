from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class LRPairFeatures:
    """Feature matrix plus pair metadata for LR pair ranking."""

    pairs: pd.DataFrame
    X: np.ndarray
    feature_names: list[str]
    encoder: str
    pca_model: object | None = None


def make_lr_pair_features(
    pairs: pd.DataFrame,
    embeddings: pd.DataFrame,
    *,
    encoder: str = "pca128_absdiff_hadamard_v1",
    pca_model: object | None = None,
    fit_pca: bool = False,
    pca_components: int = 128,
) -> LRPairFeatures:
    """Build deterministic two-vector LR pair features from protein embeddings."""

    if encoder != "pca128_absdiff_hadamard_v1":
        raise ValueError("Only `pca128_absdiff_hadamard_v1` is supported.")
    pair_frame = _canonical_pair_columns(pairs)
    emb_lookup = _embedding_lookup(embeddings)
    ligand = _stack_embeddings(pair_frame["ligand_gene"], emb_lookup, role="ligand")
    receptor = _stack_embeddings(pair_frame["receptor_gene"], emb_lookup, role="receptor")
    ligand_z, receptor_z, pca_model = _maybe_pca(ligand, receptor, pca_model=pca_model, fit_pca=fit_pca, n_components=pca_components)
    cosine = _cosine(ligand_z, receptor_z)[:, None]
    euclidean = np.linalg.norm(ligand_z - receptor_z, axis=1)[:, None]
    ligand_length = _length_feature(pair_frame, embeddings, pair_frame["ligand_gene"], prefix="ligand")[:, None]
    receptor_length = _length_feature(pair_frame, embeddings, pair_frame["receptor_gene"], prefix="receptor")[:, None]
    ligand_role = pd.to_numeric(pair_frame.get("ligand_role_score", pd.Series([0.0] * len(pair_frame))), errors="coerce").fillna(0.0).to_numpy()[:, None]
    receptor_role = pd.to_numeric(pair_frame.get("receptor_role_score", pd.Series([0.0] * len(pair_frame))), errors="coerce").fillna(0.0).to_numpy()[:, None]
    blocks = [
        ligand_z,
        receptor_z,
        np.abs(ligand_z - receptor_z),
        ligand_z * receptor_z,
        cosine,
        euclidean,
        ligand_length,
        receptor_length,
        ligand_role,
        receptor_role,
    ]
    X = np.hstack(blocks).astype(np.float32)
    feature_names = (
        [f"ligand_z_{i}" for i in range(ligand_z.shape[1])]
        + [f"receptor_z_{i}" for i in range(receptor_z.shape[1])]
        + [f"absdiff_z_{i}" for i in range(ligand_z.shape[1])]
        + [f"hadamard_z_{i}" for i in range(ligand_z.shape[1])]
        + ["cosine", "euclidean", "ligand_length_log1p", "receptor_length_log1p", "ligand_role_score", "receptor_role_score"]
    )
    return LRPairFeatures(pair_frame.reset_index(drop=True), X, feature_names, encoder, pca_model=pca_model)


def _canonical_pair_columns(pairs: pd.DataFrame) -> pd.DataFrame:
    out = pairs.copy()
    aliases = {
        "ligand_gene": ("ligand_gene", "ligand"),
        "receptor_gene": ("receptor_gene", "receptor"),
    }
    for canonical, names in aliases.items():
        if canonical not in out.columns:
            for name in names:
                if name in out.columns:
                    out[canonical] = out[name]
                    break
    missing = [col for col in ("ligand_gene", "receptor_gene") if col not in out.columns]
    if missing:
        raise ValueError(f"Pair table is missing required columns: {missing}")
    out["ligand_gene"] = out["ligand_gene"].astype(str)
    out["receptor_gene"] = out["receptor_gene"].astype(str)
    return out


def _embedding_lookup(embeddings: pd.DataFrame) -> dict[str, np.ndarray]:
    missing = [col for col in ("gene_id", "embedding") if col not in embeddings.columns]
    if missing:
        raise ValueError(f"Embedding table is missing required columns: {missing}")
    return {str(row.gene_id): np.asarray(row.embedding, dtype=np.float32) for row in embeddings.itertuples(index=False)}


def _stack_embeddings(genes: Sequence[str], lookup: dict[str, np.ndarray], *, role: str) -> np.ndarray:
    missing = [gene for gene in genes if str(gene) not in lookup]
    if missing:
        shown = ", ".join(map(str, missing[:5]))
        raise ValueError(f"Missing {role} embeddings for {len(missing)} genes: {shown}")
    return np.vstack([lookup[str(gene)] for gene in genes]).astype(np.float32)


def _maybe_pca(ligand: np.ndarray, receptor: np.ndarray, *, pca_model: object | None, fit_pca: bool, n_components: int):
    if ligand.shape[1] <= n_components and pca_model is None:
        return ligand, receptor, None
    if pca_model is None:
        if not fit_pca:
            return ligand[:, :n_components], receptor[:, :n_components], None
        from sklearn.decomposition import PCA

        n = min(n_components, ligand.shape[1], max(1, ligand.shape[0] + receptor.shape[0] - 1))
        pca_model = PCA(n_components=n, random_state=0)
        pca_model.fit(np.vstack([ligand, receptor]))
    return pca_model.transform(ligand).astype(np.float32), pca_model.transform(receptor).astype(np.float32), pca_model


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide((a * b).sum(axis=1), denom, out=np.zeros(len(a), dtype=np.float32), where=denom > 0)


def _length_feature(pairs: pd.DataFrame, embeddings: pd.DataFrame, genes: Sequence[str], *, prefix: str) -> np.ndarray:
    col = f"{prefix}_length"
    if col in pairs.columns:
        return np.log1p(pairs[col].astype(float).to_numpy())
    if "sequence_length" in embeddings.columns:
        lookup = {str(row.gene_id): float(row.sequence_length) for row in embeddings.itertuples(index=False)}
        return np.log1p(np.asarray([lookup.get(str(gene), 0.0) for gene in genes], dtype=float))
    return np.zeros(len(pairs), dtype=float)
