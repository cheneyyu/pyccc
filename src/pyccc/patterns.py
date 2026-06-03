from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans, SpectralClustering

from .analysis import CCCResult, compute_pathway_communication

PatternMode = Literal["outgoing", "incoming", "all"]
NetworkSimilarity = Literal["functional", "structural"]
PathwayEmbeddingMethod = Literal["auto", "umap", "mds"]
PathwayClusteringMethod = Literal["spectral", "kmeans"]


@dataclass
class CommunicationPatterns:
    """Latent communication patterns analogous to CellChat W/H summaries."""

    mode: PatternMode
    n_patterns: int
    matrix: pd.DataFrame
    group_pattern: pd.DataFrame
    pathway_pattern: pd.DataFrame
    group_weights: pd.DataFrame
    pathway_weights: pd.DataFrame
    reconstruction_error: float


def compute_communication_patterns(
    result: CCCResult,
    *,
    mode: PatternMode = "outgoing",
    n_patterns: int = 3,
    significant_only: bool = False,
    random_state: int | None = 0,
    max_iter: int = 1000,
) -> CommunicationPatterns:
    """Factorize pathway communication into latent group and pathway patterns.

    The input matrix is cell-group-by-pathway communication strength. Outgoing
    mode uses sender groups, incoming mode uses receiver groups, and all mode
    adds sender and receiver contributions before factorization.
    """

    if mode not in {"outgoing", "incoming", "all"}:
        raise ValueError("`mode` must be 'outgoing', 'incoming', or 'all'.")
    if n_patterns < 1:
        raise ValueError("`n_patterns` must be positive.")

    matrix = communication_pattern_matrix(result, mode=mode, significant_only=significant_only)
    if matrix.empty or matrix.to_numpy().sum() <= 0:
        raise ValueError("No pathway communication signal is available for pattern factorization.")
    n_components = min(n_patterns, matrix.shape[0], matrix.shape[1])

    model = NMF(n_components=n_components, init="nndsvda", random_state=random_state, max_iter=max_iter)
    group_weights_arr = model.fit_transform(matrix.to_numpy(dtype=float))
    pathway_weights_arr = model.components_
    names = [f"Pattern {i + 1}" for i in range(n_components)]
    group_weights = pd.DataFrame(group_weights_arr, index=matrix.index, columns=names)
    pathway_weights = pd.DataFrame(pathway_weights_arr, index=names, columns=matrix.columns)

    group_pattern = (
        group_weights.reset_index(names="group")
        .melt(id_vars="group", var_name="pattern", value_name="weight")
        .sort_values(["pattern", "weight"], ascending=[True, False])
        .reset_index(drop=True)
    )
    pathway_pattern = (
        pathway_weights.T.reset_index(names="pathway")
        .melt(id_vars="pathway", var_name="pattern", value_name="weight")
        .sort_values(["pattern", "weight"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return CommunicationPatterns(
        mode=mode,
        n_patterns=n_components,
        matrix=matrix,
        group_pattern=group_pattern,
        pathway_pattern=pathway_pattern,
        group_weights=group_weights,
        pathway_weights=pathway_weights,
        reconstruction_error=float(model.reconstruction_err_),
    )


def select_communication_pattern_number(
    result: CCCResult,
    *,
    mode: PatternMode = "outgoing",
    k_range: range | list[int] | tuple[int, ...] = range(2, 8),
    n_runs: int = 5,
    significant_only: bool = False,
    random_state: int | None = 0,
    max_iter: int = 1000,
) -> pd.DataFrame:
    """Scan candidate pattern numbers for NMF diagnostics.

    The returned table contains reconstruction error, explained fraction of
    Frobenius norm, and a simple component-stability score across random
    initializations. Higher explained/stability and lower error are preferred.
    """

    if n_runs < 1:
        raise ValueError("`n_runs` must be positive.")
    matrix = communication_pattern_matrix(result, mode=mode, significant_only=significant_only)
    if matrix.empty or matrix.to_numpy().sum() <= 0:
        raise ValueError("No pathway communication signal is available for pattern selection.")

    values = matrix.to_numpy(dtype=float)
    total_norm_sq = float(np.square(values).sum()) or 1.0
    max_components = min(matrix.shape)
    rng = np.random.default_rng(random_state)
    rows = []
    for k in k_range:
        if k < 1 or k > max_components:
            continue
        errors = []
        components = []
        for _ in range(n_runs):
            seed = int(rng.integers(0, np.iinfo(np.int32).max))
            model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=max_iter)
            model.fit_transform(values)
            errors.append(float(model.reconstruction_err_))
            components.append(model.components_)
        mean_error = float(np.mean(errors))
        rows.append(
            {
                "k": int(k),
                "mean_reconstruction_error": mean_error,
                "std_reconstruction_error": float(np.std(errors)),
                "explained": max(0.0, 1.0 - (mean_error**2 / total_norm_sq)),
                "stability": _component_stability(components),
                "n_runs": int(n_runs),
            }
        )
    if not rows:
        raise ValueError("No valid `k_range` values fit the communication matrix dimensions.")
    return pd.DataFrame(rows)


def communication_pattern_matrix(result: CCCResult, *, mode: PatternMode = "outgoing", significant_only: bool = False) -> pd.DataFrame:
    """Return the group-by-pathway matrix used for pattern factorization."""

    if mode not in {"outgoing", "incoming", "all"}:
        raise ValueError("`mode` must be 'outgoing', 'incoming', or 'all'.")
    df = compute_pathway_communication(result, significant_only=significant_only)
    if mode == "outgoing":
        df = df.rename(columns={"source": "group"})
    elif mode == "incoming":
        df = df.rename(columns={"target": "group"})
    else:
        df = pd.concat([df.rename(columns={"source": "group"}), df.rename(columns={"target": "group"})], ignore_index=True)

    mat = df.pivot_table(index="group", columns="pathway", values="prob", aggfunc="sum", fill_value=0.0)
    mat = mat.reindex(index=result.groups, fill_value=0.0)
    mat = mat.loc[:, mat.sum(axis=0) > 0]
    return mat


def compute_pathway_similarity(
    result: CCCResult,
    *,
    similarity: NetworkSimilarity = "functional",
    significant_only: bool = False,
    min_prob: float = 0.0,
    thresh: float | None = None,
) -> pd.DataFrame:
    """Return pairwise signaling-pathway network similarity.

    `similarity="functional"` compares source-target matrices with cell-group
    identities preserved. `similarity="structural"` ignores cell-group labels by
    comparing sorted edge-weight profiles, approximating CellChat's structural
    network-similarity view in a Python-native deterministic form.
    """

    vectors, pathways = _pathway_network_vectors(result, similarity=similarity, significant_only=significant_only, min_prob=min_prob, thresh=thresh)
    if len(pathways) == 0:
        return pd.DataFrame(dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    denom = np.outer(norms, norms)
    sim = np.divide(vectors @ vectors.T, denom, out=np.zeros((len(pathways), len(pathways)), dtype=float), where=denom > 0)
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(sim, index=pathways, columns=pathways)


def compute_pathway_embedding(
    result: CCCResult,
    *,
    similarity: NetworkSimilarity = "functional",
    significant_only: bool = False,
    min_prob: float = 0.0,
    thresh: float | None = None,
    method: PathwayEmbeddingMethod = "auto",
    n_neighbors: int | None = None,
    min_dist: float = 0.3,
    random_state: int | None = 0,
    remove_isolates: bool = True,
) -> pd.DataFrame:
    """Embed signaling pathways by source-target network similarity.

    `method="umap"` mirrors CellChat's `netEmbedding` workflow using UMAP on
    the pathway-similarity profiles. `method="auto"` uses UMAP when enough
    pathways are available and falls back to deterministic MDS for tiny inputs.
    """

    sim = compute_pathway_similarity(result, similarity=similarity, significant_only=significant_only, min_prob=min_prob, thresh=thresh)
    sim = _remove_isolated_pathways(sim) if remove_isolates else sim
    if sim.empty:
        return pd.DataFrame(
            {
                "pathway": pd.Series(dtype=str),
                "dim1": pd.Series(dtype=float),
                "dim2": pd.Series(dtype=float),
                "prob": pd.Series(dtype=float),
                "count": pd.Series(dtype=int),
                "embedding_method": pd.Series(dtype=str),
            }
        )
    dist = np.sqrt(np.clip(2.0 - 2.0 * sim.to_numpy(dtype=float), 0.0, None))
    coords, method_used = _pathway_embedding_coords(
        sim,
        dist,
        method=method,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    summary = result.pathway_summary(significant_only=significant_only).set_index("pathway")
    frame = pd.DataFrame({"pathway": sim.index.astype(str), "dim1": coords[:, 0], "dim2": coords[:, 1]})
    frame["prob"] = frame["pathway"].map(summary["prob"]).fillna(0.0).astype(float)
    frame["count"] = frame["pathway"].map(summary["count"]).fillna(0).astype(int)
    frame["embedding_method"] = method_used
    return frame.sort_values("prob", ascending=False).reset_index(drop=True)


def compute_pathway_clusters(
    result: CCCResult,
    *,
    similarity: NetworkSimilarity = "functional",
    significant_only: bool = False,
    min_prob: float = 0.0,
    thresh: float | None = None,
    method: PathwayClusteringMethod = "spectral",
    n_clusters: int | None = None,
    embedding: pd.DataFrame | None = None,
    embedding_method: PathwayEmbeddingMethod = "auto",
    n_neighbors: int | None = None,
    min_dist: float = 0.3,
    random_state: int | None = 0,
    remove_isolates: bool = True,
) -> pd.DataFrame:
    """Classify signaling pathways by network similarity, like CellChat `netClustering`."""

    if method not in {"spectral", "kmeans"}:
        raise ValueError("`method` must be either 'spectral' or 'kmeans'.")
    sim = compute_pathway_similarity(result, similarity=similarity, significant_only=significant_only, min_prob=min_prob, thresh=thresh)
    sim = _remove_isolated_pathways(sim) if remove_isolates else sim
    if sim.empty:
        return pd.DataFrame(
            {
                "pathway": pd.Series(dtype=str),
                "cluster": pd.Series(dtype=int),
                "n_clusters": pd.Series(dtype=int),
                "cluster_method": pd.Series(dtype=str),
                "prob": pd.Series(dtype=float),
                "count": pd.Series(dtype=int),
            }
        )
    pathways = sim.index.astype(str).tolist()
    n = len(pathways)
    k = _resolve_cluster_count(sim, n_clusters)
    if k <= 1:
        labels = np.ones(n, dtype=int)
    elif k >= n:
        labels = np.arange(1, n + 1, dtype=int)
    elif method == "spectral":
        affinity = sim.to_numpy(dtype=float).copy()
        affinity = np.maximum(affinity, affinity.T)
        np.fill_diagonal(affinity, 1.0)
        model = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="kmeans", random_state=random_state)
        labels = model.fit_predict(affinity) + 1
    else:
        if embedding is None:
            embedding = compute_pathway_embedding(
                result,
                similarity=similarity,
                significant_only=significant_only,
                min_prob=min_prob,
                thresh=thresh,
                method=embedding_method,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
                remove_isolates=remove_isolates,
            )
        coords = embedding.set_index("pathway").reindex(pathways)[["dim1", "dim2"]].fillna(0.0).to_numpy(dtype=float)
        labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(coords) + 1

    summary = result.pathway_summary(significant_only=significant_only).set_index("pathway")
    frame = pd.DataFrame({"pathway": pathways, "cluster": labels.astype(int)})
    frame["n_clusters"] = int(k)
    frame["cluster_method"] = method
    frame["prob"] = frame["pathway"].map(summary["prob"]).fillna(0.0).astype(float)
    frame["count"] = frame["pathway"].map(summary["count"]).fillna(0).astype(int)
    return frame.sort_values(["cluster", "prob"], ascending=[True, False]).reset_index(drop=True)


def _component_stability(components: list[np.ndarray]) -> float:
    if len(components) < 2:
        return np.nan
    scores = []
    normalized = [_row_normalize(component) for component in components]
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            sim = normalized[i] @ normalized[j].T
            scores.append(float(np.mean(np.max(sim, axis=1))))
    return float(np.mean(scores)) if scores else np.nan


def _row_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _pathway_network_vectors(
    result: CCCResult,
    *,
    similarity: NetworkSimilarity,
    significant_only: bool,
    min_prob: float,
    thresh: float | None,
) -> tuple[np.ndarray, list[str]]:
    if similarity not in {"functional", "structural"}:
        raise ValueError("`similarity` must be either 'functional' or 'structural'.")
    if min_prob < 0:
        raise ValueError("`min_prob` must be non-negative.")
    if thresh is not None and not 0 <= thresh <= 0.25:
        raise ValueError("`thresh` must be between 0 and 0.25.")

    df = result.significant() if significant_only else result.interactions.copy()
    df = df[pd.to_numeric(df["prob"], errors="coerce").fillna(0.0) > min_prob].copy()
    if df.empty:
        return np.empty((0, len(result.groups) ** 2), dtype=float), []
    totals = df.groupby("pathway", observed=True)["prob"].sum().sort_values(ascending=False)
    pathways = [str(pathway) for pathway in totals.index]
    vectors = []
    for pathway in pathways:
        sub = df[df["pathway"].astype(str) == pathway]
        mat = sub.pivot_table(index="source", columns="target", values="prob", aggfunc="sum", fill_value=0.0)
        mat = mat.reindex(index=result.groups, columns=result.groups, fill_value=0.0)
        arr = mat.to_numpy(dtype=float).ravel()
        if thresh:
            arr = _trim_vector(arr, thresh)
        if similarity == "structural":
            arr = np.sort(arr)[::-1]
        vectors.append(arr)
    return np.vstack(vectors), pathways


def _trim_vector(values: np.ndarray, thresh: float) -> np.ndarray:
    out = values.copy()
    positive = out[out > 0]
    if positive.size == 0:
        return out
    cutoff = np.quantile(positive, thresh)
    out[out < cutoff] = 0.0
    return out


def _remove_isolated_pathways(sim: pd.DataFrame) -> pd.DataFrame:
    if sim.empty:
        return sim
    keep = sim.sum(axis=0) > 1.0 + 1e-12
    if keep.any():
        return sim.loc[keep, keep]
    return sim


def _pathway_embedding_coords(
    sim: pd.DataFrame,
    dist: np.ndarray,
    *,
    method: PathwayEmbeddingMethod,
    n_neighbors: int | None,
    min_dist: float,
    random_state: int | None,
) -> tuple[np.ndarray, str]:
    if method not in {"auto", "umap", "mds"}:
        raise ValueError("`method` must be one of: 'auto', 'umap', or 'mds'.")
    if min_dist <= 0:
        raise ValueError("`min_dist` must be positive.")
    n = sim.shape[0]
    if method == "mds" or n < 4:
        return _classical_mds(dist), "mds"
    if method in {"auto", "umap"}:
        try:
            return _umap_embedding(sim.to_numpy(dtype=float), n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state), "umap"
        except ImportError:
            if method == "umap":
                raise
    return _classical_mds(dist), "mds"


def _umap_embedding(values: np.ndarray, *, n_neighbors: int | None, min_dist: float, random_state: int | None) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - dependency comes through scanpy in normal installs
        raise ImportError("Install `umap-learn` to use `method='umap'` pathway embedding.") from exc
    n = values.shape[0]
    resolved_neighbors = int(np.ceil(np.sqrt(n)) + 1) if n_neighbors is None else int(n_neighbors)
    resolved_neighbors = min(max(resolved_neighbors, 2), max(n - 1, 2))
    model = umap.UMAP(
        n_neighbors=resolved_neighbors,
        n_components=2,
        metric="correlation",
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = model.fit_transform(values)
    return np.asarray(coords, dtype=float)


def _resolve_cluster_count(sim: pd.DataFrame, n_clusters: int | None) -> int:
    n = sim.shape[0]
    if n == 0:
        return 0
    if n_clusters is not None:
        if n_clusters < 1:
            raise ValueError("`n_clusters` must be positive.")
        return min(int(n_clusters), n)
    if n < 3:
        return 1
    return _infer_cluster_count(sim.to_numpy(dtype=float), max_clusters=min(n - 1, 10))


def _infer_cluster_count(affinity: np.ndarray, *, max_clusters: int) -> int:
    affinity = np.maximum(np.asarray(affinity, dtype=float), np.asarray(affinity, dtype=float).T)
    np.fill_diagonal(affinity, 1.0)
    degree = affinity.sum(axis=1)
    keep = degree > 0
    if keep.sum() < 3:
        return 1
    affinity = affinity[np.ix_(keep, keep)]
    degree = affinity.sum(axis=1)
    inv_sqrt = np.zeros_like(degree, dtype=float)
    inv_sqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    laplacian = np.eye(affinity.shape[0]) - (inv_sqrt[:, None] * affinity * inv_sqrt[None, :])
    eigvals = np.sort(np.linalg.eigvalsh(laplacian))
    limit = min(max_clusters + 1, len(eigvals))
    if limit <= 2:
        return 1
    gaps = np.diff(eigvals[:limit])
    if gaps.size == 0 or not np.isfinite(gaps).any():
        return 1
    return max(2, min(int(np.argmax(gaps[1:]) + 2) if gaps.size > 1 else 2, max_clusters))


def _classical_mds(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    if n == 1:
        return np.zeros((1, 2), dtype=float)
    if n == 2:
        half = float(dist[0, 1]) / 2.0
        return np.array([[-half, 0.0], [half, 0.0]], dtype=float)

    d2 = np.square(dist)
    centered = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centered @ d2 @ centered
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = eigvals > 0
    coords = eigvecs[:, keep][:, :2] * np.sqrt(eigvals[keep][:2])
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])), constant_values=0.0)
    return coords.astype(float)
