from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ESMC_300M_MODEL_NAME = "biohub/esmc-300m-2024-12"


def embed_proteins_esmc(
    proteins: pd.DataFrame,
    *,
    model_name: str = ESMC_300M_MODEL_NAME,
    batch_size: int = 8,
    device: str = "auto",
    pooling: str = "mean",
    cache_dir: str | Path | None = None,
    model_revision: str | None = None,
    backend: str = "auto",
    fake_dim: int = 32,
    max_sequence_length: int | None = 2048,
) -> pd.DataFrame:
    """Embed protein sequences with ESMC or a deterministic hash backend.

    The default `backend="auto"` loads ESMC-300M through the official `esm`
    package and therefore requires the experimental `predict` extra. Tests and
    toy examples can use `backend="hash"` to avoid downloads while preserving
    cache semantics.
    """

    _validate_protein_frame(proteins)
    if backend not in {"auto", "esmc", "hash"}:
        raise ValueError("`backend` must be one of: auto, esmc, hash.")
    if pooling != "mean":
        raise ValueError("Only mean pooling is supported.")
    cache_root = Path(cache_dir) if cache_dir is not None else None
    rows = []
    resolved_backend = _resolve_embedding_backend(backend, model_name)
    model_bundle = None
    for chunk in _chunks(proteins.reset_index(drop=True), batch_size):
        missing = []
        for row in chunk.itertuples(index=False):
            seq_hash = _sequence_hash(row.protein_sequence)
            cached = _read_cached_embedding(
                cache_root,
                seq_hash,
                model_name=model_name,
                model_revision=model_revision,
                pooling=pooling,
                backend=resolved_backend,
                max_sequence_length=max_sequence_length,
            )
            if cached is None:
                missing.append(row)
            else:
                rows.append(_embedding_row(row, seq_hash, cached, model_name=model_name, model_revision=model_revision, pooling=pooling, backend=resolved_backend))
        if missing:
            if resolved_backend == "hash":
                embeddings = [_hash_embedding(row.protein_sequence, dim=fake_dim) for row in missing]
            else:
                if model_bundle is None:
                    model_bundle = _load_embedding_model(model_name=model_name, model_revision=model_revision, device=device)
                embeddings = _embed_with_model(missing, model_bundle=model_bundle, max_sequence_length=max_sequence_length)
            for row, emb in zip(missing, embeddings, strict=True):
                seq_hash = _sequence_hash(row.protein_sequence)
                _write_cached_embedding(
                    cache_root,
                    seq_hash,
                    emb,
                    model_name=model_name,
                    model_revision=model_revision,
                    pooling=pooling,
                    backend=resolved_backend,
                    max_sequence_length=max_sequence_length,
                )
                rows.append(_embedding_row(row, seq_hash, emb, model_name=model_name, model_revision=model_revision, pooling=pooling, backend=resolved_backend))
    return pd.DataFrame(rows)


def _validate_protein_frame(proteins: pd.DataFrame) -> None:
    missing = [col for col in ("gene_id", "protein_id", "protein_sequence") if col not in proteins.columns]
    if missing:
        raise ValueError(f"Protein table is missing required columns: {missing}")


def _chunks(frame: pd.DataFrame, batch_size: int) -> Sequence[pd.DataFrame]:
    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive.")
    return [frame.iloc[i : i + batch_size] for i in range(0, len(frame), batch_size)]


def _sequence_hash(sequence: str) -> str:
    return hashlib.sha256(str(sequence).encode("utf-8")).hexdigest()


def _resolve_embedding_backend(backend: str, model_name: str) -> str:
    return "hash" if backend == "hash" or model_name in {"hash", "fake"} else "esmc"


def _cache_path(
    cache_root: Path | None,
    seq_hash: str,
    *,
    model_name: str,
    model_revision: str | None,
    pooling: str,
    backend: str,
    max_sequence_length: int | None,
) -> Path | None:
    if cache_root is None:
        return None
    model_key = hashlib.sha1(f"{model_name}|{model_revision or ''}|{pooling}|{backend}|maxlen={max_sequence_length or ''}".encode("utf-8")).hexdigest()[:16]
    return cache_root / model_key / f"{seq_hash}.npz"


def _read_cached_embedding(
    cache_root: Path | None,
    seq_hash: str,
    *,
    model_name: str,
    model_revision: str | None,
    pooling: str,
    backend: str,
    max_sequence_length: int | None,
) -> np.ndarray | None:
    path = _cache_path(
        cache_root,
        seq_hash,
        model_name=model_name,
        model_revision=model_revision,
        pooling=pooling,
        backend=backend,
        max_sequence_length=max_sequence_length,
    )
    if path is None or not path.exists():
        return None
    return np.load(path)["embedding"]


def _write_cached_embedding(
    cache_root: Path | None,
    seq_hash: str,
    embedding: np.ndarray,
    *,
    model_name: str,
    model_revision: str | None,
    pooling: str,
    backend: str,
    max_sequence_length: int | None,
) -> None:
    path = _cache_path(
        cache_root,
        seq_hash,
        model_name=model_name,
        model_revision=model_revision,
        pooling=pooling,
        backend=backend,
        max_sequence_length=max_sequence_length,
    )
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": model_name,
        "model_revision": model_revision or "",
        "pooling": pooling,
        "sequence_hash": seq_hash,
        "embedding_backend": backend,
        "max_sequence_length": max_sequence_length,
    }
    np.savez_compressed(path, embedding=np.asarray(embedding, dtype=np.float32), metadata=json.dumps(metadata))


def _embedding_row(row, seq_hash: str, embedding: np.ndarray, *, model_name: str, model_revision: str | None, pooling: str, backend: str = "esmc") -> dict[str, object]:
    out = {
        "gene_id": str(row.gene_id),
        "protein_id": str(row.protein_id),
        "sequence_hash": seq_hash,
        "embedding": np.asarray(embedding, dtype=np.float32),
        "model_name": model_name,
        "model_revision": model_revision or "",
        "embedding_backend": backend,
        "pooling": pooling,
        "sequence_length": int(len(str(row.protein_sequence))),
    }
    if hasattr(row, "species"):
        out["species"] = str(row.species)
    return out


def _hash_embedding(sequence: str, *, dim: int) -> np.ndarray:
    digest = hashlib.sha256(str(sequence).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def _load_embedding_model(*, model_name: str, model_revision: str | None, device: str):
    if _is_esmc_300m(model_name):
        return _load_esm_package_model(device=device)
    return _load_transformers_model(model_name=model_name, model_revision=model_revision, device=device)


def _is_esmc_300m(model_name: str) -> bool:
    return str(model_name).lower() in {"biohub/esmc-300m-2024-12", "esmc_300m", "esmc-300m-2024-12"}


def _load_esm_package_model(*, device: str):
    try:
        import torch
        from esm.models.esmc import ESMC, _BatchedESMProteinTensor
        from esm.sdk.api import LogitsConfig
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install DB-free prediction support with `pyccc[predict]` to embed proteins with ESMC.") from exc

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    model = ESMC.from_pretrained("esmc_300m").to(resolved_device)
    model.eval()
    return {"backend": "esm_package", "torch": torch, "model": model, "device": resolved_device, "BatchedTensor": _BatchedESMProteinTensor, "LogitsConfig": LogitsConfig}


def _load_transformers_model(*, model_name: str, model_revision: str | None, device: str):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install DB-free prediction support with `pyccc[predict]` to embed proteins with ESMC.") from exc

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, revision=model_revision, trust_remote_code=True).to(resolved_device)
    model.eval()
    return {"backend": "transformers", "torch": torch, "tokenizer": tokenizer, "model": model, "device": resolved_device}


def _embed_with_model(rows, *, model_bundle, max_sequence_length: int | None) -> list[np.ndarray]:
    if model_bundle.get("backend") == "esm_package":
        return _embed_with_esm_package(rows, model_bundle=model_bundle, max_sequence_length=max_sequence_length)
    return _embed_with_transformers(rows, model_bundle=model_bundle, max_sequence_length=max_sequence_length)


def _embed_with_esm_package(rows, *, model_bundle, max_sequence_length: int | None) -> list[np.ndarray]:
    torch = model_bundle["torch"]
    model = model_bundle["model"]
    BatchedTensor = model_bundle["BatchedTensor"]
    LogitsConfig = model_bundle["LogitsConfig"]
    sequences = []
    lengths = []
    for row in rows:
        sequence = str(row.protein_sequence)
        if max_sequence_length is not None:
            sequence = sequence[: int(max_sequence_length)]
        sequences.append(sequence)
        lengths.append(len(sequence))
    with torch.no_grad():
        tokens = model._tokenize(sequences)
        output = model.logits(BatchedTensor(sequence=tokens), LogitsConfig(sequence=True, return_embeddings=True))
        hidden = output.embeddings.detach().cpu().numpy()
    embeddings = []
    for idx, length in enumerate(lengths):
        arr = hidden[idx]
        if arr.shape[0] >= length + 2:
            arr = arr[1 : length + 1]
        elif arr.shape[0] > length:
            arr = arr[:length]
        embeddings.append(arr.mean(axis=0).astype(np.float32))
    return embeddings


def _embed_with_transformers(rows, *, model_bundle, max_sequence_length: int | None) -> list[np.ndarray]:
    torch = model_bundle["torch"]
    tokenizer = model_bundle["tokenizer"]
    model = model_bundle["model"]
    resolved_device = model_bundle["device"]
    sequences = [" ".join(str(row.protein_sequence)) for row in rows]
    with torch.no_grad():
        tokenizer_kwargs = {"return_tensors": "pt", "padding": True, "truncation": True}
        if max_sequence_length is not None:
            tokenizer_kwargs["max_length"] = int(max_sequence_length)
        encoded = tokenizer(sequences, **tokenizer_kwargs).to(resolved_device)
        output = model(**encoded)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, tuple):
            hidden = output[0]
        if hidden is None:
            raise RuntimeError("The ESMC model output did not expose `last_hidden_state`.")
        mask = encoded.get("attention_mask")
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask_f = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
    return [item.detach().cpu().numpy().astype(np.float32) for item in pooled]
