from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .lr_resources import load_training_lr_resources, training_lr_to_cellchatdb
from .sequence import load_protein_fasta


@dataclass(frozen=True)
class LRTrainingTable:
    """Normalized LR interactions plus optional protein sequence table."""

    interactions: pd.DataFrame
    proteins: pd.DataFrame
    positive_evidence: set[str]

    def to_cellchatdb(self, *, name: str = "training_lr"):
        return training_lr_to_cellchatdb(self.interactions, name=name)


def build_lr_training_table(
    resources: pd.DataFrame | Sequence[Mapping[str, object] | str | Path],
    *,
    protein_fasta_by_species: Mapping[str, str | Path] | None = None,
    positive_evidence: set[str] | None = None,
    drop_complexes: str = "partial",
) -> LRTrainingTable:
    """Build a normalized cross-species LR table for predictor training."""

    if drop_complexes not in {"partial", "none"}:
        raise ValueError("`drop_complexes` must be one of: partial, none.")
    interactions = resources.copy() if isinstance(resources, pd.DataFrame) else load_training_lr_resources(resources)
    positive_evidence = set(positive_evidence or {"curated_direct", "curated_inferred"})
    interactions["is_positive_label"] = [
        bool(set(_split_semicolon_values(value)) & positive_evidence)
        for value in interactions["evidence_type"]
    ]
    proteins = _load_species_proteins(protein_fasta_by_species or {})
    if not proteins.empty:
        interactions = _attach_sequences(interactions, proteins)
    if drop_complexes == "partial":
        interactions = _drop_partial_complexes(interactions)
    return LRTrainingTable(interactions.reset_index(drop=True), proteins.reset_index(drop=True), set(positive_evidence))


def _load_species_proteins(protein_fasta_by_species: Mapping[str, str | Path]) -> pd.DataFrame:
    frames = []
    for species, path in protein_fasta_by_species.items():
        frame = load_protein_fasta(path)
        frame["species"] = str(species)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["gene_id", "protein_id", "protein_sequence", "species"])
    return pd.concat(frames, ignore_index=True)


def _attach_sequences(interactions: pd.DataFrame, proteins: pd.DataFrame) -> pd.DataFrame:
    out = interactions.copy()
    lookup = {
        (str(row.species), str(row.gene_id)): row
        for row in proteins.itertuples(index=False)
    }
    for side in ("ligand", "receptor"):
        gene_col = f"{side}_gene"
        out[f"{side}_protein_id"] = [
            str(getattr(lookup.get((str(species), str(gene)), None), "protein_id", ""))
            for species, gene in zip(out["species"], out[gene_col], strict=True)
        ]
        out[f"{side}_sequence"] = [
            str(getattr(lookup.get((str(species), str(gene)), None), "protein_sequence", ""))
            for species, gene in zip(out["species"], out[gene_col], strict=True)
        ]
    return out


def _drop_partial_complexes(interactions: pd.DataFrame) -> pd.DataFrame:
    out = interactions.copy()
    for col in (
        "ligand_sequence",
        "receptor_sequence",
        "ligand_complex_id",
        "receptor_complex_id",
        "complex_subunit_gene",
        "complex_required_subunits",
    ):
        if col not in out.columns:
            out[col] = ""
    complex_mask = (
        _complex_name_mask(out["ligand_gene"])
        | _complex_name_mask(out["receptor_gene"])
        | (out["ligand_complex_id"].astype(str).str.strip() != "")
        | (out["receptor_complex_id"].astype(str).str.strip() != "")
        | (out["complex_subunit_gene"].astype(str).str.strip() != "")
        | (_required_subunit_count(out["complex_required_subunits"]) > 1)
    )
    incomplete_subunit_metadata = _required_subunit_count(out["complex_required_subunits"]) > _observed_subunit_count(out["complex_subunit_gene"])
    if "ligand_sequence" not in out.columns:
        out["ligand_sequence"] = ""
    if "receptor_sequence" not in out.columns:
        out["receptor_sequence"] = ""
    missing_sequence = (out["ligand_sequence"].astype(str) == "") | (out["receptor_sequence"].astype(str) == "")
    return out[~((complex_mask & missing_sequence) | incomplete_subunit_metadata)].copy()


def _complex_name_mask(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.contains(r"[_|]", regex=True)


def _required_subunit_count(values: pd.Series) -> pd.Series:
    return values.map(_max_int_token).astype(int)


def _observed_subunit_count(values: pd.Series) -> pd.Series:
    return values.map(lambda value: len(_split_complex_subunits(value))).astype(int)


def _max_int_token(value: object) -> int:
    counts = []
    for token in _split_semicolon_values(value):
        try:
            counts.append(int(float(token)))
        except ValueError:
            continue
    return max(counts) if counts else 0


def _split_complex_subunits(value: object) -> list[str]:
    text = str(value).replace("|", ";").replace(",", ";")
    return [part.strip() for part in text.split(";") if part.strip()]


def _split_semicolon_values(value: object) -> list[str]:
    text = str(value).replace("|", ";")
    return [part.strip() for part in text.split(";") if part.strip()]
