from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .database import CellChatDB

REQUIRED_TRAINING_LR_COLUMNS = (
    "ligand_gene",
    "receptor_gene",
    "species",
    "taxon_id",
    "resource",
    "evidence_type",
    "annotation",
    "pathway",
)

OPTIONAL_TRAINING_LR_COLUMNS = (
    "ligand_protein_id",
    "receptor_protein_id",
    "ligand_sequence",
    "receptor_sequence",
    "ligand_role",
    "receptor_role",
    "ligand_complex_id",
    "receptor_complex_id",
    "complex_subunit_gene",
    "complex_required_subunits",
    "pmid",
    "source_url",
    "curation_type",
    "directed",
    "confidence_original",
    "license",
)

NORMALIZED_TRAINING_LR_COLUMNS = REQUIRED_TRAINING_LR_COLUMNS + OPTIONAL_TRAINING_LR_COLUMNS + (
    "support_count",
    "support_resources",
)

SUPPORTED_LR_SCHEMAS = {
    "cellchat",
    "omnipath",
    "cellphonedb",
    "flyphonedb2",
    "plantphonedb",
    "plantcellchat",
    "generic",
}

SCHEMA_COLUMN_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "generic": {
        "ligand_gene": ("ligand_gene", "ligand", "ligand_symbol", "source_genesymbol", "source_gene", "source", "gene_a"),
        "receptor_gene": ("receptor_gene", "receptor", "receptor_symbol", "target_genesymbol", "target_gene", "target", "gene_b"),
        "pathway": ("pathway", "pathway_name", "signaling_pathway", "classification"),
        "annotation": ("annotation", "category", "source_database", "database", "resource", "resources", "sources"),
        "evidence": ("evidence", "reference", "references", "pmid", "pubmed"),
        "confidence_original": ("confidence_original", "confidence", "score", "probability"),
    },
    "cellchat": {
        "ligand_gene": ("ligand", "ligand_gene", "source_genesymbol"),
        "receptor_gene": ("receptor", "receptor_gene", "target_genesymbol"),
        "pathway": ("pathway_name", "pathway", "signaling"),
        "annotation": ("annotation", "category"),
        "evidence": ("evidence", "references", "pmid"),
    },
    "omnipath": {
        "ligand_gene": ("source_genesymbol", "genesymbol_intercell_source", "source", "ligand", "ligand_gene"),
        "receptor_gene": ("target_genesymbol", "genesymbol_intercell_target", "target", "receptor", "receptor_gene"),
        "pathway": ("pathway", "category_intercell_source", "sources", "resources", "databases"),
        "annotation": ("annotation", "category_intercell_source", "sources", "resources", "databases"),
        "evidence": ("references", "references_stripped", "pmid"),
        "confidence_original": ("consensus_score", "curation_effort", "score"),
    },
    "cellphonedb": {
        "ligand_gene": ("ligand_gene", "ligand", "gene_a", "partner_a", "source_genesymbol"),
        "receptor_gene": ("receptor_gene", "receptor", "gene_b", "partner_b", "target_genesymbol"),
        "pathway": ("pathway", "classification", "annotation_strategy"),
        "annotation": ("annotation", "classification", "secreted", "receptor"),
        "evidence": ("evidence", "references", "pmid"),
    },
    "flyphonedb2": {
        "ligand_gene": ("ligand_gene", "ligand", "source", "source_gene"),
        "receptor_gene": ("receptor_gene", "receptor", "target", "target_gene"),
        "pathway": ("pathway", "pathway_name", "family"),
        "annotation": ("annotation", "category", "interaction_type"),
        "evidence": ("evidence", "pmid", "reference", "references"),
    },
    "plantphonedb": {
        "ligand_gene": ("ligand_gene", "ligand", "Ligand", "source_gene", "source"),
        "receptor_gene": ("receptor_gene", "receptor", "Receptor", "target_gene", "target"),
        "pathway": ("pathway", "Pathway", "category", "classification"),
        "annotation": ("annotation", "Annotation", "category", "classification"),
        "evidence": ("evidence", "reference", "Reference", "pmid"),
    },
    "plantcellchat": {
        "ligand_gene": ("ligand_gene", "ligand", "source", "source_gene"),
        "receptor_gene": ("receptor_gene", "receptor", "target", "target_gene"),
        "pathway": ("pathway_name", "pathway", "signaling"),
        "annotation": ("annotation", "classification", "category"),
        "evidence": ("evidence", "reference", "references", "pmid"),
    },
}


def load_training_lr_resources(resources: Sequence[Mapping[str, object] | str | Path], *, strict: bool = True) -> pd.DataFrame:
    """Load local LR resource tables into the normalized training schema.

    Each resource item is a mapping with at least `path` or `frame`, `schema`,
    `species`, and `taxon_id`. Network fetching is intentionally out of scope;
    callers provide local TSV/CSV/Parquet exports.
    """

    frames = [_normalize_one_resource(spec, strict=strict) for spec in resources]
    if not frames:
        raise ValueError("At least one LR resource is required.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[(combined["ligand_gene"] != "") & (combined["receptor_gene"] != "")]
    if combined.empty:
        raise ValueError("No valid ligand-receptor rows remain after normalization.")
    return _deduplicate_training_lr(combined)


def training_lr_to_cellchatdb(training_lr: pd.DataFrame, *, name: str = "training_lr") -> CellChatDB:
    """Convert a normalized training LR table into a CellChatDB-compatible object."""

    required = [col for col in REQUIRED_TRAINING_LR_COLUMNS if col not in training_lr.columns]
    if required:
        raise ValueError(f"Normalized training LR table is missing columns: {required}")
    lr = pd.DataFrame(
        {
            "ligand": training_lr["ligand_gene"].astype(str),
            "receptor": training_lr["receptor_gene"].astype(str),
            "pathway": training_lr["pathway"].fillna("unknown").astype(str),
            "annotation": training_lr["annotation"].fillna("").astype(str),
            "evidence": training_lr["evidence_type"].fillna("").astype(str),
        }
    )
    for col in ("resource", "support_count", "support_resources", "pmid", "source_url", "license"):
        if col in training_lr.columns:
            lr[col] = training_lr[col].fillna("").astype(str)
    return CellChatDB(lr, name=name, metadata={"training_lr": training_lr.copy()})


def _normalize_one_resource(spec: Mapping[str, object] | str | Path, *, strict: bool) -> pd.DataFrame:
    if isinstance(spec, (str, Path)):
        spec = {"path": spec, "schema": "generic"}
    schema = str(spec.get("schema", "generic")).lower()
    if schema not in SUPPORTED_LR_SCHEMAS:
        supported = ", ".join(sorted(SUPPORTED_LR_SCHEMAS))
        raise ValueError(f"Unsupported LR schema `{schema}`. Supported schemas: {supported}.")
    frame = _resource_frame(spec)
    aliases = {**SCHEMA_COLUMN_ALIASES["generic"], **SCHEMA_COLUMN_ALIASES.get(schema, {})}
    out = pd.DataFrame(index=frame.index)
    out["ligand_gene"] = _column_or_default(frame, aliases["ligand_gene"], required=True, strict=strict)
    out["receptor_gene"] = _column_or_default(frame, aliases["receptor_gene"], required=True, strict=strict)
    out["species"] = str(spec.get("species", "")).strip()
    out["taxon_id"] = str(spec.get("taxon_id", "")).strip()
    out["resource"] = str(spec.get("resource", schema)).strip() or schema
    out["evidence_type"] = str(spec.get("evidence_type", _default_evidence_type(schema))).strip()
    out["annotation"] = _column_or_default(frame, aliases.get("annotation", ()), default=str(spec.get("annotation", "")))
    out["pathway"] = _column_or_default(frame, aliases.get("pathway", ()), default=str(spec.get("pathway", "unknown")))

    for col in OPTIONAL_TRAINING_LR_COLUMNS:
        if col in {"ligand_sequence", "receptor_sequence"}:
            out[col] = _column_or_default(frame, (col,), default="")
        elif col == "pmid":
            out[col] = _column_or_default(frame, ("pmid", "pubmed", "PMID"), default="")
        elif col == "confidence_original":
            out[col] = _column_or_default(frame, aliases.get("confidence_original", (col,)), default="")
        elif col == "license":
            out[col] = str(spec.get("license", "")).strip()
        elif col == "source_url":
            out[col] = str(spec.get("source_url", "")).strip()
        elif col == "directed":
            out[col] = str(spec.get("directed", True))
        elif col == "curation_type":
            out[col] = str(spec.get("curation_type", schema))
        else:
            out[col] = _column_or_default(frame, (col,), default="")

    evidence = _column_or_default(frame, aliases.get("evidence", ()), default="")
    out["pmid"] = out["pmid"].where(out["pmid"].astype(str) != "", evidence)
    out = _clean_normalized_lr(out)
    if strict:
        _validate_normalized_lr(out, schema=schema)
    return out


def _resource_frame(spec: Mapping[str, object]) -> pd.DataFrame:
    if "frame" in spec:
        frame = spec["frame"]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("`frame` resource entries must contain a pandas DataFrame.")
        return frame.copy()
    if "path" not in spec:
        raise ValueError("Each LR resource must provide either `path` or `frame`.")
    path = Path(str(spec["path"]))
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    sep = spec.get("sep")
    if sep is None:
        sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def _column_or_default(
    frame: pd.DataFrame,
    aliases: Sequence[str],
    *,
    required: bool = False,
    strict: bool = False,
    default: str = "",
) -> pd.Series:
    for col in aliases:
        if col in frame.columns:
            return frame[col].fillna("").astype(str)
    if required and strict:
        raise ValueError(f"Could not find any required column among aliases: {list(aliases)}")
    return pd.Series([default] * len(frame), index=frame.index, dtype=str)


def _clean_normalized_lr(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        out[col] = out[col].fillna("").astype(str).str.strip()
    out["pathway"] = out["pathway"].replace("", "unknown")
    out["annotation"] = out["annotation"].fillna("").astype(str)
    out["support_count"] = 1
    out["support_resources"] = out["resource"]
    return out


def _validate_normalized_lr(frame: pd.DataFrame, *, schema: str) -> None:
    missing = [col for col in REQUIRED_TRAINING_LR_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Schema `{schema}` failed to produce required columns: {missing}")
    empty = [col for col in ("ligand_gene", "receptor_gene", "species", "taxon_id", "resource") if (frame[col].astype(str).str.strip() == "").any()]
    if empty:
        raise ValueError(f"Schema `{schema}` contains empty required values in columns: {empty}")
    if (frame["ligand_gene"].astype(str) == frame["receptor_gene"].astype(str)).any():
        raise ValueError(f"Schema `{schema}` contains self ligand-receptor rows; check interaction direction.")


def _deduplicate_training_lr(frame: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["species", "taxon_id", "ligand_gene", "receptor_gene", "pathway"]
    rows = []
    for _, sub in frame.groupby(group_cols, sort=False, dropna=False):
        row = sub.iloc[0].copy()
        support_resources = sorted({str(x) for x in sub["resource"].astype(str) if str(x)})
        row["support_count"] = int(len(sub))
        row["support_resources"] = ";".join(support_resources)
        row["resource"] = support_resources[0] if support_resources else str(row["resource"])
        rows.append(row)
    out = pd.DataFrame(rows).reset_index(drop=True)
    for col in NORMALIZED_TRAINING_LR_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col != "support_count" else 1
    return out[list(NORMALIZED_TRAINING_LR_COLUMNS)]


def _default_evidence_type(schema: str) -> str:
    if schema in {"cellchat", "omnipath", "cellphonedb", "flyphonedb2", "plantphonedb", "plantcellchat"}:
        return "curated_direct"
    return "user_supplied"
