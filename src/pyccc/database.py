from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

REQUIRED_LR_COLUMNS = ("ligand", "receptor")
LR_COLUMN_ALIASES = {
    "ligand": ("ligand", "ligand_symbol", "ligand_gene", "source_genesymbol", "source_gene", "source"),
    "receptor": ("receptor", "receptor_symbol", "receptor_gene", "target_genesymbol", "target_gene", "target"),
    "pathway": ("pathway", "pathway_name", "signaling_pathway", "signaling"),
    "annotation": ("annotation", "category", "source_database", "database", "resource", "resources", "sources"),
    "evidence": ("evidence", "reference", "references", "pmid", "pubmed"),
}
CELLCHATDB_URLS = {
    "human": "https://raw.githubusercontent.com/jinworks/CellChat/main/data/CellChatDB.human.rda",
    "mouse": "https://raw.githubusercontent.com/jinworks/CellChat/main/data/CellChatDB.mouse.rda",
    "zebrafish": "https://raw.githubusercontent.com/jinworks/CellChat/main/data/CellChatDB.zebrafish.rda",
}
OMNIPATH_ORGANISM_ALIASES = {
    "human": "human",
    "homo sapiens": "human",
    "hsapiens": "human",
    "9606": "human",
    "mouse": "mouse",
    "mus musculus": "mouse",
    "mmusculus": "mouse",
    "10090": "mouse",
    "rat": "rat",
    "rattus norvegicus": "rat",
    "rnorvegicus": "rat",
    "10116": "rat",
}


@dataclass(frozen=True)
class CellChatDB:
    """Ligand-receptor table used by pyccc.

    Required columns are `ligand` and `receptor`. Recommended columns are
    `pathway`, `annotation`, and `evidence`.
    """

    interactions: pd.DataFrame
    name: str = "custom"
    metadata: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "interactions", normalize_lr_table(self.interactions))
        object.__setattr__(self, "metadata", dict(self.metadata))


def normalize_lr_table(lr_table: pd.DataFrame, *, column_map: Mapping[str, str] | None = None) -> pd.DataFrame:
    lr = _canonicalize_lr_columns(lr_table, column_map=column_map)
    missing = [col for col in REQUIRED_LR_COLUMNS if col not in lr.columns]
    if missing:
        raise ValueError(f"LR table is missing required columns: {missing}")

    lr["ligand"] = lr["ligand"].astype(str).str.strip()
    lr["receptor"] = lr["receptor"].astype(str).str.strip()
    lr = lr[(lr["ligand"] != "") & (lr["receptor"] != "")]

    if "pathway" not in lr.columns:
        lr["pathway"] = "unknown"
    lr["pathway"] = lr["pathway"].fillna("unknown").astype(str)

    for col in ("annotation", "evidence"):
        if col not in lr.columns:
            lr[col] = ""
        lr[col] = lr[col].fillna("").astype(str)

    lr = lr.drop_duplicates(["ligand", "receptor", "pathway"]).reset_index(drop=True)
    if lr.empty:
        raise ValueError("LR table has no valid ligand-receptor rows after filtering.")
    return lr


def load_lr_table(
    path: str | Path,
    *,
    name: str | None = None,
    sep: str | None = None,
    ligand_col: str | None = None,
    receptor_col: str | None = None,
    pathway_col: str | None = None,
    annotation_col: str | None = None,
    evidence_col: str | None = None,
) -> CellChatDB:
    """Load a ligand-receptor table from CSV/TSV/Parquet.

    The table should contain `ligand` and `receptor`; common aliases such as
    `source_genesymbol` and `target_genesymbol` are detected automatically.
    Files ending in `.tsv` or `.txt` default to tab separation; other text
    files default to comma.
    """

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        lr = pd.read_parquet(path)
    else:
        if sep is None:
            sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        lr = pd.read_csv(path, sep=sep)
    column_map = {
        "ligand": ligand_col,
        "receptor": receptor_col,
        "pathway": pathway_col,
        "annotation": annotation_col,
        "evidence": evidence_col,
    }
    return CellChatDB(normalize_lr_table(lr, column_map={k: v for k, v in column_map.items() if v}), name=name or path.stem)


def load_cellchatdb(
    species: str = "human",
    *,
    category: str | Sequence[str] | None = None,
    pathways: str | Sequence[str] | None = None,
    path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    proxy: str | None = None,
) -> CellChatDB:
    """Load an official CellChat ligand-receptor database.

    When `path` is not provided, the requested `.rda` file is downloaded from
    the CellChat GitHub repository and cached locally. Supported species are
    `human`, `mouse`, and `zebrafish`. Use `category` to select CellChatDB
    annotation classes such as `"Secreted Signaling"`.
    """

    species = species.lower()
    if species not in CELLCHATDB_URLS:
        supported = ", ".join(sorted(CELLCHATDB_URLS))
        raise ValueError(f"`species` must be one of: {supported}.")

    if path is None:
        cache_root = Path(cache_dir or os.environ.get("PYCCC_CACHE_DIR", Path.home() / ".cache" / "pyccc"))
        path = cache_root / "cellchatdb" / f"CellChatDB.{species}.rda"
        if force_download or not Path(path).exists():
            _download_file(CELLCHATDB_URLS[species], Path(path), proxy=proxy)
    else:
        path = Path(path)

    try:
        import rdata
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError("Install `rdata` to read CellChat `.rda` database files.") from exc

    objects = rdata.read_rda(path)
    key = f"CellChatDB.{species}"
    if key not in objects:
        if len(objects) != 1:
            raise ValueError(f"Could not find `{key}` in {path}.")
        key = next(iter(objects))
    db = _cellchatdb_from_object(objects[key], species=species)
    return filter_lr_table(db, annotation=category, pathways=pathways)


def load_omnipath_interactions(
    organism: str = "human",
    *,
    resources: Sequence[str] | str | None = None,
    include: Sequence[str] | str | None = None,
    transmitter_categories: Sequence[str] | str | None = ("ligand",),
    receiver_categories: Sequence[str] | str | None = ("receptor",),
    interactions_params: Mapping[str, object] | None = None,
    transmitter_params: Mapping[str, object] | None = None,
    receiver_params: Mapping[str, object] | None = None,
    pathway_mode: str = "category",
    pathway_name: str = "OmniPath",
    name: str | None = None,
) -> CellChatDB:
    """Load ligand-receptor interactions from OmniPath's intercell network.

    This requires the optional `omnipath` dependency. Install from GitHub with
    `pyccc[omnipath]` or add `omnipath` to the local `uv sync` extras.
    Supported OmniPath organisms are `human`, `mouse`, and `rat`.
    """

    organism_resolved = _resolve_omnipath_organism(organism)
    try:
        from omnipath.interactions import import_intercell_network
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install OmniPath support with `pyccc[omnipath]` or `uv sync --extra omnipath`.") from exc

    interaction_kwargs = dict(interactions_params or {})
    interaction_kwargs.setdefault("organism", organism_resolved)
    interaction_kwargs.setdefault("genesymbols", True)
    if resources is not None:
        interaction_kwargs["resources"] = _as_list(resources)

    transmitter_kwargs = dict(transmitter_params or {})
    receiver_kwargs = dict(receiver_params or {})
    if transmitter_categories is not None and "categories" not in transmitter_kwargs:
        transmitter_kwargs["categories"] = _as_list(transmitter_categories)
    if receiver_categories is not None and "categories" not in receiver_kwargs:
        receiver_kwargs["categories"] = _as_list(receiver_categories)

    kwargs = {
        "interactions_params": interaction_kwargs,
        "transmitter_params": transmitter_kwargs,
        "receiver_params": receiver_kwargs,
    }
    if include is not None:
        kwargs["include"] = include
    raw = import_intercell_network(**kwargs)
    lr = _omnipath_frame_to_lr(raw, pathway_mode=pathway_mode, pathway_name=pathway_name)
    metadata = {
        "omnipath_raw": raw.copy(),
        "omnipath_params": pd.DataFrame(
            [
                {"key": "organism", "value": organism_resolved},
                {"key": "resources", "value": ";".join(_as_list(resources)) if resources is not None else ""},
                {"key": "include", "value": ";".join(_as_list(include)) if include is not None else ""},
            ]
        ),
    }
    return CellChatDB(lr, name=name or f"omnipath_{organism_resolved}", metadata=metadata)


def filter_lr_table(
    lr_table: CellChatDB | pd.DataFrame,
    *,
    annotation: str | Sequence[str] | None = None,
    pathways: str | Sequence[str] | None = None,
    resources: str | Sequence[str] | None = None,
    name: str | None = None,
) -> CellChatDB:
    """Filter an LR table by annotation, pathway, or resource/source labels."""

    if isinstance(lr_table, CellChatDB):
        lr = lr_table.interactions.copy()
        metadata = lr_table.metadata
        default_name = lr_table.name
    else:
        lr = normalize_lr_table(lr_table)
        metadata = {}
        default_name = "custom"
    if annotation is not None:
        values = set(_as_list(annotation))
        lr = lr[lr["annotation"].isin(values)]
    if pathways is not None:
        values = set(_as_list(pathways))
        lr = lr[lr["pathway"].isin(values)]
    if resources is not None:
        values = set(_as_list(resources))
        resource_cols = [col for col in ("resources", "sources", "database", "databases", "omnipath_sources") if col in lr.columns]
        if not resource_cols:
            raise ValueError("No resource/source column is available for filtering.")
        mask = pd.Series(False, index=lr.index)
        for col in resource_cols:
            mask |= lr[col].astype(str).map(lambda item: bool(values.intersection(_split_resource_tokens(item))))
        lr = lr[mask]
    if lr.empty:
        raise ValueError("No ligand-receptor rows remain after filtering.")
    return CellChatDB(lr, name=name or default_name, metadata=metadata)


def _cellchatdb_from_object(obj: dict, *, species: str) -> CellChatDB:
    interaction = obj["interaction"].copy()
    complex_table = obj.get("complex", pd.DataFrame())
    cofactor_table = obj.get("cofactor", pd.DataFrame())

    lr = interaction.rename(columns={"pathway_name": "pathway"}).copy()
    lr["cellchat_ligand"] = lr["ligand"].astype(str)
    lr["cellchat_receptor"] = lr["receptor"].astype(str)
    lr["ligand"] = lr["cellchat_ligand"].map(lambda value: _expand_cellchat_complex(value, complex_table))
    lr["receptor"] = lr["cellchat_receptor"].map(lambda value: _expand_cellchat_complex(value, complex_table))
    for col in ("agonist", "antagonist", "co_A_receptor", "co_I_receptor"):
        if col in lr.columns:
            lr[f"{col}_genes"] = lr[col].map(lambda value: _expand_cellchat_cofactor(value, cofactor_table))

    keep_cols = [
        "ligand",
        "receptor",
        "pathway",
        "annotation",
        "evidence",
        "interaction_name",
        "interaction_name_2",
        "agonist",
        "antagonist",
        "co_A_receptor",
        "co_I_receptor",
        "agonist_genes",
        "antagonist_genes",
        "co_A_receptor_genes",
        "co_I_receptor_genes",
        "cellchat_ligand",
        "cellchat_receptor",
    ]
    lr = lr[[col for col in keep_cols if col in lr.columns]]
    metadata = {name: value.copy() for name, value in obj.items() if name != "interaction" and isinstance(value, pd.DataFrame)}
    metadata["interaction_raw"] = interaction
    return CellChatDB(lr, name=f"cellchatdb_{species}", metadata=metadata)


def _canonicalize_lr_columns(lr_table: pd.DataFrame, *, column_map: Mapping[str, str] | None) -> pd.DataFrame:
    lr = lr_table.copy()
    for canonical, column in (column_map or {}).items():
        if column not in lr.columns:
            raise ValueError(f"Column `{column}` was requested for `{canonical}` but is not present.")
        lr[canonical] = lr[column]
    for canonical, aliases in LR_COLUMN_ALIASES.items():
        if canonical in lr.columns:
            continue
        for alias in aliases:
            if alias in lr.columns:
                lr[canonical] = lr[alias]
                break
    return lr


def _omnipath_frame_to_lr(raw: pd.DataFrame, *, pathway_mode: str, pathway_name: str) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("OmniPath returned no ligand-receptor rows.")
    source_col = _first_existing(raw, ["source_genesymbol", "genesymbol_intercell_source", "source"])
    target_col = _first_existing(raw, ["target_genesymbol", "genesymbol_intercell_target", "target"])
    lr = pd.DataFrame(
        {
            "ligand": raw[source_col].astype(str),
            "receptor": raw[target_col].astype(str),
            "annotation": "OmniPath",
        }
    )
    lr["pathway"] = _omnipath_pathway(raw, mode=pathway_mode, pathway_name=pathway_name)
    if "references" in raw.columns:
        lr["evidence"] = raw["references"].fillna("").astype(str)
    elif "references_stripped" in raw.columns:
        lr["evidence"] = raw["references_stripped"].fillna("").astype(str)
    else:
        lr["evidence"] = ""

    copy_cols = {
        "source": "omnipath_source_uniprot",
        "target": "omnipath_target_uniprot",
        "sources": "omnipath_sources",
        "resources": "omnipath_resources",
        "databases": "omnipath_databases",
        "category_intercell_source": "omnipath_ligand_category",
        "category_intercell_target": "omnipath_receptor_category",
        "parent_intercell_source": "omnipath_ligand_parent",
        "parent_intercell_target": "omnipath_receptor_parent",
        "is_stimulation": "omnipath_is_stimulation",
        "is_inhibition": "omnipath_is_inhibition",
        "consensus_score_intercell_source": "omnipath_ligand_consensus_score",
        "consensus_score_intercell_target": "omnipath_receptor_consensus_score",
    }
    for source, target in copy_cols.items():
        if source in raw.columns:
            lr[target] = raw[source].fillna("").astype(str)
    return lr


def _omnipath_pathway(raw: pd.DataFrame, *, mode: str, pathway_name: str) -> pd.Series:
    if mode not in {"category", "resource", "constant"}:
        raise ValueError("`pathway_mode` must be one of: category, resource, constant.")
    if mode == "category" and {"category_intercell_source", "category_intercell_target"}.issubset(raw.columns):
        source = raw["category_intercell_source"].fillna("transmitter").astype(str)
        target = raw["category_intercell_target"].fillna("receiver").astype(str)
        return source + " -> " + target
    if mode == "resource":
        resource_col = _first_existing(raw, ["resources", "sources", "databases"], required=False)
        if resource_col is not None:
            return raw[resource_col].fillna(pathway_name).astype(str).map(lambda item: _first_resource_token(item) or pathway_name)
    return pd.Series([pathway_name] * len(raw), index=raw.index)


def _first_existing(frame: pd.DataFrame, names: Sequence[str], *, required: bool = True) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(f"None of the expected columns are present: {list(names)}")
    return None


def _resolve_omnipath_organism(organism: str) -> str:
    key = str(organism).strip().lower()
    if key not in OMNIPATH_ORGANISM_ALIASES:
        allowed = ", ".join(sorted({"human", "mouse", "rat"}))
        raise ValueError(f"`organism` must be one of: {allowed}.")
    return OMNIPATH_ORGANISM_ALIASES[key]


def _as_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _split_resource_tokens(value: str) -> set[str]:
    return {token.strip() for token in str(value).replace(",", ";").split(";") if token.strip()}


def _first_resource_token(value: str) -> str:
    tokens = sorted(_split_resource_tokens(value))
    return tokens[0] if tokens else ""


def _expand_cellchat_complex(name: str, complex_table: pd.DataFrame) -> str:
    name = str(name).strip()
    if complex_table.empty or name not in complex_table.index:
        return name
    subunits = [str(value).strip() for value in complex_table.loc[name].tolist()]
    return "_".join(value for value in subunits if value)


def _expand_cellchat_cofactor(name: str, cofactor_table: pd.DataFrame) -> str:
    name = str(name).strip()
    if not name or cofactor_table.empty or name not in cofactor_table.index:
        return ""
    genes = [str(value).strip() for value in cofactor_table.loc[name].tolist()]
    return "_".join(value for value in genes if value)


def _download_file(url: str, path: Path, *, proxy: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handlers = []
    if proxy is not None:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    with opener.open(url, timeout=60) as response, path.open("wb") as handle:
        handle.write(response.read())


def toy_lr_table() -> CellChatDB:
    """A tiny LR database useful for examples and tests."""

    return CellChatDB(
        pd.DataFrame(
            {
                "ligand": ["TGFB1", "CXCL12", "CD74", "MIF", "VEGFA", "LAMA1"],
                "receptor": ["TGFBR1_TGFBR2", "CXCR4", "MIF", "CD74_CXCR4", "KDR", "ITGA6_ITGB1"],
                "pathway": ["TGFb", "CXCL", "MIF", "MIF", "VEGF", "LAMININ"],
                "annotation": ["Secreted Signaling"] * 6,
            }
        ),
        name="toy",
    )
