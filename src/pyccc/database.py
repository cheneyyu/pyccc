from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REQUIRED_LR_COLUMNS = ("ligand", "receptor")
CELLCHATDB_URLS = {
    "human": "https://raw.githubusercontent.com/jinworks/CellChat/main/data/CellChatDB.human.rda",
    "mouse": "https://raw.githubusercontent.com/jinworks/CellChat/main/data/CellChatDB.mouse.rda",
    "zebrafish": "https://raw.githubusercontent.com/jinworks/CellChat/main/data/CellChatDB.zebrafish.rda",
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


def normalize_lr_table(lr_table: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_LR_COLUMNS if col not in lr_table.columns]
    if missing:
        raise ValueError(f"LR table is missing required columns: {missing}")

    lr = lr_table.copy()
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


def load_lr_table(path: str | Path, *, name: str | None = None, sep: str | None = None) -> CellChatDB:
    """Load a ligand-receptor table from CSV/TSV/Parquet.

    The table must contain `ligand` and `receptor`. Files ending in `.tsv` or
    `.txt` default to tab separation; other text files default to comma.
    """

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        lr = pd.read_parquet(path)
    else:
        if sep is None:
            sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        lr = pd.read_csv(path, sep=sep)
    return CellChatDB(lr, name=name or path.stem)


def load_cellchatdb(
    species: str = "human",
    *,
    path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    proxy: str | None = None,
) -> CellChatDB:
    """Load an official CellChat ligand-receptor database.

    When `path` is not provided, the requested `.rda` file is downloaded from
    the CellChat GitHub repository and cached locally. Supported species are
    `human`, `mouse`, and `zebrafish`.
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
    return _cellchatdb_from_object(objects[key], species=species)


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
