from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
from pathlib import Path

import pandas as pd

import pyccc as pc


PLANTCELLCHAT_URL = "https://github.com/mrliuw/PlantCellChat/raw/main/LRI_database.rar"

ANIMAL_RESOURCES = {
    "human": {"taxon_id": 9606, "clade": "animal", "schema": "cellchat", "resource": "CellChatDB"},
    "mouse": {"taxon_id": 10090, "clade": "animal", "schema": "cellchat", "resource": "CellChatDB"},
}

PLANTCELLCHAT_RESOURCES = {
    "ath": {"species": "arabidopsis", "taxon_id": 3702},
    "osa": {"species": "rice", "taxon_id": 4530},
    "zma": {"species": "maize", "taxon_id": 4577},
    "sly": {"species": "tomato", "taxon_id": 4081},
    "gmx": {"species": "soybean", "taxon_id": 3847},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare normalized animal and plant LR resources for DB-free predictor training.")
    parser.add_argument("--output-dir", default="data/lr_training_resources")
    parser.add_argument("--source-dir", default="data/lr_training_sources/PlantCellChat")
    parser.add_argument("--plantcellchat-url", default=PLANTCELLCHAT_URL)
    parser.add_argument("--exclude-plant-code", action="append", default=["gmx"], help="PlantCellChat species code to exclude. Default excludes soybean/gmx.")
    parser.add_argument("--skip-animal", action="store_true")
    parser.add_argument("--skip-plant", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    summary_rows = []

    if not args.skip_animal:
        for species, cfg in ANIMAL_RESOURCES.items():
            raw = pc.load_cellchatdb(species).interactions.copy()
            normalized = pc.load_training_lr_resources(
                [
                    {
                        "frame": raw,
                        "schema": cfg["schema"],
                        "species": species,
                        "taxon_id": cfg["taxon_id"],
                        "clade": cfg["clade"],
                        "resource": cfg["resource"],
                        "source_url": f"pyccc.load_cellchatdb({species})",
                    }
                ],
                strict=False,
            )
            normalized = _drop_self_pairs(normalized)
            path = output / f"{species}_cellchat.normalized.tsv"
            normalized.to_csv(path, sep="\t", index=False)
            frames.append(normalized)
            summary_rows.append(_summary_row("animal", "CellChatDB", species, cfg["taxon_id"], raw, normalized, path))

    if not args.skip_plant:
        plant_paths = _plantcellchat_csvs(Path(args.source_dir), args.plantcellchat_url)
        excluded = {str(code).strip().lower() for code in args.exclude_plant_code if str(code).strip()}
        for code, cfg in PLANTCELLCHAT_RESOURCES.items():
            if code in excluded:
                continue
            raw_path = plant_paths.get(code)
            if raw_path is None:
                continue
            raw = pd.read_csv(raw_path)
            raw["evidence_type"] = raw.get("Evidence", pd.Series([""] * len(raw))).map(_plant_evidence_type)
            normalized = pc.load_training_lr_resources(
                [
                    {
                        "frame": raw,
                        "schema": "plantcellchat",
                        "species": cfg["species"],
                        "taxon_id": cfg["taxon_id"],
                        "clade": "plant",
                        "resource": "PlantCellChatDB",
                        "source_url": args.plantcellchat_url,
                    }
                ],
                strict=False,
            )
            normalized = _drop_self_pairs(normalized)
            path = output / f"{cfg['species']}_plantcellchat.normalized.tsv"
            normalized.to_csv(path, sep="\t", index=False)
            frames.append(normalized)
            summary_rows.append(_summary_row("plant", "PlantCellChatDB", cfg["species"], cfg["taxon_id"], raw, normalized, path))

    if not frames:
        raise SystemExit("No LR resources were prepared.")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output / "normalized_lr.tsv", sep="\t", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "resource_summary.tsv", sep="\t", index=False)


def _plantcellchat_csvs(source_dir: Path, url: str) -> dict[str, Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    rar_path = source_dir / "LRI_database.rar"
    extract_dir = source_dir / "extracted"
    if not rar_path.exists():
        _download(url, rar_path)
    extract_dir.mkdir(parents=True, exist_ok=True)
    if not any(extract_dir.glob("*.csv")):
        bsdtar = shutil.which("bsdtar")
        if bsdtar is None:
            raise RuntimeError("PlantCellChatDB extraction requires `bsdtar` on PATH.")
        subprocess.run([bsdtar, "-xf", str(rar_path), "-C", str(extract_dir)], check=True)
    return {path.stem.lower(): path for path in extract_dir.glob("*.csv")}


def _download(url: str, output: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "pyccc-dbfree-resource-prep"})
    with urllib.request.urlopen(request, timeout=120) as response, output.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _plant_evidence_type(value: object) -> str:
    text = str(value).upper()
    if "PMID" in text:
        return "curated_direct"
    if "KEGG" in text:
        return "curated_inferred"
    return "curated_inferred"


def _drop_self_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["ligand_gene"].astype(str) != frame["receptor_gene"].astype(str)].copy().reset_index(drop=True)


def _summary_row(clade: str, resource: str, species: str, taxon_id: int, raw: pd.DataFrame, normalized: pd.DataFrame, path: Path) -> dict[str, object]:
    return {
        "clade": clade,
        "resource": resource,
        "species": species,
        "taxon_id": int(taxon_id),
        "raw_rows": int(len(raw)),
        "normalized_rows": int(len(normalized)),
        "normalized_path": str(path),
    }


if __name__ == "__main__":
    main()
