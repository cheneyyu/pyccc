from __future__ import annotations

import argparse
import hashlib
import shutil
import textwrap
import urllib.request
from pathlib import Path

import pandas as pd

import pyccc as pc


PROTEOME_SOURCES = {
    "human": {
        "url": "https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz",
        "gene_id_regex": r"gene_symbol:([^\s]+)",
    },
    "mouse": {
        "url": "https://ftp.ensembl.org/pub/current_fasta/mus_musculus/pep/Mus_musculus.GRCm39.pep.all.fa.gz",
        "gene_id_regex": r"gene_symbol:([^\s]+)",
    },
    "arabidopsis": {
        "url": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/current/fasta/arabidopsis_thaliana/pep/Arabidopsis_thaliana.TAIR10.pep.all.fa.gz",
        "gene_id_regex": r"gene:([^\s]+)",
    },
    "rice": {
        "url": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/current/fasta/oryza_sativa/pep/Oryza_sativa.IRGSP-1.0.pep.all.fa.gz",
        "gene_id_regex": r"gene:([^\s]+)",
    },
    "maize": {
        "url": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/current/fasta/zea_mays/pep/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.pep.all.fa.gz",
        "gene_id_regex": r"gene:([^\s]+)",
    },
    "tomato": {
        "url": "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/current/fasta/solanum_lycopersicum/pep/Solanum_lycopersicum.SL3.0.pep.all.fa.gz",
        "gene_id_regex": r"gene:([^\s]+)",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and standardize protein FASTA files for DB-free LR predictor training.")
    parser.add_argument("--output-dir", default="data/proteomes/training")
    parser.add_argument("--raw-dir", default="data/proteomes/training_raw")
    parser.add_argument("--lr-table", default="data/lr_training_resources/normalized_lr.tsv")
    parser.add_argument("--species", action="append", choices=sorted(PROTEOME_SOURCES), help="Species to prepare. Defaults to all training species.")
    args = parser.parse_args()

    output = Path(args.output_dir)
    raw_dir = Path(args.raw_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    lr = pd.read_csv(args.lr_table, sep="\t") if args.lr_table else pd.DataFrame()
    species_names = args.species or list(PROTEOME_SOURCES)

    summary_rows = []
    for species in species_names:
        cfg = PROTEOME_SOURCES[species]
        url = cfg["url"]
        raw_path = raw_dir / Path(url).name
        if not raw_path.exists():
            _download(url, raw_path)
        proteins = pc.load_protein_fasta(raw_path, gene_id_regex=cfg["gene_id_regex"], select="longest")
        table_path = output / f"{species}.protein_table.tsv"
        fasta_path = output / f"{species}.longest_protein.fa"
        proteins.to_csv(table_path, sep="\t", index=False)
        _write_fasta(proteins, fasta_path)
        summary_rows.append(_summary_row(species, url, raw_path, proteins, lr, table_path, fasta_path))

    pd.DataFrame(summary_rows).to_csv(output / "training_proteome_summary.tsv", sep="\t", index=False)


def _download(url: str, output: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "pyccc-dbfree-proteome-prep"})
    with urllib.request.urlopen(request, timeout=300) as response, output.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _write_fasta(proteins: pd.DataFrame, output: Path) -> None:
    with output.open("w", encoding="utf-8") as handle:
        for row in proteins.sort_values(["gene_id", "protein_id"]).itertuples(index=False):
            handle.write(f">{row.protein_id} gene={row.gene_id}\n")
            handle.write("\n".join(textwrap.wrap(str(row.protein_sequence), width=80)))
            handle.write("\n")


def _summary_row(
    species: str,
    url: str,
    raw_path: Path,
    proteins: pd.DataFrame,
    lr: pd.DataFrame,
    table_path: Path,
    fasta_path: Path,
) -> dict[str, object]:
    sub = lr[lr["species"].astype(str) == species].copy() if not lr.empty and "species" in lr.columns else pd.DataFrame()
    lr_genes = set()
    if not sub.empty:
        lr_genes = set(sub["ligand_gene"].astype(str)) | set(sub["receptor_gene"].astype(str))
    protein_genes = set(proteins["gene_id"].astype(str))
    matched = lr_genes & protein_genes
    return {
        "species": species,
        "source_url": url,
        "raw_path": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "protein_table": str(table_path),
        "protein_fasta": str(fasta_path),
        "n_protein_genes": int(proteins["gene_id"].nunique()),
        "n_lr_genes": int(len(lr_genes)),
        "n_matched_lr_genes": int(len(matched)),
        "matched_lr_gene_fraction": float(len(matched) / len(lr_genes)) if lr_genes else 0.0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
