from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc

import pyccc as pc
from dbfree_validation_utils import load_manifest, manifest_results_dir, read_tsv, write_tsv


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare target-species proteins and expression/protein gene matching for DB-free validation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protein-fasta", default=None)
    parser.add_argument("--cds-fasta", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--gene-id-regex", default=r"gene=([^\s]+)")
    parser.add_argument("--transcript-id-regex", default=r"transcript=([^\s]+)")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    results_dir = manifest_results_dir(manifest, args.results_dir)
    protein_fasta = args.protein_fasta or manifest.get("protein_fasta")
    if args.cds_fasta:
        proteins = pc.load_cds_translations(args.cds_fasta, gene_id_regex=args.gene_id_regex, transcript_id_regex=args.transcript_id_regex)
        protein_source = str(args.cds_fasta)
    elif protein_fasta:
        proteins = pc.load_protein_fasta(protein_fasta, gene_id_regex=args.gene_id_regex)
        protein_source = str(protein_fasta)
    else:
        raise ValueError("Provide --protein-fasta, --cds-fasta, or protein_fasta in the manifest.")

    expression_genes = _expression_genes_from_prepared_sections(results_dir)
    match = _gene_match_table(expression_genes, proteins)
    min_fraction = float(manifest.get("publishable_gene_match_min", 0.0))
    matched = int((match["in_expression"] & match["in_proteins"]).sum())
    n_expression = int(match["in_expression"].sum())
    matched_fraction = matched / n_expression if n_expression else 0.0
    summary = pd.DataFrame(
        [
            {
                "dataset": manifest["name"],
                "species": manifest.get("species", ""),
                "protein_source": protein_source,
                "n_expression_genes": n_expression,
                "n_protein_genes": int(match["in_proteins"].sum()),
                "n_matched_genes": matched,
                "matched_fraction": matched_fraction,
                "publishable_gene_match_min": min_fraction,
                "publishable_gate_passed": bool(matched_fraction >= min_fraction),
            }
        ]
    )
    write_tsv(proteins, results_dir / "target_proteins.tsv")
    write_tsv(match, results_dir / "gene_protein_match.tsv")
    write_tsv(summary, results_dir / "gene_protein_match_summary.tsv")


def _expression_genes_from_prepared_sections(results_dir: Path) -> set[str]:
    paths_path = results_dir / "prepared_paths.tsv"
    if not paths_path.exists():
        return set()
    paths = read_tsv(paths_path)
    genes: set[str] = set()
    for row in paths.itertuples(index=False):
        adata = sc.read_h5ad(str(row.prepared_path), backed="r")
        if "gene_id" in adata.var:
            genes.update(adata.var["gene_id"].astype(str))
        else:
            genes.update(adata.var_names.astype(str))
        adata.file.close()
    return genes


def _gene_match_table(expression_genes: set[str], proteins: pd.DataFrame) -> pd.DataFrame:
    protein_genes = set(proteins["gene_id"].astype(str))
    rows = [
        {
            "gene_id": gene,
            "in_expression": gene in expression_genes,
            "in_proteins": gene in protein_genes,
        }
        for gene in sorted(expression_genes | protein_genes)
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
