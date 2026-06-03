from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pandas as pd
import scanpy as sc

import pyccc as pc
from dbfree_validation_utils import load_manifest, manifest_results_dir, read_tsv, write_tsv


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare target-species proteins and expression/protein gene matching for DB-free validation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protein-fasta", default=None, help="Override raw protein FASTA input.")
    parser.add_argument("--cds-fasta", default=None, help="Override raw CDS FASTA input.")
    parser.add_argument("--output-protein-fasta", default=None, help="Override normalized longest-protein FASTA output.")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--gene-id-regex", default=None)
    parser.add_argument("--protein-id-regex", default=None)
    parser.add_argument("--transcript-id-regex", default=None)
    parser.add_argument("--gene-id-replace", action="append", default=[], metavar="PATTERN=REPLACEMENT")
    parser.add_argument("--select", default=None, choices=["longest", "all"])
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    results_dir = manifest_results_dir(manifest, args.results_dir)
    source_cfg = dict(manifest.get("protein_source", {}))
    protein_fasta = args.protein_fasta or source_cfg.get("local_path") or source_cfg.get("path")
    cds_fasta = args.cds_fasta or source_cfg.get("cds_local_path") or source_cfg.get("cds_path")
    if not protein_fasta and not cds_fasta and not source_cfg:
        protein_fasta = manifest.get("protein_fasta")
    gene_id_regex = _first_not_none(args.gene_id_regex, source_cfg.get("gene_id_regex"), r"gene=([^\s]+)")
    protein_id_regex = _first_not_none(args.protein_id_regex, source_cfg.get("protein_id_regex"), None)
    transcript_id_regex = _first_not_none(args.transcript_id_regex, source_cfg.get("transcript_id_regex"), r"transcript=([^\s]+)")
    gene_id_replacements = _gene_id_replacements(args.gene_id_replace, source_cfg.get("gene_id_replacements"))
    select = str(_first_not_none(args.select, source_cfg.get("select"), "longest"))
    output_target = args.output_protein_fasta or source_cfg.get("prepared_fasta") or (manifest.get("protein_fasta") if source_cfg else None)
    output_protein_fasta = Path(str(output_target)) if output_target else None
    if args.cds_fasta:
        cds_fasta = args.cds_fasta
    if cds_fasta:
        proteins = pc.load_cds_translations(
            cds_fasta,
            gene_id_regex=gene_id_regex,
            transcript_id_regex=transcript_id_regex,
            gene_id_replacements=gene_id_replacements,
            select=select,
        )
        protein_source = str(cds_fasta)
    elif protein_fasta:
        proteins = pc.load_protein_fasta(
            protein_fasta,
            gene_id_regex=gene_id_regex,
            protein_id_regex=protein_id_regex,
            gene_id_replacements=gene_id_replacements,
            select=select,
        )
        protein_source = str(protein_fasta)
    else:
        raise ValueError("Provide --protein-fasta, --cds-fasta, manifest protein_source.local_path, or manifest protein_fasta.")

    expression_genes = _expression_genes_from_prepared_sections(results_dir)
    match = _gene_match_table(expression_genes, proteins)
    restrict_to_expression = bool(source_cfg.get("restrict_to_expression", False))
    proteins_for_prediction = _prediction_proteins(proteins, expression_genes=expression_genes, restrict_to_expression=restrict_to_expression)
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
                "prepared_protein_fasta": str(output_protein_fasta or ""),
                "gene_id_regex": gene_id_regex or "",
                "gene_id_replacements": _replacement_summary(gene_id_replacements),
                "select": select,
                "restrict_to_expression": restrict_to_expression,
                "n_expression_genes": n_expression,
                "n_protein_genes": int(match["in_proteins"].sum()),
                "n_prediction_protein_genes": int(proteins_for_prediction["gene_id"].nunique()),
                "n_matched_genes": matched,
                "matched_fraction": matched_fraction,
                "publishable_gene_match_min": min_fraction,
                "publishable_gate_passed": bool(matched_fraction >= min_fraction),
            }
        ]
    )
    if output_protein_fasta is not None:
        _write_protein_fasta(proteins_for_prediction, output_protein_fasta)
    write_tsv(proteins, results_dir / "target_proteins.tsv")
    write_tsv(proteins_for_prediction, results_dir / "target_proteins_for_prediction.tsv")
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


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _gene_id_replacements(cli_values: list[str], manifest_values: object) -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    values = cli_values or manifest_values or []
    if isinstance(values, dict):
        values = list(values.items())
    for item in values:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            replacements.append((str(item[0]), str(item[1])))
            continue
        text = str(item)
        if "=" not in text:
            raise ValueError(f"Gene ID replacement must be PATTERN=REPLACEMENT, got {text!r}.")
        pattern, replacement = text.split("=", 1)
        replacements.append((pattern, replacement))
    return replacements


def _replacement_summary(replacements: list[tuple[str, str]]) -> str:
    return ";".join(f"{pattern}->{replacement}" for pattern, replacement in replacements)


def _prediction_proteins(proteins: pd.DataFrame, *, expression_genes: set[str], restrict_to_expression: bool) -> pd.DataFrame:
    if not restrict_to_expression or not expression_genes:
        return proteins.copy()
    return proteins[proteins["gene_id"].astype(str).isin(expression_genes)].copy().reset_index(drop=True)


def _write_protein_fasta(proteins: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in proteins.sort_values(["gene_id", "protein_id"]).itertuples(index=False):
            sequence = str(row.protein_sequence)
            if not sequence:
                continue
            handle.write(f">{row.protein_id} gene={row.gene_id}\n")
            handle.write("\n".join(textwrap.wrap(sequence, width=80)))
            handle.write("\n")


if __name__ == "__main__":
    main()
