import gzip

import numpy as np
import pandas as pd
from anndata import AnnData

import pyccc as pc


def test_load_cds_translations_selects_longest_and_flags_invalid(tmp_path):
    fasta = tmp_path / "cds.fa"
    fasta.write_text(
        ">tx1 gene=G1 transcript=T1\nATGAAATAA\n"
        ">tx2 gene=G1 transcript=T2\nATGAAAAAATAA\n"
        ">tx3 gene=G2 transcript=T3\nATGAA\n",
        encoding="utf-8",
    )

    proteins = pc.load_cds_translations(fasta, select="longest")

    assert proteins["gene_id"].tolist() == ["G1", "G2"]
    assert proteins.loc[proteins["gene_id"] == "G1", "transcript_id"].iloc[0] == "T2"
    assert proteins.loc[proteins["gene_id"] == "G2", "valid_translation"].iloc[0] == False


def test_load_protein_fasta_and_match_expression_genes(tmp_path):
    fasta = tmp_path / "protein.fa"
    fasta.write_text(">p1 gene=G1\nMPEPTIDE\n>p2 gene=G2\nMCCCC\n", encoding="utf-8")
    proteins = pc.load_protein_fasta(fasta)
    adata = AnnData(np.array([[1, 0]], dtype=float), var=pd.DataFrame({"gene_id": ["G1", "G3"]}, index=["x1", "x2"]))

    matched = pc.match_expression_genes(adata, proteins, gene_id_key="gene_id")

    assert proteins["protein_length"].tolist() == [8, 5]
    assert matched.set_index("gene_id").loc["G1", "in_expression"]
    assert not matched.set_index("gene_id").loc["G2", "in_expression"]


def test_load_gzipped_protein_fasta_normalizes_gene_ids_and_selects_longest(tmp_path):
    fasta = tmp_path / "protein.fa.gz"
    with gzip.open(fasta, "wt", encoding="utf-8") as handle:
        handle.write(
            ">p1 OriGeneID=SoyZH13_01G000001\nMPEPTIDE\n"
            ">p2 OriGeneID=SoyZH13_01G000001\nMPEPTIDEVV\n"
            ">p3 OriGeneID=SoyZH13_01G000002\nMCCCC\n"
        )

    proteins = pc.load_protein_fasta(
        fasta,
        gene_id_regex=r"OriGeneID=(SoyZH13_\d+G\d+)",
        gene_id_replacements=[(r"^SoyZH13_", "SoyZH13-")],
    )

    assert proteins["gene_id"].tolist() == ["SoyZH13-01G000001", "SoyZH13-01G000002"]
    assert proteins.loc[proteins["gene_id"] == "SoyZH13-01G000001", "protein_id"].iloc[0] == "p2"
