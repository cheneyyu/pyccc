from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

STANDARD_CODE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYXBZUOJ*")


def load_cds_translations(
    path: str | Path,
    *,
    gene_id_regex: str | None = r"gene=([^\s]+)",
    transcript_id_regex: str | None = r"transcript=([^\s]+)",
    gene_id_replacements: Sequence[tuple[str, str]] | None = None,
    genetic_code: int = 1,
    select: str = "longest",
) -> pd.DataFrame:
    """Read CDS FASTA, translate records, and optionally keep longest isoforms."""

    if genetic_code != 1:
        raise ValueError("Only the standard genetic code (`genetic_code=1`) is supported in this release.")
    records = []
    for header, sequence in _read_fasta(path):
        cds = _clean_dna(sequence)
        transcript_id = _regex_or_default(transcript_id_regex, header, _first_token(header))
        gene_id = _apply_replacements(_regex_or_default(gene_id_regex, header, transcript_id), gene_id_replacements)
        protein = _translate_cds(cds)
        stop_count = protein.count("*")
        valid = len(cds) > 0 and len(cds) % 3 == 0 and "N" not in cds and stop_count <= 1 and (stop_count == 0 or protein.endswith("*"))
        records.append(
            {
                "gene_id": gene_id,
                "transcript_id": transcript_id,
                "protein_id": transcript_id,
                "cds_sequence": cds,
                "protein_sequence": protein.rstrip("*"),
                "cds_length": len(cds),
                "protein_length": len(protein.rstrip("*")),
                "stop_codon_count": stop_count,
                "valid_translation": bool(valid),
                "selected_isoform": False,
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"No FASTA records found in {path}.")
    return _select_isoforms(frame, select=select)


def load_protein_fasta(
    path: str | Path,
    *,
    gene_id_regex: str | None = r"gene=([^\s]+)",
    protein_id_regex: str | None = None,
    gene_id_replacements: Sequence[tuple[str, str]] | None = None,
    select: str = "longest",
) -> pd.DataFrame:
    """Read protein FASTA and return the same protein table shape as CDS input."""

    records = []
    for header, sequence in _read_fasta(path):
        protein = _clean_protein(sequence)
        protein_id = _regex_or_default(protein_id_regex, header, _first_token(header))
        gene_id = _apply_replacements(_regex_or_default(gene_id_regex, header, protein_id), gene_id_replacements)
        records.append(
            {
                "gene_id": gene_id,
                "transcript_id": protein_id,
                "protein_id": protein_id,
                "cds_sequence": "",
                "protein_sequence": protein.rstrip("*"),
                "cds_length": 0,
                "protein_length": len(protein.rstrip("*")),
                "stop_codon_count": protein.count("*"),
                "valid_translation": bool(protein) and set(protein).issubset(VALID_AA),
                "selected_isoform": False,
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"No FASTA records found in {path}.")
    return _select_isoforms(frame, select=select)


def match_expression_genes(adata, proteins: pd.DataFrame, *, gene_id_key: str | None = None) -> pd.DataFrame:
    """Report which protein-table genes match the expression matrix identifiers."""

    if gene_id_key is None:
        expression_genes = pd.Index(adata.var_names.astype(str))
    else:
        if gene_id_key not in adata.var:
            raise KeyError(f"`{gene_id_key}` is not present in adata.var.")
        expression_genes = pd.Index(adata.var[gene_id_key].astype(str))
    protein_genes = pd.Index(proteins["gene_id"].astype(str))
    return pd.DataFrame(
        {
            "gene_id": sorted(set(expression_genes).union(set(protein_genes))),
        }
    ).assign(
        in_expression=lambda df: df["gene_id"].isin(expression_genes),
        in_proteins=lambda df: df["gene_id"].isin(protein_genes),
    )


def _select_isoforms(frame: pd.DataFrame, *, select: str) -> pd.DataFrame:
    if select not in {"longest", "all"}:
        raise ValueError("`select` must be one of: longest, all.")
    out = frame.copy()
    if select == "all":
        out["selected_isoform"] = True
        return out.reset_index(drop=True)
    order = out.sort_values(["gene_id", "valid_translation", "protein_length", "cds_length"], ascending=[True, False, False, False])
    chosen = order.drop_duplicates("gene_id", keep="first").index
    out = out.loc[chosen].copy()
    out["selected_isoform"] = True
    return out.sort_values("gene_id").reset_index(drop=True)


def _read_fasta(path: str | Path) -> Iterable[tuple[str, str]]:
    header = None
    chunks: list[str] = []
    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def _regex_or_default(pattern: str | None, text: str, default: str) -> str:
    if pattern is None:
        return default
    match = re.search(pattern, text)
    return match.group(1) if match else default


def _apply_replacements(value: str, replacements: Sequence[tuple[str, str]] | None) -> str:
    out = str(value)
    for pattern, replacement in replacements or ():
        out = re.sub(pattern, replacement, out)
    return out


def _open_text(path: str | Path):
    fasta_path = Path(path)
    if fasta_path.suffix == ".gz":
        return gzip.open(fasta_path, "rt")
    return fasta_path.open()


def _first_token(header: str) -> str:
    return header.split()[0]


def _clean_dna(sequence: str) -> str:
    return re.sub("[^ACGTN]", "", sequence.upper().replace("U", "T"))


def _clean_protein(sequence: str) -> str:
    return re.sub("[^A-Z*]", "", sequence.upper())


def _translate_cds(cds: str) -> str:
    amino = []
    for idx in range(0, len(cds) - 2, 3):
        amino.append(STANDARD_CODE.get(cds[idx : idx + 3], "X"))
    return "".join(amino)
