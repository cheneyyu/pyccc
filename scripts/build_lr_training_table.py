from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import pyccc as pc


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cross-species pyccc LR training table from normalized LR TSV files and protein FASTA files.")
    parser.add_argument("--normalized-lr", action="append", required=True, help="Normalized LR TSV. May be passed more than once.")
    parser.add_argument("--protein-fasta", action="append", default=[], metavar="SPECIES=PATH", help="Species-specific protein FASTA. May be passed more than once.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-evidence", default="curated_direct,curated_inferred")
    parser.add_argument("--drop-complexes", default="partial", choices=["partial", "none"])
    parser.add_argument("--allow-missing-sequences", action="store_true", help="Keep LR rows without both ligand and receptor protein sequences.")
    args = parser.parse_args()

    resources = pd.concat([pd.read_csv(path, sep="\t") for path in args.normalized_lr], ignore_index=True)
    protein_fasta_by_species = _parse_species_paths(args.protein_fasta)
    positive_evidence = {item.strip() for item in args.positive_evidence.split(",") if item.strip()}
    training = pc.build_lr_training_table(
        resources,
        protein_fasta_by_species=protein_fasta_by_species,
        positive_evidence=positive_evidence,
        drop_complexes=args.drop_complexes,
        require_sequences=not args.allow_missing_sequences,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training.interactions.to_csv(output / "training_interactions.tsv", sep="\t", index=False)
    training.proteins.to_csv(output / "training_proteins.tsv", sep="\t", index=False)
    (output / "training_metadata.json").write_text(
        json.dumps(
            {
                "normalized_lr": [str(Path(path)) for path in args.normalized_lr],
                "protein_fasta_by_species": {species: str(path) for species, path in protein_fasta_by_species.items()},
                "positive_evidence": sorted(training.positive_evidence),
                "drop_complexes": args.drop_complexes,
                "require_sequences": not args.allow_missing_sequences,
                "n_input_interactions": int(training.interactions.attrs.get("n_input_interactions", training.interactions.shape[0])),
                "n_after_complex_filter": int(training.interactions.attrs.get("n_after_complex_filter", training.interactions.shape[0])),
                "n_after_sequence_filter": int(training.interactions.attrs.get("n_after_sequence_filter", training.interactions.shape[0])),
                "n_interactions": int(training.interactions.shape[0]),
                "n_proteins": int(training.proteins.shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _parse_species_paths(values: list[str]) -> dict[str, str]:
    out = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("`--protein-fasta` entries must use SPECIES=PATH.")
        species, path = value.split("=", 1)
        species = species.strip()
        path = path.strip()
        if not species or not path:
            raise SystemExit("`--protein-fasta` entries must use non-empty SPECIES=PATH.")
        out[species] = path
    return out


if __name__ == "__main__":
    main()
