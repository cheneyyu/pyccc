from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Subset training proteomes to LR endpoints plus reproducible background proteins for DB-free model training.")
    parser.add_argument("--training-interactions", default="data/lr_training/training_interactions.tsv")
    parser.add_argument("--training-proteins", default="data/lr_training/training_proteins.tsv")
    parser.add_argument("--output", default="data/lr_training/training_proteins_model.tsv")
    parser.add_argument("--summary", default="data/lr_training/training_protein_subset_summary.json")
    parser.add_argument("--background-per-species", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    interactions = pd.read_csv(args.training_interactions, sep="\t")
    proteins = pd.read_csv(args.training_proteins, sep="\t")
    subset, summary = subset_training_proteins(
        interactions,
        proteins,
        background_per_species=args.background_per_species,
        random_state=args.random_state,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(args.output, sep="\t", index=False)
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def subset_training_proteins(
    interactions: pd.DataFrame,
    proteins: pd.DataFrame,
    *,
    background_per_species: int = 2000,
    random_state: int = 0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if background_per_species < 0:
        raise ValueError("`background_per_species` must be non-negative.")
    rng = np.random.default_rng(random_state)
    endpoint_genes = {
        str(species): set(sub["ligand_gene"].astype(str)) | set(sub["receptor_gene"].astype(str))
        for species, sub in interactions.groupby("species", sort=False)
    }
    selected_frames = []
    species_summary = {}
    for species, sub in proteins.groupby("species", sort=False):
        species = str(species)
        sub = sub.copy()
        endpoints = endpoint_genes.get(species, set())
        endpoint_frame = sub[sub["gene_id"].astype(str).isin(endpoints)].copy()
        background = sub[~sub["gene_id"].astype(str).isin(endpoints)].copy()
        if background_per_species and not background.empty:
            n_background = min(int(background_per_species), int(background["gene_id"].nunique()))
            sampled_genes = sorted(rng.choice(sorted(background["gene_id"].astype(str).unique()), size=n_background, replace=False))
            background = background[background["gene_id"].astype(str).isin(sampled_genes)].copy()
        else:
            background = background.iloc[0:0].copy()
        selected = pd.concat([endpoint_frame, background], ignore_index=True)
        selected["training_role_source"] = np.where(selected["gene_id"].astype(str).isin(endpoints), "lr_endpoint", "background")
        selected_frames.append(selected)
        species_summary[species] = {
            "n_lr_endpoint_genes": int(endpoint_frame["gene_id"].nunique()),
            "n_background_genes": int(background["gene_id"].nunique()),
            "n_selected_genes": int(selected["gene_id"].nunique()),
        }
    if not selected_frames:
        raise ValueError("No training proteins were selected.")
    out = pd.concat(selected_frames, ignore_index=True)
    out = out.sort_values(["species", "training_role_source", "gene_id", "protein_id"]).reset_index(drop=True)
    return out, {
        "background_per_species": int(background_per_species),
        "random_state": int(random_state),
        "n_selected_rows": int(len(out)),
        "n_selected_genes": int(out["gene_id"].nunique()),
        "species": species_summary,
    }


if __name__ == "__main__":
    main()
