import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import pyccc as pc


def test_load_training_lr_resources_supports_expected_local_schemas():
    specs = [
        ("cellchat", {"ligand": "TGFB1", "receptor": "TGFBR1", "pathway_name": "TGFb", "annotation": "Secreted Signaling"}),
        ("omnipath", {"source_genesymbol": "CXCL12", "target_genesymbol": "CXCR4", "sources": "CellPhoneDB", "references": "PMID:1"}),
        ("cellphonedb", {"gene_a": "MIF", "gene_b": "CD74", "classification": "cytokine"}),
        ("flyphonedb2", {"ligand": "spz", "receptor": "Tl", "family": "Toll"}),
        ("plantphonedb", {"Ligand": "RALF1", "Receptor": "FER", "Pathway": "RALF"}),
        ("plantcellchat", {"ligand": "PSK1", "receptor": "PSKR1", "pathway_name": "PSK"}),
        ("generic", {"ligand_gene": "VEGFA", "receptor_gene": "KDR", "pathway": "VEGF"}),
    ]
    resources = [
        {"frame": pd.DataFrame([row]), "schema": schema, "species": f"species_{schema}", "taxon_id": i + 1}
        for i, (schema, row) in enumerate(specs)
    ]

    normalized = pc.load_training_lr_resources(resources)

    assert normalized.shape[0] == len(specs)
    assert {"ligand_gene", "receptor_gene", "species", "taxon_id", "resource", "evidence_type", "support_count"}.issubset(normalized.columns)
    assert set(normalized["evidence_type"]) == {"curated_direct", "user_supplied"}
    db = pc.training_lr_to_cellchatdb(normalized)
    assert db.interactions.shape[0] == len(specs)


def test_load_training_lr_resources_deduplicates_with_support_counts():
    frame = pd.DataFrame(
        [
            {"ligand_gene": "L1", "receptor_gene": "R1", "pathway": "P"},
            {"ligand_gene": "L1", "receptor_gene": "R1", "pathway": "P"},
        ]
    )

    normalized = pc.load_training_lr_resources(
        [
            {"frame": frame.iloc[[0]], "schema": "generic", "species": "human", "taxon_id": 9606, "resource": "A"},
            {"frame": frame.iloc[[1]], "schema": "generic", "species": "human", "taxon_id": 9606, "resource": "B"},
        ]
    )

    assert normalized.shape[0] == 1
    assert normalized.loc[0, "support_count"] == 2
    assert normalized.loc[0, "support_resources"] == "A;B"


def test_load_training_lr_resources_strict_missing_columns_fails():
    with pytest.raises(ValueError, match="required column"):
        pc.load_training_lr_resources(
            [{"frame": pd.DataFrame({"not_ligand": ["A"]}), "schema": "generic", "species": "x", "taxon_id": 1}],
            strict=True,
        )


def test_build_lr_training_table_script_outputs_sequences_and_metadata(tmp_path):
    root = Path(__file__).resolve().parents[1]
    normalized = pd.DataFrame(
        {
            "ligand_gene": ["L1", "L2"],
            "receptor_gene": ["R1", "R2"],
            "species": ["human", "mouse"],
            "taxon_id": [9606, 10090],
            "resource": ["fixture", "fixture"],
            "evidence_type": ["curated_direct", "user_supplied"],
            "annotation": ["", ""],
            "pathway": ["P1", "P2"],
        }
    )
    lr_path = tmp_path / "normalized.tsv"
    normalized.to_csv(lr_path, sep="\t", index=False)
    human_fasta = tmp_path / "human.fa"
    human_fasta.write_text(">pL1 gene=L1\nMCCCC\n>pR1 gene=R1\nMAVVV\n", encoding="utf-8")
    mouse_fasta = tmp_path / "mouse.fa"
    mouse_fasta.write_text(">pL2 gene=L2\nMDDDD\n>pR2 gene=R2\nMIIII\n", encoding="utf-8")
    out_dir = tmp_path / "training"

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "build_lr_training_table.py"),
            "--normalized-lr",
            str(lr_path),
            "--protein-fasta",
            f"human={human_fasta}",
            "--protein-fasta",
            f"mouse={mouse_fasta}",
            "--output-dir",
            str(out_dir),
        ],
        check=True,
        cwd=root,
    )

    interactions = pd.read_csv(out_dir / "training_interactions.tsv", sep="\t")
    proteins = pd.read_csv(out_dir / "training_proteins.tsv", sep="\t")
    metadata = json.loads((out_dir / "training_metadata.json").read_text(encoding="utf-8"))

    assert interactions["is_positive_label"].tolist() == [True, False]
    assert set(interactions["ligand_sequence"]) == {"MCCCC", "MDDDD"}
    assert set(proteins["species"]) == {"human", "mouse"}
    assert metadata["n_interactions"] == 2

    emb_path = tmp_path / "training_embeddings.tsv"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "embed_proteome_esmc.py"),
            "--protein-table",
            str(out_dir / "training_proteins.tsv"),
            "--output",
            str(emb_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--backend",
            "hash",
        ],
        check=True,
        cwd=root,
    )
    embeddings = pd.read_csv(emb_path, sep="\t")
    assert embeddings.shape[0] == proteins.shape[0]
    assert embeddings["embedding"].str.contains(",").all()
