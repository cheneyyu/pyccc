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
        ("plantcellchat", {"Ligand": "PSK1", "Receptor": "PSKR1", "Signal": "PSK", "Interaction_type": "Interacting protein"}),
        ("generic", {"ligand_gene": "VEGFA", "receptor_gene": "KDR", "pathway": "VEGF"}),
    ]
    resources = [
        {
            "frame": pd.DataFrame([row]),
            "schema": schema,
            "species": f"species_{schema}",
            "taxon_id": i + 1,
            "clade": "animal" if schema != "plantcellchat" else "plant",
        }
        for i, (schema, row) in enumerate(specs)
    ]

    normalized = pc.load_training_lr_resources(resources)

    assert normalized.shape[0] == len(specs)
    assert {"ligand_gene", "receptor_gene", "species", "taxon_id", "clade", "resource", "evidence_type", "support_count"}.issubset(normalized.columns)
    assert set(normalized["evidence_type"]) == {"curated_direct", "user_supplied"}
    assert set(normalized["clade"]) == {"animal", "plant"}
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


def test_load_training_lr_resources_merges_duplicate_provenance_fields():
    frame_a = pd.DataFrame(
        [
            {
                "ligand_gene": "L1",
                "receptor_gene": "R1",
                "pathway": "P",
                "evidence_type": "curated_direct",
                "PMID": "1",
                "source_url": "https://a.example/lr",
                "License": "license-a",
                "is_directed": True,
            }
        ]
    )
    frame_b = pd.DataFrame(
        [
            {
                "ligand_gene": "L1",
                "receptor_gene": "R1",
                "pathway": "P",
                "evidence_type": "curated_inferred",
                "PMID": "2",
                "source_url": "https://b.example/lr",
                "License": "license-b",
                "is_directed": True,
            }
        ]
    )

    normalized = pc.load_training_lr_resources(
        [
            {"frame": frame_a, "schema": "generic", "species": "human", "taxon_id": 9606, "resource": "A"},
            {"frame": frame_b, "schema": "generic", "species": "human", "taxon_id": 9606, "resource": "B"},
        ]
    )

    row = normalized.iloc[0]
    assert row["resource"] == "A;B"
    assert row["support_resources"] == "A;B"
    assert row["evidence_type"] == "curated_direct;curated_inferred"
    assert row["pmid"] == "1;2"
    assert row["source_url"] == "https://a.example/lr;https://b.example/lr"
    assert row["license"] == "license-a;license-b"


def test_load_training_lr_resources_preserves_clade_from_table_alias():
    frame = pd.DataFrame(
        [
            {
                "ligand_gene": "PSK1",
                "receptor_gene": "PSKR1",
                "pathway": "PSK",
                "kingdom": "plant",
            }
        ]
    )

    normalized = pc.load_training_lr_resources(
        [{"frame": frame, "schema": "generic", "species": "soybean", "taxon_id": 3847, "resource": "plant_fixture"}],
    )

    assert normalized.loc[0, "clade"] == "plant"


def test_load_training_lr_resources_rejects_explicitly_undirected_rows():
    frame = pd.DataFrame(
        [
            {
                "ligand_gene": "L1",
                "receptor_gene": "R1",
                "pathway": "P",
                "directed": False,
            }
        ]
    )

    with pytest.raises(ValueError, match="undirected"):
        pc.load_training_lr_resources(
            [{"frame": frame, "schema": "generic", "species": "human", "taxon_id": 9606, "resource": "fixture"}],
            strict=True,
        )


def test_load_training_lr_resources_strict_missing_columns_fails():
    with pytest.raises(ValueError, match="required column"):
        pc.load_training_lr_resources(
            [{"frame": pd.DataFrame({"not_ligand": ["A"]}), "schema": "generic", "species": "x", "taxon_id": 1}],
            strict=True,
        )


def test_build_lr_training_table_drops_incomplete_complex_metadata():
    interactions = pd.DataFrame(
        {
            "ligand_gene": ["L1", "L2", "L3"],
            "receptor_gene": ["R1", "R2A_R2B", "R3A_R3B"],
            "species": ["human", "human", "human"],
            "taxon_id": [9606, 9606, 9606],
            "resource": ["fixture", "fixture", "fixture"],
            "evidence_type": ["curated_direct;user_supplied", "curated_direct", "user_supplied"],
            "annotation": ["", "", ""],
            "pathway": ["P1", "P2", "P3"],
            "ligand_sequence": ["ML1", "ML2", "ML3"],
            "receptor_sequence": ["MR1", "", "MR3"],
            "receptor_complex_id": ["", "R2_complex", "R3_complex"],
            "complex_required_subunits": ["", "2", "2"],
            "complex_subunit_gene": ["", "R2A", "R3A;R3B"],
        }
    )

    training = pc.build_lr_training_table(interactions, drop_complexes="partial")

    assert training.interactions["ligand_gene"].tolist() == ["L1", "L3"]
    assert training.interactions["is_positive_label"].tolist() == [True, False]


def test_build_lr_training_table_can_require_complete_sequences():
    interactions = pd.DataFrame(
        {
            "ligand_gene": ["ABA", "L1"],
            "receptor_gene": ["R0", "R1"],
            "species": ["arabidopsis", "arabidopsis"],
            "taxon_id": [3702, 3702],
            "resource": ["plant_fixture", "plant_fixture"],
            "evidence_type": ["curated_direct", "curated_direct"],
            "annotation": ["", ""],
            "pathway": ["ABA", "protein_pair"],
            "ligand_sequence": ["", "MCCCC"],
            "receptor_sequence": ["MRRRR", "MIIII"],
        }
    )

    training = pc.build_lr_training_table(interactions, require_sequences=True)

    assert training.interactions["ligand_gene"].tolist() == ["L1"]
    assert training.interactions.attrs["n_input_interactions"] == 2
    assert training.interactions.attrs["n_after_sequence_filter"] == 1


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
    assert set(embeddings["species"]) == {"human", "mouse"}
    assert embeddings["embedding"].str.contains(",").all()
