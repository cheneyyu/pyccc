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
