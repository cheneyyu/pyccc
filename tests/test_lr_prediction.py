import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from joblib import load

import pyccc as pc
from pyccc.lr_prediction import _training_pairs


def _fixture_training():
    interactions = pd.DataFrame(
        {
            "species": ["toy_a", "toy_a", "toy_b", "toy_b"],
            "clade": ["clade_a", "clade_a", "clade_b", "clade_b"],
            "ligand_gene": ["L1", "L2", "L3", "L4"],
            "receptor_gene": ["R1", "R2", "R3", "R4"],
            "resource": ["fixture_a", "fixture_a", "fixture_b", "fixture_b"],
            "ligand_family": ["lfam1", "lfam2", "lfam3", "lfam4"],
            "receptor_family": ["rfam1", "rfam2", "rfam3", "rfam4"],
            "ligand_role_score": [0.9, 0.8, 0.7, 0.6],
            "receptor_role_score": [0.9, 0.8, 0.7, 0.6],
            "ligand_expression_fraction": [0.8, 0.7, 0.6, 0.5],
            "receptor_expression_fraction": [0.75, 0.65, 0.55, 0.45],
            "evidence_type": ["curated_direct"] * 4,
            "is_positive_label": [True] * 4,
        }
    )
    embeddings = pd.DataFrame(
        {
            "gene_id": ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"],
            "embedding": [
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.8, 0.2, 0.0], dtype=np.float32),
                np.array([0.2, 0.8, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.8, 0.2], dtype=np.float32),
                np.array([1.0, 0.0, 0.2], dtype=np.float32),
                np.array([0.7, 0.1, 0.2], dtype=np.float32),
            ],
            "sequence_length": [100, 120, 130, 140, 300, 330, 340, 350],
            "model_name": [pc.ESMC_300M_MODEL_NAME] * 8,
            "model_revision": ["fixture-rev"] * 8,
            "embedding_backend": ["esmc"] * 8,
            "pooling": ["mean"] * 8,
        }
    )
    return interactions, embeddings


def test_generate_candidates_can_add_embedding_neighbors_under_budget():
    genes = ["L1", "L2", "R1", "R2", "R3"]
    adata = AnnData(
        np.ones((3, len(genes)), dtype=float),
        var=pd.DataFrame({"gene_id": genes}, index=genes),
    )
    proteins = pd.DataFrame({"gene_id": genes, "protein_id": genes})
    roles = pd.DataFrame(
        {
            "gene_id": genes,
            "protein_id": genes,
            "ligand_like_score": [0.95, 0.85, 0.05, 0.05, 0.05],
            "receptor_like_score": [0.05, 0.05, 0.95, 0.85, 0.75],
        }
    )
    embeddings = pd.DataFrame(
        {
            "gene_id": genes,
            "embedding": [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([-1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([1.0, 0.0], dtype=np.float32),
            ],
        }
    )

    candidates = pc.generate_lr_candidates_dbfree(
        adata,
        proteins,
        roles,
        embeddings=embeddings,
        gene_id_key="gene_id",
        expression_min_fraction=0.0,
        max_candidate_pairs=4,
        nearest_neighbor_pairs=2,
    )

    assert len(candidates) <= 4
    assert (candidates["candidate_strategy"].str.contains("embedding_nearest_neighbor")).any()
    assert ((candidates["ligand_gene"] == "L1") & (candidates["receptor_gene"] == "R3")).any()
    assert "embedding_cosine" in candidates.columns

    predicted = pc.build_predicted_lr_table(
        candidates.assign(model_score=1.0, calibrated_probability=1.0),
        density_prior=1.0,
        min_score=0.0,
        max_pairs=4,
    )
    assert (predicted.interactions["candidate_strategy"].str.contains("embedding_nearest_neighbor")).any()


def test_training_pairs_exclude_positive_family_pair_neighbors():
    interactions = pd.DataFrame(
        {
            "species": ["toy"] * 4,
            "clade": ["toy_clade"] * 4,
            "resource": ["fixture"] * 4,
            "ligand_gene": ["L1", "L2", "L3", "L4"],
            "receptor_gene": ["R1", "R2", "R3", "R4"],
            "ligand_family": ["lfam_shared", "lfam_shared", "lfam_b", "lfam_c"],
            "receptor_family": ["rfam_shared", "rfam_shared", "rfam_b", "rfam_c"],
        }
    )
    embeddings = pd.DataFrame(
        {
            "gene_id": ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"],
            "embedding": [np.array([i, i + 1], dtype=np.float32) for i in range(8)],
        }
    )

    pairs = _training_pairs(
        interactions,
        embeddings=embeddings,
        negative_ratio=2,
        negative_strategy="pu_degree_matched",
        easy_negative_fraction=0.0,
        excluded_homology_radius="family_pair",
        random_state=4,
    )
    negatives = pairs[pairs["label"].astype(int) == 0]
    positive_family_pairs = set(zip(interactions["ligand_family"], interactions["receptor_family"], strict=True))

    assert not negatives.empty
    assert all(
        (row.ligand_family, row.receptor_family) not in positive_family_pairs
        for row in negatives.itertuples(index=False)
    )
    assert negatives["degree_matching"].astype(bool).all()
    assert set(negatives["excluded_homology_radius"]) == {"family_pair"}
    assert set(negatives["positive_resource_blacklist_for_fold"]) == {"fixture"}


def test_train_score_lr_link_predictor_sklearn_fixture(tmp_path):
    interactions, embeddings = _fixture_training()

    card = pc.train_lr_link_predictor(
        interactions,
        embeddings,
        model="sklearn",
        output_dir=tmp_path,
        negative_ratio=1,
        negative_repeats=2,
        validation_splits=["leave_species_out", "leave_resource_out", "leave_family_out", "leave_clade_out"],
    )
    scores = pc.score_lr_candidates(pd.DataFrame({"ligand_gene": ["L1"], "receptor_gene": ["R1"]}), embeddings, model=tmp_path)

    assert (tmp_path / "lr_link_model.joblib").exists()
    assert (tmp_path / "model_card.md").exists()
    assert (tmp_path / "density_prior.tsv").exists()
    assert "top-100 enrichment" in (tmp_path / "model_card.md").read_text(encoding="utf-8")
    payload = load(tmp_path / "lr_link_model.joblib")
    assert "calibrator" in payload
    assert card["metrics"]["n_positive"] == 4
    assert card["metrics"]["n_negative"] > 0
    assert card["negative_sampling"]["negative_strategy"] == "pu_degree_matched"
    assert card["negative_sampling"]["excluded_homology_radius"] == "family_pair"
    assert card["negative_sampling"]["positive_resource_blacklist_for_fold"] == "fixture_a;fixture_b"
    assert card["negative_repeats"] == 2
    assert card["negative_repeat_report"]["status"] == "ok"
    assert card["negative_repeat_report"]["summary"]["random_stratified"]["n_usable_repeats"] == 2
    assert "pr_auc_std" in card["negative_repeat_report"]["summary"]["random_stratified"]
    assert "top_100" in card["negative_repeat_report"]["summary"]["random_stratified"]["top_k_recall_mean"]
    assert "top_100" in card["negative_repeat_report"]["summary"]["random_stratified"]["top_k_enrichment_mean"]
    assert card["species_included"] == ["toy_a", "toy_b"]
    assert card["training_resources"] == ["fixture_a", "fixture_b"]
    assert card["embedding_model"]["model_name"] == [pc.ESMC_300M_MODEL_NAME]
    assert card["embedding_model"]["model_revision"] == ["fixture-rev"]
    assert card["embedding_model"]["embedding_backend"] == ["esmc"]
    assert card["embedding_model"]["pooling"] == ["mean"]
    assert card["pair_model_params"]["max_iter"] == 100
    assert card["final_model_training"] == "all_pairs_after_validation"
    assert card["validation_feature_encoder_fit"] == "train_split_only"
    assert card["calibration_method"] in {"isotonic", "skipped"}
    assert card["density_prior_groupby"] == "clade"
    assert "leave_species_out" in card["validation_report"]
    assert "leave_resource_out" in card["validation_report"]
    assert "leave_family_out" in card["validation_report"]
    assert card["validation_report"]["random_stratified"]["status"] == "ok"
    assert card["validation_report"]["leave_species_out"]["status"] == "ok"
    assert card["validation_report"]["leave_resource_out"]["status"] == "ok"
    assert card["validation_report"]["leave_family_out"]["status"] == "ok"
    assert card["validation_report"]["leave_clade_out"]["status"] == "ok"
    assert "degree_prior" in card["validation_report"]["random_stratified"]["baseline_pr_auc"]
    assert "expression_only" in card["validation_report"]["random_stratified"]["baseline_pr_auc"]
    assert "density_matched_random" in card["validation_report"]["random_stratified"]["baseline_pr_auc"]
    assert card["validation_report"]["leave_resource_out"]["folds"][0]["positive_resource_blacklist_for_fold"] in {"fixture_a", "fixture_b"}
    assert card["validation_report"]["random_stratified"]["family_failure_cases"]
    family_case = card["validation_report"]["random_stratified"]["family_failure_cases"][0]
    assert {"ligand_family", "receptor_family", "n_pairs", "failure_score"}.issubset(family_case)
    assert "top_100" in card["metrics"]["top_k_recall"]
    assert "top_100" in card["metrics"]["top_k_enrichment"]
    assert "degree_prior" in card["metrics"]["baseline_top_k_recall"]
    assert "degree_prior" in card["metrics"]["baseline_top_k_enrichment"]
    assert "family_pair_transfer" in card["validation_report"]["random_stratified"]["baseline_pr_auc"]
    assert "embedding_cosine" in card["validation_report"]["leave_species_out"]["summary"]["mean_baseline_pr_auc"]
    assert "top_100" in card["validation_report"]["leave_species_out"]["summary"]["mean_top_k_precision"]
    assert "top_100" in card["validation_report"]["leave_species_out"]["summary"]["mean_top_k_recall"]
    assert "top_100" in card["validation_report"]["leave_species_out"]["summary"]["mean_top_k_enrichment"]
    assert "role_only" in card["validation_report"]["leave_species_out"]["summary"]["mean_baseline_top_k_precision"]
    assert "role_only" in card["validation_report"]["leave_species_out"]["summary"]["mean_baseline_top_k_recall"]
    assert "role_only" in card["validation_report"]["leave_species_out"]["summary"]["mean_baseline_top_k_enrichment"]
    assert "expression_only" in card["validation_report"]["leave_species_out"]["summary"]["mean_baseline_top_k_precision"]
    assert "density_matched_random" in card["validation_report"]["leave_species_out"]["summary"]["mean_baseline_top_k_precision"]
    gates = pc.evaluate_lr_model_quality_gates(card, required_splits=["leave_species_out"], min_pr_auc_delta=-1.0)
    assert gates.attrs["passed"]
    assert {"degree_prior", "embedding_cosine", "expression_only", "role_only", "density_matched_random", "random"}.issubset(set(gates["baseline"]))
    top_k_gates = pc.evaluate_lr_model_quality_gates(
        card,
        required_splits=["leave_species_out"],
        min_pr_auc_delta=-1.0,
        top_k="top_100",
        min_top_k_delta=-1.0,
    )
    assert top_k_gates.attrs["passed"]
    assert "delta_top_k_precision" in top_k_gates.columns
    strict_gates = pc.evaluate_lr_model_quality_gates(tmp_path, required_splits=["leave_species_out"], min_pr_auc_delta=2.0)
    assert not strict_gates.attrs["passed"]
    assert {"model_score", "calibrated_probability"}.issubset(scores.columns)
    assert {"nearest_reference_ligand", "nearest_reference_receptor", "nearest_reference_species", "nearest_reference_resource"}.issubset(scores.columns)
    assert scores.loc[0, "nearest_reference_ligand"] != ""
    assert np.isfinite(scores.loc[0, "nearest_reference_distance"])
    assert scores.loc[0, "model_score"] >= 0.0


@pytest.mark.predict
def test_lightgbm_predictor_import_path_when_available(tmp_path):
    pytest.importorskip("lightgbm")
    interactions, embeddings = _fixture_training()

    card = pc.train_lr_link_predictor(interactions, embeddings, model="lightgbm", output_dir=tmp_path, negative_ratio=1)

    assert card["model_type"] == "lightgbm"
