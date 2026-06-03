import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import importlib.util
import os
import shutil
import subprocess
import warnings
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy import sparse
from anndata import AnnData

import pyccc as pc
import pyccc.analysis as analysis
import pyccc.plotting as cp
from pyccc.database import _cellchatdb_from_object


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def make_adata():
    x = np.array(
        [
            [5, 0, 0, 0, 1, 0],
            [4, 0, 0, 0, 1, 0],
            [0, 3, 2, 2, 0, 1],
            [0, 4, 3, 3, 0, 1],
            [0, 0, 4, 0, 5, 2],
            [0, 0, 5, 0, 4, 2],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {
            "cell_type": ["A", "A", "B", "B", "C", "C"],
            "condition": ["ctrl", "stim", "ctrl", "stim", "ctrl", "stim"],
        },
        index=[f"cell{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["TGFB1", "TGFBR1", "TGFBR2", "CXCR4", "CXCL12", "CD74"])
    return AnnData(x, obs=obs, var=var)


def test_compute_communication_and_network():
    lr = pd.DataFrame(
        {
            "ligand": ["TGFB1", "CXCL12", "CD74"],
            "receptor": ["TGFBR1_TGFBR2", "CXCR4", "CXCR4"],
            "pathway": ["TGFb", "CXCL", "MIF"],
        }
    )
    res = pc.compute_communication(make_adata(), "cell_type", lr, min_pct=0.0, aggregate="mean")
    assert set(["source", "target", "ligand", "receptor", "prob"]).issubset(res.interactions.columns)
    assert float(res.network().loc["A", "B"]) > 0
    assert "TGFb" in set(res.pathway_summary()["pathway"])


def test_cellchatdb_object_conversion_expands_complexes():
    obj = {
        "interaction": pd.DataFrame(
            {
                "interaction_name": ["TGFB1_TGFbR1_R2"],
                "pathway_name": ["TGFb"],
                "ligand": ["TGFB1"],
                "receptor": ["TGFbR1_R2"],
                "annotation": ["Secreted Signaling"],
                "evidence": ["test"],
                "agonist": ["TGFb agonist"],
                "interaction_name_2": ["TGFB1 - (TGFBR1+TGFBR2)"],
            }
        ),
        "complex": pd.DataFrame(
            {"subunit_1": ["TGFBR1"], "subunit_2": ["TGFBR2"], "subunit_3": [""], "subunit_4": [""]},
            index=["TGFbR1_R2"],
        ),
        "cofactor": pd.DataFrame({"cofactor1": ["THBS1"], "cofactor2": [""], "cofactor3": [""]}, index=["TGFb agonist"]),
    }
    db = _cellchatdb_from_object(obj, species="human")
    assert db.name == "cellchatdb_human"
    assert db.interactions.loc[0, "receptor"] == "TGFBR1_TGFBR2"
    assert db.interactions.loc[0, "agonist_genes"] == "THBS1"
    assert "interaction_raw" in db.metadata


def test_official_human_skin_factor_label_mapping_matches_cellchat_levels():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "cellchat_official_plotnine_demo.py"
    spec = importlib.util.spec_from_file_location("cellchat_official_plotnine_demo", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert [module.CELL_LABELS[i] for i in range(1, 4)] == ["APOE+ FIB", "FBN1+ FIB", "COL11A1+ FIB"]


def test_sparse_mean_matches_dense_and_validates_options():
    adata = make_adata()
    lr = pc.toy_lr_table()
    dense = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean")

    sparse_adata = adata.copy()
    sparse_adata.X = sparse.csr_matrix(sparse_adata.X)
    sparse_res = pc.compute_communication(sparse_adata, "cell_type", lr, min_pct=0.0, aggregate="mean")

    cols = ["source", "target", "ligand", "receptor", "prob"]
    pd.testing.assert_frame_equal(dense.interactions[cols], sparse_res.interactions[cols])

    with pytest.raises(ValueError, match="min_pct"):
        pc.compute_communication(adata, "cell_type", lr, min_pct=1.5)
    with pytest.raises(ValueError, match="layer"):
        pc.compute_communication(adata, "cell_type", lr, layer="counts", use_raw=True)


def test_gene_symbols_key_supports_cellxgene_style_var_names():
    adata = make_adata()
    adata.var["symbol"] = adata.var_names
    adata.var_names = [f"ENSG{i:011d}" for i in range(adata.n_vars)]
    lr = pd.DataFrame(
        {
            "ligand": ["TGFB1", "CXCL12"],
            "receptor": ["TGFBR1_TGFBR2", "CXCR4"],
            "pathway": ["TGFb", "CXCL"],
        }
    )

    with pytest.raises(ValueError, match="adata.var_names"):
        pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean")

    res = pc.compute_communication(
        adata,
        "cell_type",
        lr,
        gene_symbols_key="symbol",
        min_pct=0.0,
        aggregate="mean",
        n_permutations=3,
        random_state=11,
    )
    assert not res.interactions.empty
    assert {"TGFB1", "CXCL12"}.issubset(set(res.interactions["ligand"]))
    assert res.interactions["pvalue"].notna().all()

    markers = pc.identify_overexpressed_genes(adata, "cell_type", gene_symbols_key="symbol", min_pct=0.0, min_logfc=0.0)
    assert "TGFB1" in set(markers["gene"])


def test_gene_symbols_key_accepts_categorical_symbols():
    adata = make_adata()
    adata.var["symbol"] = pd.Categorical(adata.var_names.astype(str))
    adata.var_names = [f"ENSG{i:011d}" for i in range(adata.n_vars)]

    res = pc.compute_communication(
        adata,
        "cell_type",
        pc.toy_lr_table(),
        gene_symbols_key="symbol",
        min_pct=0.0,
        aggregate="mean",
    )

    assert not res.interactions.empty


def test_gene_symbols_key_validates_duplicate_symbols():
    adata = make_adata()
    adata.var["symbol"] = ["TGFB1", "TGFB1", "TGFBR2", "CXCR4", "CXCL12", "CD74"]

    with pytest.raises(ValueError, match="duplicated gene names"):
        pc.compute_communication(adata, "cell_type", pc.toy_lr_table(), gene_symbols_key="symbol")


def test_signaling_expression_frame_matches_dotplot_semantics():
    adata = make_adata()
    lr = pc.toy_lr_table()
    res = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean")

    frame = pc.signaling_expression_frame(adata, result=res, signaling="TGFb")
    assert {"group", "gene", "mean_expression", "pct_expressed", "scaled_expression", "n_cells"}.issubset(frame.columns)
    assert list(frame["group"].cat.categories) == ["A", "B", "C"]
    assert list(frame["gene"].cat.categories) == ["TGFB1", "TGFBR1", "TGFBR2"]
    tgfb_a = frame[(frame["group"].astype(str) == "A") & (frame["gene"].astype(str) == "TGFB1")].iloc[0]
    assert tgfb_a["mean_expression"] == pytest.approx(4.5)
    assert tgfb_a["pct_expressed"] == pytest.approx(1.0)

    symbol_adata = adata.copy()
    symbol_adata.var["symbol"] = symbol_adata.var_names
    symbol_adata.var_names = [f"ENSG{i:011d}" for i in range(symbol_adata.n_vars)]
    symbol_frame = pc.signaling_expression_frame(symbol_adata, "cell_type", features=["TGFB1"], gene_symbols_key="symbol")
    assert list(symbol_frame["gene"].cat.categories) == ["TGFB1"]

    with pytest.raises(KeyError, match="Requested genes"):
        pc.signaling_expression_frame(adata, "cell_type", features=["NOT_A_GENE"])


def test_cellchat_score_method_and_parallel_permutations():
    adata = make_adata()
    lr = pc.toy_lr_table()
    cellchat = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", score_method="cellchat")
    assert not cellchat.interactions.empty
    assert cellchat.interactions["prob"].between(0, 1).all()

    serial = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", n_permutations=6, random_state=7, n_jobs=1)
    parallel = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", n_permutations=6, random_state=7, n_jobs=2)
    pd.testing.assert_series_equal(serial.interactions["pvalue"], parallel.interactions["pvalue"])


def test_array_backend_validates_and_falls_back_to_cpu(monkeypatch):
    adata = make_adata()
    lr = pc.toy_lr_table()

    with pytest.raises(ValueError, match="array_backend"):
        pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", array_backend="not-a-backend")

    cpu = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", array_backend="cpu")

    def missing_cupy():
        raise ModuleNotFoundError("cupy")

    monkeypatch.setattr(analysis, "_cupy_module", missing_cupy)
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        fallback = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", array_backend="cupy")

    cols = ["source", "target", "ligand", "receptor", "prob"]
    pd.testing.assert_frame_equal(cpu.interactions[cols], fallback.interactions[cols])


def test_cupy_backend_sparse_mean_matches_cpu_without_fallback():
    try:
        analysis._cupy_module.cache_clear()
        analysis._cupy_module()
    except Exception as exc:
        pytest.skip(f"CuPy/CUDA unavailable: {exc}")

    adata = make_adata()
    adata.X = sparse.csr_matrix(adata.X)
    lr = pc.toy_lr_table()

    cpu = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", array_backend="cpu")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gpu = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", array_backend="cupy")

    assert not [warning for warning in caught if "CuPy" in str(warning.message)]
    cols = ["source", "target", "ligand", "receptor", "prob"]
    pd.testing.assert_frame_equal(cpu.interactions[cols], gpu.interactions[cols])


def test_cupy_batched_permutations_and_sketches_match_cpu_without_fallback():
    try:
        analysis._cupy_module.cache_clear()
        analysis._cupy_module()
    except Exception as exc:
        pytest.skip(f"CuPy/CUDA unavailable: {exc}")

    adata = make_adata()
    adata.X = sparse.csr_matrix(adata.X)
    lr = pc.toy_lr_table()

    perm_cpu = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", n_permutations=5, random_state=19, array_backend="cpu")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        perm_gpu = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", n_permutations=5, random_state=19, array_backend="cupy")
    assert not [warning for warning in caught if "CuPy" in str(warning.message)]

    perm_cols = ["source", "target", "ligand", "receptor", "pathway", "prob", "pvalue"]
    pd.testing.assert_frame_equal(perm_cpu.interactions[perm_cols], perm_gpu.interactions[perm_cols], check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)

    sketch_cpu = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", downsample_per_group=1, downsample_repeats=4, random_state=23, array_backend="cpu")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sketch_gpu = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", downsample_per_group=1, downsample_repeats=4, random_state=23, array_backend="cupy")
    assert not [warning for warning in caught if "CuPy" in str(warning.message)]

    sketch_cols = ["source", "target", "ligand", "receptor", "pathway", "prob", "prob_std", "stability", "sketch_repeats", "sketch_present"]
    pd.testing.assert_frame_equal(sketch_cpu.interactions[sketch_cols], sketch_gpu.interactions[sketch_cols], check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)


def test_cellchat_score_matches_reference_formula_with_cofactors():
    x = np.array(
        [
            [2, 0, 0, 0, 3],
            [4, 0, 0, 0, 3],
            [0, 3, 3, 6, 0],
            [0, 6, 6, 6, 0],
        ],
        dtype=float,
    )
    obs = pd.DataFrame({"cell_type": ["A", "A", "B", "B"]}, index=[f"cell{i}" for i in range(4)])
    var = pd.DataFrame(index=["LIG", "REC1", "REC2", "COA", "AGO"])
    adata = AnnData(x, obs=obs, var=var)
    lr = pd.DataFrame(
        {
            "ligand": ["LIG"],
            "receptor": ["REC1_REC2"],
            "pathway": ["TEST"],
            "co_A_receptor_genes": ["COA"],
            "agonist_genes": ["AGO"],
        }
    )

    res = pc.compute_communication(
        adata,
        "cell_type",
        lr,
        min_pct=0.0,
        aggregate="mean",
        score_method="cellchat",
        cofactor_adjust=True,
    )
    row = res.interactions[(res.interactions["source"] == "A") & (res.interactions["target"] == "B")].iloc[0]
    assert row["ligand_expr"] == pytest.approx(0.5)
    assert row["receptor_expr"] == pytest.approx(0.75)
    assert row["prob"] == pytest.approx(0.9)

    weighted = pc.compute_communication(
        adata,
        "cell_type",
        lr,
        min_pct=0.0,
        aggregate="mean",
        score_method="cellchat",
        cofactor_adjust=True,
        population_size=True,
    )
    weighted_row = weighted.interactions[(weighted.interactions["source"] == "A") & (weighted.interactions["target"] == "B")].iloc[0]
    assert weighted_row["prob"] == pytest.approx(0.225)


def test_cellchat_reference_comparison_runner_matches_pyccc_on_toy():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "cellchat_reference_comparison.py"
    spec = importlib.util.spec_from_file_location("cellchat_reference_comparison", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    adata = make_adata()
    lr = pc.toy_lr_table()
    pyccc_res = pc.compute_communication(
        adata,
        "cell_type",
        lr,
        min_pct=0.0,
        aggregate="mean",
        score_method="cellchat",
        cofactor_adjust=True,
    )
    ref_res = module.compute_cellchat_formula_reference(
        adata,
        "cell_type",
        lr,
        aggregate="mean",
        cofactor_adjust=True,
    )
    metrics = module.comparison_metrics(pyccc_res, ref_res)
    assert metrics["pearson"].min() > 0.999999
    assert metrics["max_abs_diff"].max() < 1e-12


def test_truncated_mean_and_overexpressed_gate():
    adata = make_adata()
    lr = pc.toy_lr_table()

    tri = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="tri_mean")
    truncated = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="truncated_mean", trim=0.25)
    assert not truncated.interactions.empty
    assert set(tri.interactions[["source", "target", "ligand", "receptor"]].itertuples(index=False, name=None)) == set(
        truncated.interactions[["source", "target", "ligand", "receptor"]].itertuples(index=False, name=None)
    )

    markers = pc.identify_overexpressed_genes(
        adata,
        "cell_type",
        candidate_genes=["TGFB1", "CXCL12", "CXCR4", "TGFBR1", "TGFBR2"],
        min_pct=0.0,
        min_logfc=0.0,
    )
    assert {"group", "gene", "logfc", "overexpressed"}.issubset(markers.columns)
    assert {"TGFB1", "CXCL12"}.issubset(set(markers.loc[markers["overexpressed"], "gene"]))

    oe = pc.identify_overexpressed_interactions(lr, markers, ["A", "B", "C"])
    assert set(oe["source"] + "->" + oe["target"]) == {"A->B", "C->B"}

    gated = pc.compute_communication(
        adata,
        "cell_type",
        lr,
        min_pct=0.0,
        aggregate="truncatedMean",
        trim=0.25,
        de_gate=True,
        de_min_pct=0.0,
        de_min_logfc=0.0,
    )
    assert set(gated.interactions["ligand"]) == {"TGFB1", "CXCL12"}
    assert set(gated.interactions["source"] + "->" + gated.interactions["target"]) == {"A->B", "C->B"}


def test_clipped_mean_matches_sparse_and_cupy_when_available():
    adata = make_adata()
    lr = pc.toy_lr_table()
    dense = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="clipped_mean", clip_quantile=0.95)

    sparse_adata = adata.copy()
    sparse_adata.X = sparse.csr_matrix(sparse_adata.X)
    sparse_res = pc.compute_communication(sparse_adata, "cell_type", lr, min_pct=0.0, aggregate="clipped_mean", clip_quantile=0.95)

    cols = ["source", "target", "ligand", "receptor", "prob"]
    pd.testing.assert_frame_equal(dense.interactions[cols], sparse_res.interactions[cols])

    try:
        analysis._cupy_module.cache_clear()
        analysis._cupy_module()
    except Exception as exc:
        pytest.skip(f"CuPy/CUDA unavailable: {exc}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cupy_res = pc.compute_communication(sparse_adata, "cell_type", lr, min_pct=0.0, aggregate="clipped_mean", clip_quantile=0.95, array_backend="cupy")
    assert not [warning for warning in caught if "CuPy" in str(warning.message)]
    pd.testing.assert_frame_equal(dense.interactions[cols], cupy_res.interactions[cols])


def test_gated_mean_matches_sparse_and_cupy_when_available():
    adata = make_adata()
    lr = pc.toy_lr_table()
    dense = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="gated_mean", clip_quantile=0.95)

    sparse_adata = adata.copy()
    sparse_adata.X = sparse.csr_matrix(sparse_adata.X)
    sparse_res = pc.compute_communication(sparse_adata, "cell_type", lr, min_pct=0.0, aggregate="gated_mean", clip_quantile=0.95)

    cols = ["source", "target", "ligand", "receptor", "prob"]
    pd.testing.assert_frame_equal(dense.interactions[cols], sparse_res.interactions[cols])

    try:
        analysis._cupy_module.cache_clear()
        analysis._cupy_module()
    except Exception as exc:
        pytest.skip(f"CuPy/CUDA unavailable: {exc}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cupy_res = pc.compute_communication(sparse_adata, "cell_type", lr, min_pct=0.0, aggregate="gated_mean", clip_quantile=0.95, array_backend="cupy")
    assert not [warning for warning in caught if "CuPy" in str(warning.message)]
    pd.testing.assert_frame_equal(dense.interactions[cols], cupy_res.interactions[cols])


def test_export_cellchat_bridge_tables_and_rds_smoke(tmp_path):
    adata = make_adata()
    lr = pc.toy_lr_table()
    result = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", score_method="cellchat")
    out_dir = pc.export_cellchat(result, tmp_path / "cellchat_export", lr_table=lr, group_sizes=adata.obs["cell_type"].value_counts())

    interactions = pd.read_csv(out_dir / "pyccc_cellchat_interactions.tsv", sep="\t")
    lr_frame = pd.read_csv(out_dir / "pyccc_cellchat_lr.tsv", sep="\t")
    groups = pd.read_csv(out_dir / "pyccc_cellchat_groups.tsv", sep="\t")
    assert {"source", "target", "interaction_name", "prob", "pval"}.issubset(interactions.columns)
    assert {"interaction_name", "interaction_name_2", "pathway_name", "ligand", "receptor"}.issubset(lr_frame.columns)
    assert list(groups["group"]) == result.groups
    assert (out_dir / "pyccc_to_cellchat.R").exists()

    rscript = shutil.which("Rscript")
    cellchat_lib = Path(__file__).resolve().parents[1] / ".r-lib" / "CellChat"
    if rscript is None or not cellchat_lib.exists():
        pytest.skip("Rscript or local CellChat R package unavailable")
    env = os.environ.copy()
    env["R_LIBS_USER"] = str(cellchat_lib.parent)
    rds_path = tmp_path / "pyccc_cellchat.rds"
    completed = subprocess.run([rscript, str(out_dir / "pyccc_to_cellchat.R"), str(out_dir), str(rds_path)], cwd=Path(__file__).resolve().parents[1], env=env, text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert rds_path.exists()


def test_export_merged_cellchat_bridge_and_rds_smoke(tmp_path):
    adata = make_adata()
    lr = pc.toy_lr_table()
    ctrl = adata[adata.obs["condition"] == "ctrl"].copy()
    stim = adata[adata.obs["condition"] == "stim"].copy()
    diff = pc.compare_samples(stim, ctrl, "cell_type", lr, label_a="stim", label_b="ctrl", min_pct=0.0, aggregate="mean", score_method="cellchat")

    out_dir = pc.export_cellchat_merged(
        diff,
        tmp_path / "cellchat_merged_export",
        lr_table=lr,
        group_sizes_a=stim.obs["cell_type"].value_counts(),
        group_sizes_b=ctrl.obs["cell_type"].value_counts(),
    )

    samples = pd.read_csv(out_dir / "pyccc_cellchat_samples.tsv", sep="\t")
    diff_frame = pd.read_csv(out_dir / "pyccc_cellchat_diff_interactions.tsv", sep="\t")
    assert list(samples["role"]) == ["a", "b"]
    assert {"source", "target", "ligand", "receptor", "prob_a", "prob_b", "delta_prob"}.issubset(diff_frame.columns)
    assert (out_dir / "stim" / "pyccc_cellchat_interactions.tsv").exists()
    assert (out_dir / "ctrl" / "pyccc_cellchat_interactions.tsv").exists()
    assert (out_dir / "pyccc_to_merged_cellchat.R").exists()

    rscript = shutil.which("Rscript")
    cellchat_lib = Path(__file__).resolve().parents[1] / ".r-lib" / "CellChat"
    if rscript is None or not cellchat_lib.exists():
        pytest.skip("Rscript or local CellChat R package unavailable")
    env = os.environ.copy()
    env["R_LIBS_USER"] = str(cellchat_lib.parent)
    rds_path = tmp_path / "pyccc_merged_cellchat.rds"
    plot_dir = tmp_path / "merged_plots"
    completed = subprocess.run(
        [rscript, str(out_dir / "pyccc_to_merged_cellchat.R"), str(out_dir), str(rds_path), str(plot_dir)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr
    assert rds_path.exists()
    manifest = pd.read_csv(plot_dir / "pyccc_merged_cellchat_plot_manifest.tsv", sep="\t")
    assert "compare_interactions_count" in set(manifest["id"])
    assert (manifest["status"] == "ok").any()


def test_wilcoxon_overexpressed_genes_has_adjusted_pvalues():
    markers = pc.identify_overexpressed_genes(
        make_adata(),
        "cell_type",
        method="wilcoxon",
        min_pct=0.0,
        min_logfc=0.0,
        candidate_genes=["TGFB1", "CXCL12"],
    )
    assert "padj" in markers.columns
    assert markers["padj"].notna().any()


def test_downsample_sketch_outputs_stability():
    adata = make_adata()
    lr = pc.toy_lr_table()
    res = pc.compute_communication(
        adata,
        "cell_type",
        lr,
        min_pct=0.0,
        aggregate="mean",
        downsample_per_group=1,
        downsample_repeats=4,
        random_state=3,
    )
    assert {"prob_std", "stability", "sketch_repeats", "sketch_present"}.issubset(res.interactions.columns)
    assert res.interactions["stability"].between(0, 1).all()
    assert set(res.interactions["sketch_repeats"]) == {4}

    permuted = pc.compute_communication(
        adata,
        "cell_type",
        lr,
        min_pct=0.0,
        aggregate="mean",
        downsample_per_group=1,
        downsample_repeats=3,
        n_permutations=2,
        random_state=5,
    )
    assert permuted.interactions["pvalue"].between(0, 1).all()


def test_cpu_batched_sketch_matches_loop_fallback(monkeypatch):
    adata = make_adata()
    adata.X = sparse.csr_matrix(adata.X)
    lr = pc.toy_lr_table()

    optimized = pc.compute_communication(
        adata,
        "cell_type",
        lr,
        min_pct=0.0,
        aggregate="mean",
        score_method="cellchat",
        downsample_per_group=1,
        downsample_repeats=4,
        random_state=23,
        array_backend="cpu",
    )

    def force_loop(*args, **kwargs):
        raise RuntimeError("forced fallback")

    monkeypatch.setattr(analysis, "_repeated_downsample_rows_cpu_batched", force_loop)
    with pytest.warns(RuntimeWarning, match="CPU batched sketches failed"):
        fallback = pc.compute_communication(
            adata,
            "cell_type",
            lr,
            min_pct=0.0,
            aggregate="mean",
            score_method="cellchat",
            downsample_per_group=1,
            downsample_repeats=4,
            random_state=23,
            array_backend="cpu",
        )

    cols = ["source", "target", "ligand", "receptor", "pathway", "prob", "prob_std", "stability", "sketch_repeats", "sketch_present"]
    pd.testing.assert_frame_equal(optimized.interactions[cols], fallback.interactions[cols], check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)


def test_population_size_weighting_scales_probabilities():
    adata = make_adata()
    lr = pc.toy_lr_table()
    base = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean")
    weighted = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", population_size=True)
    assert weighted.interactions["prob"].sum() < base.interactions["prob"].sum()


def test_spatial_distance_decay_scales_probabilities():
    adata = make_adata()
    adata.obsm["spatial"] = np.array([[0, 0], [0.2, 0.1], [1, 0], [1.2, 0.1], [0.5, 1], [0.7, 1.1]], dtype=float)
    lr = pc.toy_lr_table()
    base = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean")
    spatial = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", spatial_key="spatial", distance_decay=0.5)
    assert spatial.interactions["prob"].sum() < base.interactions["prob"].sum()


def test_cofactor_adjustment_changes_probabilities():
    adata = make_adata()
    lr = pd.DataFrame(
            {
                "ligand": ["TGFB1"],
                "receptor": ["TGFBR1_TGFBR2"],
                "pathway": ["TGFb"],
                "co_A_receptor_genes": ["TGFBR1"],
            }
        )
    base = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean")
    adjusted = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean", cofactor_adjust=True)
    assert adjusted.interactions["prob"].sum() > base.interactions["prob"].sum()


def test_compute_communication_aggregates_only_required_genes(monkeypatch):
    base = make_adata()
    extras = np.tile(np.array([[100.0, 200.0]]), (base.n_obs, 1))
    adata = AnnData(
        np.column_stack([base.X, extras]),
        obs=base.obs.copy(),
        var=pd.DataFrame(index=list(base.var_names) + ["UNUSED1", "UNUSED2"]),
    )
    minimal = adata[:, ["TGFB1", "TGFBR1", "TGFBR2", "CXCR4"]].copy()
    lr = pd.DataFrame(
        {
            "ligand": ["TGFB1"],
            "receptor": ["TGFBR1_TGFBR2"],
            "pathway": ["TGFb"],
            "co_A_receptor_genes": ["CXCR4"],
        }
    )

    seen = []
    original = analysis._group_expression_from_labels

    def capture(x, var_names, labels, groups, *, aggregate, trim, clip_quantile=0.99, array_backend="cpu"):
        seen.append(list(var_names.astype(str)))
        return original(x, var_names, labels, groups, aggregate=aggregate, trim=trim, clip_quantile=clip_quantile, array_backend=array_backend)

    monkeypatch.setattr(analysis, "_group_expression_from_labels", capture)
    full = pc.compute_communication(
        adata,
        "cell_type",
        lr,
        min_pct=0.0,
        aggregate="mean",
        cofactor_adjust=True,
        n_permutations=2,
        random_state=13,
        n_jobs=1,
    )
    compact = pc.compute_communication(
        minimal,
        "cell_type",
        lr,
        min_pct=0.0,
        aggregate="mean",
        cofactor_adjust=True,
        n_permutations=2,
        random_state=13,
        n_jobs=1,
    )

    expected = {"TGFB1", "TGFBR1", "TGFBR2", "CXCR4"}
    assert seen
    assert all(set(names) == expected for names in seen)
    cols = ["source", "target", "ligand", "receptor", "pathway", "ligand_expr", "receptor_expr", "prob", "pvalue"]
    pd.testing.assert_frame_equal(full.interactions[cols], compact.interactions[cols])


def test_differential_and_plots_smoke():
    adata = make_adata()
    lr = pc.toy_lr_table()
    ctrl = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="ctrl", min_pct=0.0, aggregate="mean")
    stim = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="stim", min_pct=0.0, aggregate="mean")
    diff = pc.compare_communication(stim, ctrl, label_a="stim", label_b="ctrl")
    assert "delta_prob" in diff.interactions.columns
    assert {"stability_a", "stability_b"}.isdisjoint(diff.interactions.columns)
    assert diff.network_delta.shape == (3, 3)
    assert diff.count_delta.shape == (3, 3)
    assert {"pathway", "prob_a", "prob_b", "delta_prob", "delta_count"}.issubset(diff.pathway_changes.columns)
    assert {"source", "target", "prob_a", "prob_b", "delta_prob", "delta_count"}.issubset(diff.source_target_changes.columns)
    assert diff.differential_network(measure="count").shape == (3, 3)
    changes = pc.signaling_changes(diff, "A")
    similarity_rank = pc.rank_pathway_similarity(diff, method="mds")
    pairwise_embedding = pc.pairwise_pathway_embedding(diff, method="mds")
    assert {"pathway", "delta_outgoing", "delta_incoming", "abs_delta_total", "direction"}.issubset(changes.columns)
    assert {"pathway", "distance", "dim1_a", "dim2_a", "dim1_b", "dim2_b", "delta_prob"}.issubset(similarity_rank.columns)
    assert {"pathway", "condition", "dim1", "dim2", "prob", "count"}.issubset(pairwise_embedding.columns)
    with pytest.raises(ValueError, match="group"):
        pc.signaling_changes(diff, "missing")

    for plotter in (
        cp.net_circle,
        cp.net_heatmap,
        cp.net_chord,
        cp.net_chord_gene,
        cp.net_individual,
        cp.net_hierarchy,
        cp.pathway_heatmap,
        cp.bubble,
        cp.dotplot,
        cp.signaling_role_scatter,
        cp.signaling_role_heatmap,
        cp.signaling_role_network,
        cp.pathway_embedding,
        cp.pathway_river,
        cp.annotation_bar,
        cp.rank_signaling,
    ):
        ax = plotter(stim)
        assert ax is not None
        plt.close(ax.figure)
    ax = cp.lr_contribution(stim, "TGFb")
    assert ax is not None
    plt.close(ax.figure)
    fig, axes = cp.lr_contribution_multi(stim, pathways=["TGFb", "CXCL"], top_n=2)
    assert fig is not None
    assert len(axes) == 2
    plt.close(fig)
    for ax in (
        cp.net_individual(stim, pathway="TGFb", layout="hierarchy", vertex_receiver=[2]),
        cp.net_individual(stim, pathway="TGFb", layout="chord"),
        cp.net_individual(stim, ligand="TGFB1", receptor="TGFBR1_TGFBR2", layout="circle"),
    ):
        assert ax is not None
        plt.close(ax.figure)
    with pytest.raises(ValueError, match="layout"):
        cp.net_individual(stim, pathway="TGFb", layout="bad")
    for ax in (
        cp.signaling_gene_expression(adata, result=stim, signaling="TGFb"),
        cp.signaling_gene_expression(adata, "cell_type", features=["TGFB1"], kind="violin"),
        cp.signaling_gene_expression(adata, "cell_type", features=["TGFB1"], kind="bar"),
    ):
        assert ax is not None
        plt.close(ax.figure)

    for plotter in (cp.compare_interactions, cp.diff_network_circle, cp.diff_heatmap, cp.diff_bubble, cp.rank_signaling_compare, cp.diff_pathway_rank, cp.diff_source_target_rank):
        ax = plotter(diff)
        assert ax is not None
        plt.close(ax.figure)
    ax = cp.signaling_changes_scatter(diff, "A")
    assert ax is not None
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert xlim[0] < 0 < xlim[1]
    assert ylim[0] < 0 < ylim[1]
    plt.close(ax.figure)

    for ax in (
        cp.compare_interactions(diff, value="count"),
        cp.rank_signaling_compare(diff, stacked=True),
        cp.diff_network_circle(diff, measure="count"),
        cp.diff_heatmap(diff, measure="count"),
        cp.diff_pathway_rank(diff, measure="count"),
        cp.diff_source_target_rank(diff, measure="count"),
        cp.pathway_embedding_pairwise(diff, method="mds"),
        cp.pathway_similarity_rank(diff, method="mds"),
    ):
        assert ax is not None
        plt.close(ax.figure)
    role_fig, role_axes = cp.signaling_role_heatmap_compare(diff, mode="incoming")
    assert role_fig is not None
    assert len(role_axes) == 2
    plt.close(role_fig)
    role_fig, role_axes = plt.subplots(1, 2)
    out_fig, out_axes = cp.signaling_role_heatmap_compare(diff, mode="all", axes=role_axes, cluster_rows=True, cluster_cols=True)
    assert out_fig is role_fig
    assert len(out_axes) == 2
    plt.close(role_fig)
    fig, axes = cp.key_plot_gallery(stim, diff)
    assert fig is not None
    assert axes.shape == (3, 3)
    plt.close(fig)


def test_compare_samples_direct_ann_data_inputs():
    adata = make_adata()
    lr = pc.toy_lr_table()
    ctrl = adata[adata.obs["condition"] == "ctrl"].copy()
    stim = adata[adata.obs["condition"] == "stim"].copy()

    diff = pc.compare_samples(stim, ctrl, "cell_type", lr, label_a="stim", label_b="ctrl", min_pct=0.0, aggregate="mean")
    assert diff.label_a == "stim"
    assert diff.label_b == "ctrl"
    assert diff.network_delta.shape == (3, 3)
    assert "delta_prob" in diff.interactions.columns

    ctrl_ab = ctrl[ctrl.obs["cell_type"].isin(["A", "B"])].copy()
    union_diff = pc.compare_samples(stim, ctrl_ab, "cell_type", lr, label_a="stim", label_b="ctrl", align_groups="union", min_pct=0.0, aggregate="mean")
    assert union_diff.groups == ["A", "B", "C"]
    assert union_diff.network_b.loc["C"].sum() == 0

    strict_diff = pc.compare_samples(stim, ctrl, "cell_type", lr, align_groups="strict", min_pct=0.0, aggregate="mean")
    assert strict_diff.network_delta.shape == (3, 3)
    with pytest.raises(ValueError, match="identical ordered cell groups"):
        pc.compare_samples(stim, ctrl_ab, "cell_type", lr, align_groups="strict", min_pct=0.0, aggregate="mean")


def test_differential_preserves_sketch_confidence_columns():
    adata = make_adata()
    lr = pc.toy_lr_table()
    ctrl = pc.compute_communication(
        adata,
        "cell_type",
        lr,
        condition_key="condition",
        condition="ctrl",
        min_pct=0.0,
        aggregate="mean",
        downsample_per_group=1,
        downsample_repeats=2,
    )
    stim = pc.compute_communication(
        adata,
        "cell_type",
        lr,
        condition_key="condition",
        condition="stim",
        min_pct=0.0,
        aggregate="mean",
        downsample_per_group=1,
        downsample_repeats=2,
    )
    diff = pc.compare_communication(stim, ctrl, label_a="stim", label_b="ctrl")
    assert {"prob_std_a", "prob_std_b", "stability_a", "stability_b"}.issubset(diff.interactions.columns)


def test_zero_and_empty_plots_smoke():
    adata = make_adata()
    lr = pc.toy_lr_table()
    res = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, aggregate="mean")
    zero_diff = pc.compare_communication(res, res, label_a="a", label_b="b")

    assert cp.diff_heatmap(zero_diff) is not None
    assert cp.diff_bubble(zero_diff) is not None
    assert cp.bubble(res, size_by="pvalue") is not None
    assert cp.bubble(res, max_pvalue=0.05) is not None
    assert cp.diff_bubble(zero_diff, max_pvalue=0.05) is not None
    assert cp.net_heatmap(res, cluster_rows=True, cluster_cols=True) is not None
    assert cp.net_chord_gene(res, level="pathway", top_n=2) is not None
    with pytest.raises(ValueError, match="level"):
        cp.net_chord_gene(res, level="bad")
    assert cp.signaling_role_scatter(res, pathways=["TGFb"]) is not None
    assert cp.signaling_role_heatmap(res, cluster_rows=True, cluster_cols=True) is not None
    assert cp.pathway_embedding(res, similarity="structural", top_label=2, label_pathways=["TGFb"]) is not None
    assert cp.pathway_embedding(res, cluster=True, n_clusters=2, method="mds") is not None
    assert cp.signaling_gene_expression(adata, result=res, signaling="TGFb", min_pct=0.5, standard_scale=False) is not None
    assert cp.diff_heatmap(zero_diff, cluster_rows=True, cluster_cols=True) is not None
    assert cp.pathway_heatmap(res, pathways=["missing"]) is not None

    no_edges = pc.compute_communication(adata, "cell_type", lr, min_pct=0.0, min_expr=999.0, aggregate="mean")
    assert no_edges.interactions.empty
    assert "pvalue" in no_edges.interactions.columns
    assert cp.net_chord(no_edges) is not None
    assert cp.net_chord_gene(no_edges) is not None


def test_spatial_network_smoke():
    adata = make_adata()
    adata.obsm["spatial"] = np.array([[0, 0], [0.2, 0.1], [1, 0], [1.2, 0.1], [0.5, 1], [0.7, 1.1]], dtype=float)
    res = pc.compute_communication(adata, "cell_type", pc.toy_lr_table(), min_pct=0.0, aggregate="mean")
    assert cp.spatial_network(adata, res) is not None


def test_cellchat_report_smoke(tmp_path):
    adata = make_adata()
    lr = pc.toy_lr_table()
    ctrl = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="ctrl", min_pct=0.0, aggregate="mean")
    stim = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="stim", min_pct=0.0, aggregate="mean")
    diff = pc.compare_communication(stim, ctrl, label_a="stim", label_b="ctrl")
    outfile = pc.save_cellchat_report(stim, tmp_path / "report.pdf", diff=diff)
    assert outfile.exists()
    assert outfile.stat().st_size > 0


def test_liana_conversion():
    tbl = pd.DataFrame(
        {
            "source": ["A"],
            "target": ["B"],
            "ligand_complex": ["TGFB1"],
            "receptor_complex": ["TGFBR1_TGFBR2"],
            "magnitude_rank": [0.5],
        }
    )
    res = pc.from_liana_results(tbl, groupby="cell_type")
    assert res.lr_name == "liana"
    assert res.interactions.loc[0, "prob"] > 0


def test_communication_patterns():
    res = pc.compute_communication(make_adata(), "cell_type", pc.toy_lr_table(), min_pct=0.0, aggregate="mean")
    patterns = pc.compute_communication_patterns(res, n_patterns=2, mode="outgoing")
    selection = pc.select_communication_pattern_number(res, k_range=range(1, 3), n_runs=2)
    assert patterns.n_patterns == 2
    assert set(["group", "pattern", "weight"]).issubset(patterns.group_pattern.columns)
    assert set(["pathway", "pattern", "weight"]).issubset(patterns.pathway_pattern.columns)
    assert patterns.matrix.shape[0] == len(res.groups)
    assert set(["k", "explained", "stability"]).issubset(selection.columns)
    similarity = pc.compute_pathway_similarity(res)
    structural = pc.compute_pathway_similarity(res, similarity="structural")
    embedding = pc.compute_pathway_embedding(res)
    umap_or_mds = pc.compute_pathway_embedding(res, method="auto", n_neighbors=2)
    clusters = pc.compute_pathway_clusters(res, n_clusters=2)
    kmeans_clusters = pc.compute_pathway_clusters(res, method="kmeans", n_clusters=2, embedding=embedding)
    assert similarity.shape[0] == similarity.shape[1] >= 1
    assert structural.shape == similarity.shape
    assert np.allclose(np.diag(similarity), 1.0)
    assert np.allclose(similarity, similarity.T)
    assert {"pathway", "dim1", "dim2", "prob", "count", "embedding_method"}.issubset(embedding.columns)
    assert len(embedding) == similarity.shape[0]
    assert set(umap_or_mds["embedding_method"]).issubset({"umap", "mds"})
    assert {"pathway", "cluster", "n_clusters", "cluster_method", "prob", "count"}.issubset(clusters.columns)
    assert clusters["cluster"].between(1, 2).all()
    assert set(kmeans_clusters["cluster_method"]) == {"kmeans"}
    with pytest.raises(ValueError, match="similarity"):
        pc.compute_pathway_similarity(res, similarity="bad")
    with pytest.raises(ValueError, match="method"):
        pc.compute_pathway_clusters(res, method="bad")


@pytest.mark.skipif(importlib.util.find_spec("plotnine") is None, reason="plotnine extra is not installed")
def test_plotnine_smoke():
    import pyccc.ggplot as cg

    adata = make_adata()
    lr = pc.toy_lr_table()
    ctrl = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="ctrl", min_pct=0.0, aggregate="mean")
    stim = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="stim", min_pct=0.0, aggregate="mean")
    diff = pc.compare_communication(stim, ctrl, label_a="stim", label_b="ctrl")
    patterns = pc.compute_communication_patterns(stim, n_patterns=2)
    selection = pc.select_communication_pattern_number(stim, k_range=range(1, 3), n_runs=2)

    for plot in (
        cg.bubble(stim, top_n=5),
        cg.bubble(stim, top_n=5, facet_by="pathway"),
        cg.bubble(stim, top_n=5, max_pvalue=0.05),
        cg.diff_bubble(diff, top_n=5),
        cg.diff_bubble(diff, top_n=5, facet_by="pathway"),
        cg.diff_bubble(diff, top_n=5, max_pvalue=0.05),
        cg.diff_pathway_rank(diff),
        cg.diff_pathway_rank(diff, measure="count"),
        cg.diff_source_target_rank(diff),
        cg.signaling_changes_scatter(diff, "A"),
        cg.pathway_embedding_pairwise(diff, method="mds"),
        cg.pathway_similarity_rank(diff, method="mds"),
        cg.dotplot(stim),
        cg.signaling_role_scatter(stim),
        cg.signaling_role_scatter(stim, pathways=["TGFb"], significant_only=False),
        cg.signaling_role_heatmap_compare(diff, mode="outgoing"),
        cg.lr_contribution(stim, "TGFb"),
        cg.lr_contribution(stim, "missing"),
        cg.lr_contribution_multi(stim, pathways=["TGFb", "CXCL"], top_n=2),
        cg.pathway_embedding(stim),
        cg.pathway_embedding(stim, similarity="structural", top_label=2, label_pathways=["TGFb"]),
        cg.pathway_embedding(stim, cluster=True, n_clusters=2, method="mds"),
        cg.signaling_gene_expression(adata, result=stim, signaling="TGFb"),
        cg.signaling_gene_expression(adata, "cell_type", features=["TGFB1"], kind="violin"),
        cg.signaling_gene_expression(adata, "cell_type", features=["TGFB1"], kind="bar"),
        cg.pathway_heatmap(stim, cluster_rows=True, cluster_cols=True),
        cg.rank_signaling(stim),
        cg.annotation_bar(stim),
        cg.rank_signaling_compare(diff),
        cg.rank_signaling_compare(diff, stacked=True),
        cg.compare_interactions(diff),
        cg.pathway_river(stim, top_n=5),
        cg.pattern_dot(patterns),
        cg.pattern_river(patterns),
        cg.pattern_number_plot(selection),
    ):
        fig = plot.draw()
        assert fig is not None


@pytest.mark.skipif(importlib.util.find_spec("plotnine") is None, reason="plotnine extra is not installed")
def test_plotnine_rank_plots_use_value_order():
    import pyccc.ggplot as cg

    adata = make_adata()
    lr = pc.toy_lr_table()
    ctrl = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="ctrl", min_pct=0.0, aggregate="mean")
    stim = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="stim", min_pct=0.0, aggregate="mean")
    diff = pc.compare_communication(stim, ctrl, label_a="stim", label_b="ctrl")

    rank = cg.rank_signaling(stim)
    expected_rank = stim.pathway_summary().head(30).sort_values("prob", ascending=True)["pathway"].astype(str).tolist()
    assert list(rank.data["pathway"].cat.categories) == expected_rank

    compare = cg.rank_signaling_compare(diff)
    merged = (
        stim.pathway_summary()[["pathway", "prob"]]
        .rename(columns={"prob": "stim"})
        .merge(ctrl.pathway_summary()[["pathway", "prob"]].rename(columns={"prob": "ctrl"}), on="pathway", how="outer")
        .fillna(0.0)
    )
    merged["total"] = merged["stim"] + merged["ctrl"]
    expected_compare = merged.sort_values("total", ascending=False).head(30).sort_values("total", ascending=True)["pathway"].astype(str).tolist()
    assert list(compare.data["pathway"].cat.categories) == expected_compare
    assert list(compare.data["condition"].cat.categories) == ["stim", "ctrl"]


@pytest.mark.skipif(importlib.util.find_spec("plotnine") is None, reason="plotnine extra is not installed")
def test_plotnine_pattern_number_recommends_elbow():
    import pyccc.ggplot as cg

    selection = pd.DataFrame(
        {
            "k": [2, 3, 4, 5],
            "mean_reconstruction_error": [7.1, 3.0, 2.25, 1.7],
            "explained": [0.976, 0.995, 0.997, 0.998],
            "stability": [1.0, 1.0, 1.0, 1.0],
        }
    )

    assert cg._recommended_pattern_k(selection) == 3
    plot = cg.pattern_number_plot(selection)
    assert list(plot.data["metric"].cat.categories) == ["reconstruction error", "explained fraction", "component stability"]
    assert set(plot.data.loc[plot.data["k"] == 3, "metric"].astype(str)) == {"reconstruction error", "explained fraction", "component stability"}


@pytest.mark.skipif(importlib.util.find_spec("plotly") is None, reason="interactive extra is not installed")
def test_interactive_river_smoke(tmp_path):
    adata = make_adata()
    res = pc.compute_communication(adata, "cell_type", pc.toy_lr_table(), min_pct=0.0, aggregate="mean")
    patterns = pc.compute_communication_patterns(res, n_patterns=2)

    pathway_fig = pc.interactive_pathway_river(res, top_n=5)
    pattern_fig = pc.interactive_pattern_river(patterns)
    assert pathway_fig.data
    assert pattern_fig.data

    out = pc.save_interactive_html(pathway_fig, tmp_path / "river.html", include_plotlyjs="cdn")
    assert out.exists()
    assert "Plotly" in out.read_text()
