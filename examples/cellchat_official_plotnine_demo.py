from __future__ import annotations

import os
import urllib.request
import warnings
from pathlib import Path

import pandas as pd
import rdata
import matplotlib.pyplot as plt
from anndata import AnnData
from scipy import sparse

import pyccc as pc
import pyccc.plotting as cp
import pyccc.ggplot as cg

OFFICIAL_TUTORIAL = "https://github.com/jinworks/CellChat/blob/main/tutorial/CellChat-vignette.Rmd"
HUMAN_SKIN_RDA_URL = "https://ndownloader.figshare.com/files/25950872"

CELL_LABELS = {
    1: "APOE+ FIB",
    2: "FBN1+ FIB",
    3: "COL11A1+ FIB",
    4: "Inflam. FIB",
    5: "cDC1",
    6: "cDC2",
    7: "LC",
    8: "Inflam. DC",
    9: "TC",
    10: "Inflam. TC",
    11: "CD40LG+ TC",
    12: "NKT",
}


def main() -> None:
    cache_dir = Path(os.environ.get("PYCCC_CACHE_DIR", Path.home() / ".cache" / "pyccc")) / "cellchat_official"
    out_dir = Path(os.environ.get("PYCCC_OUTPUT_DIR", "."))
    proxy = os.environ.get("PYCCC_PROXY") or os.environ.get("https_proxy") or os.environ.get("http_proxy")

    data_path = cache_dir / "data_humanSkin_CellChat.rda"
    _download_if_missing(HUMAN_SKIN_RDA_URL, data_path, proxy=proxy)

    adata = read_official_human_skin(data_path)
    db = pc.load_cellchatdb("human", cache_dir=cache_dir, force_download=False, proxy=proxy)
    secreted = pc.CellChatDB(
        db.interactions[db.interactions["annotation"].eq("Secreted Signaling")].copy(),
        name=f"{db.name}_secreted",
        metadata=db.metadata,
    )

    cellchat_kwargs = dict(
        min_pct=0.05,
        aggregate="tri_mean",
        score_method="cellchat",
        cofactor_adjust=True,
    )
    ls = pc.compute_communication(
        adata,
        "cell_type",
        secreted,
        condition_key="condition",
        condition="LS",
        **cellchat_kwargs,
    )
    nl = pc.compute_communication(
        adata,
        "cell_type",
        secreted,
        condition_key="condition",
        condition="NL",
        **cellchat_kwargs,
    )
    diff = pc.compare_communication(ls, nl, label_a="LS", label_b="NL")
    patterns = pc.compute_communication_patterns(ls, mode="outgoing", n_patterns=3)
    selection = pc.select_communication_pattern_number(ls, mode="outgoing", k_range=range(2, 6), n_runs=3)

    out_dir.mkdir(parents=True, exist_ok=True)
    _save_outputs(out_dir, adata, secreted, ls, diff, patterns, selection)

    print(f"Official tutorial: {OFFICIAL_TUTORIAL}")
    print(f"Official data: {HUMAN_SKIN_RDA_URL}")
    print(f"AnnData: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"CellChatDB v2 Secreted Signaling interactions: {len(secreted.interactions)}")
    print(f"LS interactions: {len(ls.interactions)}; NL interactions: {len(nl.interactions)}")
    print("Top LS pathways:")
    print(ls.pathway_summary().head(8).to_string(index=False))
    print(f"Saved plotnine figures to {out_dir.resolve()}")


def read_official_human_skin(path: str | Path) -> AnnData:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = rdata.read_rda(path)["data_humanSkin"]

    data = obj["data"]
    matrix = sparse.csc_matrix((data.x, data.i, data.p), shape=tuple(data.Dim)).T.tocsr()
    meta = obj["meta"].copy()
    meta["cell_type"] = meta["labels"].astype(int).map(CELL_LABELS)
    meta["cluster_id"] = meta["labels"].astype(str)
    var = pd.DataFrame(index=pd.Index(data.Dimnames[0], name="gene"))
    return AnnData(matrix, obs=meta, var=var)


def _download_if_missing(url: str, path: Path, *, proxy: str | None) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handlers = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    opener = urllib.request.build_opener(*handlers)
    with opener.open(url, timeout=120) as response, path.open("wb") as handle:
        handle.write(response.read())


def _save_outputs(out_dir: Path, adata: AnnData, lr_table: pc.CellChatDB, ls, diff, patterns, selection) -> None:
    targets = ["cDC1", "cDC2", "LC", "Inflam. DC", "TC", "Inflam. TC", "CD40LG+ TC"]
    chemokines = ["CCL", "CXCL"]

    cg.bubble(
        ls,
        sources=["Inflam. FIB"],
        targets=targets,
        pathways=chemokines,
        top_n=35,
        title="Official CellChat skin data: Inflam. FIB chemokines",
    ).save(out_dir / "pyccc_cellchat_official_bubble_chemokine.png", width=9.0, height=5.8, dpi=220, verbose=False)
    cg.bubble(
        ls,
        top_n=18,
        top_pairs=8,
        compact_pairs=True,
        title="Official CellChat skin data: top ligand-receptor programs",
    ).save(out_dir / "pyccc_cellchat_official_bubble_facet_pathway.png", width=8.8, height=6.6, dpi=240, verbose=False)
    for pathway in ["MIF", "CypA", "GALECTIN", "CXCL", "TNF", "ANNEXIN"]:
        cg.bubble(
            ls,
            pathways=[pathway],
            top_n=8,
            top_pairs=8,
            compact_pairs=True,
            title=f"Official CellChat skin data: {pathway} ligand-receptor programs",
        ).save(out_dir / f"pyccc_cellchat_official_bubble_{pathway.lower()}_programs.png", width=7.2, height=4.8, dpi=240, verbose=False)
    cg.pathway_heatmap(
        ls,
        top_pairs=28,
        compact_pairs=True,
        cluster_rows=True,
        cluster_cols=True,
        title="Official CellChat skin data: pathway heatmap",
    ).save(out_dir / "pyccc_cellchat_official_pathway_heatmap.png", width=9.5, height=8.0, dpi=220, verbose=False)
    cg.pathway_river(ls, top_n=10, title="Official CellChat skin data: source-to-pathway river").save(
        out_dir / "pyccc_cellchat_official_pathway_river.png", width=9.0, height=5.8, dpi=220, verbose=False
    )
    cg.signaling_role_scatter(ls, title="Official CellChat skin data: outgoing vs incoming roles").save(
        out_dir / "pyccc_cellchat_official_signaling_role_scatter.png", width=6.8, height=5.6, dpi=220, verbose=False
    )
    cg.lr_contribution(ls, "MIF", top_n=12, title="Official CellChat skin data: MIF LR contribution").save(
        out_dir / "pyccc_cellchat_official_lr_contribution_mif.png", width=6.8, height=4.8, dpi=220, verbose=False
    )
    cg.lr_contribution_multi(
        ls,
        pathways=["MIF", "CypA", "GALECTIN", "ANNEXIN"],
        top_n=8,
        title="Official CellChat skin data: LR contribution by pathway",
    ).save(
        out_dir / "pyccc_cellchat_official_lr_contribution_multi.png", width=10.0, height=4.2, dpi=220, verbose=False
    )
    cg.pathway_embedding(
        ls,
        cluster=True,
        label_pathways=["MIF", "CypA", "GALECTIN", "CXCL", "COMPLEMENT", "FGF", "TNF", "IL4"],
        title="Official CellChat skin data: pathway network embedding groups",
    ).save(
        out_dir / "pyccc_cellchat_official_pathway_embedding.png", width=7.0, height=5.8, dpi=220, verbose=False
    )
    fig, ax = plt.subplots(figsize=(7.4, 7.4), constrained_layout=True)
    cp.net_chord_gene(
        ls,
        sources=["Inflam. FIB"],
        targets=targets,
        pathways=["CXCL"],
        top_n=14,
        ax=ax,
        title="Official CellChat skin data: CXCL LR-mediated chord",
    )
    fig.savefig(out_dir / "pyccc_cellchat_official_lr_chord_cxcl.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    cp.net_individual(
        ls,
        pathway="CXCL",
        layout="hierarchy",
        vertex_receiver=targets,
        ax=axes[0],
        title="CXCL individual LR hierarchy",
    )
    cp.net_individual(ls, pathway="CXCL", layout="circle", ax=axes[1], title="CXCL individual LR circle")
    cp.net_individual(ls, pathway="CXCL", layout="chord", ax=axes[2], title="CXCL individual LR chord")
    fig.savefig(out_dir / "pyccc_cellchat_official_lr_individual_cxcl.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    cg.signaling_gene_expression(
        adata,
        result=ls,
        signaling="CXCL",
        lr_table=lr_table,
        enriched_only=False,
        title="Official CellChat skin data: CXCL signaling gene expression",
    ).save(
        out_dir / "pyccc_cellchat_official_gene_expression_cxcl.png", width=7.8, height=5.4, dpi=220, verbose=False
    )
    cg.rank_signaling_compare(diff, top_n=12, title="Official CellChat skin data: LS vs NL pathway rank").save(
        out_dir / "pyccc_cellchat_official_rank_compare.png", width=8.5, height=6.0, dpi=220, verbose=False
    )
    cg.rank_signaling_compare(diff, top_n=12, stacked=True, title="Official CellChat skin data: stacked pathway rank").save(
        out_dir / "pyccc_cellchat_official_rank_compare_stacked.png", width=8.5, height=6.0, dpi=220, verbose=False
    )
    cg.signaling_changes_scatter(
        diff,
        "Inflam. DC",
        exclude_pathways=["MIF"],
        title="Official CellChat skin data: signaling changes of Inflam. DC",
    ).save(
        out_dir / "pyccc_cellchat_official_signaling_changes_inflam_dc.png", width=7.2, height=5.6, dpi=220, verbose=False
    )
    cg.signaling_role_heatmap_compare(
        diff,
        mode="outgoing",
        title="Official CellChat skin data: outgoing role heatmap comparison",
    ).save(
        out_dir / "pyccc_cellchat_official_role_heatmap_compare_outgoing.png", width=9.8, height=7.2, dpi=220, verbose=False
    )
    cg.pathway_embedding_pairwise(
        diff,
        label_pathways=["TNF", "CSF", "GALECTIN"],
        title="Official CellChat skin data: pairwise pathway embedding",
    ).save(
        out_dir / "pyccc_cellchat_official_pathway_embedding_pairwise.png", width=7.2, height=5.6, dpi=220, verbose=False
    )
    cg.pathway_similarity_rank(
        diff,
        top_n=16,
        title="Official CellChat skin data: pathway manifold distance rank",
    ).save(
        out_dir / "pyccc_cellchat_official_pathway_similarity_rank.png", width=7.2, height=5.6, dpi=220, verbose=False
    )
    cg.diff_pathway_rank(diff, top_n=20, title="Official CellChat skin data: differential pathway rank").save(
        out_dir / "pyccc_cellchat_official_diff_pathway_rank.png", width=8.5, height=6.0, dpi=220, verbose=False
    )
    cg.diff_source_target_rank(diff, top_n=24, title="Official CellChat skin data: differential source-target rank").save(
        out_dir / "pyccc_cellchat_official_diff_source_target_rank.png", width=8.5, height=6.0, dpi=220, verbose=False
    )
    cg.diff_bubble(diff, top_n=40, top_pairs=20, compact_pairs=True, title="Official CellChat skin data: differential LR bubble").save(
        out_dir / "pyccc_cellchat_official_diff_bubble.png", width=11.0, height=7.0, dpi=220, verbose=False
    )
    cg.pattern_dot(patterns, title="Official CellChat skin data: outgoing pattern dot").save(
        out_dir / "pyccc_cellchat_official_pattern_dot.png", width=8.4, height=5.5, dpi=220, verbose=False
    )
    cg.pattern_river(patterns, top_pathways=14, title="Official CellChat skin data: outgoing pattern river").save(
        out_dir / "pyccc_cellchat_official_pattern_river.png", width=9.2, height=5.8, dpi=220, verbose=False
    )
    cg.pattern_number_plot(selection, title="Official CellChat skin data: pattern number diagnostics").save(
        out_dir / "pyccc_cellchat_official_pattern_number.png", width=6.5, height=7.0, dpi=220, verbose=False
    )
    try:
        import pyccc.interactive as ci

        ci.save_interactive_html(
            ci.interactive_pathway_river(ls, top_n=16, title="Official CellChat skin data: interactive pathway river"),
            out_dir / "pyccc_cellchat_official_pathway_river.html",
        )
        ci.save_interactive_html(
            ci.interactive_pattern_river(patterns, title="Official CellChat skin data: interactive outgoing pattern river"),
            out_dir / "pyccc_cellchat_official_pattern_river.html",
        )
    except ImportError:
        return


if __name__ == "__main__":
    main()
