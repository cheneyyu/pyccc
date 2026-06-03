import numpy as np
import pandas as pd
from anndata import AnnData

import pyccc as pc
import pyccc.ggplot as cg


def make_demo_adata():
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
            "cell_type": ["Sender", "Sender", "Receiver", "Receiver", "Stroma", "Stroma"],
            "condition": ["ctrl", "stim", "ctrl", "stim", "ctrl", "stim"],
        },
        index=[f"cell{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["TGFB1", "TGFBR1", "TGFBR2", "CXCR4", "CXCL12", "CD74"])
    return AnnData(x, obs=obs, var=var)


adata = make_demo_adata()
lr = pc.toy_lr_table()
ctrl = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="ctrl", min_pct=0.0, aggregate="mean")
stim = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="stim", min_pct=0.0, aggregate="mean")
diff = pc.compare_communication(stim, ctrl, label_a="stim", label_b="ctrl")
patterns = pc.compute_communication_patterns(stim, n_patterns=2)
selection = pc.select_communication_pattern_number(stim, k_range=range(1, 3), n_runs=3)

cg.bubble(stim, top_n=10).save("pyccc_plotnine_bubble.png", width=8, height=5, dpi=180, verbose=False)
cg.bubble(stim, top_n=10, facet_by="pathway").save("pyccc_plotnine_bubble_facet_pathway.png", width=9, height=5, dpi=180, verbose=False)
cg.pathway_heatmap(stim).save("pyccc_plotnine_pathway_heatmap.png", width=8, height=5, dpi=180, verbose=False)
cg.pathway_river(stim, top_n=6).save("pyccc_plotnine_river.png", width=8, height=5, dpi=180, verbose=False)
cg.signaling_role_scatter(stim).save("pyccc_plotnine_signaling_role_scatter.png", width=6.5, height=5.2, dpi=180, verbose=False)
cg.lr_contribution(stim, "TGFb").save("pyccc_plotnine_lr_contribution.png", width=6.2, height=4.2, dpi=180, verbose=False)
cg.lr_contribution_multi(stim, pathways=["TGFb", "CXCL"], top_n=4).save("pyccc_plotnine_lr_contribution_multi.png", width=8.2, height=2.8, dpi=180, verbose=False)
cg.pathway_embedding(stim, cluster=True, n_clusters=2, method="mds").save("pyccc_plotnine_pathway_embedding.png", width=6.5, height=5.2, dpi=180, verbose=False)
cg.signaling_gene_expression(adata, result=stim, signaling="TGFb").save("pyccc_plotnine_gene_expression_tgfb.png", width=6.8, height=4.8, dpi=180, verbose=False)
cg.signaling_role_heatmap_compare(diff, mode="outgoing").save("pyccc_plotnine_role_heatmap_compare_outgoing.png", width=8.5, height=5.2, dpi=180, verbose=False)
cg.diff_bubble(diff, top_n=10).save("pyccc_plotnine_diff_bubble.png", width=8, height=5, dpi=180, verbose=False)
cg.compare_interactions(diff).save("pyccc_plotnine_compare_interactions.png", width=6, height=4, dpi=180, verbose=False)
cg.rank_signaling_compare(diff).save("pyccc_plotnine_rank_compare.png", width=8, height=5, dpi=180, verbose=False)
cg.rank_signaling_compare(diff, stacked=True).save("pyccc_plotnine_rank_compare_stacked.png", width=8, height=5, dpi=180, verbose=False)
cg.signaling_changes_scatter(diff, "Sender").save("pyccc_plotnine_signaling_changes_sender.png", width=6.6, height=5.2, dpi=180, verbose=False)
cg.pathway_embedding_pairwise(diff, method="mds").save("pyccc_plotnine_pathway_embedding_pairwise.png", width=6.8, height=5.2, dpi=180, verbose=False)
cg.pathway_similarity_rank(diff, method="mds").save("pyccc_plotnine_pathway_similarity_rank.png", width=6.8, height=4.6, dpi=180, verbose=False)
cg.diff_pathway_rank(diff).save("pyccc_plotnine_diff_pathway_rank.png", width=7, height=4.5, dpi=180, verbose=False)
cg.diff_source_target_rank(diff).save("pyccc_plotnine_diff_source_target_rank.png", width=7, height=4.5, dpi=180, verbose=False)
cg.annotation_bar(stim).save("pyccc_plotnine_annotation_bar.png", width=7, height=4, dpi=180, verbose=False)
cg.pattern_dot(patterns).save("pyccc_plotnine_pattern_dot.png", width=8, height=5, dpi=180, verbose=False)
cg.pattern_river(patterns).save("pyccc_plotnine_pattern_river.png", width=8, height=5, dpi=180, verbose=False)
cg.pattern_number_plot(selection).save("pyccc_plotnine_pattern_number.png", width=6, height=7, dpi=180, verbose=False)
