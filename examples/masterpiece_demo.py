import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData

import pyccc as pc
import pyccc.plotting as cp


def make_masterpiece_adata(seed=7):
    rng = np.random.default_rng(seed)
    cell_types = ["T cell", "Myeloid", "Fibroblast", "Endothelial", "Tumor"]
    conditions = ["control", "treated"]
    genes = [
        "CXCL9",
        "CXCL10",
        "CXCL12",
        "CCL2",
        "TGFB1",
        "VEGFA",
        "MIF",
        "CD74",
        "COL1A1",
        "LAMA1",
        "TNF",
        "IL6",
        "IFNG",
        "PDGFA",
        "JAG1",
        "CXCR3",
        "CXCR4",
        "CCR2",
        "TGFBR1",
        "TGFBR2",
        "KDR",
        "CXCR2",
        "ITGA6",
        "ITGB1",
        "TNFRSF1A",
        "IL6R",
        "IFNGR1",
        "PDGFRA",
        "NOTCH1",
        "ACKR3",
    ]
    rows = []
    obs = []
    for condition in conditions:
        for cell_type in cell_types:
            for _ in range(42):
                base = rng.gamma(0.9, 0.25, len(genes))
                obs.append({"cell_type": cell_type, "condition": condition})
                rows.append(base)
    x = np.vstack(rows)
    gene_idx = {g: i for i, g in enumerate(genes)}
    obs_df = pd.DataFrame(obs)

    def boost(mask, names, amount):
        for name in names:
            x[mask, gene_idx[name]] += rng.gamma(amount, 0.55, mask.sum())

    for condition in conditions:
        cond = obs_df["condition"].to_numpy() == condition
        treated = condition == "treated"
        boost(cond & (obs_df["cell_type"].to_numpy() == "T cell"), ["IFNG", "CXCL10", "TNF"], 5.0 if treated else 2.8)
        boost(cond & (obs_df["cell_type"].to_numpy() == "Myeloid"), ["CCL2", "MIF", "IL6", "TGFB1"], 4.2 if treated else 3.0)
        boost(cond & (obs_df["cell_type"].to_numpy() == "Fibroblast"), ["CXCL12", "COL1A1", "LAMA1", "TGFB1", "PDGFA", "JAG1"], 4.8 if treated else 3.4)
        boost(cond & (obs_df["cell_type"].to_numpy() == "Endothelial"), ["VEGFA", "KDR", "ACKR3", "NOTCH1"], 3.6 if treated else 2.7)
        boost(cond & (obs_df["cell_type"].to_numpy() == "Tumor"), ["CXCL9", "VEGFA", "MIF", "CD74"], 4.4 if treated else 3.1)

        boost(cond & (obs_df["cell_type"].to_numpy() == "T cell"), ["CXCR3", "CXCR4", "IFNGR1", "TNFRSF1A"], 3.8)
        boost(cond & (obs_df["cell_type"].to_numpy() == "Myeloid"), ["CCR2", "CD74", "TGFBR1", "TGFBR2", "IL6R"], 3.4)
        boost(cond & (obs_df["cell_type"].to_numpy() == "Fibroblast"), ["TGFBR1", "TGFBR2", "PDGFRA", "NOTCH1"], 3.2)
        boost(cond & (obs_df["cell_type"].to_numpy() == "Endothelial"), ["KDR", "CXCR4", "ACKR3", "ITGA6", "ITGB1"], 3.6)
        boost(cond & (obs_df["cell_type"].to_numpy() == "Tumor"), ["CD74", "CXCR4", "TGFBR1", "TGFBR2", "NOTCH1"], 3.3)

    obs_df.index = [f"cell{i}" for i in range(len(obs_df))]
    return AnnData(x, obs=obs_df, var=pd.DataFrame(index=genes))


def make_lr_table():
    return pd.DataFrame(
        {
            "ligand": [
                "CXCL9",
                "CXCL10",
                "CXCL12",
                "CXCL12",
                "CCL2",
                "TGFB1",
                "VEGFA",
                "MIF",
                "MIF",
                "COL1A1",
                "LAMA1",
                "TNF",
                "IL6",
                "IFNG",
                "PDGFA",
                "JAG1",
            ],
            "receptor": [
                "CXCR3",
                "CXCR3",
                "CXCR4",
                "ACKR3",
                "CCR2",
                "TGFBR1_TGFBR2",
                "KDR",
                "CD74_CXCR4",
                "CD74",
                "ITGA6_ITGB1",
                "ITGA6_ITGB1",
                "TNFRSF1A",
                "IL6R",
                "IFNGR1",
                "PDGFRA",
                "NOTCH1",
            ],
            "pathway": [
                "CXCL",
                "CXCL",
                "CXCL",
                "CXCL",
                "CCL",
                "TGFb",
                "VEGF",
                "MIF",
                "MIF",
                "COLLAGEN",
                "LAMININ",
                "TNF",
                "IL6",
                "IFN-II",
                "PDGF",
                "NOTCH",
            ],
        }
    )


adata = make_masterpiece_adata()
lr = make_lr_table()
cellchat_kwargs = dict(min_pct=0.08, aggregate="tri_mean", score_method="cellchat")
control = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="control", **cellchat_kwargs)
treated = pc.compute_communication(adata, "cell_type", lr, condition_key="condition", condition="treated", **cellchat_kwargs)
diff = pc.compare_communication(treated, control, label_a="treated", label_b="control")

fig, _ = cp.masterpiece_gallery(
    treated,
    diff,
    title="PyCCC communication landscape",
    subtitle="Synthetic treated-vs-control tissue atlas with CellChat-style visual summaries",
    outfile="pyccc_masterpiece_gallery.png",
)

network_fig, network_ax = plt.subplots(figsize=(7.5, 7.5))
cp.net_circle(treated, ax=network_ax, title="Treated communication network")
cp.save_figure(network_fig, "pyccc_masterpiece_network.png", dpi=420)

diff_fig, diff_ax = plt.subplots(figsize=(7.5, 7.5))
cp.diff_network_circle(diff, ax=diff_ax, title="Differential communication network")
cp.save_figure(diff_fig, "pyccc_masterpiece_diff_network.png", dpi=420)

bubble_fig, bubble_ax = plt.subplots(figsize=(10.5, 7.0))
cp.bubble(
    treated,
    ax=bubble_ax,
    top_n=14,
    top_pairs=9,
    compact_pairs=True,
    show_size_legend=False,
    title="Top ligand-receptor programs",
)
cp.save_figure(bubble_fig, "pyccc_masterpiece_bubble.png", dpi=420)
