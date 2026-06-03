from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from .analysis import CCCResult, compute_pathway_communication
from .patterns import CommunicationPatterns

NODE_COLORS = ["#9a5b4f", "#2c7fb8", "#8dd3c7", "#fdb462", "#7b6bb1", "#4daf4a", "#e78ac3", "#a6cee3", "#b3a2c8", "#fb9a99"]
TERMINAL_COLOR = "#6b7280"


def interactive_pathway_river(
    result: CCCResult,
    *,
    mode: str = "outgoing",
    pathways: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    top_n: int | None = 12,
    min_prob: float = 0.0,
    significant_only: bool = False,
    title: str | None = None,
    width: int = 1100,
    height: int = 720,
):
    """Interactive Plotly Sankey from cell groups to signaling pathways."""

    go = _go()
    flows = _pathway_flow_table(
        result,
        mode=mode,
        pathways=pathways,
        groups=groups,
        top_n=top_n,
        min_prob=min_prob,
        significant_only=significant_only,
    )
    if flows.empty:
        flows = pd.DataFrame({"group": ["No interactions"], "pathway": ["No interactions"], "prob": [0.0]})

    group_order = [group for group in result.groups if group in set(flows["group"])]
    group_order += [group for group in flows["group"].drop_duplicates() if group not in set(group_order)]
    pathway_order = flows.groupby("pathway", observed=True)["prob"].sum().sort_values(ascending=False).index.tolist()
    labels = group_order + pathway_order
    index = {label: i for i, label in enumerate(labels)}
    group_colors = _palette(group_order)
    node_colors = [group_colors[label] for label in group_order] + [TERMINAL_COLOR] * len(pathway_order)

    links = flows.sort_values(["group", "pathway"]).copy()
    link_colors = [_rgba(group_colors[group], 0.32) for group in links["group"]]
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node={
                    "label": labels,
                    "color": node_colors,
                    "pad": 14,
                    "thickness": 18,
                    "line": {"color": "white", "width": 0.5},
                },
                link={
                    "source": [index[group] for group in links["group"]],
                    "target": [index[pathway] for pathway in links["pathway"]],
                    "value": links["prob"].astype(float).tolist(),
                    "color": link_colors,
                    "customdata": links[["group", "pathway", "prob"]].to_numpy(),
                    "hovertemplate": "%{customdata[0]} -> %{customdata[1]}<br>probability=%{customdata[2]:.4g}<extra></extra>",
                },
            )
        ]
    )
    fig.update_layout(title_text=title or f"Interactive pathway river ({mode})", width=width, height=height, font={"size": 13})
    return fig


def interactive_pattern_river(
    patterns: CommunicationPatterns,
    *,
    top_groups: int | None = 12,
    top_pathways: int | None = 18,
    title: str | None = None,
    width: int = 1200,
    height: int = 760,
):
    """Interactive Plotly Sankey for cell groups -> latent patterns -> pathways."""

    go = _go()
    group_weights, pathway_weights = _pattern_tables(patterns, top_groups=top_groups, top_pathways=top_pathways)
    groups = group_weights.index.tolist()
    pattern_names = group_weights.columns.tolist()
    pathways = pathway_weights.columns.tolist()
    labels = groups + pattern_names + pathways
    index = {label: i for i, label in enumerate(labels)}
    pattern_colors = _palette(pattern_names)
    group_colors = [TERMINAL_COLOR] * len(groups)
    pathway_colors = [TERMINAL_COLOR] * len(pathways)
    node_colors = group_colors + [pattern_colors[label] for label in pattern_names] + pathway_colors

    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    colors: list[str] = []
    hovers: list[tuple[str, str, float]] = []
    for group in groups:
        for pattern in pattern_names:
            weight = float(group_weights.loc[group, pattern])
            if weight <= 0:
                continue
            sources.append(index[group])
            targets.append(index[pattern])
            values.append(weight)
            colors.append(_rgba(pattern_colors[pattern], 0.30))
            hovers.append((group, pattern, weight))
    for pattern in pattern_names:
        for pathway in pathways:
            weight = float(pathway_weights.loc[pattern, pathway])
            if weight <= 0:
                continue
            sources.append(index[pattern])
            targets.append(index[pathway])
            values.append(weight)
            colors.append(_rgba(pattern_colors[pattern], 0.30))
            hovers.append((pattern, pathway, weight))

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node={
                    "label": labels,
                    "color": node_colors,
                    "pad": 13,
                    "thickness": 18,
                    "line": {"color": "white", "width": 0.5},
                },
                link={
                    "source": sources,
                    "target": targets,
                    "value": values,
                    "color": colors,
                    "customdata": hovers,
                    "hovertemplate": "%{customdata[0]} -> %{customdata[1]}<br>loading=%{customdata[2]:.4g}<extra></extra>",
                },
            )
        ]
    )
    fig.update_layout(title_text=title or f"Interactive pattern river ({patterns.mode})", width=width, height=height, font={"size": 13})
    return fig


def save_interactive_html(fig, path: str | Path, *, include_plotlyjs: str | bool = "cdn") -> Path:
    """Save a Plotly figure as a standalone or CDN-backed HTML document."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=include_plotlyjs, full_html=True)
    return path


def _go():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError("Install interactive plotting support with `uv sync --extra interactive`.") from exc
    return go


def _pathway_flow_table(
    result: CCCResult,
    *,
    mode: str,
    pathways: Sequence[str] | None,
    groups: Sequence[str] | None,
    top_n: int | None,
    min_prob: float,
    significant_only: bool,
) -> pd.DataFrame:
    if mode not in {"outgoing", "incoming", "all"}:
        raise ValueError("`mode` must be 'outgoing', 'incoming', or 'all'.")
    df = compute_pathway_communication(result, significant_only=significant_only)
    if pathways is not None:
        df = df[df["pathway"].isin(pathways)]
    if mode == "outgoing":
        df = df.rename(columns={"source": "group"})
    elif mode == "incoming":
        df = df.rename(columns={"target": "group"})
    else:
        df = pd.concat([df.rename(columns={"source": "group"}), df.rename(columns={"target": "group"})], ignore_index=True)
    if groups is not None:
        df = df[df["group"].isin(groups)]
    flows = df.groupby(["group", "pathway"], observed=True)["prob"].sum().reset_index()
    flows = flows[flows["prob"] >= min_prob]
    if top_n is not None:
        keep = flows.groupby("pathway", observed=True)["prob"].sum().sort_values(ascending=False).head(top_n).index
        flows = flows[flows["pathway"].isin(keep)]
    return flows


def _pattern_tables(
    patterns: CommunicationPatterns,
    *,
    top_groups: int | None,
    top_pathways: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = patterns.group_weights.sum(axis=1).sort_values(ascending=False)
    pathways = patterns.pathway_weights.sum(axis=0).sort_values(ascending=False)
    if top_groups is not None:
        groups = groups.head(top_groups)
    if top_pathways is not None:
        pathways = pathways.head(top_pathways)
    return patterns.group_weights.loc[groups.index], patterns.pathway_weights.loc[:, pathways.index]


def _palette(labels: Sequence[str]) -> dict[str, str]:
    return {label: NODE_COLORS[i % len(NODE_COLORS)] for i, label in enumerate(labels)}


def _rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    r, g, b = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
