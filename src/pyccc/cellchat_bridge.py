from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .analysis import CCCResult
from .database import CellChatDB, normalize_lr_table


_R_HELPER_DIRS = (
    Path(__file__).resolve().parents[2] / "inst" / "r",
    Path(__file__).resolve().parent / "r",
)


def export_cellchat(
    result: CCCResult,
    out_dir: str | Path,
    *,
    lr_table: CellChatDB | pd.DataFrame | None = None,
    group_sizes: Mapping[str, int | float] | pd.Series | None = None,
    pvalue_default: float = 0.0,
    copy_r_helper: bool = True,
) -> Path:
    """Export a pyccc result as CellChat-compatible bridge files.

    The exported directory can be converted to a minimal CellChat R object with
    ``inst/r/pyccc_to_cellchat.R``. The bridge preserves LR/source/target
    probabilities and p-values; expression matrices are intentionally not
    reconstructed.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    interactions, lr = _cellchat_interaction_tables(result, lr_table=lr_table, pvalue_default=pvalue_default)
    groups = _cellchat_group_table(result.groups, group_sizes)

    interactions.to_csv(out / "pyccc_cellchat_interactions.tsv", sep="\t", index=False)
    lr.to_csv(out / "pyccc_cellchat_lr.tsv", sep="\t", index=False)
    groups.to_csv(out / "pyccc_cellchat_groups.tsv", sep="\t", index=False)

    metadata = {
        "format": "pyccc_cellchat_bridge",
        "version": 1,
        "groupby": result.groupby,
        "condition": result.condition or "",
        "lr_name": result.lr_name,
        "pvalue_cutoff": result.pvalue_cutoff,
        "n_groups": len(groups),
        "n_lr": len(lr),
        "n_interactions": len(interactions),
        "files": {
            "interactions": "pyccc_cellchat_interactions.tsv",
            "lr": "pyccc_cellchat_lr.tsv",
            "groups": "pyccc_cellchat_groups.tsv",
            "r_helper": "pyccc_to_cellchat.R",
        },
    }
    (out / "pyccc_cellchat_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    if copy_r_helper:
        _copy_r_helper("pyccc_to_cellchat.R", out)
    return out


def export_cellchat_merged(
    diff,
    out_dir: str | Path,
    *,
    lr_table: CellChatDB | pd.DataFrame | None = None,
    group_sizes_a: Mapping[str, int | float] | pd.Series | None = None,
    group_sizes_b: Mapping[str, int | float] | pd.Series | None = None,
    pvalue_default: float = 0.0,
    copy_r_helper: bool = True,
) -> Path:
    """Export a two-sample differential result for CellChat comparison plots.

    The exported directory contains two single-sample CellChat bridge
    directories plus a helper script that converts them into a merged CellChat
    object with ``mergeCellChat``. Differential TSV tables are also written for
    users who want to inspect pyccc's comparison directly.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    label_a = str(diff.label_a)
    label_b = str(diff.label_b)
    dir_a = _safe_dir_name(label_a) or "sample_a"
    dir_b = _safe_dir_name(label_b) or "sample_b"
    if dir_a == dir_b:
        dir_a = f"{dir_a}_a"
        dir_b = f"{dir_b}_b"

    export_cellchat(
        diff.a,
        out / dir_a,
        lr_table=lr_table,
        group_sizes=group_sizes_a,
        pvalue_default=pvalue_default,
        copy_r_helper=copy_r_helper,
    )
    export_cellchat(
        diff.b,
        out / dir_b,
        lr_table=lr_table,
        group_sizes=group_sizes_b,
        pvalue_default=pvalue_default,
        copy_r_helper=copy_r_helper,
    )

    samples = pd.DataFrame(
        [
            {"role": "a", "label": label_a, "directory": dir_a},
            {"role": "b", "label": label_b, "directory": dir_b},
        ]
    )
    samples.to_csv(out / "pyccc_cellchat_samples.tsv", sep="\t", index=False)
    diff.interactions.to_csv(out / "pyccc_cellchat_diff_interactions.tsv", sep="\t", index=False)
    diff.pathway_changes.to_csv(out / "pyccc_cellchat_diff_pathways.tsv", sep="\t", index=False)
    diff.source_target_changes.to_csv(out / "pyccc_cellchat_diff_source_targets.tsv", sep="\t", index=False)
    diff.network_delta.to_csv(out / "pyccc_cellchat_network_delta.tsv", sep="\t")
    diff.count_delta.to_csv(out / "pyccc_cellchat_count_delta.tsv", sep="\t")

    metadata = {
        "format": "pyccc_cellchat_merged_bridge",
        "version": 1,
        "label_a": label_a,
        "label_b": label_b,
        "groupby": diff.a.groupby,
        "n_groups": len(diff.groups),
        "n_interactions_diff": int(len(diff.interactions)),
        "samples": samples.to_dict(orient="records"),
        "files": {
            "samples": "pyccc_cellchat_samples.tsv",
            "diff_interactions": "pyccc_cellchat_diff_interactions.tsv",
            "diff_pathways": "pyccc_cellchat_diff_pathways.tsv",
            "diff_source_targets": "pyccc_cellchat_diff_source_targets.tsv",
            "network_delta": "pyccc_cellchat_network_delta.tsv",
            "count_delta": "pyccc_cellchat_count_delta.tsv",
            "r_helper": "pyccc_to_merged_cellchat.R",
        },
    }
    (out / "pyccc_cellchat_merged_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    if copy_r_helper:
        _copy_r_helper("pyccc_to_merged_cellchat.R", out)
    return out


def _cellchat_interaction_tables(result: CCCResult, *, lr_table: CellChatDB | pd.DataFrame | None, pvalue_default: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = result.interactions.copy()
    if frame.empty:
        required = ["source", "target", "ligand", "receptor", "pathway", "prob"]
        frame = pd.DataFrame(columns=required)
    if "pvalue" not in frame.columns:
        frame["pvalue"] = pvalue_default
    frame["pvalue"] = pd.to_numeric(frame["pvalue"], errors="coerce").fillna(pvalue_default)
    frame["prob"] = pd.to_numeric(frame["prob"], errors="coerce").fillna(0.0)

    lr = _cellchat_lr_table(frame, lr_table)
    lr_index = lr.set_index(["ligand", "receptor", "pathway_name"], drop=False)

    interaction_name = []
    interaction_name_2 = []
    pathway_name = []
    annotation = []
    evidence = []
    for row in frame.itertuples(index=False):
        key = (str(row.ligand), str(row.receptor), str(row.pathway))
        if key in lr_index.index:
            lr_row = lr_index.loc[key]
            if isinstance(lr_row, pd.DataFrame):
                lr_row = lr_row.iloc[0]
            interaction_name.append(str(lr_row["interaction_name"]))
            interaction_name_2.append(str(lr_row["interaction_name_2"]))
            pathway_name.append(str(lr_row["pathway_name"]))
            annotation.append(str(lr_row.get("annotation", "")))
            evidence.append(str(lr_row.get("evidence", "")))
        else:
            name = _make_interaction_name(str(row.ligand), str(row.receptor), str(row.pathway))
            interaction_name.append(name)
            interaction_name_2.append(f"{row.ligand} - {row.receptor}")
            pathway_name.append(str(row.pathway))
            annotation.append(str(getattr(row, "annotation", "")))
            evidence.append(str(getattr(row, "evidence", "")))

    out = pd.DataFrame(
        {
            "source": frame["source"].astype(str),
            "target": frame["target"].astype(str),
            "interaction_name": interaction_name,
            "interaction_name_2": interaction_name_2,
            "pathway_name": pathway_name,
            "ligand": frame["ligand"].astype(str),
            "receptor": frame["receptor"].astype(str),
            "prob": frame["prob"].astype(float),
            "pval": frame["pvalue"].astype(float),
            "annotation": annotation,
            "evidence": evidence,
        }
    )
    return out, lr


def _cellchat_lr_table(frame: pd.DataFrame, lr_table: CellChatDB | pd.DataFrame | None) -> pd.DataFrame:
    if lr_table is None:
        agg = {}
        if "annotation" in frame.columns:
            agg["annotation"] = ("annotation", "first")
        if "evidence" in frame.columns:
            agg["evidence"] = ("evidence", "first")
        if agg:
            lr = frame.groupby(["ligand", "receptor", "pathway"], observed=True, sort=False).agg(**agg).reset_index()
        else:
            lr = frame[["ligand", "receptor", "pathway"]].drop_duplicates().copy()
    elif isinstance(lr_table, CellChatDB):
        lr = lr_table.interactions.copy()
    else:
        lr = normalize_lr_table(lr_table)

    lr = lr.rename(columns={"pathway": "pathway_name"}).copy()
    for col in ("annotation", "evidence"):
        if col not in lr.columns:
            lr[col] = ""
        lr[col] = lr[col].fillna("").astype(str)
    if "interaction_name" not in lr.columns:
        lr["interaction_name"] = [_make_interaction_name(lig, rec, path) for lig, rec, path in zip(lr["ligand"], lr["receptor"], lr["pathway_name"])]
    if "interaction_name_2" not in lr.columns:
        lr["interaction_name_2"] = lr["ligand"].astype(str) + " - " + lr["receptor"].astype(str)

    keep = ["interaction_name", "interaction_name_2", "pathway_name", "ligand", "receptor", "annotation", "evidence"]
    extra = [col for col in lr.columns if col not in keep and pd.api.types.is_string_dtype(lr[col])]
    lr = lr[keep + extra].drop_duplicates("interaction_name").reset_index(drop=True)
    return lr


def _cellchat_group_table(groups: list[str], group_sizes: Mapping[str, int | float] | pd.Series | None) -> pd.DataFrame:
    if group_sizes is None:
        sizes = {group: 1 for group in groups}
    elif isinstance(group_sizes, pd.Series):
        sizes = {str(idx): value for idx, value in group_sizes.items()}
    else:
        sizes = {str(key): value for key, value in group_sizes.items()}
    return pd.DataFrame({"group": [str(group) for group in groups], "n_cells": [max(int(sizes.get(str(group), 1)), 1) for group in groups]})


def _make_interaction_name(ligand: str, receptor: str, pathway: str) -> str:
    raw = f"{ligand}_{receptor}_{pathway}"
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw)


def _copy_r_helper(name: str, out: Path) -> None:
    for directory in _R_HELPER_DIRS:
        helper = directory / name
        if helper.exists():
            shutil.copy2(helper, out / name)
            return


def _safe_dir_name(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in label.strip())
    return cleaned.strip("._-")
