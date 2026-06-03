from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .database import CellChatDB


def estimate_lr_density_prior(
    training_table,
    *,
    groupby: str = "clade",
    stat: str = "median",
    min_species_edges: int = 100,
) -> pd.DataFrame:
    """Estimate robust LR edge-density priors from normalized species resources."""

    interactions = getattr(training_table, "interactions", training_table).copy()
    required = {"species", "ligand_gene", "receptor_gene", "resource"}
    missing = sorted(required - set(interactions.columns))
    if missing:
        raise ValueError(f"Training table is missing columns required for density estimation: {missing}")
    if groupby not in interactions.columns:
        interactions[groupby] = "all"
    rows = []
    for species, sub in interactions.groupby("species", sort=False):
        ligand_count = sub["ligand_gene"].astype(str).nunique()
        receptor_count = sub["receptor_gene"].astype(str).nunique()
        edge_count = sub[["ligand_gene", "receptor_gene"]].drop_duplicates().shape[0]
        density = edge_count / max(ligand_count * receptor_count, 1)
        rows.append(
            {
                "species": species,
                groupby: str(sub[groupby].iloc[0]),
                "ligand_count": ligand_count,
                "receptor_count": receptor_count,
                "edge_count": edge_count,
                "species_density": density,
                "resources": ";".join(sorted(set(sub["resource"].astype(str)))),
            }
        )
    species_frame = pd.DataFrame(rows)
    eligible = species_frame[species_frame["edge_count"] >= min_species_edges]
    if eligible.empty:
        eligible = species_frame
    prior_rows = []
    for key, sub in eligible.groupby(groupby, sort=False):
        values = sub["species_density"].astype(float).to_numpy()
        density = _summary(values, stat=stat)
        prior_rows.append(
            {
                groupby: key,
                "density_prior": float(density),
                "density_stat": stat,
                "density_source_species": ";".join(sub["species"].astype(str)),
                "density_source_resources": ";".join(sorted(set(";".join(sub["resources"]).split(";")))),
                "n_species": int(len(sub)),
            }
        )
    return pd.DataFrame(prior_rows)


def build_predicted_lr_table(
    scored_pairs: pd.DataFrame,
    *,
    roles: pd.DataFrame | None = None,
    density_prior: pd.DataFrame | float | str = "auto",
    species_hint: str = "unknown",
    min_score: float = 0.50,
    max_pairs: int = 50000,
    max_pairs_per_ligand: int = 200,
    max_pairs_per_receptor: int = 200,
    allow_low_score_density_fill: bool = False,
    model_name: str = "universal_esmc300m_lgbm_v0",
    model_version: str = "0",
    model_revision: str = "",
    feature_encoder: str = "pca128_absdiff_hadamard_v1",
    candidate_strategy: str = "role_expression_cross_product",
    name: str = "dbfree_predicted",
) -> CellChatDB:
    """Select scored candidate pairs into a CellChatDB-compatible predicted LR table."""

    pairs = _canonical_scores(scored_pairs)
    rho, density_meta = _resolve_density_prior(density_prior, species_hint=species_hint)
    ligand_count = pairs["ligand_gene"].nunique()
    receptor_count = pairs["receptor_gene"].nunique()
    candidate_pair_count = len(pairs)
    candidate_grid_size = max(ligand_count * receptor_count, 1)
    raw_target_pair_count = int(round(rho * candidate_grid_size))
    k_target = raw_target_pair_count
    k_target = min(max_pairs, max(k_target, 1 if (pairs["model_score"] >= min_score).any() else 0))
    eligible = pairs.copy() if allow_low_score_density_fill else pairs[pairs["model_score"] >= min_score].copy()
    eligible = eligible.sort_values("model_score", ascending=False)
    selected_rows = []
    ligand_degree: dict[str, int] = {}
    receptor_degree: dict[str, int] = {}
    for row in eligible.itertuples(index=False):
        lig = str(row.ligand_gene)
        rec = str(row.receptor_gene)
        if ligand_degree.get(lig, 0) >= max_pairs_per_ligand or receptor_degree.get(rec, 0) >= max_pairs_per_receptor:
            continue
        selected_rows.append(row._asdict())
        ligand_degree[lig] = ligand_degree.get(lig, 0) + 1
        receptor_degree[rec] = receptor_degree.get(rec, 0) + 1
        if len(selected_rows) >= k_target:
            break
    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        raise ValueError("No candidate LR pairs passed score and density filters.")
    selected = selected.reset_index(drop=True)
    selected["density_rank"] = np.arange(1, len(selected) + 1)
    selected["confidence"] = selected.get("calibrated_probability", selected["model_score"]).astype(float)
    if "calibrated_probability" not in selected.columns:
        selected["calibrated_probability"] = selected["model_score"]
    global_warnings = []
    if str(density_meta.get("density_mode", "")) == "auto_default":
        global_warnings.append("auto_default_density_prior")
    if len(selected) < k_target:
        global_warnings.append("density_target_not_reached")
    if "heuristic" in str(model_name):
        global_warnings.append("heuristic_pair_ranker")
    warnings_out = _append_warning(selected.get("warning", pd.Series([""] * len(selected))), global_warnings)
    out = pd.DataFrame(
        {
            "ligand": selected["ligand_gene"].astype(str),
            "receptor": selected["receptor_gene"].astype(str),
            "pathway": "DB-free predicted",
            "annotation": "Predicted LR",
            "evidence": "ESMC-300M + LightGBM link prediction",
            "evidence_type": "embedding_link_prediction",
            "confidence": selected["confidence"].astype(float),
            "model_score": selected["model_score"].astype(float),
            "calibrated_probability": selected["calibrated_probability"].astype(float),
            "model_name": model_name,
            "model_version": model_version,
            "model_revision": model_revision,
            "feature_encoder": feature_encoder,
            "density_prior": float(rho),
            "density_rank": selected["density_rank"].astype(int),
            "candidate_strategy": candidate_strategy,
            "ligand_role_score": selected.get("ligand_role_score", pd.Series([0.0] * len(selected))).astype(float),
            "receptor_role_score": selected.get("receptor_role_score", pd.Series([0.0] * len(selected))).astype(float),
            "nearest_reference_ligand": selected.get("nearest_reference_ligand", pd.Series([""] * len(selected))).astype(str),
            "nearest_reference_receptor": selected.get("nearest_reference_receptor", pd.Series([""] * len(selected))).astype(str),
            "nearest_reference_lr": selected.get("nearest_reference_lr", pd.Series([""] * len(selected))).astype(str),
            "nearest_reference_species": selected.get("nearest_reference_species", pd.Series([""] * len(selected))).astype(str),
            "nearest_reference_resource": selected.get("nearest_reference_resource", pd.Series([""] * len(selected))).astype(str),
            "nearest_reference_pathway": selected.get("nearest_reference_pathway", pd.Series([""] * len(selected))).astype(str),
            "nearest_reference_distance": pd.to_numeric(selected.get("nearest_reference_distance", pd.Series([np.nan] * len(selected))), errors="coerce"),
            "warning": warnings_out.astype(str),
        }
    )
    summary = pd.DataFrame(
        [
            {
                **density_meta,
                "candidate_ligand_count": ligand_count,
                "candidate_receptor_count": receptor_count,
                "candidate_pair_count": candidate_pair_count,
                "candidate_grid_size": candidate_grid_size,
                "target_pair_count": raw_target_pair_count,
                "capped_target_pair_count": k_target,
                "selected_pair_count": len(out),
                "achieved_density": float(len(out) / candidate_grid_size),
                "density_delta": float(len(out) / candidate_grid_size - rho),
                "density_ratio": float((len(out) / candidate_grid_size) / rho) if rho > 0 else np.nan,
                "score_threshold": float(out["model_score"].min()),
                "min_score": min_score,
                "max_pairs": max_pairs,
                "warning": ";".join(global_warnings),
            }
        ]
    )
    metadata = {"prediction_summary": summary, "scored_pairs": pairs.copy()}
    if roles is not None:
        metadata["roles"] = roles.copy()
    return CellChatDB(out, name=name, metadata=metadata)


def evaluate_predicted_lr_density_prior(
    predicted_db: CellChatDB | pd.DataFrame | dict[str, object] | str | Path,
    *,
    max_abs_delta: float | None = None,
    max_fold_error: float = 2.0,
) -> pd.DataFrame:
    """Evaluate whether a predicted LR table stays close to its density prior."""

    summary = _prediction_summary_frame(predicted_db)
    rows = []
    for idx, row in summary.iterrows():
        prior = float(row.get("density_prior", np.nan))
        achieved = float(row.get("achieved_density", np.nan))
        if pd.isna(achieved) and pd.notna(prior):
            grid = float(row.get("candidate_grid_size", row.get("candidate_ligand_count", 0) * row.get("candidate_receptor_count", 0)))
            selected = float(row.get("selected_pair_count", np.nan))
            achieved = selected / grid if grid > 0 else np.nan
        delta = achieved - prior if pd.notna(achieved) and pd.notna(prior) else np.nan
        ratio = achieved / prior if pd.notna(achieved) and pd.notna(prior) and prior > 0 else np.nan
        fold_error = max(ratio, 1.0 / ratio) if pd.notna(ratio) and ratio > 0 else np.nan
        abs_delta_limit = float(max_abs_delta) if max_abs_delta is not None else max(0.01, prior * 0.5) if pd.notna(prior) else np.nan
        delta_pass = bool(pd.notna(delta) and abs(delta) <= abs_delta_limit)
        fold_pass = bool(pd.notna(fold_error) and fold_error <= max_fold_error)
        passed = delta_pass or fold_pass
        rows.append(
            {
                "row": int(idx),
                "density_prior": prior,
                "achieved_density": achieved,
                "density_delta": delta,
                "density_ratio": ratio,
                "density_fold_error": fold_error,
                "max_abs_delta": abs_delta_limit,
                "max_fold_error": float(max_fold_error),
                "candidate_ligand_count": int(row.get("candidate_ligand_count", 0)),
                "candidate_receptor_count": int(row.get("candidate_receptor_count", 0)),
                "selected_pair_count": int(row.get("selected_pair_count", 0)),
                "passed": passed,
                "reason": "ok" if passed else "density outside prior tolerance",
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["passed"] = bool(out["passed"].all()) if not out.empty else False
    return out


def _summary(values: np.ndarray, *, stat: str) -> float:
    if stat == "median":
        return float(np.median(values))
    if stat == "mean":
        return float(np.mean(values))
    if stat == "trimmed_median":
        if len(values) > 4:
            values = np.sort(values)[1:-1]
        return float(np.median(values))
    raise ValueError("`stat` must be one of: median, mean, trimmed_median.")


def _append_warning(values: pd.Series, warnings_: Sequence[str]) -> pd.Series:
    clean = [str(item) for item in warnings_ if str(item)]
    if not clean:
        return values.fillna("").astype(str)

    def combine(value) -> str:
        parts = [part for part in str(value or "").split(";") if part]
        for warning in clean:
            if warning not in parts:
                parts.append(warning)
        return ";".join(parts)

    return values.fillna("").map(combine)


def _prediction_summary_frame(predicted_db: CellChatDB | pd.DataFrame | dict[str, object] | str | Path) -> pd.DataFrame:
    if isinstance(predicted_db, CellChatDB):
        if "prediction_summary" not in predicted_db.metadata:
            raise ValueError("CellChatDB metadata does not contain `prediction_summary`.")
        return predicted_db.metadata["prediction_summary"].copy()
    if isinstance(predicted_db, pd.DataFrame):
        return predicted_db.copy()
    if isinstance(predicted_db, dict):
        if "prediction_summary" in predicted_db:
            value = predicted_db["prediction_summary"]
            if isinstance(value, pd.DataFrame):
                return value.copy()
            return pd.DataFrame(value if isinstance(value, list) else [value])
        return pd.DataFrame([predicted_db])
    path = Path(predicted_db)
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and "prediction_summary" in value:
            value = value["prediction_summary"]
        return pd.DataFrame(value if isinstance(value, list) else [value])
    return pd.read_csv(path, sep="\t")


def _canonical_scores(scored_pairs: pd.DataFrame) -> pd.DataFrame:
    pairs = scored_pairs.copy()
    for canonical, aliases in {"ligand_gene": ("ligand_gene", "ligand"), "receptor_gene": ("receptor_gene", "receptor")}.items():
        if canonical not in pairs.columns:
            for alias in aliases:
                if alias in pairs.columns:
                    pairs[canonical] = pairs[alias]
                    break
    if "model_score" not in pairs.columns:
        if "score" in pairs.columns:
            pairs["model_score"] = pairs["score"]
        else:
            raise ValueError("Scored pairs must contain `model_score` or `score`.")
    missing = [col for col in ("ligand_gene", "receptor_gene") if col not in pairs.columns]
    if missing:
        raise ValueError(f"Scored pairs are missing required columns: {missing}")
    pairs["model_score"] = pairs["model_score"].astype(float)
    return pairs


def _resolve_density_prior(density_prior: pd.DataFrame | float | str, *, species_hint: str) -> tuple[float, dict[str, object]]:
    if isinstance(density_prior, (float, int)):
        rho = float(density_prior)
        return rho, {"density_prior": rho, "density_source_species": "", "density_source_resources": "", "density_mode": "explicit"}
    if isinstance(density_prior, str):
        if density_prior != "auto":
            return float(density_prior), {"density_prior": float(density_prior), "density_source_species": "", "density_source_resources": "", "density_mode": "explicit"}
        rho = 0.01
        return rho, {"density_prior": rho, "density_source_species": "", "density_source_resources": "", "density_mode": "auto_default"}
    frame = density_prior.copy()
    if species_hint in set(frame.get("clade", pd.Series(dtype=str)).astype(str)):
        row = frame[frame["clade"].astype(str) == str(species_hint)].iloc[0]
    elif species_hint in set(frame.get("species_hint", pd.Series(dtype=str)).astype(str)):
        row = frame[frame["species_hint"].astype(str) == str(species_hint)].iloc[0]
    else:
        row = frame.iloc[0]
    rho = float(row["density_prior"])
    return rho, {
        "density_prior": rho,
        "density_source_species": str(row.get("density_source_species", "")),
        "density_source_resources": str(row.get("density_source_resources", "")),
        "density_mode": "table",
    }
