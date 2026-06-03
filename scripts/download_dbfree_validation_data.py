from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

from dbfree_validation_utils import (
    load_manifest,
    manifest_data_dir,
    manifest_results_dir,
    parse_size_bytes,
    section_file_name,
    section_local_path,
    selected_sections,
    sha256_file,
    size_ratio_ok,
    utc_now,
    write_tsv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real-data DB-free CCC validation sections from a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", default=None, help="Override manifest data_dir root.")
    parser.add_argument("--results-dir", default=None, help="Override manifest results_dir root.")
    parser.add_argument("--section", action="append", default=[], help="Section name to download. May be repeated.")
    parser.add_argument("--smoke-only", action="store_true", help="Download only sections marked smoke=true.")
    parser.add_argument("--include-proteome", action="store_true", help="Also download the manifest protein_source asset.")
    parser.add_argument("--proteome-only", action="store_true", help="Download only the manifest protein_source asset.")
    parser.add_argument("--dry-run", action="store_true", help="Write the manifest table without downloading files.")
    parser.add_argument("--force", action="store_true", help="Re-download files even when the local file already exists.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    data_dir = manifest_data_dir(manifest, args.data_dir)
    results_dir = manifest_results_dir(manifest, args.results_dir)
    rows = []
    if not args.proteome_only:
        for section in selected_sections(manifest, section_names=args.section, smoke_only=args.smoke_only):
            rows.append(_download_section(manifest, section, data_dir=data_dir, dry_run=args.dry_run, force=args.force))
    if args.include_proteome or args.proteome_only:
        rows.append(_download_proteome(manifest, data_dir=data_dir, dry_run=args.dry_run, force=args.force))

    out = pd.DataFrame(rows)
    combined = _append_download_manifest(results_dir.parent / "download_manifest.tsv", out)
    write_tsv(combined, results_dir.parent / "download_manifest.tsv")
    write_tsv(out, results_dir / "download_manifest.tsv")


def _download_section(
    manifest: dict[str, object],
    section: dict[str, object],
    *,
    data_dir: Path,
    dry_run: bool,
    force: bool,
) -> dict[str, object]:
    url = str(section.get("h5ad_url") or section.get("url") or "")
    if not url:
        raise ValueError(f"Section {section.get('name')} has no h5ad_url.")
    expected_size = section.get("expected_size", "")
    expected_bytes = parse_size_bytes(expected_size)
    local_path = section_local_path(section, data_dir)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    status = "dry_run"
    actual_bytes = 0
    checksum = ""
    if not dry_run:
        if force or not local_path.exists():
            _download_url(url, local_path)
        actual_bytes = local_path.stat().st_size
        checksum = sha256_file(local_path)
        status = "ok" if size_ratio_ok(actual_bytes, expected_bytes) else "size_mismatch"
    return {
        "asset_type": "spatial_h5ad",
        "dataset": str(manifest["name"]),
        "species": str(manifest.get("species", "")),
        "section_id": str(section["name"]),
        "file_name": section_file_name(section),
        "source_url": url,
        "source_page": str(manifest.get("source_page", "")),
        "local_path": str(local_path),
        "expected_size": str(expected_size),
        "expected_bytes": expected_bytes if expected_bytes is not None else "",
        "actual_bytes": actual_bytes,
        "sha256": checksum,
        "status": status,
        "downloaded_at": utc_now(),
    }


def _download_proteome(
    manifest: dict[str, object],
    *,
    data_dir: Path,
    dry_run: bool,
    force: bool,
) -> dict[str, object]:
    source = dict(manifest.get("protein_source", {}))
    url = str(source.get("url") or "")
    if not url:
        raise ValueError(f"Manifest {manifest.get('name')} has no protein_source.url.")
    expected_size = source.get("expected_size", "")
    expected_bytes = parse_size_bytes(expected_size)
    local_path = _protein_source_local_path(source, data_dir)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    status = "dry_run"
    actual_bytes = 0
    checksum = ""
    if not dry_run:
        if force or not local_path.exists():
            _download_url(url, local_path)
        actual_bytes = local_path.stat().st_size
        checksum = sha256_file(local_path)
        status = "ok" if size_ratio_ok(actual_bytes, expected_bytes) else "size_mismatch"
    return {
        "asset_type": "protein_fasta",
        "dataset": str(manifest["name"]),
        "species": str(manifest.get("species", "")),
        "section_id": "proteome",
        "file_name": _protein_source_file_name(source),
        "source_url": url,
        "source_page": str(source.get("source_page") or manifest.get("source_page", "")),
        "local_path": str(local_path),
        "expected_size": str(expected_size),
        "expected_bytes": expected_bytes if expected_bytes is not None else "",
        "actual_bytes": actual_bytes,
        "sha256": checksum,
        "status": status,
        "downloaded_at": utc_now(),
    }


def _protein_source_file_name(source: dict[str, object]) -> str:
    if source.get("file_name"):
        return str(source["file_name"])
    url = str(source.get("url", ""))
    return url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1] or "protein.fa.gz"


def _protein_source_local_path(source: dict[str, object], data_dir: Path) -> Path:
    if source.get("local_path"):
        return Path(str(source["local_path"]))
    return data_dir / "raw" / _protein_source_file_name(source)


def _download_url(url: str, output: Path) -> None:
    tmp = output.with_suffix(output.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        tmp.replace(output)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _append_download_manifest(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path, sep="\t")
        combined = pd.concat([old, new_rows], ignore_index=True)
    else:
        combined = new_rows.copy()
    combined = _normalize_manifest_schema(combined)
    keys = [col for col in ("asset_type", "dataset", "section_id", "source_url") if col in combined.columns]
    if keys:
        combined = combined.drop_duplicates(keys, keep="last")
    return combined.sort_values([col for col in ("dataset", "section_id") if col in combined.columns]).reset_index(drop=True)


def _normalize_manifest_schema(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "asset_type" not in out:
        out["asset_type"] = ""
    asset = out["asset_type"].fillna("").astype(str)
    file_name = out.get("file_name", pd.Series([""] * len(out))).fillna("").astype(str)
    section_id = out.get("section_id", pd.Series([""] * len(out))).fillna("").astype(str)
    inferred = asset.copy()
    inferred[(inferred == "") & file_name.str.endswith(".h5ad")] = "spatial_h5ad"
    inferred[(inferred == "") & section_id.eq("proteome")] = "protein_fasta"
    out["asset_type"] = inferred
    for col in ("expected_bytes", "actual_bytes"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


if __name__ == "__main__":
    main()
