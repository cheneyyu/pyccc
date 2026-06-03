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
    parser.add_argument("--dry-run", action="store_true", help="Write the manifest table without downloading files.")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    data_dir = manifest_data_dir(manifest, args.data_dir)
    results_dir = manifest_results_dir(manifest, args.results_dir)
    rows = []
    for section in selected_sections(manifest, section_names=args.section, smoke_only=args.smoke_only):
        rows.append(_download_section(manifest, section, data_dir=data_dir, dry_run=args.dry_run))

    out = pd.DataFrame(rows)
    write_tsv(out, results_dir.parent / "download_manifest.tsv")
    write_tsv(out, results_dir / "download_manifest.tsv")


def _download_section(
    manifest: dict[str, object],
    section: dict[str, object],
    *,
    data_dir: Path,
    dry_run: bool,
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
        _download_url(url, local_path)
        actual_bytes = local_path.stat().st_size
        checksum = sha256_file(local_path)
        status = "ok" if size_ratio_ok(actual_bytes, expected_bytes) else "size_mismatch"
    return {
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


def _download_url(url: str, output: Path) -> None:
    tmp = output.with_suffix(output.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        tmp.replace(output)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
