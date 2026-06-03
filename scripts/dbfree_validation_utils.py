from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_manifest(path: str | Path) -> dict[str, object]:
    """Load a JSON-compatible YAML manifest without adding a runtime YAML dependency."""

    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{manifest_path} must be JSON-compatible YAML. Keep manifests as JSON objects "
            "or install a YAML parser and extend this loader."
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} did not contain a manifest object.")
    if not manifest.get("name"):
        raise ValueError(f"{manifest_path} is missing `name`.")
    if not isinstance(manifest.get("sections"), list) or not manifest["sections"]:
        raise ValueError(f"{manifest_path} must contain a non-empty `sections` list.")
    return manifest


def selected_sections(
    manifest: dict[str, object],
    *,
    section_names: Iterable[str] | None = None,
    smoke_only: bool = False,
) -> list[dict[str, object]]:
    sections = [dict(item) for item in manifest.get("sections", [])]
    if section_names:
        wanted = {str(name) for name in section_names}
        sections = [section for section in sections if str(section.get("name")) in wanted]
        missing = wanted - {str(section.get("name")) for section in sections}
        if missing:
            raise ValueError(f"Manifest has no sections: {sorted(missing)}")
    if smoke_only:
        sections = [section for section in sections if bool(section.get("smoke", False))]
    if not sections:
        raise ValueError("No sections selected.")
    return sections


def manifest_results_dir(manifest: dict[str, object], override: str | Path | None = None) -> Path:
    if override is not None:
        root = Path(override)
        return root / str(manifest["name"]) if root.name != str(manifest["name"]) else root
    return Path(str(manifest.get("results_dir", Path("results/dbfree_validation") / str(manifest["name"]))))


def manifest_data_dir(manifest: dict[str, object], override: str | Path | None = None) -> Path:
    if override is not None:
        root = Path(override)
        return root / str(manifest["name"]) if root.name != str(manifest["name"]) else root
    return Path(str(manifest.get("data_dir", Path("data/dbfree_validation") / str(manifest["name"]))))


def section_file_name(section: dict[str, object]) -> str:
    if section.get("file_name"):
        return str(section["file_name"])
    url = str(section.get("h5ad_url") or section.get("url") or "")
    if "/" in url:
        return url.rstrip("/").rsplit("/", 1)[-1]
    return f"{section['name']}.h5ad"


def section_local_path(section: dict[str, object], data_dir: Path) -> Path:
    if section.get("local_path"):
        return Path(str(section["local_path"]))
    return data_dir / "raw" / section_file_name(section)


def prepared_section_path(section: dict[str, object], results_dir: Path) -> Path:
    return results_dir / "prepared" / f"{section['name']}.h5ad"


def parse_size_bytes(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    text = str(value).strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?B?|bytes?)?", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse file size: {value!r}")
    number = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    factors = {"B": 1, "BYTE": 1, "BYTES": 1, "K": 1000, "KB": 1000, "M": 1000**2, "MB": 1000**2, "G": 1000**3, "GB": 1000**3, "T": 1000**4, "TB": 1000**4}
    return int(number * factors[unit])


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_tsv(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, sep="\t", index=False)


def read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def checksum_short(path: str | Path) -> str:
    file_path = Path(path)
    return sha256_file(file_path)[:16] if file_path.exists() else ""


def size_ratio_ok(actual: int, expected: int | None, *, tolerance: float = 0.05) -> bool:
    if expected is None or expected <= 0:
        return True
    return math.isclose(float(actual), float(expected), rel_tol=tolerance)
