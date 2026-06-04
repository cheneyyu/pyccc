from __future__ import annotations

from pathlib import Path


PACKAGE_MODEL_ROOT = Path(__file__).resolve().parent / "models"
DEFAULT_DBFREE_ROLE_MODEL = PACKAGE_MODEL_ROOT / "universal_esmc300m_lgbm_role_classifiers_v0"
DEFAULT_DBFREE_PAIR_MODEL = PACKAGE_MODEL_ROOT / "universal_esmc300m_lgbm_pair_ranker_v0"


def resolve_dbfree_model_path(model: str | Path, *, expected_file: str | None = None) -> Path:
    """Resolve explicit or bundled DB-free model paths."""

    path = Path(model)
    if _model_path_exists(path, expected_file=expected_file):
        return path
    bundled = {
        DEFAULT_DBFREE_ROLE_MODEL.name: DEFAULT_DBFREE_ROLE_MODEL,
        DEFAULT_DBFREE_PAIR_MODEL.name: DEFAULT_DBFREE_PAIR_MODEL,
    }.get(path.name)
    if bundled is not None and _model_path_exists(bundled, expected_file=expected_file):
        return bundled
    return path


def _model_path_exists(path: Path, *, expected_file: str | None) -> bool:
    if expected_file is None:
        return path.exists()
    return (path / expected_file).exists()
