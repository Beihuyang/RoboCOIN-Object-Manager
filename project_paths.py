"""Portable path storage and compatibility helpers for project metadata."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PORTABLE_ROOTS = (
    "objects",
    "RoboCOIN_datasets",
    "models",
    "sam3_weights",
    "ontology",
    "templates",
)


def portable_path(path: str | Path, root: Path = PROJECT_ROOT) -> str:
    """Store project-owned paths relative to the project directory.

    Paths outside the project are retained as absolute paths because silently
    shortening them would make them impossible to resolve. Normal project data,
    model, and result paths are always returned in portable POSIX form.
    """
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        return candidate.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(candidate)


def _relocated_project_path(path: Path, root: Path) -> Path | None:
    """Map an absolute path saved under an older checkout to this checkout."""
    parts = path.parts
    project_name = root.name
    if project_name in parts:
        index = len(parts) - 1 - tuple(reversed(parts)).index(project_name)
        return root.joinpath(*parts[index + 1:])
    for anchor in PORTABLE_ROOTS:
        if anchor in parts:
            index = parts.index(anchor)
            return root.joinpath(*parts[index:])
    return None


def resolve_project_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve relative metadata paths and relocate stale absolute paths."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return (root / candidate).resolve()
    if candidate.exists():
        return candidate.resolve()
    relocated = _relocated_project_path(candidate, root.resolve())
    return relocated.resolve() if relocated is not None else candidate.resolve()


def same_project_path(left: str | Path, right: str | Path) -> bool:
    """Compare paths after applying checkout-relocation compatibility."""
    return resolve_project_path(left) == resolve_project_path(right)


def portable_saved_string(value: str, root: Path = PROJECT_ROOT) -> str:
    """Convert a saved absolute project path to relative form when recognized."""
    path = Path(value)
    if not path.is_absolute():
        return value
    relocated = _relocated_project_path(path, root.resolve())
    if relocated is None:
        return value
    return portable_path(relocated, root)
