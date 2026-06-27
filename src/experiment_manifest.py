"""Small JSON manifests for simulator experiment provenance.

Captures library versions, git commit, topology, seeds, and flags alongside the
generated artifacts so runs are reproducible.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

__all__ = [
    "try_git_rev_parse_head",
    "library_versions",
    "build_base_manifest",
    "write_manifest_json",
]


def try_git_rev_parse_head(repo_root: Optional[Path] = None) -> str:
    """Return the current git commit hash, or an empty string if unavailable."""
    cwd = str(repo_root) if repo_root is not None else None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def library_versions() -> dict[str, str]:
    """Return Qiskit and Aer versions (empty strings when not importable)."""
    out: dict[str, str] = {}
    try:
        import qiskit

        out["qiskit"] = str(getattr(qiskit, "__version__", ""))
    except Exception:
        out["qiskit"] = ""
    try:
        import qiskit_aer

        out["qiskit_aer"] = str(getattr(qiskit_aer, "__version__", ""))
    except Exception:
        out["qiskit_aer"] = ""
    return out


def build_base_manifest(
    *,
    script: str,
    argv: list[str],
    extra: Optional[Mapping[str, Any]] = None,
    repo_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Assemble the common provenance fields for an experiment manifest sidecar."""
    payload: dict[str, Any] = {
        "schema": "experiment_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "script": script,
        "argv": list(argv),
        "git_commit": try_git_rev_parse_head(repo_root),
        "versions": library_versions(),
    }
    if extra:
        payload["experiment"] = dict(extra)
    return payload


def write_manifest_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as pretty-printed, sorted JSON, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
