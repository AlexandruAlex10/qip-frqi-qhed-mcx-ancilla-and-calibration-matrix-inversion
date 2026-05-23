"""
Mock NISQ-style backends and YAML experiment presets (no IBM credentials).

Primary path for thesis-style evidence: :class:`~qiskit.providers.fake_provider.GenericBackendV2`
with a **linear** coupling map and ``num_qubits`` matching the FRQI circuit, combined with
``NoiseModel.from_backend`` so Aer uses snapshot-like gate errors and readout channels.

Optional named ``Fake*V2`` backends from ``qiskit.providers.fake_provider`` are supported when
their qubit count is sufficient for the circuit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from qiskit.transpiler import CouplingMap


def linear_coupling_edges(num_qubits: int) -> List[List[int]]:
    """Undirected line as Qiskit coupling edges (both directions)."""
    if num_qubits < 2:
        return []
    edges: List[List[int]] = []
    for i in range(num_qubits - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    return edges


def make_generic_linear_backend(
    num_qubits: int,
    *,
    seed: int = 42,
    basis_gates: Optional[List[str]] = None,
) -> Any:
    """``GenericBackendV2`` on a line topology (good when FRQI width exceeds small fakes)."""
    from qiskit.providers.fake_provider import GenericBackendV2

    bg = basis_gates or ["cx", "rz", "sx", "id", "x"]
    return GenericBackendV2(
        num_qubits,
        basis_gates=bg,
        coupling_map=linear_coupling_edges(num_qubits),
        seed=int(seed),
    )


def load_named_fake_backend_v2(name: str) -> Any:
    """Instantiate ``FakeFooV2`` from ``qiskit.providers.fake_provider`` by class name."""
    import importlib

    mod = importlib.import_module("qiskit.providers.fake_provider")
    if not hasattr(mod, name):
        known = sorted(x for x in dir(mod) if x.startswith("Fake") and x.endswith("V2"))
        raise ValueError(f"Unknown fake backend {name!r}. Examples: {known[:8]} ...")
    cls = getattr(mod, name)
    return cls()


def noise_model_from_backend(backend: Any) -> Any:
    """Aer :class:`~qiskit_aer.noise.NoiseModel` built from a ``BackendV2`` snapshot."""
    from qiskit_aer.noise import NoiseModel

    return NoiseModel.from_backend(backend)


def load_yaml_preset(path: Path | str) -> Dict[str, Any]:
    """Load a YAML preset document (requires PyYAML)."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load experiment presets (pip install pyyaml).") from exc

    p = Path(path)
    with p.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None or not isinstance(raw, dict):
        raise ValueError(f"Preset {p} must be a YAML mapping at the top level.")
    return raw


def build_noise_model_from_preset(
    doc: Mapping[str, Any],
    *,
    scale: float = 1.0,
) -> Tuple[Any, Dict[str, Any]]:
    """Return ``(noise_model, flat_metadata)`` using keys under ``doc['noise']``."""
    from src.noise_models import build_noise_model

    noise = doc.get("noise", {})
    if not isinstance(noise, dict):
        raise ValueError("Preset must contain a 'noise' mapping.")

    def _f(key: str, default: float = 0.0) -> float:
        v = noise.get(key, default)
        if v is None:
            return float(default)
        return float(v) * float(scale)

    def _opt_float(key: str) -> Optional[float]:
        v = noise.get(key)
        if v is None:
            return None
        return float(v)

    nm = build_noise_model(
        p_depol_1q=_f("p_depol_1q", 0.0),
        p_depol_2q=_f("p_depol_2q", 0.0),
        t1=_opt_float("t1"),
        t2=_opt_float("t2"),
        gate_time_1q=_opt_float("gate_time_1q"),
        gate_time_2q=_opt_float("gate_time_2q"),
        readout_prob_01=_f("readout_prob_01", 0.0),
        readout_prob_10=_f("readout_prob_10", 0.0),
    )
    meta = {
        "preset_name": str(doc.get("name", "")),
        "preset_description": str(doc.get("description", "")),
        "noise_scale": float(scale),
    }
    return nm, meta


def coupling_map_from_backend(backend: Any) -> CouplingMap:
    """Best-effort :class:`~qiskit.transpiler.CouplingMap` for a ``BackendV2``."""
    cm = getattr(backend, "coupling_map", None)
    if cm is None:
        raise ValueError("Backend has no coupling_map.")
    if isinstance(cm, CouplingMap):
        return cm
    return CouplingMap(cm)
