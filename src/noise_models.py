"""Qiskit Aer noise models and thin runners for density-matrix / shot simulation.

Also provides partial trace onto the FRQI data register (color + position),
fidelity against the ideal FRQI reference, single-qubit readout calibration +
linear-inversion mitigation, and helpers for shot experiments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
from qiskit.transpiler import CouplingMap

from qiskit_aer.noise import NoiseModel as NoiseModelType


from src.frqi import build_frqi_statevector, validate_grayscale_image
from src.improved import frqi_structural_num_qubits_vchain

__all__ = [
    "transpile_for_aer_noise",
    "TranspileConfig",
    "build_noise_model",
    "run_density_matrix",
    "run_shots_counts",
    "reduced_frqi_density_matrix",
    "frqi_reduced_state_fidelity",
    "estimate_single_qubit_readout_matrix",
    "invert_readout_matrix",
    "counts_to_prob_vector",
    "mitigate_linear",
    "run_readout_mitigation_slice_demo",
    "noisy_frqi_metrics_row",
]

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error
except Exception as exc:  # pragma: no cover
    AerSimulator = None  # type: ignore[misc, assignment]
    NoiseModel = None  # type: ignore[misc, assignment]
    ReadoutError = None  # type: ignore[misc, assignment]
    depolarizing_error = None  # type: ignore[misc, assignment]
    thermal_relaxation_error = None  # type: ignore[misc, assignment]
    _AER_IMPORT_ERROR = exc
else:
    _AER_IMPORT_ERROR = None

# Default Aer-friendly basis used whenever a caller does not specify one.
_DEFAULT_BASIS_GATES: Tuple[str, ...] = ("id", "rz", "sx", "x", "cx")


def _require_aer() -> None:
    if AerSimulator is None or NoiseModel is None:
        raise RuntimeError("qiskit-aer is required for noisy simulation.") from _AER_IMPORT_ERROR


def _as_density_matrix(obj: Any) -> DensityMatrix:
    if isinstance(obj, DensityMatrix):
        return obj
    return DensityMatrix(obj)


def transpile_for_aer_noise(
    qc: QuantumCircuit,
    *,
    basis_gates: Sequence[str] = _DEFAULT_BASIS_GATES,
    optimization_level: int = 0,
    coupling_map: Optional[Any] = None,
    backend: Optional[Any] = None,
    seed_transpiler: Optional[int] = None,
    layout_method: Optional[str] = None,
    routing_method: Optional[str] = None,
) -> QuantumCircuit:
    """Transpile a circuit for Aer execution.

    The target is chosen by precedence:

    - ``backend`` set: transpile directly for that backend (coupling + basis +
      calibration-derived noise; recommended for mock NISQ snapshots).
    - else ``coupling_map`` set: transpile onto that topology with ``basis_gates``.
    - else (both ``None``): the legacy path, a **fully connected** coupling map
      with ``layout_method="trivial"`` so logical qubit order is preserved.

    Returns
    -------
    QuantumCircuit
        The transpiled circuit.
    """
    kwargs: Dict[str, Any] = {"optimization_level": int(optimization_level)}
    if seed_transpiler is not None:
        kwargs["seed_transpiler"] = int(seed_transpiler)

    if backend is not None:
        return transpile(qc, backend=backend, **kwargs)

    if coupling_map is None:
        kwargs["coupling_map"] = CouplingMap.from_full(qc.num_qubits)
        kwargs["basis_gates"] = list(basis_gates)
        kwargs["layout_method"] = layout_method or "trivial"
        kwargs["routing_method"] = routing_method or "basic"
        return transpile(qc, **kwargs)

    kwargs["coupling_map"] = coupling_map
    kwargs["basis_gates"] = list(basis_gates)
    if layout_method is not None:
        kwargs["layout_method"] = layout_method
    if routing_method is not None:
        kwargs["routing_method"] = routing_method
    return transpile(qc, **kwargs)


@dataclass(frozen=True)
class TranspileConfig:
    """Bundle of transpilation options threaded through the Aer runners.

    Centralizes the basis-gate default and the seven ``transpile_*`` knobs so
    each runner builds one config instead of repeating the same kwargs.
    """

    basis_gates: Sequence[str] = _DEFAULT_BASIS_GATES
    optimization_level: int = 0
    coupling_map: Optional[Any] = None
    backend: Optional[Any] = None
    seed_transpiler: Optional[int] = None
    layout_method: Optional[str] = None
    routing_method: Optional[str] = None

    @classmethod
    def from_runner_kwargs(
        cls,
        *,
        backend: Optional[Any] = None,
        coupling_map: Optional[Any] = None,
        basis_gates: Optional[Sequence[str]] = None,
        optimization_level: int = 0,
        seed_transpiler: Optional[int] = None,
        layout_method: Optional[str] = None,
        routing_method: Optional[str] = None,
    ) -> "TranspileConfig":
        """Build a config from a runner's ``transpile_*`` keyword arguments.

        ``basis_gates is None`` falls back to :data:`_DEFAULT_BASIS_GATES`.
        """
        return cls(
            basis_gates=tuple(basis_gates) if basis_gates is not None else _DEFAULT_BASIS_GATES,
            optimization_level=int(optimization_level),
            coupling_map=coupling_map,
            backend=backend,
            seed_transpiler=seed_transpiler,
            layout_method=layout_method,
            routing_method=routing_method,
        )

    def apply(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Transpile ``qc`` using :func:`transpile_for_aer_noise` with this config."""
        return transpile_for_aer_noise(
            qc,
            basis_gates=self.basis_gates,
            optimization_level=self.optimization_level,
            coupling_map=self.coupling_map,
            backend=self.backend,
            seed_transpiler=self.seed_transpiler,
            layout_method=self.layout_method,
            routing_method=self.routing_method,
        )


def build_noise_model(
    *,
    p_depol_1q: float = 0.0,
    p_depol_2q: float = 0.0,
    t1: Optional[float] = None,
    t2: Optional[float] = None,
    gate_time_1q: Optional[float] = None,
    gate_time_2q: Optional[float] = None,
    readout_prob_01: float = 0.0,
    readout_prob_10: float = 0.0,
    single_qubit_gates: Sequence[str] = ("id", "rz", "sx", "x", "h", "delay"),
    two_qubit_gates: Sequence[str] = ("cx",),
) -> NoiseModelType:
    """Build a :class:`NoiseModel` with depolarizing, optional thermal, and readout error.

    Thermal relaxation is applied on single-qubit gates only and requires
    ``t1``, ``t2``, ``gate_time_1q`` in the **same unit**. ``gate_time_2q`` is
    accepted for API symmetry but unused (two-qubit gates get depolarizing only).

    Readout convention: ``readout_prob_01`` is ``P(meas=1 | true=0)`` and
    ``readout_prob_10`` is ``P(meas=0 | true=1)``; the same single-qubit
    confusion is attached to every qubit.
    """
    _require_aer()
    assert NoiseModel is not None and depolarizing_error is not None
    assert thermal_relaxation_error is not None and ReadoutError is not None

    _ = gate_time_2q  # Reserved for future two-qubit thermal modeling.
    noise_model = NoiseModel()
    use_thermal = (
        t1 is not None
        and t2 is not None
        and gate_time_1q is not None
        and t1 > 0
        and t2 > 0
        and gate_time_1q >= 0
    )

    def _one_qubit_error() -> Any:
        errs: List[Any] = []
        if p_depol_1q > 0:
            errs.append(depolarizing_error(p_depol_1q, 1))
        if use_thermal:
            errs.append(thermal_relaxation_error(t1, t2, gate_time_1q))  # type: ignore[arg-type]
        if not errs:
            return None
        out = errs[0]
        for e in errs[1:]:
            out = out.compose(e)
        return out

    def _two_qubit_error() -> Any:
        if p_depol_2q <= 0:
            return None
        return depolarizing_error(p_depol_2q, 2)

    e1 = _one_qubit_error()
    if e1 is not None:
        for g in single_qubit_gates:
            noise_model.add_all_qubit_quantum_error(e1, [g])

    e2 = _two_qubit_error()
    if e2 is not None:
        for g in two_qubit_gates:
            noise_model.add_all_qubit_quantum_error(e2, [g])

    if readout_prob_01 > 0 or readout_prob_10 > 0:
        p01 = float(np.clip(readout_prob_01, 0.0, 1.0))
        p10 = float(np.clip(readout_prob_10, 0.0, 1.0))
        # Aer expects a row-stochastic matrix indexed by the true state, so we
        # transpose our P(measured=i | true=j) confusion before handing it over.
        confusion = [[1.0 - p01, p10], [p01, 1.0 - p10]]
        ro = ReadoutError(np.asarray(confusion).T.tolist())
        add_ro = getattr(noise_model, "add_all_qubit_readout_error", None)
        if callable(add_ro):
            add_ro(ro)
        else:  # pragma: no cover
            raise RuntimeError("NoiseModel missing add_all_qubit_readout_error; update qiskit-aer.")

    return noise_model


def run_density_matrix(
    qc: QuantumCircuit,
    noise_model: Optional[Any] = None,
    *,
    transpile_first: bool = True,
    seed_simulator: Optional[int] = None,
    transpile_backend: Optional[Any] = None,
    transpile_coupling_map: Optional[Any] = None,
    transpile_basis_gates: Optional[Sequence[str]] = None,
    transpile_optimization_level: int = 0,
    seed_transpiler: Optional[int] = None,
    transpile_layout_method: Optional[str] = None,
    transpile_routing_method: Optional[str] = None,
) -> DensityMatrix:
    """Run ``qc`` on ``AerSimulator(method='density_matrix')`` and return the final state."""
    _require_aer()
    assert AerSimulator is not None
    if transpile_first:
        cfg = TranspileConfig.from_runner_kwargs(
            backend=transpile_backend,
            coupling_map=transpile_coupling_map,
            basis_gates=transpile_basis_gates,
            optimization_level=transpile_optimization_level,
            seed_transpiler=seed_transpiler,
            layout_method=transpile_layout_method,
            routing_method=transpile_routing_method,
        )
        tqc = cfg.apply(qc)
    else:
        tqc = qc
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    circ = tqc.copy()
    circ.save_density_matrix(label="dm")
    opts: Dict[str, Any] = {}
    if seed_simulator is not None:
        opts["seed_simulator"] = int(seed_simulator)
    job = sim.run(circ, shots=1, **opts)
    res = job.result()
    # Result.data(0) vs Result.data() varies across Qiskit 1.x point releases.
    try:
        raw = res.data(0)["dm"]
    except Exception:
        raw = res.data()["dm"]
    return _as_density_matrix(raw)


def run_shots_counts(
    qc: QuantumCircuit,
    noise_model: Optional[Any] = None,
    *,
    shots: int = 4096,
    transpile_first: bool = True,
    seed_simulator: Optional[int] = None,
    transpile_backend: Optional[Any] = None,
    transpile_coupling_map: Optional[Any] = None,
    transpile_basis_gates: Optional[Sequence[str]] = None,
    transpile_optimization_level: int = 0,
    seed_transpiler: Optional[int] = None,
    transpile_layout_method: Optional[str] = None,
    transpile_routing_method: Optional[str] = None,
) -> Dict[str, int]:
    """Run ``qc`` (which must include measurements) and return raw shot counts."""
    _require_aer()
    assert AerSimulator is not None
    if transpile_first:
        cfg = TranspileConfig.from_runner_kwargs(
            backend=transpile_backend,
            coupling_map=transpile_coupling_map,
            basis_gates=transpile_basis_gates,
            optimization_level=transpile_optimization_level,
            seed_transpiler=seed_transpiler,
            layout_method=transpile_layout_method,
            routing_method=transpile_routing_method,
        )
        tqc = cfg.apply(qc)
    else:
        tqc = qc
    sim = AerSimulator(noise_model=noise_model)
    opts: Dict[str, Any] = {}
    if seed_simulator is not None:
        opts["seed_simulator"] = int(seed_simulator)
    job = sim.run(tqc, shots=int(shots), **opts)
    return job.result().get_counts()


def reduced_frqi_density_matrix(rho: Any, m: int, total_qubits: int) -> DensityMatrix:
    """Trace out flag + v-chain ancillas, keeping qubits ``0..m`` (color + position).

    Raises
    ------
    ValueError
        If ``rho`` does not have ``total_qubits`` qubits.
    """
    rho_dm = _as_density_matrix(rho)
    if rho_dm.num_qubits != total_qubits:
        raise ValueError(f"Expected {total_qubits} qubits in rho, got {rho_dm.num_qubits}.")
    trace_out = list(range(m + 1, total_qubits))
    if not trace_out:
        return rho_dm
    return partial_trace(rho_dm, trace_out)


def frqi_reduced_state_fidelity(
    rho_full: Any,
    image: np.ndarray,
    *,
    m: int,
    total_qubits: int,
) -> float:
    """Return the fidelity of the reduced noisy state to the ideal FRQI data register."""
    img = validate_grayscale_image(image)
    ref = DensityMatrix(build_frqi_statevector(img))
    red = reduced_frqi_density_matrix(rho_full, m, total_qubits)
    return float(state_fidelity(red, ref))


def estimate_single_qubit_readout_matrix(
    num_qubits: int,
    measured_qubit: int,
    noise_model: Optional[Any],
    *,
    shots: int = 8000,
    seed_simulator: Optional[int] = None,
    transpile_backend: Optional[Any] = None,
    transpile_coupling_map: Optional[Any] = None,
    transpile_basis_gates: Optional[Sequence[str]] = None,
    transpile_optimization_level: int = 0,
    seed_transpiler: Optional[int] = None,
    transpile_layout_method: Optional[str] = None,
    transpile_routing_method: Optional[str] = None,
) -> np.ndarray:
    """Estimate the 2x2 readout confusion ``A[i,j] ~= P(measured=i | true=j)``.

    Prepares ``|0>`` and ``|1>`` on ``measured_qubit``, measures, and
    column-normalizes the empirical frequencies.
    """
    _require_aer()
    assert AerSimulator is not None
    sim = AerSimulator(noise_model=noise_model)
    opts: Dict[str, Any] = {}
    if seed_simulator is not None:
        opts["seed_simulator"] = int(seed_simulator)
    cfg = TranspileConfig.from_runner_kwargs(
        backend=transpile_backend,
        coupling_map=transpile_coupling_map,
        basis_gates=transpile_basis_gates,
        optimization_level=transpile_optimization_level,
        seed_transpiler=seed_transpiler,
        layout_method=transpile_layout_method,
        routing_method=transpile_routing_method,
    )

    def _cal_prep(true_one: bool) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits, 1)
        if true_one:
            qc.x(measured_qubit)
        qc.measure(measured_qubit, 0)
        return cfg.apply(qc)

    est = np.zeros((2, 2), dtype=np.float64)
    for j, true_one in enumerate((False, True)):
        qc = _cal_prep(true_one)
        job = sim.run(qc, shots=int(shots), **opts)
        cts = job.result().get_counts()
        est[0, j] = int(cts.get("0", 0)) / float(shots)
        est[1, j] = int(cts.get("1", 0)) / float(shots)
    for j in range(2):
        col = est[:, j].sum()
        if col > 0:
            est[:, j] /= col
    return est


def invert_readout_matrix(confusion: np.ndarray, *, rcond: float = 1e-6) -> np.ndarray:
    """Return the Moore-Penrose inverse of a readout confusion matrix."""
    mat = np.asarray(confusion, dtype=np.float64)
    return np.linalg.pinv(mat, rcond=rcond)


def counts_to_prob_vector(counts: Mapping[str, int], num_measured: int, shots: int) -> np.ndarray:
    """Convert bitstring counts to a length-``2**num_measured`` probability vector (little-endian)."""
    dim = 2**num_measured
    p = np.zeros(dim, dtype=np.float64)
    for bits, n in counts.items():
        idx = int(bits, 2) if isinstance(bits, str) else int(bits)
        p[idx] += float(n) / float(shots)
    return p


def mitigate_linear(probs: np.ndarray, inv_matrix: np.ndarray) -> np.ndarray:
    """Apply ``inv_matrix @ probs``, then clip negatives and renormalize."""
    out = inv_matrix @ np.asarray(probs, dtype=np.float64)
    out = np.clip(out, 0.0, None)
    s = out.sum()
    if s > 0:
        out /= s
    return out


def run_readout_mitigation_slice_demo(
    m: int,
    address_bits: int,
    theta: float,
    *,
    mode: str = "v-chain",
    noise_model: Optional[Any] = None,
    shots: int = 12000,
    shots_cal: Optional[int] = None,
    seed_simulator: int = 123,
    rcond: float = 1e-6,
    transpile_backend: Optional[Any] = None,
    transpile_coupling_map: Optional[Any] = None,
    transpile_basis_gates: Optional[Sequence[str]] = None,
    transpile_optimization_level: int = 0,
    seed_transpiler: Optional[int] = None,
    transpile_layout_method: Optional[str] = None,
    transpile_routing_method: Optional[str] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Run a shot demo on a single FRQI slice: raw vs mitigated ``P(color=1)``.

    Calibrates the color-qubit readout matrix, inverts it, then measures
    :func:`src.improved.build_single_address_ry_slice` and mitigates.

    Returns
    -------
    tuple
        ``(p1_raw, p1_mitigated, confusion, inv_confusion)``.
    """
    from src.improved import build_single_address_ry_slice

    cal_shots = int(shots_cal) if shots_cal is not None else int(shots)
    data_shots = int(shots)

    base = build_single_address_ry_slice(m, address_bits, theta, mode=mode)
    n = base.num_qubits
    confusion = estimate_single_qubit_readout_matrix(
        n,
        0,
        noise_model,
        shots=cal_shots,
        seed_simulator=seed_simulator,
        transpile_backend=transpile_backend,
        transpile_coupling_map=transpile_coupling_map,
        transpile_basis_gates=transpile_basis_gates,
        transpile_optimization_level=transpile_optimization_level,
        seed_transpiler=seed_transpiler,
        transpile_layout_method=transpile_layout_method,
        transpile_routing_method=transpile_routing_method,
    )
    inv_c = invert_readout_matrix(confusion, rcond=rcond)

    qc = QuantumCircuit(n, 1)
    qc.compose(base, qubits=list(range(n)), inplace=True)
    qc.measure(0, 0)
    cfg = TranspileConfig.from_runner_kwargs(
        backend=transpile_backend,
        coupling_map=transpile_coupling_map,
        basis_gates=transpile_basis_gates,
        optimization_level=transpile_optimization_level,
        seed_transpiler=seed_transpiler,
        layout_method=transpile_layout_method,
        routing_method=transpile_routing_method,
    )
    tqc = cfg.apply(qc)

    _require_aer()
    assert AerSimulator is not None
    sim = AerSimulator(noise_model=noise_model)
    job = sim.run(tqc, shots=data_shots, seed_simulator=seed_simulator)
    cts = job.result().get_counts()
    probs = counts_to_prob_vector(cts, 1, data_shots)
    mit = mitigate_linear(probs, inv_c)
    return float(probs[1]), float(mit[1]), confusion, inv_c


def noisy_frqi_metrics_row(
    image: np.ndarray,
    *,
    method: str,
    noise_model: Optional[Any],
    seed_simulator: Optional[int] = None,
    return_recon: bool = False,
    transpile_backend: Optional[Any] = None,
    transpile_coupling_map: Optional[Any] = None,
    transpile_basis_gates: Optional[Sequence[str]] = None,
    transpile_optimization_level: int = 0,
    seed_transpiler: Optional[int] = None,
    transpile_layout_method: Optional[str] = None,
    transpile_routing_method: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a density-matrix noisy FRQI experiment and return a metrics row.

    Computes fidelity to the ideal state plus PSNR/SSIM on the mixed-state
    reconstruction. With ``return_recon=True`` the result also includes the
    reconstructed image under key ``recon``.

    Parameters
    ----------
    method : str
        Either ``"vchain"`` or ``"naive"``.

    Raises
    ------
    ValueError
        If ``method`` is not recognized.
    """
    from src.frqi import reconstruct_image_from_reduced_density_matrix, required_position_qubits
    from src.metrics import psnr, ssim_uint8

    img = validate_grayscale_image(image)
    n = int(img.shape[0])
    m = required_position_qubits(n)
    if method == "vchain":
        from src.improved import build_frqi_prep_vchain

        qc = build_frqi_prep_vchain(img)
        total = frqi_structural_num_qubits_vchain(m)
    elif method == "naive":
        from src.improved import build_frqi_prep_naive

        qc = build_frqi_prep_naive(img)
        total = qc.num_qubits
    else:
        raise ValueError("method must be 'vchain' or 'naive'.")

    t0 = time.perf_counter()
    rho = run_density_matrix(
        qc,
        noise_model,
        seed_simulator=seed_simulator,
        transpile_backend=transpile_backend,
        transpile_coupling_map=transpile_coupling_map,
        transpile_basis_gates=transpile_basis_gates,
        transpile_optimization_level=transpile_optimization_level,
        seed_transpiler=seed_transpiler,
        transpile_layout_method=transpile_layout_method,
        transpile_routing_method=transpile_routing_method,
    )
    wall = time.perf_counter() - t0
    red = reduced_frqi_density_matrix(rho, m, total_qubits=total)
    fid = float(state_fidelity(red, DensityMatrix(build_frqi_statevector(img))))
    recon = reconstruct_image_from_reduced_density_matrix(red, n)
    p = psnr(img, recon)
    try:
        s = ssim_uint8(img, recon)
    except ValueError:
        s = float("nan")

    out: Dict[str, Any] = {
        "m": m,
        "total_qubits": total,
        "wall_s": wall,
        "fidelity": fid,
        "psnr": p,
        "ssim": s,
    }
    if return_recon:
        out["recon"] = recon
    return out
