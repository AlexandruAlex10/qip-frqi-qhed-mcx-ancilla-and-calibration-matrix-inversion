"""Quantum circuit resource counts after transpilation."""

from __future__ import annotations

from typing import Any, Optional

from qiskit import transpile

__all__ = ["transpiled_circuit_stats"]


def transpiled_circuit_stats(
    qc: Any,
    basis_gates: list[str],
    opt_level: int,
    *,
    backend: Optional[Any] = None,
    coupling_map: Optional[Any] = None,
) -> dict[str, int | float]:
    """Transpile ``qc`` and report qubit count, depth, size, CX, and single-qubit gates.

    The transpile target follows precedence ``backend`` > ``coupling_map`` >
    ``basis_gates`` only. Single-qubit totals depend on the chosen basis.

    Returns
    -------
    dict
        Keys ``num_qubits``, ``depth``, ``size``, ``cx``, ``single_qubit_gates``.
    """
    if backend is not None:
        tqc = transpile(qc, backend=backend, optimization_level=opt_level)
    elif coupling_map is not None:
        tqc = transpile(qc, basis_gates=basis_gates, coupling_map=coupling_map, optimization_level=opt_level)
    else:
        tqc = transpile(qc, basis_gates=basis_gates, optimization_level=opt_level)
    ops = tqc.count_ops()
    cx = int(ops.get("cx", 0))
    single = sum(int(v) for k, v in ops.items() if k != "cx")
    return {
        "num_qubits": int(tqc.num_qubits),
        "depth": int(tqc.depth()),
        "size": int(tqc.size()),
        "cx": cx,
        "single_qubit_gates": int(single),
    }
