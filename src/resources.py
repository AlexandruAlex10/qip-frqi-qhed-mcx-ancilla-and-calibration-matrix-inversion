"""
Quantum circuit resource counts after transpilation.
"""

from __future__ import annotations

from typing import Any, Optional

from qiskit import transpile


def transpiled_circuit_stats(
    qc: Any,
    basis_gates: list[str],
    opt_level: int,
    *,
    backend: Optional[Any] = None,
    coupling_map: Optional[Any] = None,
) -> dict[str, int | float]:
    """Transpile ``qc`` and return qubit count, depth, size, and CX count.

    ``basis_gates`` and ``opt_level`` are passed to :func:`qiskit.transpile` when ``backend``
    is not provided. If ``backend`` is provided, transpilation targets that backend directly
    (coupling + basis gates from the mock snapshot).

    Additional single-qubit totals are included for reporting (gate names depend
    on the chosen basis).
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
        "single_qubit_gates": int(single)
    }
