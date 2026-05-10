"""Quantum circuit resource counts after transpilation."""

from __future__ import annotations

from typing import Any

from qiskit import transpile


def transpiled_circuit_stats(qc: Any, basis_gates: list[str], opt_level: int) -> dict[str, int | float]:
    """Transpile ``qc`` and return qubit count, depth, size, and CX count.

    ``basis_gates`` and ``opt_level`` are passed to :func:`qiskit.transpile`.
    Additional single-qubit totals are included for reporting (gate names depend
    on the chosen basis).
    """
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
