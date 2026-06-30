"""Draw a small MCX v-chain stub circuit for figure reuse.

Saves PNG to outputs/stub_mcx_vchain.png (creates outputs/ if needed).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from qiskit import QuantumCircuit

    # 3 controls + 1 target X + ancilla count for v-chain = num_ctrl - 2 = 1
    num_ctrl = 3
    ancilla = num_ctrl - 2
    total = num_ctrl + 1 + max(ancilla, 0)
    qc = QuantumCircuit(total)
    # qubit order: controls 0..num_ctrl-1, target at num_ctrl, ancilla at end
    ctrl = list(range(num_ctrl))
    targ = num_ctrl
    anc = num_ctrl + 1 if ancilla > 0 else None

    if ancilla > 0:
        qc.mcx(ctrl, targ, ancilla_qubits=anc)
    else:
        qc.mcx(ctrl, targ)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "stub_mcx_vchain.png"
    print("Drawing MCX v-chain stub circuit...", flush=True)
    # Use matplotlib to save with custom DPI
    fig = qc.draw(output="mpl", style="iqp", fold=40)
    fig.savefig(str(out), dpi=160, bbox_inches="tight")
    print("Wrote", out, flush=True)


if __name__ == "__main__":
    main()
