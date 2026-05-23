# Structural baseline vs empirical reference

This note freezes two comparison layers for gate-level FRQI preparation. It aligns with the repository indexing in `build_frqi_statevector` (`src/frqi.py`): basis **|position, color⟩** with the **color qubit as the least significant bit** in the computational basis index.

## 1. Empirical reference (black-box synthesis)

**Definition:** Build the exact FRQI statevector, then let Qiskit synthesize state preparation via `QuantumCircuit.initialize(state)` (`maybe_build_qiskit_circuit` in `src/frqi.py`).

**Role:** Anchor for transpiled **depth**, **CX count**, and **gate size** after decomposition to the fixed basis **`cx`, `rz`, `sx`** at optimization level **3** (same as `scripts/frqi_qhed_sobel.py` and `src/resources.py`).

**Important limitation (thesis Methods):** These metrics describe **automatic synthesis** of the full state, not a transparent sequence of “one rotation per pixel” on an address register. They must **not** be quoted as the structural cost of multi-controlled FRQI prep.

See the initialize-only columns in `outputs/frqi_qhed_sobel_metrics.csv` (produced by the demo script).

## 2. Structural baseline (thesis gate model)

**Definition:** Explicit FRQI preparation as a **loop over pixels** \(p = 0, \ldots, N_{\text{pix}}-1\) with \(N_{\text{pix}} = N \times N\) for an \(N \times N\) image ( \(N\) a power of two).

- **Position register:** \(m = 2\log_2 N\) qubits encoding row/column (same information content as `required_position_qubits` in `src/frqi.py`).
- **Color register:** one qubit (Ry angle \(\theta_p = (\pi/2)\cdot (\text{intensity}_p / 255)\) in \([0, \pi/2]\)).
- **Per-pixel operation (conceptual):** apply **Ry(\(\theta_p\))** on the color qubit **conditioned on** the position register being exactly the bitstring of index \(p\).

Equivalently, each step is an **\(m\)-controlled Ry** (one multi-controlled rotation per pixel, applied sequentially). Unitarity is preserved because only one address matches at a time along the idealized construction; the thesis compares **decompositions** of that controlled rotation, not a different state.

### 2.1 Naive baseline (for complexity comparison)

**Documented choice:** Treat the **naive** structural baseline as the **standard decomposition** of an \(m\)-controlled single-qubit rotation into **\(m\)-controlled Pauli roots and Toffoli / multi-controlled X** ladders **without** dedicated ancilla — i.e. **exponential-in-\(m\)** Toffoli count in the worst case for textbook Barenco-style constructions of multi-controlled \(X\) (used inside controlled-\(R_y\) synthesis). This yields **\(\Theta(2^m)\)** two-qubit-rich layers **per** multi-controlled slice in the worst case, hence **per pixel** scaling **\(\Theta(N^2)\)** in \(m = 2\log_2 N\) since \(2^m = N^2\).

This is a **conceptual** worst case for “no ancilla assistance”; the exact gate count depends on the chosen decomposition library, but the scaling differentiates the thesis story from ancilla-assisted constructions.

### 2.2 Improved construction — fixed ancilla MCX variant

**Documented choice:** Implement the address-matching and controlled Ry using **Qiskit `MCXGate` with `mode="v-chain"`** (linear-depth **dirty-ancilla** chain / “v-chain” pattern as in Qiskit’s multi-controlled X documentation).

**Pattern per pixel \(p\)** (standard textbook structure):

1. Compare address register to index \(p\) using multi-controlled **X** into a **flag ancilla** (or workspace), implemented as **`MCXGate(..., mode="v-chain")`** with **\(m-2\)** ancilla qubits for \(m \ge 3\) (Qiskit’s v-chain requirement; smaller \(m\) uses fewer or no ancilla).
2. Apply **Ry(\(\theta_p\))** on the color qubit **controlled on the flag** (a single C-Ry once the flag is set).
3. **Uncompute** the flag via the **adjoint** of step 1 to return ancillas and parity workspace.

**Ancilla budget (upper bound, serialized pixels):** **\(m-2\)** workspace qubits suffice for one v-chain MCX at a time; they are **reused** across pixels. Total line count (data + ancilla) is therefore **\(m + 1 + (m-2) = 2m - 1\)** for \(m \ge 3\), plus any routing slack you document in layout.

**Complexity sketch (v-chain vs naive):**

- **Per MCX (v-chain):** **\(O(m)\)** CX and **\(O(m)\)** depth in the number of controls (see Qiskit / Maslov et al. multi-controlled X constructions).
- **Per pixel:** \(O(m)\) for compare + \(O(m)\) for uncompute \(\Rightarrow O(m)\) CX and depth up to constants.
- **Full FRQI prep:** **\(N_{\text{pix}} \cdot O(m) = O(N^2 \log N)\)** CX and depth for **sequential** pixel loop (dominant term; constants depend on basis and layout).

### 2.3 Failure modes (ablation hooks)

- **Routing and coupling map:** On hardware, long chains may lose CX savings if **SWAP insertion** dominates.
- **Ancilla availability:** If **\(m-2\)** clean or dirty ancillas are unavailable, fallback modes increase depth or force no-ancilla decompositions.
- **Readout dominates:** For very small demos, **SPAM** can mask gate-depth improvements unless mitigation (readout spec) is applied.

## 3. QHED attachment (scope note)

Gate-level **QHED** (Hadamard / neighbor-conditioned operations on the color qubit) is **separate** from the current **classical** FWHT + Sobel-style baseline in `src/qhed.py`. To prioritize **FRQI prep** diagrams and counts; edge extraction may remain **classical post-processing** on reconstructed images until a dedicated quantum QHED circuit is in scope.

## 4. Figures

Toy vs v-chain slice and block-level FRQI prep: **`thesis/design/figures/`** (`toy_naive_vs_vchain.svg|pdf`, `block_frqi_prep.svg|pdf`). Generate PDFs and SVGs via **`scripts/toy_and_block_level_diagrams.py`**.
