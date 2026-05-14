# Readout mitigation design (calibration matrix inversion)

**Simple readout correction** via an estimated **confusion (assignment) matrix** and its **inverse** applied to measured outcome probabilities.

## 1. Which bits are measured

**Primary (thesis default):** Measure **only the color qubit** after FRQI preparation (and after any optional **swap** of color into a dedicated readout line if required by hardware).

- **Dimension:** outcomes \(\{0,1\}\) → confusion matrix **\(M \in \mathbb{R}^{2\times 2}\)** for that qubit.
- **Rationale:** FRQI reconstruction from shots uses **\(P(1)\)** vs **\(P(0)\)** per prepared pixel slice (or aggregated experiments); position qubits may be **implicit** in how circuits are run (e.g. one circuit per address for calibration) or traced out depending on the implementation choice.

**Debugging / diagnostics (optional):** Measure the **full** \((m+1)\)-qubit register (or a subset including ancillas) for **process debugging** only. Then **\(M\)** has dimension **\(2^k \times 2^k\)** for **\(k\)** measured qubits. This explodes quickly; use only on simulators or tiny \(m\).

**Ancilla measurement:** Workspace ancillas for v-chain MCX should be **uncomputed** before measurement in the ideal circuit; if any ancilla remains entangled, include them in **\(k\)** or **reset** them before measurement (document the chosen invariant).

## 2. Workflow

1. **Choose \(k\)** measured qubits (default \(k=1\): color only).
2. **Calibration circuits:** For each computational basis state **\(|b\rangle\)** on those \(k\) qubits, \(b = 0,\ldots,2^k-1\), prepare **\(|b\rangle\)** (or the full register with ancillas in a known reset state), **measure** \(k\) bits, and accumulate counts into a **\(2^k \times 2^k\)** empirical matrix **\(\hat{M}\)** where \(\hat{M}_{o,b}\) is the fraction of outcomes **\(o\)** when truth was **\(b\)** (column stochastic).
3. **Regularized inversion:** Form **\(W \approx \hat{M}^{-1}\)** (or **Moore–Penrose pseudoinverse** if **\(\hat{M}\)** is singular). Practical recipes: small **Tikhonov** ridge on the diagonal, or **maximum-likelihood** inversion (e.g. **ignis**-style) if using Qiskit libraries.
4. **Apply to data:** For each FRQI-related experiment, collect a **length-\(2^k\)** histogram **\(\vec{f}\)** of raw counts (normalized to probabilities **\(\tilde{p}\)**). Report **\(\hat{p} = W \tilde{p}\)** then **renormalize** \(\hat{p}\) to sum to 1 and clip negatives to 0 if needed (document clipping as a thesis limitation).
5. **Shot budget:** Allocate shots across calibration **\(2^k\)** states and FRQI runs; record **\(N_{\text{cal}}\)** and **\(N_{\text{data}}\)** in Methods.

## 3. Software path (next steps)

**Prototype:**

- **Qiskit:** Build calibration circuits with **`QuantumCircuit`** + **`measure`**, run on **Aer** / hardware, assemble counts with **`Sampler`** or **`backend.run`**.  
- Use **`qiskit.result.mitigation`** / **`CompleteMeasFitter`** (package split may vary by Qiskit 1.x; check installed **`qiskit-experiments`** or **`mthree`** if available) **or** manual **NumPy** inversion of **\(\hat{M}\)** for \(k=1\) or \(k=2\) where **\(2^k\)** is tiny.

**Minimal path (always available):** **NumPy** `np.linalg.inv` on **\(\hat{M} + \epsilon I\)** for small **\(k\)**, with explicit unit tests against known injected readout bias in simulation.

## 4. Bibliographic pointers

- IBM Qiskit documentation: **readout error mitigation**, **assignment matrices**, and **`MCXGate`** modes (for consistency with the gate-level FRQI prep note).
- Qiskit textbook / Qiskit Experiments: **measurement error mitigation** chapter (version pinned in `requirements.txt`).
