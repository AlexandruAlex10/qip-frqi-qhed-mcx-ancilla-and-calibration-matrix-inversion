"""Generate toy and block-level FRQI prep diagrams.

Writes PDF and SVG under thesis/design/figures/ using matplotlib.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "thesis" / "design" / "figures"


def _save_both(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "svg"):
        out = FIG_DIR / f"{stem}.{fmt}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.08, format=fmt)
        print("Wrote", out)


def draw_toy_naive_vs_ancilla() -> None:
    """3 address + 1 color toy: naive m-controlled Ry vs ancilla+v-chain slice."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))

    for ax, title in zip(
        axes,
        (
            "Naive structural view (no ancilla)",
            "Improved slice (v-chain MCX + ancilla)",
        ),
    ):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis("off")
        ax.set_title(title, fontsize=12, pad=12)

    # --- Left: wires + one big box ---
    ax = axes[0]
    ys = [6.2, 5.0, 3.8, 2.6]  # q3,q2,q1,q0 color at bottom
    labels = [r"$q_2$" + "\n" + r"(addr)", r"$q_1$" + "\n" + r"(addr)", r"$q_0$" + "\n" + r"(addr)", r"$c$" + "\n" + r"(color)"]
    for y, lab in zip(ys, labels):
        ax.plot([0.6, 9.0], [y, y], color="black", linewidth=1.0)
        ax.text(0.1, y, lab, va="center", fontsize=10)

    box = FancyBboxPatch(
        (2.6, 2.35),
        5.6,
        4.5,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        linewidth=1.2,
        edgecolor="#1a5276",
        facecolor="#d4e6f1",
    )
    ax.add_patch(box)
    ax.text(
        5.4,
        4.6,
        r"$m$-controlled $R_y(\theta)$" + "\n" + r"($m{=}3$ here)",
        ha="center",
        va="center",
        fontsize=11,
        color="#1a5276",
    )
    # controls stubs into box
    for y in ys[:-1]:
        ax.plot([2.6, 2.6], [y, 4.6], color="#1a5276", linestyle="--", linewidth=0.9)
    ax.plot([2.6, 2.6], [ys[-1], 3.0], color="#1a5276", linestyle="--", linewidth=0.9)

    # --- Right: ancilla wire + blocks ---
    ax = axes[1]
    ys2 = [6.4, 5.2, 4.0, 2.8, 1.4]
    labels2 = [
        r"$q_2$",
        r"$q_1$",
        r"$q_0$",
        r"$c$",
        r"$a$" + "\n" + "(flag / ancilla workspace)",
    ]
    for y, lab in zip(ys2, labels2):
        ax.plot([0.5, 9.2], [y, y], color="black", linewidth=1.0)
        ax.text(0.05, y, lab, va="center", fontsize=10)

    def round_box(x0, y0, w, h, text, fc, ec):
        p = FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.1,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(p)
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=9.5)

    round_box(1.2, 0.95, 2.5, 6.6, "", "#ffffff", "#cccccc")  # spacer column visual
    round_box(1.4, 5.05, 2.15, 1.9, 'MCX\nv-chain', "#fdebd0", "#af601a")
    round_box(4.0, 1.95, 2.35, 1.35, r"ctrl-$R_y$", "#d5f5e3", "#1e8449")
    round_box(6.6, 5.05, 2.15, 1.9, r"MCX$^\dagger$", "#fdebd0", "#af601a")

    # vertical connectors (schematic)
    ax.annotate(
        "",
        xy=(5.15, 3.25),
        xytext=(2.45, 5.8),
        arrowprops=dict(arrowstyle="-", color="#566573", lw=1.0),
    )
    ax.annotate(
        "",
        xy=(6.55, 5.8),
        xytext=(5.15, 3.25),
        arrowprops=dict(arrowstyle="-", color="#566573", lw=1.0),
    )

    ax.text(
        5.0,
        0.55,
        "Uncompute returns ancilla; reuse across pixels.",
        ha="center",
        fontsize=9,
        style="italic",
        color="#566573",
    )

    fig.tight_layout()
    _save_both(fig, "toy_naive_vs_vchain")
    plt.close(fig)


def draw_block_full_frqi() -> None:
    """Block-level FRQI prep."""
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Block-level FRQI preparation (structural view)", fontsize=13, pad=14)

    # Registers
    rb = FancyBboxPatch(
        (0.6, 7.0),
        8.8,
        2.2,
        boxstyle="round,pad=0.03,rounding_size=0.2",
        linewidth=1.2,
        edgecolor="#1f618d",
        facecolor="#eaf2f8",
    )
    ax.add_patch(rb)
    ax.text(
        5.0,
        8.1,
        r"Registers: $m$ position qubits + $1$ color qubit  (basis $| \mathrm{addr}, c \rangle$)",
        ha="center",
        va="center",
        fontsize=11,
    )

    # Start
    sb = FancyBboxPatch(
        (0.8, 5.35),
        2.0,
        1.1,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.0,
        edgecolor="#566573",
        facecolor="#f8f9f9",
    )
    ax.add_patch(sb)
    ax.text(1.8, 5.9, r"$|0\rangle^{\otimes (m+1)}$", ha="center", va="center", fontsize=11)

    # Loop box
    lb = FancyBboxPatch(
        (3.3, 1.2),
        5.8,
        5.6,
        boxstyle="round,pad=0.04,rounding_size=0.2",
        linewidth=1.4,
        edgecolor="#117864",
        facecolor="#e9f7ef",
    )
    ax.add_patch(lb)
    ax.text(
        6.2,
        6.35,
        r"For each pixel $p = 0 \ldots N_{\mathrm{pix}}-1$",
        ha="center",
        fontsize=11.5,
        color="#0e6655",
    )

    steps = [
        (3.55, 4.55, 5.3, 1.05, "Address\ncompare", "#fdebd0", "#af601a"),
        (3.55, 3.25, 5.3, 1.05, r"ctrl-$R_y(\theta_p)$", "#d5f5e3", "#1e8449"),
        (3.55, 1.95, 5.3, 1.05, "Uncompute\ncompare", "#fdebd0", "#af601a"),
    ]
    for x, y, w, h, txt, fc, ec in steps:
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.05,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10)

    ax.text(
        6.2,
        1.45,
        r"v-chain uses $\leq m - 2$ ancilla qubits (reused each iteration)",
        ha="center",
        fontsize=9.5,
        color="#0e6655",
    )

    # Arrows
    ar = FancyArrowPatch(
        (2.8, 5.9),
        (3.25, 5.9),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.1,
        color="black",
    )
    ax.add_patch(ar)
    ar2 = FancyArrowPatch(
        (6.2, 1.05),
        (6.2, 0.55),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.1,
        color="black",
    )
    ax.add_patch(ar2)
    ax.text(6.45, 0.35, r"FRQI state $|\psi_{\mathrm{FRQI}}\rangle$", fontsize=11, ha="center")

    _save_both(fig, "block_frqi_prep")
    plt.close(fig)


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    draw_toy_naive_vs_ancilla()
    draw_block_full_frqi()


if __name__ == "__main__":
    main()
