"""Plots for invariant research.

Only two plot families live here:
    1. one-method invariance maps;
    2. pairwise comparison maps.

Tracking is never performed in this module.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np


def get_pyplot(show=False):
    show = bool(show)
    if show and threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Interactive plotting must run on the main thread.")

    import matplotlib
    if not show:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt
    if not show and str(matplotlib.get_backend()).lower() != "agg":
        plt.switch_backend("Agg")
    return plt


def _scatter_map(ax, x, y, values, *, cmap, invalid=None, symmetric=False):
    finite = np.isfinite(values)
    if np.any(finite):
        v = np.asarray(values[finite], dtype=float)
        kwargs = {}
        if symmetric:
            vmax = float(np.max(np.abs(v)))
            if not np.isfinite(vmax) or vmax == 0.0:
                vmax = 1.0
            kwargs.update(vmin=-vmax, vmax=vmax)
        scatter = ax.scatter(
            np.asarray(x)[finite],
            np.asarray(y)[finite],
            c=v,
            cmap=cmap,
            marker="s",
            s=34,
            linewidths=0,
            **kwargs,
        )
    else:
        scatter = None

    if invalid is not None:
        invalid = np.asarray(invalid, dtype=bool)
        if np.any(invalid):
            ax.scatter(
                np.asarray(x)[invalid],
                np.asarray(y)[invalid],
                marker="s",
                s=20,
                color="0.88",
                linewidths=0,
                zorder=0,
            )
    return scatter


def plot_invariance_map(
    tracking,
    metrics,
    method,
    output_path,
    *,
    show=False,
    vmin=-14.0,
    vmax=0.0,
    delta=0.0,
):
    """Plot log10 invariant drift for one method."""
    plt = get_pyplot(show)
    x = np.asarray(tracking["x_mm"])
    y = np.asarray(tracking["y_mm"])
    values = np.asarray(metrics["log10D"], dtype=float)
    valid = np.asarray(metrics["valid"], dtype=bool)

    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    shown = np.where(valid, values, np.nan)
    finite = np.isfinite(shown)
    if np.any(finite):
        scatter = ax.scatter(
            x[finite],
            y[finite],
            c=shown[finite],
            cmap="viridis",
            vmin=float(vmin),
            vmax=float(vmax),
            marker="s",
            s=34,
            linewidths=0,
        )
    else:
        scatter = None

    if np.any(~valid):
        ax.scatter(
            x[~valid],
            y[~valid],
            marker="s",
            s=20,
            color="0.88",
            linewidths=0,
            zorder=0,
        )
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(r"$\log_{10} D_{I_x}$")

    ax.set_xlabel(r"$x_0$ [mm]")
    ax.set_ylabel(r"$y_0$ [mm]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{method}: horizontal invariant tracking | "
        + rf"$\delta={float(delta):g}$"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def plot_comparison_map(
    tracking,
    metrics1,
    metrics2,
    name1,
    name2,
    output_path,
    *,
    show=False,
    delta=0.0,
):
    """Plot log10(D2)-log10(D1): red=name1 better, blue=name2 better."""
    plt = get_pyplot(show)
    x = np.asarray(tracking["x_mm"])
    y = np.asarray(tracking["y_mm"])
    valid1 = np.asarray(metrics1["valid"], dtype=bool)
    valid2 = np.asarray(metrics2["valid"], dtype=bool)
    both = valid1 & valid2

    score = np.full(len(x), np.nan)
    score[both] = (
        np.asarray(metrics2["log10D"], dtype=float)[both]
        - np.asarray(metrics1["log10D"], dtype=float)[both]
    )

    fig, ax = plt.subplots(figsize=(9.0, 6.8))
    scatter = _scatter_map(
        ax,
        x,
        y,
        score,
        cmap="bwr",
        invalid=~both,
        symmetric=True,
    )
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(
            f"log10(D_{name2}) - log10(D_{name1})\n"
            f"red = {name1} better, blue = {name2} better"
        )

    ax.set_xlabel(r"$x_0$ [mm]")
    ax.set_ylabel(r"$y_0$ [mm]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{name1} vs {name2} | "
        + rf"$\delta={float(delta):g}$"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    if show:
        plt.show()
    plt.close(fig)

    return {
        "path": output_path,
        "score": score,
        "valid": both,
        "name1_better": int(np.count_nonzero(score[both] > 0.0)),
        "name2_better": int(np.count_nonzero(score[both] < 0.0)),
        "ties": int(np.count_nonzero(score[both] == 0.0)),
    }
