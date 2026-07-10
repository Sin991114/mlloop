"""Minimal matplotlib helpers producing small SVG charts for reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ACCENT = "#4C6EF5"
WARN = "#E8590C"
MUTED = "#94A3B8"


def new_fig(width: float = 6.4, height: float = 3.4):
    return plt.subplots(figsize=(width, height), dpi=100)


def save_svg(fig, path: Path | str) -> str:
    """Save a figure as SVG and return its file name (relative to its directory)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path.name
