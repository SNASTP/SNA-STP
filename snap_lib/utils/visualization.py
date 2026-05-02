"""
Visualization Utilities
=======================

Plotting functions for SNA-STP analysis results.
"""

import numpy as np
from typing import Dict, List, Optional


def plot_structure_matrix(
    M: np.ndarray,
    title: str = "Structure Matrix",
    var_names: Optional[List[str]] = None,
    figsize: tuple = (10, 4),
    save_path: Optional[str] = None,
):
    """
    Visualize a structure matrix as a heatmap.

    Args:
        M: Structure matrix.
        title: Plot title.
        var_names: Variable names for x-axis labeling.
        figsize: Figure size.
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(M, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)

    n_cols = M.shape[1]
    n_inputs = int(np.log2(n_cols))

    # Label rows
    n_rows = M.shape[0]
    n_outputs = int(np.log2(n_rows))
    if n_outputs <= 4:
        row_labels = []
        for j in range(n_rows):
            bits = []
            for i in range(n_outputs):
                bits.append(str(1 - ((j >> (n_outputs - 1 - i)) & 1)))
            row_labels.append("".join(bits))
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels)

    # Label columns (if not too many)
    if n_inputs <= 6:
        col_labels = []
        for j in range(n_cols):
            bits = []
            for i in range(n_inputs):
                bits.append(str(1 - ((j >> (n_inputs - 1 - i)) & 1)))
            col_labels.append("".join(bits))
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)

    ax.set_title(title)
    ax.set_xlabel("Input combinations")
    ax.set_ylabel("Output")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shapley_values(
    shapley: Dict[str, float],
    title: str = "Shapley Values (from SNA-STP)",
    figsize: tuple = (8, 5),
    save_path: Optional[str] = None,
):
    """
    Bar chart of Shapley values.

    Args:
        shapley: Dict mapping feature names to Shapley values.
        title: Plot title.
        figsize: Figure size.
        save_path: Save path.
    """
    import matplotlib.pyplot as plt

    names = list(shapley.keys())
    values = list(shapley.values())

    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(values))[::-1]
    names = [names[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]

    colors = ["#2196F3" if v >= 0 else "#F44336" for v in values]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Shapley Value")
    ax.set_title(title)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(
    importance: List[Dict],
    title: str = "Feature Importance (SNA-STP)",
    figsize: tuple = (8, 5),
    save_path: Optional[str] = None,
):
    """
    Bar chart of feature importance scores.

    Args:
        importance: List of dicts from LogicExtractor.extract_important_features().
        title: Plot title.
        figsize: Figure size.
        save_path: Save path.
    """
    import matplotlib.pyplot as plt

    names = [f["feature"] for f in importance]
    scores = [f["influence"] for f in importance]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(names)), scores, color="#4CAF50", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Influence Score")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_trajectory(
    trajectory: List[Dict],
    title: str = "Input Trajectory Through Network",
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
):
    """
    Visualize the state trajectory of an input through the network.

    Args:
        trajectory: From SNAPAnalyzer.trace_trajectory().
        title: Plot title.
        figsize: Figure size.
        save_path: Save path.
    """
    import matplotlib.pyplot as plt

    layers = [t["layer"] for t in trajectory]
    names = [t["name"] for t in trajectory]
    indices = [t["state_index"] for t in trajectory]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(layers, indices, "o-", color="#673AB7", markersize=10, linewidth=2)

    for i, (l, idx, name) in enumerate(zip(layers, indices, names)):
        state_str = str(trajectory[i].get("state", ""))
        ax.annotate(
            f"{name}\nδ^{idx}",
            (l, idx),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("State Index")
    ax.set_title(title)
    ax.set_xticks(layers)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
