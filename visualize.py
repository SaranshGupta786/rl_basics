"""Matplotlib helpers for GridWorld visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from gridworld import Cell, GridWorld


def _save_or_show(fig: plt.Figure, save_path: Path | None) -> None:
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {save_path}")
    else:
        plt.show()


def plot_grid(
    env: GridWorld,
    *,
    path: Iterable[tuple[int, int]] | None = None,
    values: dict[tuple[int, int], float] | None = None,
    title: str = "GridWorld",
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    base = np.zeros((env.height, env.width))

    for row in range(env.height):
        for col in range(env.width):
            cell = Cell(env.grid[row, col])
            if cell == Cell.WALL:
                base[row, col] = np.nan

    if values is not None:
        value_grid = np.full((env.height, env.width), np.nan)
        for (row, col), value in values.items():
            value_grid[row, col] = value
        im = ax.imshow(value_grid, cmap="YlGn", origin="upper")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Value")
    else:
        ax.imshow(base, cmap="Greys", vmin=0, vmax=1, origin="upper")

    for row in range(env.height):
        for col in range(env.width):
            cell = Cell(env.grid[row, col])
            if cell == Cell.WALL:
                ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=True, color="#333333"))
            elif cell == Cell.START:
                ax.text(col, row, "S", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
            elif cell == Cell.GOAL:
                ax.text(col, row, "G", ha="center", va="center", fontsize=14, fontweight="bold", color="white")

    if path:
        path = list(path)
        xs = [col for _, col in path]
        ys = [row for row, _ in path]
        ax.plot(xs, ys, color="#e74c3c", linewidth=2, marker="o", markersize=6)

    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_xticklabels(range(env.width))
    ax.set_yticklabels(range(env.height))
    ax.grid(True, color="white", linewidth=0.5)
    ax.set_title(title)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_learning_curve(
    rewards: list[float],
    *,
    window: int = 20,
    title: str = "Q-learning progress",
    save_path: Path | None = None,
) -> None:
    series = np.array(rewards, dtype=float)
    if len(series) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(series, kernel, mode="valid")
        x = np.arange(window - 1, len(series))
    else:
        smoothed = series
        x = np.arange(len(series))

    fig, _ = plt.subplots(figsize=(8, 4))
    plt.plot(series, alpha=0.25, label="Episode reward")
    plt.plot(x, smoothed, linewidth=2, label=f"{window}-episode average")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path)
