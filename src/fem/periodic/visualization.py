"""Lightweight Matplotlib visualization for periodic 2D mode fields."""

from __future__ import annotations

from typing import Any

import numpy as np

from .results import PeriodicMode, PeriodicModeSet


def visualize(
    mode: PeriodicMode,
    component: str = "Ey",
    quantity: str = "real",
    *,
    ax: Any = None,
    cmap: str = "RdBu_r",
    show_mesh: bool = False,
    colorbar: bool = True,
    show: bool = True,
    slice_axis: str | None = None,
    slice_fraction: float = 0.5,
    max_points: int = 5000,
) -> Any:
    """Create and show a Matplotlib figure for one selected mode."""

    if not isinstance(mode, PeriodicMode):
        raise TypeError("mode must be a PeriodicMode.")
    if not isinstance(show, (bool, np.bool_)):
        raise TypeError("show must be a boolean.")
    import matplotlib.pyplot as plt

    values = mode.fields.quantity(component, quantity)
    if mode.fields.dimension == 3:
        if isinstance(max_points, bool) or int(max_points) < 1:
            raise ValueError("max_points must be a positive integer.")
        points = np.asarray(mode.fields.coordinates)
        selected = np.ones(points.shape[0], dtype=bool)
        if slice_axis is not None:
            axis_name = str(slice_axis).strip().lower()
            if axis_name not in ("x", "y", "z"):
                raise ValueError("slice_axis must be None, 'x', 'y', or 'z'.")
            fraction = float(slice_fraction)
            if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError("slice_fraction must lie in [0, 1].")
            axis = {"x": 0, "y": 1, "z": 2}[axis_name]
            coordinate = points[:, axis]
            plane = coordinate.min() + fraction * np.ptp(coordinate)
            distance = np.abs(coordinate - plane)
            tolerance = max(0.02 * np.ptp(coordinate), float(distance.min()) * 1.001)
            selected = distance <= tolerance
        indices = np.flatnonzero(selected)
        if indices.size > int(max_points):
            indices = indices[:: int(np.ceil(indices.size / int(max_points)))]
        if ax is None:
            figure = plt.figure(figsize=(8.0, 6.5))
            ax = figure.add_subplot(111, projection="3d")
        elif not hasattr(ax, "get_zlim"):
            raise ValueError("A 3D Matplotlib axes is required for 3D modes.")
        artist = ax.scatter(
            points[indices, 0],
            points[indices, 1],
            points[indices, 2],
            c=np.asarray(values)[indices],
            cmap=cmap,
            s=14,
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.set_title(
            f"Mode {mode.index} {mode.polarization or ''}: {quantity}({component}), "
            f"neff={mode.neff:.6g}"
        )
        if colorbar:
            ax.figure.colorbar(artist, ax=ax, shrink=0.75)
        if show:
            plt.show()
        return ax.figure, ax

    import matplotlib.tri as mtri

    if ax is None:
        _, ax = plt.subplots()
    owners = mode.fields.sample_element_indices
    element_count = mode.fields.mesh_cells.shape[0]
    sums = np.bincount(owners, weights=values, minlength=element_count)
    counts = np.bincount(owners, minlength=element_count)
    cell_values = sums / np.maximum(counts, 1)
    points = mode.fields.mesh_points
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], mode.fields.mesh_cells)
    artist = ax.tripcolor(triangulation, facecolors=cell_values, shading="flat", cmap=cmap)
    if show_mesh:
        ax.triplot(triangulation, color="black", linewidth=0.25, alpha=0.45)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"Mode {mode.index} {mode.polarization or ''}: {quantity}({component}), "
        f"neff={mode.neff:.6g}"
    )
    if colorbar:
        ax.figure.colorbar(artist, ax=ax)
    if show:
        plt.show()
    return ax.figure, ax


__all__ = ["visualize"]
