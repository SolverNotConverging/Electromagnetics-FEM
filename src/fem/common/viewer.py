"""Interactive Qt/PyVista result viewer shared by every solver family.

One window shows any result: choose the case, mode, field component and
quantity, and toggle the mesh and the material regions. 1D/2D fields are drawn
with Matplotlib; 3D periodic cells use PyVista, with arrows for the E and H
vector fields. Frequency sweeps add S-parameter spectra and scattering results
with an NF2FF contour add far-field patterns. The window only reads public
result data, so it also works for results loaded from HDF5.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

_WAVE_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "E", "H")
_QUANTITIES = ("real", "imag", "magnitude", "phase")
_STATIC_QUANTITIES = ("real", "magnitude")
_SPECTRUM_QUANTITIES = ("db", "magnitude", "phase", "real", "imag")
_SPECTRUM = "S-parameters"
_FARFIELD = ("Far field (polar)", "Far field (rectangular)")
_FARFIELD_QUANTITIES = {"directivity (dB)": ("directivity", True), "gain (dB)": ("gain", True),
                        "realized gain (dB)": ("realized_gain", True),
                        "power density (W/m/rad)": ("power_density", False)}
_MAX_ARROWS = 2500


def _relative_permittivity(description: Any) -> complex:
    """Return the leading relative permittivity of a recorded material description."""
    if not isinstance(description, dict):
        return complex("nan")
    for key in ("eps_r", "epsilon"):
        if key in description:
            value = np.asarray(description[key], dtype=complex).ravel()
            return complex(value[0]) if value.size else complex("nan")
    if "permittivity" in description:
        return _relative_permittivity(description["permittivity"])
    if "tensor" in description:
        return complex(np.asarray(description["tensor"], dtype=complex).ravel()[0])
    if "material" in description:
        return _relative_permittivity(description["material"])
    return complex("nan")


def cell_materials(mesh_data: Any) -> tuple[np.ndarray, list[str]] | None:
    """Return a region index per mesh cell and one label per region.

    Uses the cell tags, region names and configured materials that every solver
    records in ``mesh_data``. Returns None when the mesh carries no tags.
    """
    if mesh_data is None:
        return None
    metadata = mesh_data.metadata
    tags = metadata.get("element_tags")
    if tags is None or len(tags) != len(mesh_data.elements):
        return None
    tags = np.asarray(tags)
    names = dict(metadata.get("physical_names") or {})
    context = metadata.get("context") or {}
    background = context.get("background_material", context.get("background"))
    materials = {"background": background, "exterior": background}
    for key in ("regions", "materials"):
        for region in context.get(key) or ():
            if isinstance(region, dict) and "name" in region:
                materials[region["name"]] = region.get("material", region.get("permittivity", region))
    unique = sorted(set(tags.tolist()))
    labels = []
    for tag in unique:
        name = names.get(tag, f"region {tag}")
        epsilon = _relative_permittivity(materials.get(name))
        labels.append(name if np.isnan(epsilon.real) else f"{name} (εr={epsilon.real:.4g})")
    index = np.searchsorted(np.asarray(unique), tags)
    return index, labels


class _Source:
    """Uniform view of a result: named cases, each optionally with modes."""

    def __init__(self, result: Any):
        self.result = result
        name = type(result).__name__
        self.spectrum = name == "ScatteringSweepResult"
        if self.spectrum:
            self.cases = list(result.results)
        elif name == "PeriodicSweepResult":
            self.cases = list(result)
        else:
            self.cases = [result]
        first = self.cases[0]
        self.has_modes = hasattr(first, "mode") and hasattr(first, "__len__")
        self.family = type(first).__module__.rsplit(".", 1)[0]
        self.static = self.family == "fem.electrostatics"
        self.is_3d = self.family == "fem.periodic" and getattr(first, "dimension", None) == 3

    def case_labels(self) -> list[str]:
        labels = [_SPECTRUM] if self.spectrum else []
        frequencies = getattr(self.result, "frequencies", None)
        for index in range(len(self.cases)):
            if frequencies is not None:
                labels.append(f"Case {index}: {float(frequencies[index]) / 1e9:.6g} GHz")
            else:
                labels.append(f"Case {index}")
        return labels if self.spectrum or len(labels) > 1 else ["Result"]

    def components(self, case: Any) -> list[str]:
        if self.static:
            two_d = case.coordinates.shape[1] == 2
            return ["potential", "Ex", *(["Ey"] if two_d else []), "E",
                    "Dx", *(["Dy"] if two_d else []), "D"]
        fields = case.mode(0).fields if self.has_modes else getattr(case, "fields", None)
        if fields is not None and hasattr(fields, "components"):
            names = list(fields.components)
            return names + [n for n in ("E", "H") if n not in names]
        names = list(_WAVE_COMPONENTS)
        if getattr(case, "nf2ff", None) is not None:
            names += list(_FARFIELD)
        return names

    def mesh_data(self, case: Any) -> Any:
        mesh = getattr(case, "mesh_data", None)
        return mesh[0] if isinstance(mesh, tuple) else mesh

    def info(self, case: Any, mode: int) -> str:
        if self.static:
            return f"energy = {case.energy:.6g} J   (static field: E and D; no H)"
        if self.has_modes or hasattr(case, "neff"):
            selected = case.mode(mode) if self.has_modes else case
            neff = complex(selected.neff)
            return f"neff = {neff.real:.6g} {neff.imag:+.3g}j"
        parts = []
        s = getattr(case, "s_parameters", None)
        if hasattr(s, "items"):
            for key, value in s.items():
                value = complex(np.asarray(value).ravel()[0])
                name = f"S[{key[0]}, {key[1]}←{key[2]}]" if isinstance(key, tuple) and len(key) == 3 else f"S{key}"
                parts.append(f"{name} = {abs(value):.4g} ∠ {np.degrees(np.angle(value)):.1f}°")
        return "   ".join(parts[:8])


def _application():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    return app if app is not None else QApplication([])


def _display_points(source: _Source, points: np.ndarray) -> np.ndarray:
    """Plot coordinates in the orientation each family uses (scattering puts z horizontal)."""
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    return points[:, ::-1] if source.family == "fem.scattering" and points.shape[1] == 2 else points


def _draw_mesh(ax: Any, source: _Source, mesh: Any) -> None:
    points = _display_points(source, mesh.coordinates)
    cells = np.asarray(mesh.elements)
    if points.shape[1] == 1:
        ax.vlines(points[:, 0], 0, 1, transform=ax.get_xaxis_transform(), color="black",
                  linewidth=0.4, alpha=0.5)
    else:
        ax.triplot(points[:, 0], points[:, 1], cells[:, :3], color="black", linewidth=0.3, alpha=0.6)


def _material_figure(source: _Source, mesh: Any, regions: tuple[np.ndarray, list[str]], case: Any = None):
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    index, labels = regions
    colours = _region_colours(labels)
    figure = Figure()
    ax = figure.subplots()
    points = _display_points(source, mesh.coordinates)
    cells = np.asarray(mesh.elements)
    if points.shape[1] == 1:
        for cell, region in zip(cells, index):
            ax.axvspan(points[cell, 0].min(), points[cell, 0].max(), color=colours[region], linewidth=0)
        ax.set_yticks([])
    else:
        ax.tripcolor(points[:, 0], points[:, 1], cells[:, :3], facecolors=index.astype(float),
                     cmap=ListedColormap(colours), vmin=-0.5, vmax=len(labels) - 0.5)
        ax.set_aspect("equal", adjustable="box")
    axes = [f"{name} (m)" for name in mesh.axes]
    if source.family == "fem.scattering":
        axes = axes[::-1]
    ax.set_xlabel(axes[0])
    if len(axes) > 1:
        ax.set_ylabel(axes[1])
    handles = [Patch(color=colours[i], label=label) for i, label in enumerate(labels)]
    handles += _draw_scene_lines(ax, getattr(case, "scene", None))
    ax.legend(handles=handles, loc="upper right", fontsize="small", framealpha=0.85)
    ax.set_title("Material regions")
    return figure


_LINE_STYLES = {"pec": ("black", "-", 2.5), "pmc": ("tab:purple", "-", 2.0),
                "wave_port": ("tab:red", "--", 1.5), "pml": ("tab:green", ":", 1.5)}


def _draw_scene_lines(ax: Any, scene: Any) -> list[Any]:
    """Draw scattering PEC/PMC sheets, ports and PML interfaces (z horizontal)."""
    from matplotlib.lines import Line2D

    handles, seen = [], set()
    for line in getattr(scene, "lines", ()) or ():
        colour, style, width = _LINE_STYLES.get(line.kind, ("grey", "-", 1.0))
        x, z = np.asarray(line.endpoints).T
        ax.plot(z, x, color=colour, linestyle=style, linewidth=width)
        if line.kind not in seen:
            seen.add(line.kind)
            handles.append(Line2D([], [], color=colour, linestyle=style, linewidth=width,
                                  label=line.kind.replace("_", " ").upper() if line.kind in ("pec", "pmc", "pml")
                                  else "wave port"))
    return handles


class ResultViewer:
    """Window with case/mode/component/quantity selectors and mesh/material toggles."""

    def __init__(self, result: Any, *, title: str | None = None):
        from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QMainWindow,
                                       QSpinBox, QVBoxLayout, QWidget)

        self._app = _application()
        self._source = _Source(result)
        self._window = QMainWindow()
        self._window.setWindowTitle(title or f"FEM viewer — {type(result).__name__}")
        self._window.resize(1100, 760)
        self._canvas: Any = None
        self._plotter: Any = None

        central = QWidget()
        self._layout = QVBoxLayout(central)
        bar = QHBoxLayout()
        self._case = QComboBox()
        self._case.addItems(self._source.case_labels())
        self._mode = QSpinBox()
        self._component = QComboBox()
        self._quantity = QComboBox()
        self._mesh = QCheckBox("Mesh")
        self._material = QCheckBox("Material")
        self._info = QLabel()
        for label, widget in (("Case", self._case), ("Mode", self._mode),
                              ("Component", self._component), ("Quantity", self._quantity)):
            bar.addWidget(QLabel(label))
            bar.addWidget(widget)
        bar.addWidget(self._mesh)
        bar.addWidget(self._material)
        bar.addStretch(1)
        self._layout.addLayout(bar)
        self._body = QVBoxLayout()
        self._layout.addLayout(self._body, 1)
        self._layout.addWidget(self._info)
        self._window.setCentralWidget(central)
        self._status = self._window.statusBar()

        self._case.currentIndexChanged.connect(self._case_changed)
        self._mode.valueChanged.connect(self._render)
        self._quantity.currentIndexChanged.connect(self._render)
        self._component.currentIndexChanged.connect(self._component_changed)
        self._mesh.toggled.connect(self._render)
        self._material.toggled.connect(self._render)
        self._case_changed()

    # -- state -------------------------------------------------------------
    @property
    def _is_spectrum(self) -> bool:
        return self._source.spectrum and self._case.currentIndex() == 0

    def _current_case(self) -> Any:
        index = self._case.currentIndex() - (1 if self._source.spectrum else 0)
        return self._source.cases[max(index, 0)]

    def _quantities(self) -> list[str]:
        if self._is_spectrum:
            return list(_SPECTRUM_QUANTITIES)
        if self._component.currentText() in _FARFIELD:
            return list(_FARFIELD_QUANTITIES)
        return list(_STATIC_QUANTITIES if self._source.static else _QUANTITIES)

    def _set_quantities(self) -> None:
        want = self._quantities()
        if [self._quantity.itemText(i) for i in range(self._quantity.count())] != want:
            self._quantity.blockSignals(True)
            self._quantity.clear()
            self._quantity.addItems(want)
            self._quantity.blockSignals(False)

    def _case_changed(self) -> None:
        for widget in (self._mode, self._component):
            widget.blockSignals(True)
        try:
            self._component.clear()
            if self._is_spectrum:
                self._component.addItems(["S11", "S21"])
            else:
                case = self._current_case()
                self._component.addItems(self._source.components(case))
                self._mode.setRange(0, max(len(case) - 1, 0) if self._source.has_modes else 0)
                default = self._component.findText("potential" if self._source.static else "E")
                if default >= 0:
                    self._component.setCurrentIndex(default)
        finally:
            for widget in (self._mode, self._component):
                widget.blockSignals(False)
        self._component_changed()

    def _component_changed(self) -> None:
        self._set_quantities()
        far = self._component.currentText() in _FARFIELD
        field_view = not self._is_spectrum and not far
        self._mode.setEnabled(self._source.has_modes and field_view)
        self._mesh.setEnabled(field_view)
        self._material.setEnabled(field_view)
        self._render()

    # -- drawing -----------------------------------------------------------
    def _clear_body(self) -> None:
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
        while self._body.count():
            item = self._body.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._canvas = None

    def _render(self) -> None:
        component = self._component.currentText()
        quantity = self._quantity.currentText()
        mode = self._mode.value()
        self._clear_body()
        message = ""
        try:
            if self._is_spectrum:
                widget = self._matplotlib(self._source.result.plot(component=component, quantity=quantity))
                self._info.setText("")
            else:
                case = self._current_case()
                if component in _FARFIELD:
                    widget = self._matplotlib(self._farfield_figure(case, component, quantity))
                elif self._source.is_3d:
                    widget, message = self._pyvista(case, case.mode(mode), component, quantity)
                else:
                    widget, message = self._field_2d(case, mode, component, quantity)
                self._info.setText(self._source.info(case, mode))
            if message:
                self._status.showMessage(message)
            else:
                self._status.clearMessage()
        except Exception as exc:  # keep the window usable when a combination is invalid
            from PySide6.QtWidgets import QLabel
            widget = QLabel(f"Cannot display this selection: {exc}")
            self._status.showMessage(str(exc))
        self._body.addWidget(widget)

    def _field_figure(self, case: Any, mode: int, component: str, quantity: str):
        family = self._source.family
        if family == "fem.waveguide":
            from fem.waveguide.visualization import visualize

            figure, _ = visualize(case, mode=mode, component=component, quantity=quantity,
                                  material=False, show=False)
            return figure
        if family == "fem.periodic":
            from matplotlib.figure import Figure

            from fem.periodic.visualization import visualize

            figure = Figure()
            visualize(case.mode(mode), component=component, quantity=quantity,
                      ax=figure.subplots(), show=False)
            return figure
        return case.plot(component=component, quantity=quantity)

    def _field_2d(self, case: Any, mode: int, component: str, quantity: str):
        mesh = self._source.mesh_data(case)
        message = ""
        figure = None
        if self._material.isChecked():
            regions = cell_materials(mesh)
            if regions is None:
                message = "This result records no material regions."
            else:
                figure = _material_figure(self._source, mesh, regions, case)
        if figure is None:
            figure = self._field_figure(case, mode, component, quantity)
        if self._mesh.isChecked():
            if mesh is None:
                message = "This result records no mesh."
            else:
                _draw_mesh(figure.axes[0], self._source, mesh)
        return self._matplotlib(figure), message

    @staticmethod
    def _farfield_figure(case: Any, kind: str, quantity: str):
        from matplotlib.figure import Figure

        name, in_db = _FARFIELD_QUANTITIES[quantity]
        theta = (np.arange(720) + 0.5) * 2 * np.pi / 720
        values = np.asarray(getattr(case.far_field(theta), name), dtype=float)
        if in_db:
            values = 10 * np.log10(np.maximum(values, np.finfo(float).tiny))
        figure = Figure()
        if kind == _FARFIELD[0]:
            axis = figure.add_subplot(projection="polar")
            axis.plot(theta, values)
            axis.set_theta_zero_location("E")
        else:
            axis = figure.subplots()
            axis.plot(np.degrees(theta), values)
            axis.set_xlabel("theta (deg)")
            axis.grid(True)
        axis.set_title(quantity)
        return figure

    def _matplotlib(self, figure: Any):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from PySide6.QtWidgets import QVBoxLayout, QWidget

        canvas = FigureCanvasQTAgg(figure)
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(NavigationToolbar2QT(canvas, holder))
        layout.addWidget(canvas)
        self._canvas = canvas
        return holder

    def _pyvista(self, case: Any, mode: Any, component: str, quantity: str):
        from pyvistaqt import QtInteractor

        plotter = QtInteractor(self._window)
        message = populate_3d(plotter, mode, self._source.mesh_data(case), component, quantity,
                              mesh_overlay=self._mesh.isChecked(), material=self._material.isChecked())
        self._plotter = plotter
        return plotter.interactor, message

    # -- public ------------------------------------------------------------
    def show(self, *, block: bool = True) -> ResultViewer:
        self._window.show()
        if block:
            self._app.exec()
        return self


def populate_3d(plotter: Any, mode: Any, mesh: Any, component: str, quantity: str, *,
                mesh_overlay: bool = False, material: bool = False) -> str:
    """Draw one 3D periodic mode into a PyVista plotter; return a status message.

    ``E`` and ``H`` are drawn as arrows; Cartesian components as coloured points.
    The mesh is a wireframe and material regions are translucent coloured cells.
    """
    import pyvista as pv

    fields = mode.fields
    points = np.asarray(fields.coordinates, dtype=float)
    if component in ("E", "H"):
        add_arrows(plotter, points, [fields.component(f"{component}{axis}") for axis in "xyz"],
                   quantity, label=f"|{component}|")
    else:
        cloud = pv.PolyData(points)
        label = f"{quantity}({component})"
        cloud[label] = np.asarray(fields.quantity(component, quantity))
        plotter.add_mesh(cloud, scalars=label, cmap="RdBu_r" if quantity in ("real", "imag") else "viridis",
                         render_points_as_spheres=True, point_size=6)
    message = ""
    if mesh is not None and (mesh_overlay or material):
        grid = _tetra_grid(mesh)
        if material:
            regions = cell_materials(mesh)
            if regions is None:
                message = "This result records no material regions."
            else:
                index, labels = regions
                colours = _region_colours(labels)
                legend = []
                labels = [label.replace("εr", "eps_r") for label in labels]  # VTK fonts lack Greek
                for region, (label, colour) in enumerate(zip(labels, colours)):
                    if label.split(" (")[0] in ("background", "exterior"):
                        legend.append([label, "#8c8c8c"])
                        continue  # leave the background open so the field stays visible
                    legend.append([label, _hex(colour)])
                    cells = grid.extract_cells(np.flatnonzero(index == region))
                    plotter.add_mesh(cells, color=_hex(colour), opacity=0.35, show_edges=False)
                plotter.add_mesh(grid.extract_surface(), style="wireframe", color="grey", opacity=0.3)
                plotter.add_legend(legend, bcolor="white", face="rectangle", loc="upper left",
                                   size=(0.45, 0.05 * len(legend) + 0.02))
        if mesh_overlay:
            plotter.add_mesh(grid, style="wireframe", color="black", opacity=0.15)
    elif mesh is None and (mesh_overlay or material):
        message = "This result records no mesh."
    plotter.add_axes()
    return message


def _hex(colour: Any) -> str:
    from matplotlib.colors import to_hex

    return to_hex(colour)


def _region_colours(labels: list[str]) -> list[Any]:
    """Light grey for the background, distinct qualitative colours for other regions."""
    from matplotlib import colormaps

    palette = colormaps["tab10"].colors
    colours, others = [], 0
    for label in labels:
        if label.split(" (")[0] in ("background", "exterior"):
            colours.append((0.88, 0.88, 0.88))
        else:
            colours.append(palette[others % len(palette)])
            others += 1
    return colours


def _tetra_grid(mesh: Any):
    import pyvista as pv

    cells = np.asarray(mesh.elements)[:, :4]
    connectivity = np.hstack([np.full((len(cells), 1), 4), cells]).ravel()
    return pv.UnstructuredGrid(connectivity, np.full(len(cells), pv.CellType.TETRA),
                               np.asarray(mesh.coordinates, dtype=float))


def add_arrows(plotter: Any, points: np.ndarray, components: list[np.ndarray], quantity: str,
               *, label: str, max_arrows: int = _MAX_ARROWS) -> Any:
    """Draw a complex 3D vector field as arrows coloured by its magnitude; return the arrows.

    ``real``/``imag`` draw that part of the phasor; ``magnitude``/``phase`` draw
    the real part (the field at ``omega*t = 0``) coloured by the magnitude.
    """
    import pyvista as pv

    vectors = np.stack([np.asarray(c) for c in components], axis=1)
    shown = np.imag(vectors) if quantity == "imag" else np.real(vectors)
    magnitude = np.linalg.norm(np.abs(vectors), axis=1)
    step = max(1, int(np.ceil(len(points) / max_arrows)))
    cloud = pv.PolyData(np.asarray(points, dtype=float)[::step])
    cloud["vectors"] = shown[::step]
    cloud[label] = magnitude[::step]
    span = float(np.max(np.ptp(points, axis=0))) or 1.0
    largest = float(np.max(np.linalg.norm(shown, axis=1))) or 1.0
    arrows = cloud.glyph(orient="vectors", scale="vectors", factor=0.08 * span / largest)
    plotter.add_mesh(arrows, scalars=label, cmap="viridis")
    return arrows


def show(result: Any, *, block: bool = True) -> ResultViewer:
    """Open the interactive viewer for a result object or HDF5 archive path."""
    if isinstance(result, (str, os.PathLike)):
        from .persistence import load

        result = load(result)
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        block = False
    return ResultViewer(result).show(block=block)


__all__ = ["ResultViewer", "add_arrows", "cell_materials", "populate_3d", "show"]
