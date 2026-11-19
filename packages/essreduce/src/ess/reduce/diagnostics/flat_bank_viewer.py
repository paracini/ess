# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Viewer for high-dimensional data."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from html import escape
from typing import Any

import ipywidgets as ipw
import plopp as pp
import scipp as sc
from plopp.core.utils import scalar_to_string
from plopp.widgets.slicing import CombinedSlicer


class FlatBankViewer(ipw.VBox):
    """Image plot of data with controls for flattening and slicing its dims."""

    def __init__(
        self,
        data: Mapping[str, sc.DataArray] | sc.DataArray,
        dims: Iterable[str] | None = None,
        *,
        sliced: Iterable[str] = (),
        horizontal: Iterable[str] = (),
        vertical: Iterable[str] = (),
        cbar: bool = True,
        rasterized: bool = True,
        **fig_kwargs: Any,
    ) -> None:
        """Create a new view.

        Parameters
        ----------
        data:
            Data to show. A mapping (e.g., a data group) with one entry per bank
            adds buttons for selecting a bank. The banks may have different dims
            but must all have the same unit and no empty dimensions.
        dims:
            Order of the slicing controls. With multiple banks, dims that are not
            in a bank are ignored for that bank.
        sliced:
            Dims that start in slice mode.
        horizontal:
            Dims that start flattened onto the horizontal axis.
            Order is outermost to innermost (last dim varies fastest).
        vertical:
            Dims that start flattened onto the vertical axis.
            Order is outermost to innermost (last dim varies fastest).
        cbar:
            Whether to show a colorbar.
        rasterized:
            If ``True``, the figure is rasterized which improves rendering
            speed but reduces resolution.
        **fig_kwargs:
            Additional arguments passed to :func:`plopp.imagefigure`.

        Dims that are not in a bank are ignored for that bank.
        Dims on neither axis are sliced. If neither axis is specified, dims not
        listed in ``sliced`` start on the vertical axis, except for the last dim,
        which starts on the horizontal axis. Drag dimensions to reorder them or
        move them between axes. Remove a dimension from an axis to slice it;
        use the slicer's X/Y buttons to move it back to an axis.
        The direction controls reverse the displayed axes without changing the data.
        Scipp may reject layouts with more than six unsliced dimensions. Failed
        layouts show an error instead of leaving the previous plot visible.
        """
        # Validate every bank before constructing any controls or figures, including
        # banks that will only be displayed after the user changes the selection.
        self._banks = _validate_banks(
            {"": data} if isinstance(data, sc.DataArray) else data
        )
        self._dims = None if dims is None else tuple(dims)
        self._layout = _validate_layout(
            self._banks,
            sliced=tuple(sliced),
            horizontal=tuple(horizontal),
            vertical=tuple(vertical),
        )
        self._fig_kwargs = {"cbar": cbar, "rasterized": rasterized} | fig_kwargs
        self._figure = None
        self._figure_node = pp.Node(lambda: self._plot_data)
        self._message = ipw.HTML(
            "Add at least one dimension to each axis to show a plot.",
            layout={"display": "none"},
        )
        self._figure_box = ipw.Box([self._message])
        self._configs_box = ipw.VBox()
        self._plot_data = None
        self._unwrap_indices = None

        if isinstance(data, sc.DataArray):
            selector = None
        else:
            selector = ipw.ToggleButtons(options=list(self._banks))
            selector.observe(lambda change: self._select(change["new"]), names="value")
        # Graphs and controls are created on first selection of a bank and kept
        # afterwards, so each bank keeps its settings.
        self._views: dict[str, _BankView] = {}
        self._current = next(iter(self._banks))
        self._select(self._current)

        children = [self._figure_box, self._configs_box]
        if selector is not None:
            children.insert(0, ipw.HBox([ipw.Label("Bank:"), selector]))
        super().__init__(children)

    @property
    def configs(self) -> "_DimensionConfigs":
        """Dimension controls of the current bank."""
        return self._views[self._current].configs

    @property
    def node(self) -> pp.Node:
        """Node that produces the flattened and sliced data of the current bank."""
        return self._views[self._current].node

    def _select(self, bank: str) -> None:
        self._current = bank
        if bank not in self._views:
            da = self._banks[bank]
            dims = self._dims
            if dims is not None:
                dims = [dim for dim in dims if dim in da.dims]
            self._views[bank] = _BankView(
                da,
                dims,
                _initial_layout(da.dims, **self._layout),
                partial(self._on_bank_data, bank),
                partial(self._on_bank_error, bank),
            )
            configs = self._views[bank].configs
            for control in (configs.invert_horizontal, configs.invert_vertical):
                control.observe(
                    partial(self._update_axis_directions, bank), names="value"
                )
        view = self._views[bank]
        self._configs_box.children = [view.configs]
        # The sink does not cache (`_on_bank_data` returns None), so this calls
        # `_show` with the cached data of `view.node`, or computes it for new banks.
        view.sink()

    def _on_bank_data(self, bank: str, data: sc.DataArray | None) -> None:
        # Only the current bank's controls are visible, but be defensive anyway.
        if bank == self._current:
            try:
                self._show(data)
            except Exception as error:
                self._on_bank_error(bank, error)
                raise

    def _on_bank_error(self, bank: str, error: Exception) -> None:
        if bank == self._current:
            self._hide_figure(f"Cannot display this layout: {error}")

    def _hide_figure(self, message: str) -> None:
        self._message.value = escape(message)
        self._message.layout.display = None
        self._plot_data = None
        if self._figure is not None:
            self._figure.layout.display = "none"

    def _show(self, data: sc.DataArray | None) -> None:
        if data is None:
            self._hide_figure("Add at least one dimension to each axis to show a plot.")
            return
        self._message.layout.display = "none"

        vertical_dim, horizontal_dim = data.dims
        data = _to_plot_data(data)
        self._plot_data = data
        # Same order as used by `flatten`.
        sizes = self._views[self._current].sizes
        layout = self.configs.axis_layout
        vertical, horizontal = layout.vertical, layout.horizontal
        self._unwrap_indices = _unwrap_flat_indices_2d(
            {dim: sizes[dim] for dim in horizontal},
            {dim: sizes[dim] for dim in vertical},
        )

        # Replacing the figure widget makes the frontend re-create the canvas, which
        # flickers and shifts the layout. So use a single figure and modify that.
        if self._figure is None:
            self._figure = pp.imagefigure(self._figure_node, **self._fig_kwargs)
            self._figure_box.children = [self._message, self._figure]
            # Must come after the figure's first update, which sets Plopp's formatter.
            self._figure.ax.format_coord = self._format_coord
        else:
            self._figure.layout.display = None
            artist = self._figure.artists[self._figure_node.id]
            if artist.data.shape != data.shape:
                # Plopp's image artists cannot change their shape; replace the artist.
                self._figure.artists.pop(self._figure_node.id).remove()
                # Plopp does not unregister this when removing an artist.
                self._figure.canvas._coord_formatters.remove(artist.format_coord)

        self._figure.canvas.xlabel = horizontal_dim
        self._figure.canvas.ylabel = vertical_dim
        # Update through the graph so later renders also use the current data.
        # This re-creates the artist if it was removed above and redraws the canvas.
        self._figure_node.notify_children("data changed")
        self._update_axis_directions(self._current)

    def _update_axis_directions(self, bank: str, *_args: Any) -> None:
        if bank != self._current or self._figure is None:
            return
        changed = False
        for axis, control in (
            (self._figure.ax.xaxis, self.configs.invert_horizontal),
            (self._figure.ax.yaxis, self.configs.invert_vertical),
        ):
            if axis.get_inverted() != control.value:
                axis.set_inverted(control.value)
                changed = True
        if changed:
            self._figure.canvas.draw()

    def _format_coord(self, x: float, y: float) -> str:
        """Show the indices of all flattened dims and the value under the cursor."""
        if self._plot_data is None:
            return ""
        # Use round because axis coords are in the middle of bins.
        i, j = round(x), round(y)
        n_vertical, n_horizontal = self._plot_data.shape
        if not (0 <= i < n_horizontal and 0 <= j < n_vertical):
            return ""
        indices = ", ".join(
            f"{dim.capitalize()}: {index}"
            for dim, index in self._unwrap_indices(i, j).items()
        )
        data = self._plot_data.data["vertical", j]["horizontal", i]
        value = scalar_to_string(data)
        return f"{{{indices}}}: {value}"


class _BankView:
    """Controls and graph for a single data array."""

    def __init__(
        self,
        da: sc.DataArray,
        dims: Iterable[str] | None,
        layout: "_DimensionLayout",
        on_data: Callable[[sc.DataArray | None], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self.sizes = dict(da.sizes)
        self.configs = _DimensionConfigs(da, dims, layout)

        def compute(state: "_DimensionState") -> sc.DataArray | None:
            try:
                return _prepare_plot(da, state)
            except Exception as error:
                # Notebook callback tracebacks can be hidden. Never leave the
                # previous bank/layout visible when the computation failed.
                on_error(error)
                raise

        self.node = pp.Node(compute, self.configs.state_node)
        # Leaf node: Plopp calls it with new data whenever anything upstream changes.
        self.sink = pp.Node(on_data, self.node)


def _validate_banks(data: Mapping[str, Any]) -> dict[str, sc.DataArray]:
    if not data:
        raise ValueError("Need at least one data array.")
    for key, value in data.items():
        if not isinstance(value, sc.DataArray):
            raise TypeError(
                f"Expected all entries to be data arrays, got {type(value).__name__} "
                f"for '{key}'."
            )
        if empty := [dim for dim, size in value.sizes.items() if size == 0]:
            name = f"bank {key!r}" if key else "data array"
            raise ValueError(f"Empty dimensions {empty} in {name}.")
    return dict(data)


def _validate_layout(
    banks: Mapping[str, sc.DataArray],
    *,
    sliced: tuple[str, ...],
    horizontal: tuple[str, ...],
    vertical: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    listed = [*sliced, *horizontal, *vertical]
    if duplicates := sorted(dim for dim, count in Counter(listed).items() if count > 1):
        raise ValueError(
            f"Dims {duplicates} are listed more than once in "
            "`sliced`, `horizontal`, and `vertical`."
        )
    all_dims = {dim for da in banks.values() for dim in da.dims}
    if unknown := [dim for dim in listed if dim not in all_dims]:
        raise ValueError(f"Dims {unknown} are not in the data.")
    return {"sliced": sliced, "horizontal": horizontal, "vertical": vertical}


@dataclass(frozen=True, slots=True)
class _DimensionLayout:
    sliced: tuple[str, ...]
    vertical: tuple[str, ...]
    horizontal: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _DimensionState:
    layout: _DimensionLayout
    slices: dict[str, tuple[int, int]]


def _initial_layout(
    dims: Iterable[str],
    *,
    sliced: tuple[str, ...],
    horizontal: tuple[str, ...],
    vertical: tuple[str, ...],
) -> _DimensionLayout:
    dims = tuple(dims)
    if not horizontal and not vertical:
        flattened = tuple(dim for dim in dims if dim not in sliced)
        vertical, horizontal = flattened[:-1], flattened[-1:]
    else:
        horizontal = tuple(dim for dim in horizontal if dim in dims)
        vertical = tuple(dim for dim in vertical if dim in dims)
    return _DimensionLayout(
        sliced=tuple(dim for dim in dims if dim not in (*horizontal, *vertical)),
        vertical=vertical,
        horizontal=horizontal,
    )


def _to_plot_data(da: sc.DataArray) -> sc.DataArray:
    """Rename the dims to fixed names and use index coords.

    This way, a single figure can show any layout of flattened dims.
    Also, this returns a new data array which protects the input from
    modifications in `fig.update`.
    """
    vertical_dim, horizontal_dim = da.dims
    renamed = da.rename_dims({vertical_dim: "vertical", horizontal_dim: "horizontal"})
    return renamed.drop_coords(list(renamed.coords)).assign_coords(
        {dim: sc.arange(dim, size, unit=None) for dim, size in renamed.sizes.items()}
    )


class _DimensionConfigs(ipw.VBox):
    """Ordered axis dimensions and persistent sliders for unassigned dimensions."""

    def __init__(
        self,
        da: sc.DataArray,
        dims: Iterable[str] | None = None,
        layout: _DimensionLayout | None = None,
    ) -> None:
        # ``dims`` orders the sliders; it must not hide dimensions from the controls.
        self._dims = tuple(dict.fromkeys((*(() if dims is None else dims), *da.dims)))
        if layout is None:
            layout = _initial_layout(da.dims, sliced=(), horizontal=(), vertical=())
        self.horizontal = ipw.TagsInput(
            value=list(layout.horizontal),
            allowed_tags=list(da.dims),
            allow_duplicates=False,
        )
        self.vertical = ipw.TagsInput(
            value=list(layout.vertical),
            allowed_tags=list(da.dims),
            allow_duplicates=False,
        )
        self.invert_horizontal = ipw.Checkbox(
            description="Right to left",
            indent=False,
            layout=ipw.Layout(width="10em", flex="0 0 auto"),
        )
        self.invert_vertical = ipw.Checkbox(
            description="Top to bottom",
            indent=False,
            layout=ipw.Layout(width="10em", flex="0 0 auto"),
        )
        for tags in (self.horizontal, self.vertical):
            tags.tooltip = (
                "Drag dimensions to reorder or move between axes; remove to slice."
            )
            tags.layout = ipw.Layout(flex="1 1 auto", width="auto", min_width="0")
        self.slicers = {
            dim: CombinedSlicer(
                dim=dim, size=da.sizes[dim], coord=_slicer_coord(da, dim)
            )
            for dim in self._dims
        }
        # Keep the rows as well as the slicers so updates do not rebuild controls.
        self._slicer_rows = {
            dim: self._slicer_row(dim, slicer) for dim, slicer in self.slicers.items()
        }
        self._slicer_help = ipw.HTML(
            "Dimensions on neither axis are sliced below. "
            "Use X or Y to move a dimension to that axis."
        )
        self._slicers_box = ipw.VBox()
        self._updating = False

        # One snapshot per edit keeps axis order and slice ranges consistent.
        self.state_node = pp.Node(self._get_state)
        self.state_node.pretty_name = "DimensionConfigs"
        for dim, slicer in self.slicers.items():
            slicer.slider.observe(partial(self._on_slider_change, dim), names="value")
        self.horizontal.observe(self._on_tags_change, names="value")
        self.vertical.observe(self._on_tags_change, names="value")
        self._update_slicers()

        super().__init__(
            [
                self._axis_row("X axis", self.horizontal, self.invert_horizontal),
                self._axis_row("Y axis", self.vertical, self.invert_vertical),
                ipw.HTML(
                    "Drag dimensions to reorder (last dimension varies fastest) "
                    "or move between axes. Remove a dimension from an axis to slice it."
                ),
                self._slicer_help,
                self._slicers_box,
            ]
        )

    def _slicer_row(self, dim: str, slicer: CombinedSlicer) -> ipw.HBox:
        buttons = []
        for label, axis in (("X", self.horizontal), ("Y", self.vertical)):
            button = ipw.Button(
                description=label,
                tooltip=f"Move {dim} to the {label.lower()} axis",
                layout=ipw.Layout(width="3em", flex="0 0 auto"),
            )
            button.on_click(partial(self._move_to_axis, dim, axis))
            buttons.append(button)
        return ipw.HBox([*buttons, slicer], layout=ipw.Layout(align_items="center"))

    def _move_to_axis(self, dim: str, axis: ipw.TagsInput, _button: ipw.Button) -> None:
        # Ignore queued clicks after the row has disappeared, including double clicks.
        if dim in self.axis_layout.sliced:
            axis.value = [*axis.value, dim]

    @staticmethod
    def _axis_row(label: str, tags: ipw.TagsInput, reverse: ipw.Checkbox) -> ipw.HBox:
        # TagsInput does not render its description, so use an explicit label.
        return ipw.HBox(
            [
                ipw.Label(label, layout=ipw.Layout(width="4em", flex="0 0 auto")),
                reverse,
                tags,
            ],
            layout=ipw.Layout(align_items="center"),
        )

    @property
    def axis_layout(self) -> _DimensionLayout:
        horizontal, vertical = tuple(self.horizontal.value), tuple(self.vertical.value)
        return _DimensionLayout(
            sliced=tuple(
                dim for dim in self._dims if dim not in (*horizontal, *vertical)
            ),
            vertical=vertical,
            horizontal=horizontal,
        )

    def _get_state(self) -> _DimensionState:
        layout = self.axis_layout
        return _DimensionState(
            layout=layout,
            slices={dim: self.slicers[dim].value for dim in layout.sliced},
        )

    def _update_slicers(self) -> None:
        sliced = self.axis_layout.sliced
        self._slicers_box.children = [self._slicer_rows[dim] for dim in sliced]
        self._slicer_help.layout.display = None if sliced else "none"

    def _on_slider_change(self, dim: str, change: dict[str, Any]) -> None:
        if dim in self.axis_layout.sliced:
            self.state_node.notify_children(change)

    def _on_tags_change(self, change: dict[str, Any]) -> None:
        if self._updating:
            return
        edited = change["owner"]
        other = self.vertical if edited is self.horizontal else self.horizontal
        self._updating = True
        try:
            # TagsInput prevents duplicates in the frontend, but not in Python.
            edited.value = list(dict.fromkeys(edited.value))
            other.value = [dim for dim in other.value if dim not in edited.value]
            self._update_slicers()
        finally:
            self._updating = False
        self.state_node.notify_children(change)


def _slicer_coord(da: sc.DataArray, dim: str) -> sc.Variable:
    # Same fallback as Plopp's slice widgets.
    if dim in da.coords:
        return da.coords[dim]
    return sc.arange(dim, da.sizes[dim], unit=None)


def _prepare_plot(da: sc.DataArray, state: _DimensionState) -> sc.DataArray | None:
    if not state.layout.vertical or not state.layout.horizontal:
        return None
    # Reduce dimensionality before transpose/flatten: Scipp cannot copy some
    # non-contiguous arrays with more than six dimensions (e.g. DREAM + tof).
    return _flatten(_apply_slices(da, state.slices), state.layout)


def _flatten(da: sc.DataArray, layout: _DimensionLayout) -> sc.DataArray:
    return (
        da.transpose([*layout.vertical, *layout.horizontal])
        .flatten(layout.horizontal, to=" * ".join(layout.horizontal))
        .flatten(layout.vertical, to=" * ".join(layout.vertical))
    )


def _apply_slices(
    da: sc.DataArray, slices: Mapping[str, tuple[int, int]]
) -> sc.DataArray:
    for dim, (start, stop) in slices.items():
        da = da[dim, start] if start == stop else da[dim, start : stop + 1].sum(dim)
    return da


def _unwrap_flat_indices_2d(
    x_sizes: dict[str, int], y_sizes: dict[str, int]
) -> Callable[[int, int], dict[str, int]]:
    def unwrap(x: int, y: int) -> dict[str, int]:
        return {**_unwrap_flat_index(x, x_sizes), **_unwrap_flat_index(y, y_sizes)}

    return unwrap


def _unwrap_flat_index(index: int, sizes: dict[str, int]) -> dict[str, int]:
    indices = []
    for dim, size in reversed(sizes.items()):
        index, offset = divmod(index, size)
        indices.append((dim, offset))
    return dict(reversed(indices))  # Restore the input order.
