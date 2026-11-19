# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import math

import numpy as np
import plopp as pp
import pytest
import scipp as sc

from ess.reduce.diagnostics.flat_bank_viewer import (
    FlatBankViewer,
    _DimensionConfigs,
    _unwrap_flat_indices_2d,
)

DIMS = ("arc", "channel", "tube", "length")
SHAPE = (2, 3, 4, 5)

# Emitted by ipympl's toolbar, not by the code under test.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:Passing unrecognized arguments to super\(Toolbar\):DeprecationWarning"
)


def make_data(dims: tuple[str, ...], shape: tuple[int, ...]) -> sc.DataArray:
    return sc.DataArray(
        sc.arange("x", float(math.prod(shape)), unit="counts").fold(
            "x", dims=dims, shape=shape
        ),
        coords={
            dim: sc.arange(dim, size, unit=None)
            for dim, size in zip(dims, shape, strict=True)
        },
    )


@pytest.fixture
def da() -> sc.DataArray:
    return make_data(DIMS, SHAPE)


@pytest.fixture
def interactive_backend():
    # Without an interactive backend, Plopp creates figures that are not widgets.
    pytest.importorskip("ipympl")
    import matplotlib.pyplot as plt

    original = plt.get_backend()
    plt.switch_backend("module://ipympl.backend_nbagg")
    yield
    plt.switch_backend(original)


def slice_dim(configs: _DimensionConfigs, dim: str) -> None:
    for axis in (configs.horizontal, configs.vertical):
        axis.value = [d for d in axis.value if d != dim]


def test_defaults_assign_all_dimensions_to_axes(da: sc.DataArray) -> None:
    configs = _DimensionConfigs(da)
    assert configs.horizontal.value == ["length"]
    assert configs.vertical.value == ["arc", "channel", "tube"]
    assert configs.axis_layout.sliced == ()
    assert configs._slicers_box.children == ()
    assert configs._slicer_help.layout.display == "none"


@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
def test_adding_dimension_transfers_it_to_other_axis(
    da: sc.DataArray, axis: str
) -> None:
    configs = _DimensionConfigs(da)
    target = getattr(configs, axis)
    source = configs.vertical if axis == "horizontal" else configs.horizontal
    dim = source.value[0]
    target.value = [*target.value, dim]
    assert dim not in source.value
    assert dim not in configs.axis_layout.sliced
    assert configs._slicers_box.children == ()


def test_axis_dimensions_deduplicate_programmatic_values(da: sc.DataArray) -> None:
    configs = _DimensionConfigs(da)
    configs.horizontal.value = ["arc", "arc"]
    assert configs.horizontal.value == ["arc"]


def test_slider_order_does_not_omit_unspecified_dimensions(da: sc.DataArray) -> None:
    configs = _DimensionConfigs(da, dims=["tube", "arc"])
    configs.horizontal.value = []
    configs.vertical.value = []
    assert configs.axis_layout.sliced == ("tube", "arc", "channel", "length")
    assert tuple(row.children[-1] for row in configs._slicers_box.children) == tuple(
        configs.slicers[dim] for dim in configs.axis_layout.sliced
    )


@pytest.mark.parametrize(("axis", "button_index"), [("horizontal", 0), ("vertical", 1)])
def test_slicer_axis_buttons_preserve_range(
    da: sc.DataArray, axis: str, button_index: int
) -> None:
    configs = _DimensionConfigs(da)
    slice_dim(configs, "tube")
    configs.slicers["tube"].slider.value = (1, 2)
    row = configs._slicers_box.children[0]
    target = getattr(configs, axis)
    expected = [*target.value, "tube"]

    row.children[button_index].click()
    assert target.value == expected
    assert configs._slicers_box.children == ()
    assert configs._slicer_help.layout.display == "none"

    # Queued clicks on the hidden row must not duplicate or move the dimension.
    row.children[button_index].click()
    row.children[1 - button_index].click()
    assert target.value == expected

    slice_dim(configs, "tube")
    assert configs._slicers_box.children == (row,)
    assert configs._slicer_help.layout.display is None
    assert configs.state_node().slices == {"tube": (1, 2)}


def test_axis_transfer_emits_one_complete_state(da: sc.DataArray) -> None:
    configs = _DimensionConfigs(da)
    calls = []
    sink = pp.Node(lambda state: calls.append(state), configs.state_node)
    sink()
    calls.clear()

    configs.horizontal.value = ["length", "arc"]

    # Removing the dimension from the other axis must not trigger an intermediate plot.
    assert len(calls) == 1
    assert calls[0].layout.horizontal == ("length", "arc")
    assert calls[0].layout.vertical == ("channel", "tube")
    assert calls[0].layout.sliced == ()


@pytest.mark.usefixtures("interactive_backend")
def test_initial_axis_order_is_used_and_unassigned_dimensions_are_sliced(
    da: sc.DataArray,
) -> None:
    view = FlatBankViewer(da, horizontal=["length"], vertical=["channel", "arc"])
    assert view.configs.axis_layout.sliced == ("tube",)
    view.configs.slicers["tube"].slider.value = (1, 2)
    expected = (
        da["tube", 1:3]
        .sum("tube")
        .transpose(["channel", "arc", "length"])
        .flatten(["channel", "arc"], to="channel * arc")
    )
    assert sc.identical(view.node(), expected)
    assert view._plot_data.sizes == {"vertical": 6, "horizontal": 5}
    assert view._figure.canvas.ylabel == "channel * arc"
    assert view._figure.canvas.xlabel == "length"


@pytest.mark.usefixtures("interactive_backend")
def test_explicit_sliced_dimensions_are_excluded_from_default_axes(da: sc.DataArray):
    view = FlatBankViewer(da, sliced=["tube"])
    assert view.configs.horizontal.value == ["length"]
    assert view.configs.vertical.value == ["arc", "channel"]
    assert view.configs.axis_layout.sliced == ("tube",)


@pytest.mark.usefixtures("interactive_backend")
def test_view_updates_in_place_and_hover_follows_dimension_order(
    da: sc.DataArray,
) -> None:
    view = FlatBankViewer(da)
    figure = view._figure
    slice_dim(view.configs, "tube")
    view.configs.slicers["tube"].slider.value = (3, 3)
    assert view._figure is figure
    assert sc.identical(view._plot_data.data.sum(), da["tube", 3].data.sum())
    view.configs.vertical.value = ["channel", "arc"]
    assert view._figure is figure
    assert view._figure.canvas.ylabel == "channel * arc"
    # y=3 -> channel=1, arc=1 with the reordered dimensions.
    text = view._figure.ax.format_coord(4.2, 3.2)
    assert text.startswith("{Length: 4, Channel: 1, Arc: 1}: ")
    expected = da["arc", 1]["channel", 1]["tube", 3]["length", 4].value
    assert str(expected) in text
    assert view._plot_data["vertical", 3]["horizontal", 4].value == expected


@pytest.mark.usefixtures("interactive_backend")
@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
def test_reversing_axis_preserves_data_hover_and_zoom(
    da: sc.DataArray, axis: str
) -> None:
    view = FlatBankViewer(da)
    data = view.node()
    plotted = view._plot_data
    hover = view._format_coord(1, 2)
    ax = view._figure.ax
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(1.5, 10.5)
    control = getattr(view.configs, f"invert_{axis}")
    control.value = True
    assert ax.xaxis_inverted() == (axis == "horizontal")
    assert ax.yaxis_inverted() == (axis == "vertical")
    assert sorted(ax.get_xlim()) == [0.5, 3.5]
    assert sorted(ax.get_ylim()) == [1.5, 10.5]
    assert view.node() is data
    assert view._plot_data is plotted
    assert view._format_coord(1, 2) == hover
    control.value = False
    assert not ax.xaxis_inverted()
    assert not ax.yaxis_inverted()


def assert_figure_matches(view: FlatBankViewer, expected: np.ndarray) -> None:
    assert view._figure.layout.display is None
    reference = sc.DataArray(
        sc.array(dims=["vertical", "horizontal"], values=expected, unit="counts")
    )
    # A fresh Plopp render must not restore the figure's original data.
    for render in (False, True):
        if render:
            view._figure.view.render()
        artist = view._figure.artists[view._figure_node.id]
        np.testing.assert_array_equal(artist.data.values, expected)
        np.testing.assert_allclose(
            artist._image.get_array(), view._figure.view.colormapper.rgba(reference)
        )
        assert view._figure.canvas.xlabel == " * ".join(view.configs.horizontal.value)
        assert view._figure.canvas.ylabel == " * ".join(view.configs.vertical.value)


@pytest.mark.usefixtures("interactive_backend")
def test_image_tracks_reordering_transfers_and_slices(da: sc.DataArray) -> None:
    view = FlatBankViewer(da)
    configs = view.configs
    values = da.values
    # Explicit NumPy references avoid reconstructing the viewer's layout algorithm.
    # set_state exercises the same path as updates received from the browser.
    for control, value, expected in (
        (
            configs.vertical,
            ["tube", "arc", "channel"],
            values.transpose(2, 0, 1, 3).reshape(24, 5),
        ),
        (
            configs.vertical,
            ["tube", "channel", "arc"],
            values.transpose(2, 1, 0, 3).reshape(24, 5),
        ),
        (
            configs.horizontal,
            ["tube", "length"],
            values.transpose(1, 0, 2, 3).reshape(6, 20),
        ),
        (
            configs.horizontal,
            ["length", "tube"],
            values.transpose(1, 0, 3, 2).reshape(6, 20),
        ),
        (
            configs.vertical,
            ["arc"],
            values.sum(axis=1).transpose(0, 2, 1).reshape(2, 20),
        ),
        (
            configs.slicers["channel"].slider,
            (1, 1),
            values[:, 1].transpose(0, 2, 1).reshape(2, 20),
        ),
    ):
        control.set_state({"value": value})
        assert_figure_matches(view, expected)


@pytest.mark.usefixtures("interactive_backend")
def test_unsupported_layout_hides_old_plot_and_recovers() -> None:
    da = make_data(
        ("sumo", "module", "segment", "counter", "wire", "strip", "tof"),
        (2, 3, 2, 2, 2, 3, 4),
    )
    vertical = ["counter", "segment", "wire"]
    view = FlatBankViewer(da, horizontal=["module", "strip", "sumo"], vertical=vertical)
    expected = da.values.sum(axis=6).transpose(3, 2, 4, 1, 5, 0).reshape(8, 18)
    assert_figure_matches(view, expected)
    with pytest.raises(RuntimeError, match="more than 6 dimensions"):
        view.configs.vertical.value = [*vertical, "tof"]
    assert view._figure.layout.display == "none"
    assert view._plot_data is None
    assert "Cannot display this layout" in view._message.value
    assert "more than 6 dimensions" in view._message.value
    assert view._message.layout.display is None

    view.configs.vertical.value = vertical
    assert view._figure.layout.display is None
    assert view._message.layout.display == "none"
    assert_figure_matches(view, expected)


@pytest.mark.usefixtures("interactive_backend")
def test_reversed_axes_survive_slicing_reordering_and_autoscale(da: sc.DataArray):
    view = FlatBankViewer(da)
    view.configs.invert_horizontal.value = True
    view.configs.invert_vertical.value = True
    slice_dim(view.configs, "tube")
    view.configs.slicers["tube"].slider.value = (1, 1)
    view.configs.vertical.value = ["channel", "arc"]
    view._figure.view.fit_to_data()
    assert view._figure.ax.get_xlim() == (4.5, -0.5)
    assert view._figure.ax.get_ylim() == (5.5, -0.5)
    view.configs.horizontal.value = []
    view.configs.horizontal.value = ["length"]
    assert view._figure.ax.xaxis_inverted()
    assert view._figure.ax.yaxis_inverted()


@pytest.mark.usefixtures("interactive_backend")
@pytest.mark.parametrize(("axis", "button_index"), [("horizontal", 0), ("vertical", 1)])
def test_axis_button_restores_plot_after_emptying_axis(
    da: sc.DataArray, axis: str, button_index: int
) -> None:
    view = FlatBankViewer(da)
    figure = view._figure
    control = getattr(view.configs, axis)
    dim = control.value[0]
    control.value = []
    assert view.node() is None
    assert view._message.layout.display is None
    assert figure.layout.display == "none"
    assert view._format_coord(0, 0) == ""

    view.configs._slicer_rows[dim].children[button_index].click()
    assert control.value == [dim]
    assert view._figure is figure
    assert view._message.layout.display == "none"
    expected = (
        da.values.reshape(24, 5) if axis == "horizontal" else da.values.sum(axis=(1, 2))
    )
    assert_figure_matches(view, expected)


@pytest.mark.usefixtures("interactive_backend")
def test_can_create_reversed_plot_after_starting_with_all_dimensions_sliced(
    da: sc.DataArray,
):
    view = FlatBankViewer(da, sliced=da.dims)
    assert view._figure is None
    assert view._message.layout.display is None
    view.configs.invert_horizontal.value = True
    view.configs.invert_vertical.value = True
    view.configs.horizontal.value = ["length"]
    assert view._figure is None
    view.configs.vertical.value = ["arc"]
    assert view._message.layout.display == "none"
    assert_figure_matches(view, da.values.sum(axis=(1, 2)))
    assert view._figure.ax.xaxis_inverted()
    assert view._figure.ax.yaxis_inverted()


@pytest.mark.usefixtures("interactive_backend")
def test_view_hover_outside_data_is_empty(da: sc.DataArray) -> None:
    view = FlatBankViewer(da, horizontal=["length"], vertical=["channel", "arc"])
    for x, y in [(-1, 0), (5, 0), (0, -1), (0, 6)]:
        assert view._format_coord(x, y) == ""


@pytest.fixture
def banks(da: sc.DataArray) -> sc.DataGroup:
    return sc.DataGroup(
        {
            "a": da,
            "b": make_data(("wire", "strip"), (6, 7)),
            "c": make_data(("arc", "pixel", "tube"), (2, 3, 4)),
        }
    )


def bank_selector(view: FlatBankViewer):
    return view.children[0].children[1]


@pytest.mark.usefixtures("interactive_backend")
def test_switching_banks_preserves_settings_and_updates_image(
    banks: sc.DataGroup,
) -> None:
    view = FlatBankViewer(banks)
    selector = bank_selector(view)
    figure = view._figure
    slice_dim(view.configs, "tube")
    view.configs.slicers["tube"].slider.value = (2, 2)
    view.configs.vertical.value = ["channel", "arc"]
    view.configs.invert_horizontal.value = True
    view.configs.invert_vertical.value = True

    for name in ("b", "c"):
        selector.value = name
        assert view._figure is figure
        assert_figure_matches(view, banks[name].values.reshape(6, -1))
        assert not view._figure.ax.xaxis_inverted()
        assert not view._figure.ax.yaxis_inverted()

    selector.value = "a"
    assert view._figure is figure
    assert view.configs.vertical.value == ["channel", "arc"]
    assert view.configs.axis_layout.sliced == ("tube",)
    assert view.configs.slicers["tube"].value == (2, 2)
    assert view._figure.ax.xaxis_inverted()
    assert view._figure.ax.yaxis_inverted()
    assert_figure_matches(
        view, banks['a'].values[:, :, 2, :].transpose(1, 0, 2).reshape(6, 5)
    )


@pytest.mark.usefixtures("interactive_backend")
def test_view_ignores_updates_from_other_banks(banks: sc.DataGroup) -> None:
    view = FlatBankViewer(banks)
    selector = bank_selector(view)
    selector.value = "b"
    configs_b = view.configs
    selector.value = "a"
    shown = view._plot_data
    configs_b.vertical.value = []
    configs_b.invert_vertical.value = True
    assert not view._figure.ax.yaxis_inverted()
    assert view._plot_data is shown


@pytest.mark.usefixtures("interactive_backend")
def test_initial_layout_per_bank(banks: sc.DataGroup) -> None:
    view = FlatBankViewer(
        banks,
        horizontal=["length", "strip", "pixel"],
        vertical=["channel", "arc", "wire"],
    )
    assert view.configs.vertical.value == ["channel", "arc"]
    assert view.configs.horizontal.value == ["length"]
    assert view.configs.axis_layout.sliced == ("tube",)
    bank_selector(view).value = "b"
    assert view.configs.vertical.value == ["wire"]
    assert view.configs.horizontal.value == ["strip"]
    bank_selector(view).value = "c"
    assert view.configs.vertical.value == ["arc"]
    assert view.configs.horizontal.value == ["pixel"]
    assert view.configs.axis_layout.sliced == ("tube",)


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        ({"sliced": ["arc"], "vertical": ["arc"]}, "more than once"),
        ({"sliced": ["arc"], "horizontal": ["arc"]}, "more than once"),
        ({"horizontal": ["arc"], "vertical": ["arc"]}, "more than once"),
        ({"vertical": ["arc", "arc"]}, "more than once"),
        ({"horizontal": ["nope"]}, "not in the data"),
    ],
)
def test_view_rejects_invalid_layout(
    banks: sc.DataGroup,
    layout: dict[str, list[str]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FlatBankViewer(banks, **layout)


def test_view_rejects_empty_bank_mapping() -> None:
    with pytest.raises(ValueError, match="at least one data array"):
        FlatBankViewer({})


@pytest.mark.parametrize("container", ["array", "dict"])
def test_empty_dimensions_are_rejected_before_selecting_any_bank(
    da: sc.DataArray, container: str
) -> None:
    empty = sc.DataArray(sc.zeros(sizes={"x": 2, "y": 0}))
    if container == "array":
        data = empty
        message = r"Empty dimensions \['y'\] in data array"
    else:
        # The empty bank is not initially selected, but must still be checked.
        data = {"good": da, "empty": empty}
        message = r"Empty dimensions \['y'\] in bank 'empty'"
    with pytest.raises(ValueError, match=message):
        FlatBankViewer(data)


def test_view_rejects_non_data_array_entries(banks: sc.DataGroup) -> None:
    banks["scalar"] = sc.scalar(1.0)
    with pytest.raises(TypeError, match="scalar"):
        FlatBankViewer(banks)


@pytest.mark.parametrize(
    ("x_sizes", "y_sizes"),
    [
        ({"a": 3}, {"b": 5}),
        ({"a": 3, "2nd": 5}, {"b": 4}),
        ({"a": 2}, {"b": 4, "c": 1, "d": 3}),
        ({"x": 3, "a": 5, "g": 1}, {"b": 4, "y": 4}),
    ],
)
def test_unwrap_flat_indices_preserves_indices_and_dimension_order(x_sizes, y_sizes):
    unwrap = _unwrap_flat_indices_2d(x_sizes, y_sizes)
    for x, x_indices in enumerate(np.ndindex(*x_sizes.values())):
        for y, y_indices in enumerate(np.ndindex(*y_sizes.values())):
            expected = dict(
                zip((*x_sizes, *y_sizes), (*x_indices, *y_indices), strict=True)
            )
            actual = unwrap(x, y)
            assert actual == expected
            assert list(actual) == list(expected)
