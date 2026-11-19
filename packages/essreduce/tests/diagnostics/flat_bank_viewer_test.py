# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
import math

import numpy as np
import plopp as pp
import pytest
import scipp as sc
from traitlets import TraitError

from ess.reduce.diagnostics.flat_bank_viewer import (
    FlatBankViewer,
    _DimensionConfigs,
    _prepare_plot,
    _unwrap_flat_indices_2d,
)

DIMS = ("arc", "channel", "tube", "length")
SHAPE = (2, 3, 4, 5)


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


def view_test(func):
    """Mark a test that creates a figure."""
    func = pytest.mark.usefixtures("interactive_backend")(func)
    # Emitted by ipympl's toolbar, not by the code under test.
    return pytest.mark.filterwarnings(
        "ignore:Passing unrecognized arguments to super\\(Toolbar\\):DeprecationWarning"
    )(func)


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
def test_adding_tag_transfers_dimension_to_other_axis(
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


@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
def test_axis_tags_reject_unknown_dimensions(da: sc.DataArray, axis: str) -> None:
    configs = _DimensionConfigs(da)
    before = configs.axis_layout
    with pytest.raises(TraitError):
        getattr(configs, axis).value = ["unknown"]
    assert configs.axis_layout == before


@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
def test_axis_tags_deduplicate_programmatic_values(da: sc.DataArray, axis: str) -> None:
    configs = _DimensionConfigs(da)
    tags = getattr(configs, axis)
    assert tags.allow_duplicates is False
    tags.value = ["arc", "arc"]
    assert tags.value == ["arc"]


def test_unassigned_dimensions_show_sliders_and_keep_their_values(
    da: sc.DataArray,
) -> None:
    configs = _DimensionConfigs(da)
    slice_dim(configs, "tube")
    slicer = configs.slicers["tube"]
    row = configs._slicers_box.children[0]
    assert row.children[-1] is slicer
    assert configs._slicer_help.layout.display is None
    slicer.slider.value = (1, 2)
    configs.horizontal.value = ["tube", *configs.horizontal.value]
    assert configs._slicers_box.children == ()
    slice_dim(configs, "tube")
    assert configs._slicers_box.children == (row,)
    assert configs.state_node().slices == {"tube": (1, 2)}


def test_slider_order_does_not_omit_unspecified_dimensions(da: sc.DataArray) -> None:
    configs = _DimensionConfigs(da, dims=["tube", "arc"])
    configs.horizontal.value = []
    configs.vertical.value = []
    assert configs.axis_layout.sliced == ("tube", "arc", "channel", "length")
    assert tuple(row.children[-1] for row in configs._slicers_box.children) == tuple(
        configs.slicers[dim] for dim in configs.axis_layout.sliced
    )


@pytest.mark.parametrize(("axis", "button_index"), [("horizontal", 0), ("vertical", 1)])
def test_slicer_axis_buttons_update_once_and_preserve_range(
    da: sc.DataArray, axis: str, button_index: int
) -> None:
    configs = _DimensionConfigs(da)
    slice_dim(configs, "tube")
    configs.slicers["tube"].slider.value = (1, 2)
    row = configs._slicers_box.children[0]
    target = getattr(configs, axis)
    original = list(target.value)
    calls = []
    sink = pp.Node(lambda state: calls.append(state), configs.state_node)
    sink()
    calls.clear()

    row.children[button_index].click()

    assert target.value == [*original, "tube"]
    assert len(calls) == 1
    assert calls[0].layout == configs.axis_layout
    assert calls[0].slices == {}
    assert configs._slicers_box.children == ()
    assert configs._slicer_help.layout.display == "none"

    # Clicks already queued on the hidden row must not duplicate or move the tag.
    row.children[button_index].click()
    row.children[1 - button_index].click()
    assert target.value == [*original, "tube"]
    assert len(calls) == 1

    slice_dim(configs, "tube")
    assert configs._slicers_box.children == (row,)
    assert configs._slicer_help.layout.display is None
    assert configs.state_node().slices == {"tube": (1, 2)}


def test_each_input_edit_emits_one_consistent_state(da: sc.DataArray) -> None:
    configs = _DimensionConfigs(da)
    calls = []
    sink = pp.Node(lambda state: calls.append(state), configs.state_node)
    sink()
    calls.clear()

    for control, value in (
        (configs.vertical, ["arc", "channel"]),
        (configs.slicers["tube"].slider, (2, 2)),
        (configs.horizontal, ["arc", "length"]),
        (configs.horizontal, ["length", "arc"]),
        (configs.vertical, ["tube", "channel"]),
    ):
        control.value = value
        assert len(calls) == 1
        state = calls.pop()
        assert state.layout == configs.axis_layout
        assert state.slices == {
            dim: configs.slicers[dim].value for dim in state.layout.sliced
        }
        assert not set(state.layout.horizontal).intersection(state.layout.vertical)


def test_flatten_uses_tag_order_for_every_dimension(da: sc.DataArray) -> None:
    configs = _DimensionConfigs(da)
    node = pp.Node(_prepare_plot, da, configs.state_node)
    configs.vertical.value = ["tube", "arc", "channel"]
    expected = da.transpose(["tube", "arc", "channel", "length"]).flatten(
        ["tube", "arc", "channel"], to="tube * arc * channel"
    )
    assert sc.identical(node(), expected)
    # Reorder only non-leading tags: there is no special primary dimension.
    configs.vertical.value = ["tube", "channel", "arc"]
    expected = da.transpose(["tube", "channel", "arc", "length"]).flatten(
        ["tube", "channel", "arc"], to="tube * channel * arc"
    )
    assert sc.identical(node(), expected)


@view_test
def test_initial_axis_order_is_used_and_unassigned_dimensions_are_sliced(
    da: sc.DataArray,
) -> None:
    view = FlatBankViewer(da, horizontal=["length"], vertical=["channel", "arc"])
    assert len(view.children) == 2
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


@view_test
def test_explicit_sliced_dimensions_are_excluded_from_default_axes(da: sc.DataArray):
    view = FlatBankViewer(da, sliced=["tube"])
    assert view.configs.horizontal.value == ["length"]
    assert view.configs.vertical.value == ["arc", "channel"]
    assert view.configs.axis_layout.sliced == ("tube",)


@view_test
def test_view_updates_in_place_and_hover_follows_tag_order(da: sc.DataArray) -> None:
    view = FlatBankViewer(da)
    figure = view._figure
    slice_dim(view.configs, "tube")
    view.configs.slicers["tube"].slider.value = (3, 3)
    assert view._figure is figure
    assert sc.identical(view._plot_data.data.sum(), da["tube", 3].data.sum())
    view.configs.vertical.value = ["channel", "arc"]
    assert view._figure is figure
    assert view._figure.canvas.ylabel == "channel * arc"
    # y=3 -> channel=1, arc=1 with the reordered tags.
    text = view._figure.ax.format_coord(4.2, 3.2)
    assert text.startswith("{Length: 4, Channel: 1, Arc: 1}: ")
    expected = da["arc", 1]["channel", 1]["tube", 3]["length", 4].value
    assert str(expected) in text
    assert view._plot_data["vertical", 3]["horizontal", 4].value == expected


@view_test
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


def assert_figure_matches_inputs(view: FlatBankViewer, da: sc.DataArray) -> None:
    configs = view.configs
    if not configs.horizontal.value or not configs.vertical.value:
        assert view._figure.layout.display == "none"
        return

    # Compute an independent reference, slicing before transposing/reshaping.
    expected = da.values.copy()
    dims = list(da.dims)
    for dim in configs.axis_layout.sliced:
        start, stop = configs.slicers[dim].value
        axis = dims.index(dim)
        expected = np.take(expected, range(start, stop + 1), axis=axis).sum(axis=axis)
        dims.remove(dim)
    order = [*configs.vertical.value, *configs.horizontal.value]
    expected = expected.transpose([dims.index(dim) for dim in order]).reshape(
        math.prod(da.sizes[dim] for dim in configs.vertical.value), -1
    )

    # Check both the current artist and a fresh Plopp render, which must not
    # restore the data from when the figure was first created.
    for render in (False, True):
        if render:
            view._figure.view.render()
        artist = view._figure.artists[view._figure_node.id]
        np.testing.assert_array_equal(artist.data.values, expected)
        reference = sc.DataArray(
            sc.array(dims=["vertical", "horizontal"], values=expected, unit=da.unit)
        )
        np.testing.assert_allclose(
            artist._image.get_array(), view._figure.view.colormapper.rgba(reference)
        )
        assert view._figure.canvas.xlabel == " * ".join(configs.horizontal.value)
        assert view._figure.canvas.ylabel == " * ".join(configs.vertical.value)
        assert view._figure.ax.xaxis_inverted() == configs.invert_horizontal.value
        assert view._figure.ax.yaxis_inverted() == configs.invert_vertical.value


@view_test
def test_image_tracks_successive_input_changes_and_rerenders(da: sc.DataArray) -> None:
    view = FlatBankViewer(da)
    configs = view.configs
    # set_state simulates widget updates received from the browser.
    for control, value in (
        (configs.vertical, ["tube", "arc", "channel"]),
        (configs.horizontal, ["tube", "length"]),
        (configs.horizontal, ["length", "tube"]),
        (configs.vertical, ["arc"]),
        (configs.slicers["channel"].slider, (1, 1)),
        (configs.horizontal, ["length"]),
        (configs.invert_horizontal, True),
        (configs.invert_vertical, True),
        (configs.vertical, []),
        (configs.vertical, ["tube", "channel", "arc"]),
        (configs.vertical, ["channel", "arc"]),
    ):
        control.set_state({"value": value})
        assert_figure_matches_inputs(view, da)

    # Exercise the buttons through to the rendered image, including recovery
    # when an empty axis has hidden the plot.
    figure = view._figure
    for axis, button_index in ((configs.horizontal, 0), (configs.vertical, 1)):
        dim = axis.value[0]
        axis.value = []
        configs.slicers[dim].slider.value = (1, 1)
        assert_figure_matches_inputs(view, da)
        configs._slicer_rows[dim].children[button_index].click()
        assert axis.value == [dim]
        assert view._figure is figure
        assert_figure_matches_inputs(view, da)


@view_test
def test_unsupported_layout_hides_old_plot_and_recovers() -> None:
    da = make_data(
        ("sumo", "module", "segment", "counter", "wire", "strip", "tof"),
        (2, 3, 2, 2, 2, 3, 4),
    )
    vertical = ["counter", "segment", "wire"]
    view = FlatBankViewer(da, horizontal=["module", "strip", "sumo"], vertical=vertical)
    assert_figure_matches_inputs(view, da)
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
    assert_figure_matches_inputs(view, da)


@view_test
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


@view_test
def test_axis_directions_can_be_set_before_figure_exists(da: sc.DataArray):
    view = FlatBankViewer(da, sliced=da.dims)
    view.configs.invert_horizontal.value = True
    view.configs.invert_vertical.value = True
    view.configs.horizontal.value = ["length"]
    view.configs.vertical.value = ["arc"]
    assert view._figure.ax.xaxis_inverted()
    assert view._figure.ax.yaxis_inverted()


@view_test
@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
def test_empty_axis_hides_plot_until_a_tag_is_added(
    da: sc.DataArray, axis: str
) -> None:
    view = FlatBankViewer(da)
    tags = getattr(view.configs, axis)
    original = list(tags.value)
    tags.value = []
    assert view.node() is None
    assert view._message.layout.display is None
    assert view._figure.layout.display == "none"
    tags.value = original
    assert view._message.layout.display == "none"
    assert view._figure.layout.display is None


@view_test
def test_can_create_plot_after_starting_with_all_dimensions_sliced(da: sc.DataArray):
    view = FlatBankViewer(da, sliced=da.dims)
    assert view._figure is None
    assert view._message.layout.display is None
    view.configs.horizontal.value = ["length"]
    assert view._figure is None
    view.configs.vertical.value = ["arc"]
    assert view._figure is not None
    assert view.node().dims == ("arc", "length")
    assert view._message.layout.display == "none"


@view_test
@pytest.mark.parametrize(("x", "y"), [(-1, 0), (5, 0), (0, -1), (0, 6)])
def test_view_hover_outside_data_is_empty(da: sc.DataArray, x: float, y: float) -> None:
    view = FlatBankViewer(da, horizontal=["length"], vertical=["channel", "arc"])
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


@view_test
def test_axis_directions_are_kept_per_bank(banks: sc.DataGroup):
    view = FlatBankViewer(banks)
    configs_a = view.configs
    configs_a.invert_horizontal.value = True
    bank_selector(view).value = "b"
    assert not view._figure.ax.xaxis_inverted()
    assert not view._figure.ax.yaxis_inverted()
    configs_a.invert_vertical.value = True
    assert not view._figure.ax.yaxis_inverted()  # Inactive bank's update is ignored.
    bank_selector(view).value = "a"
    assert view._figure.ax.xaxis_inverted()
    assert view._figure.ax.yaxis_inverted()


@view_test
def test_image_tracks_bank_switches_and_rerenders(banks: sc.DataGroup):
    view = FlatBankViewer(banks)
    view.configs.vertical.value = ["channel", "arc", "tube"]
    for bank in ("b", "a", "c", "b", "a"):
        bank_selector(view).value = bank
        assert_figure_matches_inputs(view, banks[bank])


@view_test
def test_switching_banks_keeps_axis_order_and_slider_state(banks: sc.DataGroup) -> None:
    view = FlatBankViewer(banks)
    selector = bank_selector(view)
    assert list(selector.options) == ["a", "b", "c"]
    figure = view._figure
    slice_dim(view.configs, "tube")
    view.configs.slicers["tube"].slider.value = (2, 2)
    view.configs.vertical.value = ["channel", "arc"]
    expected = view._plot_data
    selector.value = "b"
    assert view._figure is figure
    assert view._configs_box.children == (view.configs,)
    assert view._plot_data.sizes == {"vertical": 6, "horizontal": 7}
    assert view.configs.horizontal.allowed_tags == ["wire", "strip"]
    selector.value = "c"
    assert view.configs.axis_layout.sliced == ()
    selector.value = "a"
    assert view.configs.vertical.value == ["channel", "arc"]
    assert view.configs.axis_layout.sliced == ("tube",)
    assert view.configs.slicers["tube"].value == (2, 2)
    assert sc.identical(view._plot_data, expected)


@view_test
def test_view_ignores_updates_from_other_banks(banks: sc.DataGroup) -> None:
    view = FlatBankViewer(banks)
    selector = bank_selector(view)
    selector.value = "b"
    configs_b = view.configs
    selector.value = "a"
    shown = view._plot_data
    configs_b.vertical.value = []
    assert view._plot_data is shown


@view_test
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


@pytest.mark.parametrize("container", ["array", "dict", "data_group"])
def test_empty_dimensions_are_rejected_before_selecting_any_bank(
    da: sc.DataArray, container: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = sc.DataArray(sc.zeros(sizes={"x": 2, "y": 0}))
    if container == "array":
        data = empty
        message = r"Empty dimensions \['y'\] in data array"
    else:
        # The empty bank is not initially selected, but must still be checked.
        data = {"good": da, "empty": empty}
        if container == "data_group":
            data = sc.DataGroup(data)
        message = r"Empty dimensions \['y'\] in bank 'empty'"
    monkeypatch.setattr(
        FlatBankViewer,
        "_select",
        lambda *_: pytest.fail("Bank selected before validation"),
    )
    with pytest.raises(ValueError, match=message):
        FlatBankViewer(data)


def test_view_rejects_non_data_array_entries(banks: sc.DataGroup) -> None:
    banks["scalar"] = sc.scalar(1.0)
    with pytest.raises(TypeError, match="scalar"):
        FlatBankViewer(banks)


def test_unwrap_flat_indices_2d_single_dim() -> None:
    unwrap = _unwrap_flat_indices_2d({'a': 3}, {'b': 5})
    assert unwrap(0, 0) == {'a': 0, 'b': 0}
    assert unwrap(1, 0) == {'a': 1, 'b': 0}
    assert unwrap(2, 0) == {'a': 2, 'b': 0}
    assert unwrap(0, 4) == {'a': 0, 'b': 4}
    assert unwrap(2, 4) == {'a': 2, 'b': 4}


def test_unwrap_flat_indices_2d_two_x_dims() -> None:
    unwrap = _unwrap_flat_indices_2d({'a': 3, '2nd': 5}, {'b': 4})
    assert unwrap(0, 0) == {'a': 0, '2nd': 0, 'b': 0}
    assert unwrap(0, 1) == {'a': 0, '2nd': 0, 'b': 1}
    assert unwrap(0, 3) == {'a': 0, '2nd': 0, 'b': 3}

    assert unwrap(1, 0) == {'a': 0, '2nd': 1, 'b': 0}
    assert unwrap(1, 1) == {'a': 0, '2nd': 1, 'b': 1}
    assert unwrap(1, 3) == {'a': 0, '2nd': 1, 'b': 3}

    assert unwrap(4, 0) == {'a': 0, '2nd': 4, 'b': 0}
    assert unwrap(4, 1) == {'a': 0, '2nd': 4, 'b': 1}
    assert unwrap(4, 3) == {'a': 0, '2nd': 4, 'b': 3}

    assert unwrap(5, 0) == {'a': 1, '2nd': 0, 'b': 0}
    assert unwrap(5, 2) == {'a': 1, '2nd': 0, 'b': 2}
    assert unwrap(5, 3) == {'a': 1, '2nd': 0, 'b': 3}

    assert unwrap(11, 0) == {'a': 2, '2nd': 1, 'b': 0}
    assert unwrap(11, 2) == {'a': 2, '2nd': 1, 'b': 2}
    assert unwrap(11, 3) == {'a': 2, '2nd': 1, 'b': 3}

    assert unwrap(14, 0) == {'a': 2, '2nd': 4, 'b': 0}
    assert unwrap(14, 1) == {'a': 2, '2nd': 4, 'b': 1}
    assert unwrap(14, 2) == {'a': 2, '2nd': 4, 'b': 2}


def test_unwrap_flat_indices_2d_three_y_dims() -> None:
    unwrap = _unwrap_flat_indices_2d({'a': 2}, {'b': 4, 'c': 1, 'd': 3})
    assert unwrap(0, 0) == {'a': 0, 'b': 0, 'c': 0, 'd': 0}
    assert unwrap(1, 0) == {'a': 1, 'b': 0, 'c': 0, 'd': 0}

    assert unwrap(0, 3) == {'a': 0, 'b': 1, 'c': 0, 'd': 0}
    assert unwrap(1, 3) == {'a': 1, 'b': 1, 'c': 0, 'd': 0}

    assert unwrap(0, 7) == {'a': 0, 'b': 2, 'c': 0, 'd': 1}
    assert unwrap(1, 7) == {'a': 1, 'b': 2, 'c': 0, 'd': 1}


def test_unwrap_flat_indices_2d_dim_order() -> None:
    unwrap = _unwrap_flat_indices_2d({'x': 3, 'a': 5, 'g': 1}, {'b': 4, 'y': 4})
    assert list(unwrap(0, 0)) == ['x', 'a', 'g', 'b', 'y']
