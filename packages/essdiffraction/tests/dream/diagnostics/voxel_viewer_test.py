# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipp as sc
from ess.dream.diagnostics import FlatVoxelViewer


@pytest.fixture
def interactive_backend():
    pytest.importorskip("ipympl")
    original = plt.get_backend()
    plt.switch_backend("module://ipympl.backend_nbagg")
    yield
    plt.switch_backend(original)


def bank(dims: tuple[str, ...]) -> sc.DataArray:
    # Match the dimensions and storage order of the GEANT4 histograms, including
    # the seventh (tof) dimension in the endcap and high-resolution banks.
    sizes = dict.fromkeys(dims, 2)
    return sc.DataArray(
        sc.arange("n", float(math.prod(sizes.values())), unit="counts").fold(
            "n", sizes=sizes
        ),
        coords={"tof": sc.arange("tof", 3.0, unit="us")},
    )


@pytest.mark.usefixtures("interactive_backend")
@pytest.mark.filterwarnings(
    "ignore:Passing unrecognized arguments to super\\(Toolbar\\):DeprecationWarning"
)
@pytest.mark.parametrize("container", ["array", "dict", "data_group"])
@pytest.mark.parametrize("extra_dim", [None, "sumo", "sector"])
def test_single_bank_does_not_require_other_banks_dimensions(
    container: str, extra_dim: str | None
) -> None:
    dims = ("module", "segment", "counter", "wire", "strip", "tof")
    da = bank(dims if extra_dim is None else (extra_dim, *dims))
    if container == "array":
        data = da
    elif container == "dict":
        data = {"bank": da}
    else:
        data = sc.DataGroup(bank=da)
    view = FlatVoxelViewer(data)
    horizontal = ["module", "strip"] + ([] if extra_dim is None else [extra_dim])
    assert view.configs.horizontal.value == horizontal
    assert view.configs.vertical.value == ["counter", "segment", "wire"]
    assert view.configs.axis_layout.sliced == ("tof",)
    expected = (
        da.sum("tof")
        .transpose(["counter", "segment", "wire", *horizontal])
        .values.reshape(8, -1)
    )
    np.testing.assert_array_equal(
        view._figure.artists[view._figure_node.id].data.values, expected
    )


@pytest.mark.usefixtures("interactive_backend")
@pytest.mark.filterwarnings(
    "ignore:Passing unrecognized arguments to super\\(Toolbar\\):DeprecationWarning"
)
def test_switch_from_mantle_to_seven_dimensional_banks_updates_image() -> None:
    dims = ("module", "segment", "counter", "wire", "strip", "tof")
    banks = {
        "mantle": bank(dims),
        "endcap_forward": bank(("sumo", *dims)),
        "high_resolution": bank(("sector", *dims)),
    }
    view = FlatVoxelViewer(banks)
    selector = view.children[0].children[1]
    assert selector.value == "mantle"

    for name in ("endcap_forward", "high_resolution", "mantle", "endcap_forward"):
        selector.value = name
        horizontal = list(view.configs.horizontal.value)
        vertical = list(view.configs.vertical.value)
        # Independent reference: integrate tof before arranging detector pixels.
        expected = banks[name].sum("tof").transpose([*vertical, *horizontal])
        values = expected.values.reshape(
            math.prod(expected.sizes[dim] for dim in vertical), -1
        )
        artist = view._figure.artists[view._figure_node.id]
        np.testing.assert_array_equal(artist.data.values, values)
        assert view._figure.canvas.xlabel == " * ".join(horizontal)
        assert view._figure.canvas.ylabel == " * ".join(vertical)
        assert view._figure.layout.display is None
        assert view._message.layout.display == "none"

    # Reordering and changing a slice on the selected endcap must also work.
    view.configs.horizontal.value = ["sumo", "strip", "module"]
    view.configs.slicers["tof"].slider.value = (1, 1)
    expected = banks["endcap_forward"]["tof", 1].transpose(
        [*view.configs.vertical.value, *view.configs.horizontal.value]
    )
    values = expected.values.reshape(8, -1)
    np.testing.assert_array_equal(
        view._figure.artists[view._figure_node.id].data.values, values
    )
