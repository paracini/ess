# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipp as sc
from ess.bifrost.diagnostics import FlatDetectorViewer


@pytest.fixture
def interactive_backend():
    pytest.importorskip("ipympl")
    original = plt.get_backend()
    plt.switch_backend("module://ipympl.backend_nbagg")
    yield
    plt.switch_backend(original)


def bank(dims: tuple[str, ...]) -> sc.DataArray:
    sizes = dict.fromkeys(dims, 2)
    return sc.DataArray(
        sc.arange("n", float(math.prod(sizes.values())), unit="counts").fold(
            "n", sizes=sizes
        )
    )


@pytest.mark.usefixtures("interactive_backend")
@pytest.mark.filterwarnings(
    "ignore:Passing unrecognized arguments to super\\(Toolbar\\):DeprecationWarning"
)
@pytest.mark.parametrize("container", ["array", "dict", "data_group"])
def test_accepts_array_and_bank_mappings(container: str) -> None:
    da = bank(("arc", "channel", "tube", "length", "tof"))
    if container == "array":
        data = da
    elif container == "dict":
        data = {"bank": da}
    else:
        data = sc.DataGroup(bank=da)
    view = FlatDetectorViewer(data)
    assert view.configs.vertical.value == ["arc", "tube"]
    assert view.configs.horizontal.value == ["channel", "length"]
    assert view.configs.axis_layout.sliced == ("tof",)
    expected = da.sum("tof").transpose(["arc", "tube", "channel", "length"])
    np.testing.assert_array_equal(
        view._figure.artists[view._figure_node.id].data.values,
        expected.values.reshape(4, 4),
    )


@pytest.mark.usefixtures("interactive_backend")
@pytest.mark.filterwarnings(
    "ignore:Passing unrecognized arguments to super\\(Toolbar\\):DeprecationWarning"
)
def test_mapping_can_switch_to_bank_with_different_dimensions() -> None:
    view = FlatDetectorViewer(
        {
            "full": bank(("arc", "channel", "tube", "length", "tof")),
            "partial": bank(("tube", "length", "energy")),
        }
    )
    view.children[0].children[1].value = "partial"
    assert view.configs.vertical.value == ["tube"]
    assert view.configs.horizontal.value == ["length"]
    assert view.configs.axis_layout.sliced == ("energy",)
    assert view._figure.canvas.xlabel == "length"
    assert view._figure.canvas.ylabel == "tube"


@pytest.mark.usefixtures("interactive_backend")
@pytest.mark.filterwarnings(
    "ignore:Passing unrecognized arguments to super\\(Toolbar\\):DeprecationWarning"
)
def test_single_partial_bank_does_not_require_all_detector_dimensions() -> None:
    view = FlatDetectorViewer({"bank": bank(("tube", "length", "tof"))})
    assert view.configs.vertical.value == ["tube"]
    assert view.configs.horizontal.value == ["length"]
    assert view.configs.axis_layout.sliced == ("tof",)
