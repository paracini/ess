# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Scipp contributors (https://github.com/scipp)
"""Flat detector viewer for BIFROST."""

from collections.abc import Mapping
from typing import Any

import scipp as sc

from ess.reduce.diagnostics.flat_bank_viewer import FlatBankViewer


class FlatDetectorViewer(FlatBankViewer):
    """Interactive 2D plot of all detector pixels."""

    def __init__(
        self,
        data: Mapping[str, sc.DataArray] | sc.DataArray,
        *,
        cbar: bool = True,
        rasterized: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a new viewer.

        Parameters
        ----------
        data:
            Histogrammed data, one entry per bank.
        cbar:
            Whether to show a colorbar.
        rasterized:
            If ``True``, the figure is rasterized which improves rendering
            speed but reduces resolution.
        **kwargs:
            Additional arguments passed to :func:`plopp.imagefigure`.
        """
        dims = (
            data.dims
            if isinstance(data, sc.DataArray)
            else {dim for bank in data.values() for dim in bank.dims}
        )
        vertical = tuple(dim for dim in ("arc", "tube") if dim in dims)
        horizontal = tuple(dim for dim in ("channel", "length") if dim in dims)
        sliced = tuple(dim for dim in dims if dim not in (*vertical, *horizontal))
        super().__init__(
            data,
            dims=(*vertical, *horizontal),
            sliced=sliced,
            vertical=vertical,
            horizontal=horizontal,
            cbar=cbar,
            rasterized=rasterized,
            **kwargs,
        )
