"""The results as xarray datasets, labelled for plotting.

Every result has a ``to_dataset()``: named dimensions (``species``, ``reaction``,
``channel``, ``x``, ``state``), coordinates, and on every variable its
``long_name`` and ``units``, which plotting libraries such as plasma-plots and
xarray's own ``.plot`` put on the axes. :func:`stack` joins the results of a
scan along the scanned parameter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import xarray as xr

__all__ = ["stack", "variable"]


class _HasDataset(Protocol):
    def to_dataset(self) -> xr.Dataset: ...


def variable(dims, values, long_name: str, units: str = "") -> xr.Variable:
    """A labelled variable: ``dims``, ``values``, ``long_name`` and ``units``."""
    attrs = {"long_name": long_name}
    if units:
        attrs["units"] = units
    return xr.Variable(dims, np.asarray(values), attrs)


def stack(
    results: Sequence[_HasDataset],
    dim: str,
    values: Sequence[Any],
    *,
    long_name: str | None = None,
    units: str = "",
    attrs: Mapping[str, Any] | None = None,
) -> xr.Dataset:
    """The datasets of a scan joined along the scanned parameter.

    Args:
        results: The results, e.g. of :meth:`GlobalModel.scan`.
        dim: The name of the new dimension, e.g. ``"power"``.
        values: The parameter's value of every result.
        long_name: Its label; ``dim`` if ``None``.
        units: Its units.
        attrs: Further attributes of the dataset.
    """
    if len(results) != len(values):
        raise ValueError("one value per result")
    joined = xr.concat(
        [result.to_dataset() for result in results],
        dim=dim,
        coords="different",
        compat="equals",
        join="outer",
        combine_attrs="override",  # keeps every variable's long_name and units
    )
    coordinate = {"long_name": long_name or dim}
    if units:
        coordinate["units"] = units
    joined = joined.assign_coords({dim: (dim, np.asarray(values), coordinate)})
    joined.attrs = dict(attrs or {})
    return joined
