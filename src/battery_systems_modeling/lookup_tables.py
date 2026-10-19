from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import interpolate, integrate


class _SocLookupTable:
    """Base class for lookup tables of a quantity vs SOC."""

    soc: np.array
    _interpolant: Callable | None = None

    def is_interpolated(self) -> bool:
        """Tests if the lookup table has already been interpolated."""
        return self._interpolant is not None

    def _interpolate(self, values: np.array) -> None:
        """Build the cubic interpolant of `values` vs SOC."""
        self._interpolant = interpolate.interp1d(
            self.soc, values, kind="cubic", fill_value="extrapolate"
        )


@dataclass
class OCVLookupTable(_SocLookupTable):
    """Open circuit voltage (Voc) lookup table. Interpolates the Voc vs SOC function."""

    Voc: np.array  # Open circuit voltages (V)
    soc: np.array  # SOCs (0,1)

    @property
    def Voc_int(self) -> Callable | None:
        """Interpolated Voc(SOC) function."""
        return self._interpolant

    def interpolate(self) -> None:
        """Interpolate the Voc(SOC) function."""
        self._interpolate(self.Voc)
        self.integrate = np.vectorize(self._integrate)

    def _integrate(self, soc: float) -> float:
        """Calculate the integral of Voc(SOC) between 0 and a given SOC."""
        return integrate.quad(self.Voc_int, 0, soc)[0]


@dataclass
class dOCVdTLookupTable(_SocLookupTable):
    """Entropic coefficient lookup table. Interpolates the dVoc/dT vs SOC function."""

    dVoc_dT: np.array  # Open circuit voltage temperature derivatives (V/K)
    soc: np.array  # SOCs (0,1)

    @property
    def dVoc_dT_int(self) -> Callable | None:
        """Interpolated dVoc/dT(SOC) function."""
        return self._interpolant

    def interpolate(self) -> None:
        """Interpolate the dVoc/dT(SOC) function."""
        self._interpolate(self.dVoc_dT)
