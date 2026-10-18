from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import interpolate, integrate


@dataclass
class OCVLookupTable:
    """Open circuit voltage (Voc) lookup table. Interpolates the Voc vs SOC function."""

    Voc: np.array  # Open circuit voltages (V)
    soc: np.array  # SOCs (0,1)

    def __post_init__(self) -> None:
        """Initialize Voc_int."""
        self.Voc_int: np.array | None = None

    def is_interpolated(self) -> bool:
        """Tests if the lookup table has already been interpolated."""
        return self.Voc_int is not None

    def interpolate(self) -> None:
        """Interpolate the Voc(SOC) function."""
        self.Voc_int = interpolate.interp1d(
            self.soc, self.Voc, kind="cubic", fill_value="extrapolate"
        )
        self.integrate = np.vectorize(self._integrate)

    def _integrate(self, soc: float) -> float:
        """Calculate the integral of Voc(SOC) between 0 and a given SOC."""
        return integrate.quad(self.Voc_int, 0, soc)[0]
