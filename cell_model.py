"""Equivalent circuit cell modeling

Reference:
C. D. Rahn and C.Y. Wang, Battery Systems Engineering. Oxford, UK: John Wiley & Sons Ltd, 2013

Model:
    Vb = Voc(SOC(t)) + ib(t) * Rs + ∆vt(t)

Where:
    Vb is the cell voltage
    Voc is the open circuit voltage which is a function of SOC dependent on cell chemistry
    ib(t) is the cell current
    Rs is the cell internal resistance
    ∆vt(t) is the voltage across the SEI layer
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import interpolate, integrate

LOGGER = logging.getLogger(__name__)


@dataclass
class CellParameters:

    """Defines the Cell Parameters."""

    Q: float  # Cell capacity (Ah)
    Ct: float  # Double layer capacitance (F)
    Rt: float  # SEI layer charge transfer resistance (Ohms)
    Rs: float  # Cell internal resistance (Ohms)
    Rd: float  # Self-discharge resistance  (Ohms)


@dataclass
class OCVLookupTable:

    """Open circuit voltage (Voc) lookup table. Interpolates the Voc vs SOC function."""

    Voc: np.array  # Open circuit voltages (V)
    soc: np.array  # SOCs (0,1)

    def __post_init__(self) -> None:
        """Initialize Voc_int."""
        self.Voc_int: Optional[np.array] = None

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


class CellModel:

    """Cell Model Definition."""

    seconds_in_hour: int = 3600

    def __init__(
        self,
        ib: np.array,
        time: np.array,
        soc0: float,
        cell_params: CellParameters,
        ocv_lu_table: OCVLookupTable,
    ) -> None:
        """Initialize cell model and perform interpolations."""
        self.ib = ib
        self.time = time
        self.soc0 = soc0
        self.cell_params = cell_params
        self.ocv_lu_table = ocv_lu_table
        if not self.ocv_lu_table.is_interpolated():
            self.ocv_lu_table.interpolate()
        self._interpolate_ib()
        self.sol: Optional[integrate.OdeResult] = None
        self.Vb: Optional[np.array] = None

    def _interpolate_ib(self) -> None:
        """Interpolate the ib(t) function."""
        self.ib_int = interpolate.interp1d(
            self.time, self.ib, kind="linear", fill_value="extrapolate"
        )

    def voc_int(self, soc: float) -> float:
        """Use the OCV lookup table to get Voc for a given SOC"""
        return self.ocv_lu_table.Voc_int(soc)

    def dW_dt(self, t: float, W: np.array) -> np.array:
        """Differential equations in W(t) where `W(t) = [SOC(t), ∆vt(t)]`"""
        ib = self.ib_int(t)
        q = self.cell_params.Q * self.seconds_in_hour
        Voc = self.voc_int(W[0])

        return np.array(
            [
                (ib - Voc / self.cell_params.Rd) * (1 / q),
                (ib - W[1] / self.cell_params.Rt) * (1 / self.cell_params.Ct),
            ]
        )

    def cell_output(self, W: np.array) -> np.array:
        """Cell output voltage given W(t) where `W(t) = [SOC(t), ∆vt(t)]`"""
        ib = self.ib_int(self.time)
        Voc = self.voc_int(W[0])

        return Voc + ib * self.cell_params.Rs + W[1]

    @property
    def soc(self) -> np.array:
        """Cell SOC(t)."""
        return self.sol.y[0]

    @property
    def soe(self) -> np.array:
        """Cell SOE(t)."""
        return self.ocv_lu_table.integrate(self.soc) / self.ocv_lu_table.integrate(1)

    @property
    def cell_power(self) -> np.array:
        """Cell output power."""
        return self.ib * self.Vb

    def solve(self, method: str = "DOP853") -> CellModel:
        """Solve for the cell voltage."""
        W0 = np.array([self.soc0, 0])
        trange = (self.time[0], self.time[-1])

        self.sol = integrate.solve_ivp(
            self.dW_dt, trange, W0, method=method, t_eval=self.time, dense_output=True
        )

        LOGGER.info(self.sol.message)
        self.Vb = self.cell_output(self.sol.y)

        return self
