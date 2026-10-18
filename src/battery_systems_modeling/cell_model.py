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

import numpy as np
from scipy import interpolate, integrate

from battery_systems_modeling.parameters import CellParameters
from battery_systems_modeling.lookup_tables import OCVLookupTable

LOGGER = logging.getLogger(__name__)
DEFAULT_SOLVER = "DOP853"


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
        self.sol: integrate.OdeResult | None = None
        self.Vb: np.array | None = None

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

    def solve(self, method: str = DEFAULT_SOLVER) -> CellModel:
        """Solve for the cell voltage."""
        W0 = np.array([self.soc0, 0])
        trange = (self.time[0], self.time[-1])

        self.sol = integrate.solve_ivp(
            self.dW_dt, trange, W0, method=method, t_eval=self.time, dense_output=True
        )

        LOGGER.info(self.sol.message)
        self.Vb = self.cell_output(self.sol.y)

        return self
