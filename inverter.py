"""Equivalent circuit cell modeling - System inversion

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
from scipy import optimize

from cell_model import CellParameters, OCVLookupTable, CellModel

LOGGER = logging.getLogger(__name__)


class CellModelInverter:

    """Performs inversion of the CellModel by solving for ib(t) given a target power"""

    time_init_lookback: float = 1.0

    def __init__(
        self,
        power: np.array,
        time: np.array,
        soc0: float,
        cell_params: CellParameters,
        ocv_lu_table: OCVLookupTable,
    ) -> None:
        """Initialize cell model inverter."""
        self.power = power
        self.time = time
        self.soc0 = soc0
        self.cell_params = cell_params
        self.ocv_lu_table = ocv_lu_table
        if not self.ocv_lu_table.is_interpolated():
            self.ocv_lu_table.interpolate()

    def _error(self, ib_new: float, ib_prev: np.array, power: np.array, time: np.array) -> float:
        """Returns the SSE for the difference between model power and set power."""
        ib = np.append(ib_prev, ib_new)
        model = CellModel(ib, time, self.soc0, self.cell_params, self.ocv_lu_table).solve()
        return sum((model.cell_power - power) ** 2)

    def _init_solution_vars(self) -> None:
        """Initialize the solution variables. We initialize ib with a value of 0.
        The time series is initialized with a value of `time_init_lookback` before the first time
        which is then prepended to `time`.
        The power series is initialized with a value equal to the first value which is then
        prepended to `power`.
        """
        self._ib_soln = np.array([0])
        self._time_soln = np.append(self.time[0] - self.time_init_lookback, self.time)
        self._power_soln = np.append(self.power[0], self.power)

    def solve(self) -> CellModel:
        """Iteratively solve the plant inversion problem. For each time step we minimize the error
        between set power and cell model power for the new value of ib(t). We then save that
        value of ib(t) and use it in the next time step t+1 to to find ib(t+1).
        """
        self._init_solution_vars()

        for i in range(1, len(self._time_soln)):
            result = optimize.minimize(
                self._error,
                x0=self._ib_soln[-1],
                args=(self._ib_soln, self._power_soln[:i+1], self._time_soln[:i+1]),
            )
            self._ib_soln = np.append(self._ib_soln, result.x[0])

        self._ib_soln = self._ib_soln[1:]

        return CellModel(
            self._ib_soln, self.time, self.soc0, self.cell_params, self.ocv_lu_table
        ).solve()
