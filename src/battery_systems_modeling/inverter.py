"""Equivalent circuit cell modeling - System inversion

Reference:
C. D. Rahn and C.Y. Wang, Battery Systems Engineering. Oxford, UK: John Wiley & Sons Ltd, 2013

Model:
    Vb = Voc(SOC(t)) + ib(t) * Rs(T) + ∆vt(t)

Where:
    Vb is the cell voltage
    Voc is the open circuit voltage which is a function of SOC dependent on cell chemistry
    ib(t) is the cell current (positive when charging)
    Rs(T) is the cell internal resistance, which depends on the cell temperature T
    ∆vt(t) is the voltage across the SEI layer
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import optimize

from battery_systems_modeling.cell_model import CellModel
from battery_systems_modeling.parameters import CellLimits, CellParameters
from battery_systems_modeling.lookup_tables import OCVLookupTable, dOCVdTLookupTable

LOGGER = logging.getLogger(__name__)
ROOT_GUESS_OFFSET = 1e-3  # Offset (A) between the two initial guesses of the secant root finder


class CellModelInverter:
    """Performs inversion of the CellModel by solving for ib(t) given a target power.

    The arguments other than `power` are the same as for `CellModel`. Cell limits are not applied
    while solving for ib(t) but are applied to the returned `CellModel`.
    """

    def __init__(
        self,
        power: np.array,
        time: np.array,
        soc0: float,
        cell_params: CellParameters,
        ocv_lu_table: OCVLookupTable,
        T0: float | None = None,
        T_amb: float | np.array | None = None,
        docv_dT_lu_table: dOCVdTLookupTable | None = None,
        limits: CellLimits | None = None,
    ) -> None:
        """Initialize cell model inverter."""
        self.power = power
        self.time = time
        self.soc0 = soc0
        self.cell_params = cell_params
        self.ocv_lu_table = ocv_lu_table
        self.T0 = T0
        self.T_amb = None if T_amb is None else np.broadcast_to(T_amb, np.shape(time))
        self.docv_dT_lu_table = docv_dT_lu_table
        self.limits = limits

        if not self.ocv_lu_table.is_interpolated():
            self.ocv_lu_table.interpolate()

    def _build_model(self, ib: np.array, time: np.array, **kwargs) -> CellModel:
        """Build a `CellModel` for the given current and time series."""
        return CellModel(
            ib,
            time,
            kwargs.pop("soc0", self.soc0),
            self.cell_params,
            self.ocv_lu_table,
            docv_dT_lu_table=self.docv_dT_lu_table,
            **kwargs,
        )

    def _initial_state(self) -> np.array:
        """Cell state W at the first time step."""
        W0 = [self.soc0, 0.0]
        if self.cell_params.thermal is not None:
            W0.append(self.T0)
        return np.array(W0)

    def _initial_voltage(self, ib: float, W: np.array) -> float:
        """Cell voltage at the first time step, where there is no time to integrate over."""
        T = W[2] if self.cell_params.thermal is not None else self.cell_params.T_ref
        return self.ocv_lu_table.Voc_int(W[0]) + ib * self.cell_params.Rs_at(T) + W[1]

    def _step(self, k: int, ib_prev: float, ib_new: float, W_prev: np.array) -> tuple:
        """Integrate from time step k-1 to k with ib(t) linear between `ib_prev` and `ib_new`.
        Returns the cell state W and the cell voltage Vb at time step k.
        """
        if k == 0:
            return W_prev, self._initial_voltage(ib_new, W_prev)

        model = self._build_model(
            np.array([ib_prev, ib_new]),
            self.time[k - 1 : k + 1],
            soc0=W_prev[0],
            dvt0=W_prev[1],
            T0=W_prev[2] if self.cell_params.thermal is not None else None,
            T_amb=None if self.T_amb is None else self.T_amb[k - 1 : k + 1],
        ).solve()

        return model.state[:, -1], model.Vb[-1]

    def _solve_step(self, k: int, ib_prev: float, W_prev: np.array) -> tuple[float, np.array]:
        """Solve for ib at time step k such that the cell power equals the target power.
        Returns ib and the cell state W at time step k.
        """

        def power_error(ib_new: float) -> float:
            return ib_new * self._step(k, ib_prev, ib_new, W_prev)[1] - self.power[k]

        root = optimize.root_scalar(
            power_error, x0=ib_prev, x1=ib_prev + ROOT_GUESS_OFFSET, method="secant"
        )
        if not root.converged:
            raise ValueError(
                f"No cell current found for target power {self.power[k]} W at t={self.time[k]} s"
            )

        return root.root, self._step(k, ib_prev, root.root, W_prev)[0]

    def solve(self) -> CellModel:
        """Iteratively solve the plant inversion problem. For each time step we find the value of
        ib(t) for which the cell model power equals the target power, starting from the cell state
        at the previous time step. The resulting ib(t) is then used to solve the `CellModel`.
        """
        ib = np.zeros(len(self.time))
        W = self._initial_state()

        for k in range(len(self.time)):
            ib[k], W = self._solve_step(k, ib[k - 1] if k else 0.0, W)

        return self._build_model(
            ib, self.time, T0=self.T0, T_amb=self.T_amb, limits=self.limits
        ).solve()
