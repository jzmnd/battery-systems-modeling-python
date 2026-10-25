"""Equivalent circuit cell modeling

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

Lumped thermal model (when `CellParameters.thermal` is set):
    Cp dT/dt = ib² Rs(T) + ∆vt² / Rt + ib T dVoc/dT(SOC) - hA (T - T_amb(t))

The entropic term is only included when a `dOCVdTLookupTable` is provided.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import interpolate, integrate

from battery_systems_modeling.parameters import CellLimits, CellParameters
from battery_systems_modeling.lookup_tables import OCVLookupTable, dOCVdTLookupTable

LOGGER = logging.getLogger(__name__)
DEFAULT_SOLVER = "DOP853"
EVENT_TERMINATION = 1  # solve_ivp status when a terminal event occurred


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
        dvt0: float = 0.0,
        T0: float | None = None,
        T_amb: float | np.array | None = None,
        docv_dT_lu_table: dOCVdTLookupTable | None = None,
        limits: CellLimits | None = None,
    ) -> None:
        """Initialize cell model and perform interpolations.

        `soc0` is the initial cell SOC and `dvt0` the initial voltage across the SEI layer.
        The cell paramters are defined in `cell_params`.
        The dependency of Voc on SOC is set by the `ocv_lu_table` lookup table.
        `T0` is the initial cell temperature (K) and `T_amb` the ambient temperature (K), either a
        scalar or an array on the `time` grid. Both are required when `cell_params.thermal` is set.
        """
        if cell_params.thermal is not None and (T0 is None or T_amb is None):
            raise ValueError("T0 and T_amb are required when thermal parameters are set")

        self.ib = ib
        self.time = time
        self.soc0 = soc0
        self.cell_params = cell_params
        self.ocv_lu_table = ocv_lu_table
        self.dvt0 = dvt0
        self.T0 = T0
        self.T_amb = None if T_amb is None else np.broadcast_to(T_amb, np.shape(time))
        self.docv_dT_lu_table = docv_dT_lu_table
        self.limits = limits or CellLimits()

        if not self.ocv_lu_table.is_interpolated():
            self.ocv_lu_table.interpolate()
        if self.docv_dT_lu_table is not None and not self.docv_dT_lu_table.is_interpolated():
            self.docv_dT_lu_table.interpolate()

        self._interpolate_ib()
        self._interpolate_T_amb()
        self._sol: integrate.OdeResult | None = None
        self.Vb: np.array | None = None
        self.termination_reason: str | None = None

    @property
    def is_thermal(self) -> bool:
        """True if the cell temperature is a state of the model."""
        return self.cell_params.thermal is not None

    def _interpolate_ib(self) -> None:
        """Interpolate the ib(t) function."""
        self.ib_int = interpolate.interp1d(
            self.time, self.ib, kind="linear", fill_value="extrapolate"
        )

    def _interpolate_T_amb(self) -> None:
        """Interpolate the ambient temperature T_amb(t) function."""
        if self.T_amb is None:
            self.T_amb_int = None
            return
        self.T_amb_int = interpolate.interp1d(
            self.time, self.T_amb, kind="linear", fill_value="extrapolate"
        )

    def voc_int(self, soc: float) -> float:
        """Use the OCV lookup table to get Voc for a given SOC"""
        return self.ocv_lu_table.Voc_int(soc)

    def _temperature(self, W: np.array) -> float | np.array:
        """Cell temperature (K) from the state W."""
        return W[2] if self.is_thermal else self.cell_params.T_ref

    def dW_dt(self, t: float, W: np.array) -> np.array:
        """Differential equations in W(t) where `W(t) = [SOC(t), ∆vt(t), T(t)]`"""
        ib = self.ib_int(t)
        q = self.cell_params.Q * self.seconds_in_hour
        Voc = self.voc_int(W[0])

        dW = [
            (ib - Voc / self.cell_params.Rd) * (1 / q),
            (ib - W[1] / self.cell_params.Rt) * (1 / self.cell_params.Ct),
        ]
        if self.is_thermal:
            dW.append(self._dT_dt(t, W, ib))

        return np.array(dW)

    def _dT_dt(self, t: float, W: np.array, ib: float) -> float:
        """Rate of change of the cell temperature with time."""
        T = W[2]
        heat = ib**2 * self.cell_params.Rs_at(T) + W[1] ** 2 / self.cell_params.Rt
        if self.docv_dT_lu_table is not None:
            heat += ib * T * self.docv_dT_lu_table.dVoc_dT_int(W[0])

        return (
            heat - self.cell_params.thermal.hA * (T - self.T_amb_int(t))
        ) / self.cell_params.thermal.Cp

    def cell_output(self, t: np.array, W: np.array) -> np.array:
        """Cell output voltage given time t and W(t) where `W(t) = [SOC(t), ∆vt(t), T(t)]`"""
        ib = self.ib_int(t)
        Voc = self.voc_int(W[0])

        return Voc + ib * self.cell_params.Rs_at(self._temperature(W)) + W[1]

    def _events(self) -> list:
        """Terminal `solve_ivp` events for the cell limits, each named by its `limit` attribute."""

        def make_event(limit: str, fn, direction: int):
            fn.limit = limit
            fn.terminal = True
            fn.direction = direction
            return fn

        limits = self.limits
        events = [
            make_event("Vb_min", lambda t, W: self.cell_output(t, W) - limits.Vb_min, -1),
            make_event("Vb_max", lambda t, W: self.cell_output(t, W) - limits.Vb_max, 1),
            make_event("soc_min", lambda t, W: W[0] - limits.soc_min, -1),
            make_event("soc_max", lambda t, W: W[0] - limits.soc_max, 1),
        ]
        if self.is_thermal:
            events.append(make_event("T_max", lambda t, W: W[2] - limits.T_max, 1))

        return events

    @property
    def state(self) -> np.array:
        """Cell state W(t) where `W(t) = [SOC(t), ∆vt(t), T(t)]` (T only when thermal)."""
        return self._sol.y

    @property
    def soc(self) -> np.array:
        """Cell SOC(t)."""
        return self._sol.y[0]

    @property
    def T(self) -> np.array:
        """Cell temperature T(t) (K)."""
        return self._temperature(self._sol.y) * np.ones_like(self.soc)

    @property
    def soe(self) -> np.array:
        """Cell SOE(t)."""
        return self.ocv_lu_table.integrate(self.soc) / self.ocv_lu_table.integrate(1)

    @property
    def cell_power(self) -> np.array:
        """Cell output power."""
        return self.ib_int(self._sol.t) * self.Vb

    def solve(self, method: str = DEFAULT_SOLVER) -> CellModel:
        """Solve for the cell voltage. If a cell limit is reached the solution ends at that time
        and `termination_reason` is set.
        """
        self.termination_reason = None
        W0 = [self.soc0, self.dvt0]
        if self.is_thermal:
            W0.append(self.T0)
        trange = (self.time[0], self.time[-1])
        events = self._events()

        self._sol = integrate.solve_ivp(
            self.dW_dt,
            trange,
            np.array(W0),
            method=method,
            t_eval=self.time,
            dense_output=True,
            events=events or None,
        )

        LOGGER.info(self._sol.message)
        self.termination_reason = self._termination_reason(events)
        self.Vb = self.cell_output(self._sol.t, self._sol.y)

        return self

    def _termination_reason(self, events: list) -> str:
        """Describe why the solver stopped."""
        if self._sol.status == EVENT_TERMINATION:
            for event, t_event in zip(events, self._sol.t_events):
                if len(t_event):
                    return f"cell limit reached: {event.limit}"
        return f"solve_ivp terminated: {self._sol.message}"
