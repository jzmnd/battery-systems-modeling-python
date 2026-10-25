from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R_GAS: float = 8.31446261815324


@dataclass
class CellParameters:
    """Defines the Cell Parameters."""

    Q: float  # Cell capacity (Ah)
    Ct: float  # Double layer capacitance (F)
    Rt: float  # SEI layer charge transfer resistance (Ohms)
    Rs: float  # Cell internal resistance (Ohms)
    Rd: float  # Self-discharge resistance (Ohms)

    T_ref: float = 300.0  # Reference temperature for cell parameters (K)
    Ea_s: float = 0.0  # Arrhenius rate for the temperature dependence of Rs (J/mol)

    thermal: ThermalParameters | None = None

    def Rs_at(self, T: float) -> float:
        """Cell internal resistance (Ohms) at an arbitrary temperature T (K)."""
        return self.Rs * np.exp((self.Ea_s / R_GAS) * (1 / T - 1 / self.T_ref))


@dataclass
class ThermalParameters:
    """Defines the Cell thermal properties."""

    Cp: float  # Heat capacity (J/K)
    hA: float  # Convective heat transfer rate (W/K)


@dataclass
class CellLimits:
    """Defines the Cell limits."""

    Vb_min: float = 0.0  # Minimum cell voltage (V)
    Vb_max: float = 15.0  # Maximum cell voltage (V)

    soc_min: float = 0.0  # Minimum cell SOC
    soc_max: float = 1.0  # Maximum cell SOC

    T_max: float = 400.0  # Maximum cell temperature (K)
