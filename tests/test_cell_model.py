import numpy as np
import pytest

from battery_systems_modeling.cell_model import CellModel
from battery_systems_modeling.lookup_tables import OCVLookupTable, dOCVdTLookupTable
from battery_systems_modeling.parameters import CellLimits, CellParameters, ThermalParameters


@pytest.fixture
def soc():
    return np.linspace(0, 1, 6)


@pytest.fixture
def t_amb():
    return 300.0


@pytest.fixture
def ocv_table(soc):
    return OCVLookupTable(Voc=3.0 + 1.2 * soc, soc=soc)


def make_params(**kwargs) -> CellParameters:
    defaults = dict(Q=2.0, Ct=2700.0, Rt=0.011, Rs=0.008, Rd=1e12)
    return CellParameters(**{**defaults, **kwargs})


def constant_current(ib: float, duration: float, n: int = 11) -> tuple[np.ndarray, np.ndarray]:
    time = np.linspace(0, duration, n)
    return np.full_like(time, ib), time


def test_thermal_parameters_require_initial_and_ambient_temperature(ocv_table, t_amb):
    ib, time = constant_current(0.0, 10.0)
    params = make_params(thermal=ThermalParameters(Cp=100.0, hA=1.0))

    with pytest.raises(ValueError):
        CellModel(ib, time, 0.5, params, ocv_table, T0=t_amb)


def test_constant_current_changes_soc_by_charge_throughput(ocv_table):
    ib, time = constant_current(-1.0, 600.0)

    model = CellModel(ib, time, 0.5, make_params(), ocv_table).solve()

    assert model.soc[-1] == pytest.approx(0.5 - 600.0 / (2.0 * 3600.0))


def test_isothermal_model_has_constant_reference_temperature(ocv_table):
    ib, time = constant_current(-1.0, 60.0)

    model = CellModel(ib, time, 0.5, make_params(T_ref=295.0), ocv_table).solve()

    np.testing.assert_allclose(model.T, 295.0)


def test_zero_current_cell_relaxes_to_ambient(ocv_table, t_amb):
    ib, time = constant_current(0.0, 100.0)
    params = make_params(thermal=ThermalParameters(Cp=100.0, hA=1.0))

    model = CellModel(ib, time, 0.5, params, ocv_table, T0=310.0, T_amb=t_amb).solve()

    assert model.T[-1] == pytest.approx(t_amb + 10.0 * np.exp(-1.0), rel=1e-6)


def test_adiabatic_cell_heats_by_joule_heating(ocv_table, t_amb):
    ib, time = constant_current(-2.0, 10.0)
    params = make_params(Rt=1e9, thermal=ThermalParameters(Cp=50.0, hA=0.0))

    model = CellModel(ib, time, 0.5, params, ocv_table, T0=t_amb, T_amb=t_amb).solve()

    assert model.T[-1] - t_amb == pytest.approx(2.0**2 * params.Rs * 10.0 / 50.0)


def test_joule_heating_increases_as_temperature_drops(ocv_table, t_amb):
    ib, time = constant_current(-2.0, 10.0)
    thermal = ThermalParameters(Cp=50.0, hA=0.0)
    params = make_params(Rt=1e9, T_ref=t_amb, Ea_s=3e4, thermal=thermal)

    cold = CellModel(ib, time, 0.5, params, ocv_table, T0=t_amb - 30, T_amb=t_amb).solve()
    warm = CellModel(ib, time, 0.5, params, ocv_table, T0=t_amb + 30, T_amb=t_amb).solve()

    assert cold.T[-1] - cold.T[0] > warm.T[-1] - warm.T[0]


def test_entropic_heating_is_only_included_with_dvoc_dt_table(ocv_table, soc, t_amb):
    ib, time = constant_current(1.0, 10.0)
    params = make_params(Rs=1e-9, Rt=1e9, thermal=ThermalParameters(Cp=100.0, hA=0.0))
    dvoc_dt = dOCVdTLookupTable(dVoc_dT=np.full_like(soc, 1e-4), soc=soc)
    args = (ib, time, 0.5, params, ocv_table)

    without = CellModel(*args, T0=t_amb, T_amb=t_amb).solve()
    with_table = CellModel(*args, T0=t_amb, T_amb=t_amb, docv_dT_lu_table=dvoc_dt).solve()

    assert without.T[-1] == pytest.approx(t_amb, abs=1e-6)
    assert with_table.T[-1] - t_amb == pytest.approx(1.0 * t_amb * 1e-4 * 10.0 / 100.0, rel=1e-2)


def test_discharge_terminates_at_minimum_voltage(ocv_table):
    ib, time = constant_current(-2.0, 3000.0, n=301)
    limits = CellLimits(Vb_min=3.3, Vb_max=4.2)

    model = CellModel(ib, time, 0.5, make_params(), ocv_table, limits=limits).solve()

    assert model.termination_reason == "cell limit reached: Vb_min"
    assert model._sol.t[-1] < time[-1]
    assert len(model.Vb) == len(model._sol.t)
    assert model.Vb[-1] == pytest.approx(3.3, abs=5e-3)


def test_no_termination_when_limits_are_not_reached(ocv_table):
    ib, time = constant_current(-1.0, 60.0)
    limits = CellLimits(Vb_min=3.0, Vb_max=4.2)

    model = CellModel(ib, time, 0.5, make_params(), ocv_table, limits=limits).solve()

    assert (
        model.termination_reason
        == "solve_ivp terminated: The solver successfully reached the end of the integration interval."
    )
    assert model._sol.t[-1] == time[-1]


def test_temperature_limit_terminates_solution(ocv_table, t_amb):
    ib, time = constant_current(-2.0, 100.0, n=101)
    params = make_params(Rt=1e9, thermal=ThermalParameters(Cp=1.0, hA=0.0))
    limits = CellLimits(Vb_min=0.0, Vb_max=10.0, T_max=t_amb + 2.0)

    model = CellModel(
        ib, time, 0.5, params, ocv_table, T0=t_amb, T_amb=t_amb, limits=limits
    ).solve()

    assert model.termination_reason == "cell limit reached: T_max"
