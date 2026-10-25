import numpy as np
import pytest

from battery_systems_modeling.inverter import CellModelInverter
from battery_systems_modeling.lookup_tables import OCVLookupTable, dOCVdTLookupTable
from battery_systems_modeling.parameters import CellLimits, CellParameters, ThermalParameters


POWER_ATOL = 1e-3  # Limited by the default tolerances of the ODE solver


@pytest.fixture
def soc():
    return np.linspace(0, 1, 6)


@pytest.fixture
def time():
    return np.linspace(0, 600, 61)


@pytest.fixture
def target_power(time):
    return np.select([time < 200, time < 400], [-4.0, 2.0], default=0.0)


@pytest.fixture
def t_amb():
    return 300.0


@pytest.fixture
def ocv_table(soc):
    return OCVLookupTable(Voc=3.0 + 1.2 * soc, soc=soc)


def make_params(**kwargs) -> CellParameters:
    defaults = dict(Q=2.0, Ct=2700.0, Rt=0.011, Rs=0.008, Rd=1e6)
    return CellParameters(**{**defaults, **kwargs})


def test_inverted_model_delivers_target_power(ocv_table, target_power, time):
    model = CellModelInverter(target_power, time, 0.5, make_params(), ocv_table).solve()

    np.testing.assert_allclose(model.cell_power, target_power, atol=POWER_ATOL)
    assert model.soc[-1] < 0.5


def test_inverted_thermal_model_delivers_target_power(ocv_table, soc, t_amb, target_power, time):
    params = make_params(Ea_s=2e4, T_ref=t_amb, thermal=ThermalParameters(Cp=50.0, hA=0.5))
    docv_dt = dOCVdTLookupTable(dVoc_dT=np.full_like(soc, 1e-4), soc=soc)

    model = CellModelInverter(
        target_power, time, 0.5, params, ocv_table, T0=t_amb, T_amb=t_amb, docv_dT_lu_table=docv_dt
    ).solve()

    np.testing.assert_allclose(model.cell_power, target_power, atol=POWER_ATOL)
    assert model.T[-1] > t_amb


def test_unreachable_power_raises(ocv_table, time):
    power = np.full_like(time, -1000.0)

    with pytest.raises(ValueError, match="No cell current found"):
        CellModelInverter(power, time, 0.5, make_params(), ocv_table).solve()


def test_limits_are_applied_to_the_returned_model(ocv_table, time):
    power = np.full_like(time, -4.0)
    limits = CellLimits(Vb_min=3.5, Vb_max=4.2)

    model = CellModelInverter(power, time, 0.5, make_params(), ocv_table, limits=limits).solve()

    assert model.termination_reason.startswith("cell limit reached")
    assert model.state.shape[1] < len(time)
