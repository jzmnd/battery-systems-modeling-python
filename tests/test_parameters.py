import pytest

from battery_systems_modeling.parameters import CellParameters


def make_params(**kwargs) -> CellParameters:
    return CellParameters(Q=2.0, Ct=2700.0, Rt=0.011, Rs=0.008, Rd=1e6, **kwargs)


def test_rs_at_reference_temperature_returns_rs():
    params = make_params(T_ref=300.0, Ea_s=2e4)

    assert params.Rs_at(300.0) == pytest.approx(params.Rs)


def test_rs_at_is_constant_when_activation_energy_is_zero():
    params = make_params(Ea_s=0.0)

    assert params.Rs_at(260.0) == pytest.approx(params.Rs)
    assert params.Rs_at(330.0) == pytest.approx(params.Rs)


def test_rs_at_follows_arrhenius_and_increases_as_temperature_drops():
    params = make_params(T_ref=300.0, Ea_s=2e4)

    assert params.Rs_at(280.0) > params.Rs > params.Rs_at(320.0)
