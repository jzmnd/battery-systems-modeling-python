import numpy as np
import pytest

from battery_systems_modeling.lookup_tables import OCVLookupTable, dOCVdTLookupTable


@pytest.fixture
def soc():
    return np.linspace(0, 1, 6)


def test_docv_dt_not_interpolated_until_requested(soc):
    table = dOCVdTLookupTable(dVoc_dT=np.zeros_like(soc), soc=soc)
    assert not table.is_interpolated()
    assert table.dVoc_dT_int is None

    table.interpolate()

    assert table.is_interpolated()


def test_docv_dt_interpolation_reproduces_table_nodes_and_midpoints(soc):
    dvoc_dt = 1e-4 * (1 + soc)
    table = dOCVdTLookupTable(dVoc_dT=dvoc_dt, soc=soc)
    table.interpolate()

    np.testing.assert_allclose(table.dVoc_dT_int(soc), dvoc_dt)
    assert table.dVoc_dT_int(0.5) == pytest.approx(1.5e-4)


def test_ocv_not_interpolated_until_requested(soc):
    table = OCVLookupTable(Voc=3.0 + soc, soc=soc)
    assert not table.is_interpolated()
    assert table.Voc_int is None

    table.interpolate()

    assert table.is_interpolated()


def test_ocv_interpolation_reproduces_table_nodes_and_midpoints(soc):
    voc = 3.0 + 0.5 * soc
    table = OCVLookupTable(Voc=voc, soc=soc)
    table.interpolate()

    np.testing.assert_allclose(table.Voc_int(soc), voc)
    assert table.Voc_int(0.5) == pytest.approx(3.25)


def test_ocv_integrate_matches_analytic_integral(soc):
    table = OCVLookupTable(Voc=3.0 + soc, soc=soc)
    table.interpolate()

    assert table.integrate(0.5) == pytest.approx(3.0 * 0.5 + 0.5**2 / 2)
