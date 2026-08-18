"""
Field-table (approximate) mode and the survey helper.
"""

import numpy as np
import pytest

import jetfuncs as jf

numba_only = pytest.mark.skipif(not jf.numba_available(), reason="numba is not installed")

CFG = dict(
    m=6.5e9,
    a=0.9,
    inc=163.0,
    mdot=5.45e-5,
    s=0.6,
    p=2.14,
    h=0.001,
    eta=0.03,
    gamma_inf=4.0,
    Nx=24,
    Ny=24,
    Nz=120,
    xmin=0.3,
    xmax=3.0e3,
    ymin=0.3,
    ymax=3.0e3,
    zmin=0.3,
    zmax=3.0e5,
    use_log_xgrid=True,
    use_log_ygrid=True,
    use_log_zgrid=True,
)


def _total_and_pixel_error(I_ref, I_test):
    good = I_ref > I_ref.max() * 1e-6
    rel = np.abs(I_test - I_ref)[good] / I_ref[good]
    return abs(I_test.sum() / I_ref.sum() - 1.0), np.percentile(rel, 99)


# ---------------------------------------------------------------- field table
@numba_only
def test_field_table_accuracy_and_convergence():
    model = jf.JetModel(**CFG)
    _, _, I_exact = model.make_image(230.0)
    errs = {}
    for ppd, n_u in ((50, 64), (200, 256)):
        model.build_field_table(points_per_decade=ppd, n_u=n_u)
        _, _, I_tab = model.make_image(230.0)
        errs[ppd] = _total_and_pixel_error(I_exact, I_tab)
        assert np.all(np.isfinite(I_tab))
        assert np.array_equal(I_exact == 0.0, I_tab == 0.0)
    # bilinear interpolation: the error falls with resolution (h^2 on production grids;
    # this deliberately coarse test grid converges more slowly), and stays small
    assert errs[200][0] < errs[50][0]
    assert errs[200][1] < errs[50][1]
    assert errs[200][0] < 1e-3
    assert errs[200][1] < 3e-3
    # clearing the table restores the exact kernel
    model.clear_field_table()
    _, _, I_again = model.make_image(230.0)
    np.testing.assert_allclose(I_again, I_exact, rtol=1e-11, atol=0.0)


@numba_only
def test_field_table_used_by_precompute_state():
    model = jf.JetModel(**CFG)
    model.build_field_table(points_per_decade=100, n_u=128)
    _, _, I_tab = model.make_image(43.0)
    model.precompute_state()
    _, _, I_cached = model.make_image(43.0)
    np.testing.assert_allclose(I_cached, I_tab, rtol=1e-11, atol=0.0)
    model.clear_state()
    model.clear_field_table()


@numba_only
def test_field_table_memory_guard_and_heating():
    model = jf.JetModel(**CFG)
    with pytest.raises(MemoryError):
        model.build_field_table(max_memory_gb=1e-9)
    with pytest.raises(ValueError):
        model.build_field_table(heating_prescription="poynting")
    # a table built for one heating prescription is not used for the other
    model.build_field_table(points_per_decade=50, n_u=64, heating_prescription="magnetic")
    _, _, I_mag_tab = model.make_image(43.0, heating_prescription="magnetic")
    _, _, I_poy = model.make_image(43.0)  # falls back to the exact kernel
    _, _, I_poy_np = model.make_image(43.0, backend="numpy")
    np.testing.assert_allclose(I_poy, I_poy_np, rtol=1e-9, atol=0.0)
    assert np.isfinite(I_mag_tab).all()


# ---------------------------------------------------------------- survey
def _flux(model, frequency=230.0):
    return float(model.make_image(frequency, backend="numpy")[2].sum())


def test_survey_in_process_and_spawned():
    # (vary mdot rather than h: this tiny model is optically thick, so h cancels in j/alpha)
    cfgs = [
        dict(
            m=10.0,
            a=0.5,
            inc=60.0,
            mdot=md,
            Nx=6,
            Ny=6,
            Nz=24,
            xmin=-5.0,
            xmax=5.0,
            ymin=-5.0,
            ymax=5.0,
            zmin=-20.0,
            zmax=20.0,
        )
        for md in (1e-3, 2e-3, 4e-3)
    ]
    serial = jf.survey(cfgs, _flux, n_workers=1)
    assert len(serial) == 3 and len(set(serial)) == 3
    parallel = jf.survey(cfgs, _flux, n_workers=2, threads_per_worker=1)
    assert parallel == serial
    kw = jf.survey(cfgs[:1], _flux, n_workers=1, func_kwargs=dict(frequency=43.0))
    assert kw[0] != serial[0]
