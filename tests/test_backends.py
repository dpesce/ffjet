"""
The compiled (numba) back end must reproduce the numpy reference implementation.

These tests use deliberately small grids; the first numba call in a session compiles
the kernels (several seconds), after which everything is fast.
"""

import numpy as np
import pytest

import jetfuncs as jf

numba_only = pytest.mark.skipif(not jf.numba_available(), reason="numba is not installed")

# a linear grid at the M87-like fiducial parameters, and a log grid exercising the
# non-default code paths (general anisotropy exponent, jet cutout, magnetic heating,
# a different inclination and spin)
CONFIGS = {
    "linear": dict(
        m=6.5e9,
        a=0.9,
        inc=163.0,
        mdot=5.45e-5,
        s=0.6,
        p=2.14,
        h=0.001,
        eta=0.03,
        gamma_inf=4.0,
        Nx=16,
        Ny=16,
        Nz=96,
        xmin=-60.0,
        xmax=60.0,
        ymin=-60.0,
        ymax=60.0,
        zmin=-240.0,
        zmax=240.0,
    ),
    "log": dict(
        m=4.1e6,
        a=0.5,
        inc=30.0,
        mdot=7e-7,
        s=0.7,
        p=2.8,
        h=0.003,
        eta=0.1,
        p_eta=2.3,
        gamma_inf=6.0,
        gamma_m=20.0,
        gamma_max=10**2.5,
        jet_cutout_fraction=0.3,
        Nx=16,
        Ny=16,
        Nz=96,
        xmin=0.3,
        xmax=1e4,
        ymin=0.3,
        ymax=1e4,
        zmin=0.3,
        zmax=1e6,
        use_log_xgrid=True,
        use_log_ygrid=True,
        use_log_zgrid=True,
    ),
}


def _compare(I_ref, I_test, rtol):
    assert I_ref.shape == I_test.shape
    assert np.all(np.isfinite(I_test))
    # the two back ends must integrate exactly the same set of cells ...
    assert np.array_equal(I_ref == 0.0, I_test == 0.0)
    # ... and agree to roundoff where there is emission
    good = I_ref > I_ref.max() * 1e-12
    assert good.sum() > 0
    np.testing.assert_allclose(I_test[good], I_ref[good], rtol=rtol, atol=0.0)


@numba_only
@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("frequency", [8.0, 230.0, 3.0e4])
def test_numba_matches_numpy(name, frequency):
    model = jf.JetModel(**CONFIGS[name])
    _, _, I_np = model.make_image(frequency, backend="numpy")
    _, _, I_nb = model.make_image(frequency, backend="numba")
    _compare(I_np, I_nb, rtol=1e-11)


@numba_only
def test_numba_matches_numpy_magnetic_heating_and_tau_stop():
    model = jf.JetModel(**CONFIGS["linear"])
    for kwargs in (dict(heating_prescription="magnetic"), dict(tau_stop=3.0)):
        _, _, I_np = model.make_image(43.0, backend="numpy", **kwargs)
        _, _, I_nb = model.make_image(43.0, backend="numba", **kwargs)
        _compare(I_np, I_nb, rtol=1e-11)


@numba_only
@pytest.mark.parametrize("name", list(CONFIGS))
def test_precomputed_state_matches_full_kernel(name):
    model = jf.JetModel(**CONFIGS[name])
    freqs = (2.0, 86.0, 1.0e5)
    full = [model.make_image(f, backend="numba")[2] for f in freqs]
    nbytes = model.precompute_state()
    assert nbytes == 44 * model.n_jet_cells
    for f, I_full in zip(freqs, full):
        _, _, I_cached = model.make_image(f, backend="numba")
        _compare(I_full, I_cached, rtol=1e-11)
    # a different heating prescription must not use the stored state
    _, _, I_mag_np = model.make_image(86.0, backend="numpy", heating_prescription="magnetic")
    _, _, I_mag_nb = model.make_image(86.0, backend="numba", heating_prescription="magnetic")
    _compare(I_mag_np, I_mag_nb, rtol=1e-11)
    model.clear_state()
    assert model._state is None


@numba_only
def test_precompute_state_memory_guard():
    model = jf.JetModel(**CONFIGS["linear"])
    with pytest.raises(MemoryError):
        model.precompute_state(max_memory_gb=1e-9)


@numba_only
def test_jet_cell_count_matches_numpy_mask():
    """The compiled interval finder must select exactly the cells the numpy loop uses."""
    model = jf.JetModel(**CONFIGS["log"])
    rH, nu, s = model.rH, model.nu, model.s
    count = 0
    for i in range(len(model.z_im_1D) - 1):
        z_im_now = model.z_mid_1D[i] + model.z_J_f
        x = model.x_im_f * model.cos_i + z_im_now * model.sin_i
        y = model.y_im_f
        z = z_im_now * model.cos_i - model.x_im_f * model.sin_i
        r = np.sqrt(x * x + y * y + z * z)
        ct = z / r
        rjet1 = rH * np.power(1.0 / (1.0 - ct), 1.0 / nu)
        rjet2 = rH * np.power(1.0 / (1.0 + ct), 1.0 / nu)
        ind = ((r <= rjet1) | (r <= rjet2)) & (r > rH * 1.01)
        r_rH_1_s = np.power(r / rH, 1.0 - s)
        arg_s = r_rH_1_s * np.sqrt(0.5 * (1.0 - ct))
        arg_c = r_rH_1_s * np.sqrt(0.5 * (1.0 + ct))
        theta_fp = np.where(
            arg_s < 1.0 / np.sqrt(2.0), 2.0 * np.arcsin(arg_s), 2.0 * np.arccos(arg_c)
        )
        cut = 2.0 * np.arcsin(model.jet_cutout_fraction / np.sqrt(2.0))
        ind &= ~((theta_fp < cut) | (theta_fp > np.pi - cut))
        count += int(ind.sum())
    assert model.n_jet_cells == count


def test_backend_selection_and_errors():
    model = jf.JetModel(**CONFIGS["linear"], backend="numpy")
    assert model._resolve_backend(None) == "numpy"
    with pytest.raises(ValueError):
        jf.JetModel(**CONFIGS["linear"], backend="fortran")
    with pytest.raises(ValueError):
        model.make_image(230.0, backend="fortran")
    with pytest.raises(ValueError):
        model.make_image(230.0, heating_prescription="poynting")
    if jf.numba_available():
        assert jf.JetModel(**CONFIGS["linear"])._resolve_backend(None) == "numba"
        assert jf.get_num_threads() >= 1
    else:
        with pytest.raises(ImportError):
            model.make_image(230.0, backend="numba")


def test_G_lookup_handles_nan_inf_and_zero():
    model = jf.JetModel(m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2)
    out = model.GIx_p(np.array([1.0, np.nan, np.inf, 0.0, -1.0]))
    assert np.isfinite(out[0])
    assert np.isnan(out[1])
    assert out[2] == 0.0
    assert out[3] == out[4] == pytest.approx(model.GI_p[0])
