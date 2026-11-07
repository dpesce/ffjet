"""
JetModel(geometry="kerr"): images along Kerr null geodesics.

Checks against results the Kerr code does not produce itself: the flat model (to which it
must reduce when the ray geometry is made flat), its own numpy reference path, the
reflection symmetry of the Kerr metric and of the jet, and convergence in the step of the
geodesic integration.

Several tolerances below follow from one estimate.  The tracer places a cell's point at
affine offset s from lambda_J, the ray's closest approach to the jet axis, with an error
of ~2.6e-7 |s| (the Runge-Kutta error of the affine parameter far out) plus the precision
of lambda_J itself, sqrt(2 eps) r_J / sin(i) (a minimum search; see test_geodesics.py).
Moving the points of a ray by that much changes a pixel by that amount over the length on
which the emission varies, which is at least ~1 r_g near the base of the jet, where r_J
is ~10 r_g, and grows with r_J: at most ~3e-6 of the pixel.  The tests allow 1e-5 of the
peak of each Stokes plane.
"""

import math

import numpy as np
import pytest

import jetfuncs as jf

numba_only = pytest.mark.skipif(not jf.numba_available(), reason="numba is not installed")

if jf.numba_available():
    from jetfuncs import _geodesics as geo
    from jetfuncs import _kernel as K
    from jetfuncs import _kernel_kerr as KK
    from jetfuncs import _kernel_pol as KP

# as in test_backends.py: an M87-like model on a linear grid, and a log grid exercising
# a general anisotropy exponent, the jet cutout, another inclination and spin
CONFIGS = {
    "linear": dict(
        m=6.5e9, a=0.9, inc=163.0, mdot=5.45e-5, s=0.6, p=2.14, h=0.001, eta=0.03,
        gamma_inf=4.0, Nx=16, Ny=16, Nz=96,
        xmin=-60.0, xmax=60.0, ymin=-60.0, ymax=60.0, zmin=-240.0, zmax=240.0,
    ),
    "log": dict(
        m=4.1e6, a=0.5, inc=30.0, mdot=7e-7, s=0.7, p=2.8, h=0.003, eta=0.1, p_eta=2.3,
        gamma_inf=6.0, gamma_m=20.0, gamma_max=10**2.5, jet_cutout_fraction=0.3,
        Nx=16, Ny=16, Nz=96,
        xmin=0.3, xmax=1e4, ymin=0.3, ymax=1e4, zmin=0.3, zmax=1e6,
        use_log_xgrid=True, use_log_ygrid=True, use_log_zgrid=True,
    ),
}

GEOMETRY_TOL = 1e-5     # see the module docstring


def _planes(S):
    return S[None] if S.ndim == 2 else S


def _assert_close_to_peak(test, ref, tol, signs=None):
    """|test - ref| <= tol * max|ref| in every Stokes plane."""
    test, ref = _planes(test), _planes(ref)
    assert test.shape == ref.shape
    assert np.all(np.isfinite(test))
    for k in range(ref.shape[0]):
        scale = np.abs(ref[k]).max()
        assert scale > 0.0
        assert np.abs(test[k] - ref[k]).max() <= tol * scale, f"Stokes {'IQUV'[k]}"


def _compare(ref, test, rtol):
    """test_backends._compare: same cells, and agreement to rtol where there is emission."""
    assert ref.shape == test.shape
    assert np.all(np.isfinite(test))
    assert np.array_equal(ref == 0.0, test == 0.0)
    good = ref > ref.max() * 1e-9
    assert good.sum() > 0
    np.testing.assert_allclose(test[good], ref[good], rtol=rtol, atol=0.0)


def _compare_pol(ref, test, rtol):
    """test_polarization._compare: each plane to rtol of its largest value where I is not negligible."""
    good = ref[0] > ref[0].max() * 1e-9
    assert good.sum() > 0
    for k in range(4):
        scale = np.max(np.abs(ref[k][good]))
        if scale == 0.0:
            continue
        assert np.max(np.abs(ref[k][good] - test[k][good])) <= rtol * scale, f"Stokes {'IQUV'[k]}"


def _flat_limit(cfg):
    """A Kerr-mode model whose rays follow the flat metric (M = a = 0) through the same jet."""
    model = jf.JetModel(**cfg, geometry="kerr")
    model._ray_metric = (0.0, 0.0)
    return model


# ---------------------------------------------------------------- the flat limit
@numba_only
@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("backend", ["numba", "numpy"])
def test_flat_limit_reproduces_the_flat_model(name, backend):
    """
    With the ray geometry made flat, the Kerr path must reproduce the flat model: the
    same cells, and images that agree to the tracer's error.  This pins the camera, the
    image orientation, the cell layout along the ray, the ZAMO-frame line-of-sight
    algebra, the transported Stokes basis and the Stokes signs.
    """
    cfg = CONFIGS[name]
    flat = jf.JetModel(**cfg)
    lim = _flat_limit(cfg)
    assert lim.n_jet_cells == flat.n_jet_cells
    np.testing.assert_array_equal(lim._get_intervals()["nint"], flat._get_intervals()["nint"])
    for frequency in (8.0, 230.0):
        _assert_close_to_peak(lim.make_image(frequency, backend=backend)[2],
                              flat.make_image(frequency, backend=backend)[2], GEOMETRY_TOL)
        _assert_close_to_peak(lim.make_image_polarized(frequency, backend=backend)[2],
                              flat.make_image_polarized(frequency, backend=backend)[2], GEOMETRY_TOL)


@numba_only
def test_flat_limit_with_the_field_table_and_the_stored_state():
    cfg = CONFIGS["log"]
    flat = jf.JetModel(**cfg)
    lim = _flat_limit(cfg)
    for model in (flat, lim):
        model.build_field_table()
    _assert_close_to_peak(lim.make_image_polarized(86.0)[2], flat.make_image_polarized(86.0)[2],
                          GEOMETRY_TOL)
    for model in (flat, lim):
        model.clear_field_table()
        model.precompute_state(polarized=True)
    _assert_close_to_peak(lim.make_image(86.0)[2], flat.make_image(86.0)[2], GEOMETRY_TOL)
    _assert_close_to_peak(lim.make_image_polarized(86.0)[2], flat.make_image_polarized(86.0)[2],
                          GEOMETRY_TOL)


@numba_only
def test_ray_parts_reduce_to_the_flat_ones():
    """
    For the flat line of sight, 1 - omega lambda = 1 and the fixed sky basis, the
    ZAMO-basis line-of-sight halves equal the flat model's Cartesian ones to roundoff;
    and the Stokes I and polarized halves agree exactly on the quantities they share.
    """
    model = jf.JetModel(**CONFIGS["linear"])
    model._ensure_polarized_tables()
    P = K.pack_params(model)
    tabs = (model._stag_lut_x, model._stag_lut_r, model._stag_lut_t, model._stag_lut_a)
    n = np.array([model.nx, model.ny, model.nz])
    a_sky = np.array([model.cos_i, 0.0, -model.sin_i])
    b_sky = np.array([0.0, -1.0, 0.0])
    iv = model._get_intervals()
    checked = 0
    for pix in np.flatnonzero(iv["nint"])[::7]:
        for i in range(iv["c_lo"][pix, 0], iv["c_hi"][pix, 0] + 1, 3):
            zi, ds = K._cell_overlap(model.z_im_1D, i, iv["seg_in"][pix, 0], iv["seg_out"][pix, 0], model.rg)
            xi, yi, zJ = model.x_im_f[pix], model.y_im_f[pix], model.z_J_f[pix]
            z_im = zi + zJ
            x = xi * model.cos_i + z_im * model.sin_i
            y = yi
            z = z_im * model.cos_i - xi * model.sin_i
            R = math.hypot(x, y)
            r = math.sqrt(R * R + z * z)
            omc = R * R / (r * (r + abs(z)))
            sgn = 1.0 if z >= 0.0 else -1.0
            chain = K._rtheta_chain(r, omc, sgn, P, *tabs)
            vr, vt, vp, gamma, alpha, Bpr, Bpt, Bpp, Bpm, gc, Kj, Ka = chain
            args = (vr, vt, vp, gamma, alpha, Bpr, Bpt, Bpp, gc, Kj, Ka, P)
            # the spherical basis at the point, in Cartesian components
            rhat = np.array([x, y, z]) / r
            that = np.array([x * z / (r * R), y * z / (r * R), -R / r])
            phat = np.array([-y / R, x / R, 0.0])
            sph = lambda v: (float(v @ rhat), float(v @ that), float(v @ phat))
            flat_I = K._ray_part(x, y, z, r, R, *args)
            kerr_I = KK._ray_part_kerr(*args, *sph(n), 1.0)
            np.testing.assert_allclose(kerr_I, flat_I, rtol=1e-12, atol=0.0)
            flat_P = KP._ray_part_pol(x, y, z, r, R, *args)
            kerr_P = KK._ray_part_pol_kerr(*args, *sph(n), 1.0, *sph(a_sky), *sph(b_sky))
            np.testing.assert_allclose(kerr_P, flat_P, rtol=1e-11, atol=1e-12)
            assert kerr_P[:5] == kerr_I
            checked += 1
    assert checked > 20


# ---------------------------------------------------------------- Kerr images
@numba_only
@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("frequency", [8.0, 230.0])
def test_numba_matches_numpy(name, frequency):
    """
    The numpy reference path takes the rays, cells and local frame from the compiled
    tracer and does the physics and the transfer itself; it must agree with the
    compiled kernels as closely as the two back ends do in flat mode.
    """
    model = jf.JetModel(**CONFIGS[name], geometry="kerr")
    _compare(model.make_image(frequency, backend="numpy")[2],
             model.make_image(frequency, backend="numba")[2], rtol=1e-11)
    _compare_pol(model.make_image_polarized(frequency, backend="numpy")[2],
                 model.make_image_polarized(frequency, backend="numba")[2], rtol=1e-9)


@numba_only
def test_numba_matches_numpy_magnetic_heating_and_tau_stop():
    model = jf.JetModel(**CONFIGS["linear"], geometry="kerr")
    for kwargs in (dict(heating_prescription="magnetic"), dict(tau_stop=3.0)):
        _compare(model.make_image(43.0, backend="numpy", **kwargs)[2],
                 model.make_image(43.0, backend="numba", **kwargs)[2], rtol=1e-11)


@numba_only
@pytest.mark.parametrize("name", list(CONFIGS))
def test_stored_state_matches_the_full_kernel(name):
    model = jf.JetModel(**CONFIGS[name], geometry="kerr")
    freqs = (2.0, 86.0, 1.0e5)
    full_I = [model.make_image(f)[2] for f in freqs]
    full_S = [model.make_image_polarized(f)[2] for f in freqs]
    assert model.precompute_state() == 48 * model.n_jet_cells
    for f, I_full in zip(freqs, full_I):
        _compare(I_full, model.make_image(f)[2], rtol=1e-11)
    assert model.precompute_state(polarized=True) == 80 * model.n_jet_cells
    for f, S_full in zip(freqs, full_S):
        _compare_pol(S_full, model.make_image_polarized(f)[2], rtol=1e-11)
    model.clear_state()


@numba_only
def test_field_table_accuracy_and_convergence():
    """
    The field table interpolates the same (r, theta) physics in Kerr mode as in flat
    mode: on the configuration of test_field_table_and_survey.py, with the bounds used
    there, its error is small and falls with resolution.
    """
    cfg = dict(
        m=6.5e9, a=0.9, inc=163.0, mdot=5.45e-5, s=0.6, p=2.14, h=0.001, eta=0.03,
        gamma_inf=4.0, Nx=24, Ny=24, Nz=120, xmin=0.3, xmax=3.0e3, ymin=0.3, ymax=3.0e3,
        zmin=0.3, zmax=3.0e5, use_log_xgrid=True, use_log_ygrid=True, use_log_zgrid=True,
    )
    model = jf.JetModel(**cfg, geometry="kerr")
    I_exact = model.make_image(230.0)[2]
    good = I_exact > I_exact.max() * 1e-6
    errs = {}
    for ppd, n_u in ((50, 64), (200, 256)):
        model.build_field_table(points_per_decade=ppd, n_u=n_u)
        I_tab = model.make_image(230.0)[2]
        assert np.array_equal(I_exact == 0.0, I_tab == 0.0)
        errs[ppd] = (abs(I_tab.sum() / I_exact.sum() - 1.0),
                     np.percentile(np.abs(I_tab - I_exact)[good] / I_exact[good], 99))
    assert errs[200][0] < errs[50][0]
    assert errs[200][1] < errs[50][1]
    assert errs[200][0] < 1e-3
    assert errs[200][1] < 3e-3
    # the table reaches every point inside the jet along the geodesics
    ft = model._ftab
    r_top = model.rH + 10.0 ** (ft["logr0"] + (ft["nr"] - 1) / ft["invdlogr"])
    assert r_top >= model._get_intervals()["rmax"].max()
    model.clear_field_table()
    np.testing.assert_allclose(model.make_image(230.0)[2], I_exact, rtol=1e-11, atol=0.0)


@numba_only
@pytest.mark.parametrize("name", list(CONFIGS))
def test_reflection_symmetry(name):
    """
    The Kerr metric and the jet are symmetric under z -> -z, which maps the observer at
    inc to the one at 180 - inc and the image to its mirror x_im -> -x_im; the
    reflection reverses handedness, so Q is unchanged and U and V change sign.
    """
    cfg = CONFIGS[name]
    S = jf.JetModel(**cfg, geometry="kerr").make_image_polarized(86.0)[2]
    S_mirror = jf.JetModel(**dict(cfg, inc=180.0 - cfg["inc"]), geometry="kerr").make_image_polarized(86.0)[2]
    signs = np.array([1.0, 1.0, -1.0, -1.0])[:, None, None]
    _assert_close_to_peak(signs * S_mirror[:, :, ::-1], S, GEOMETRY_TOL)


@numba_only
def test_kerr_images_converge_in_the_geodesic_step():
    """
    Halving both step parameters of the tracer moves the cells by the tracer's error
    (fourth order, so the difference is ~15/16 of the error at the default steps).
    """
    cfg = CONFIGS["linear"]
    model = jf.JetModel(**cfg, geometry="kerr")
    fine = jf.JetModel(**cfg, geometry="kerr")
    eps, wcap = fine._geodesic_steps
    fine._geodesic_steps = (0.5 * eps, 0.5 * wcap)
    _assert_close_to_peak(model.make_image_polarized(86.0)[2], fine.make_image_polarized(86.0)[2],
                          GEOMETRY_TOL)


@numba_only
def test_rays_inside_the_shadow_end_on_the_horizon():
    """
    Every ray of the log grid either escapes or is captured, and the captured ones are
    those whose screen points lie inside the critical curve, which for a = 0.5 seen at
    30 deg (Bardeen 1973) lies between b = 4.61 and 5.63 r_g from the center.  The four
    innermost rays are vortical (eta + lambda^2 < 0).
    """
    model = jf.JetModel(**CONFIGS["log"], geometry="kerr")
    status = model._get_intervals()["status"]
    b = np.hypot(model.x_im_f, model.y_im_f)
    assert set(np.unique(status)) <= {geo.CAPTURED, geo.ESCAPED}
    assert np.all(status[b < 4.5] == geo.CAPTURED)
    assert np.all(status[b > 5.8] == geo.ESCAPED)
    assert np.count_nonzero(status == geo.CAPTURED) > 0


# ---------------------------------------------------------------- guards
def test_geometry_must_be_flat_or_kerr():
    with pytest.raises(ValueError, match="geometry"):
        jf.JetModel(**CONFIGS["linear"], geometry="schwarzschild")


@numba_only
@pytest.mark.parametrize("inc", [0.0, 180.0])
def test_pole_on_kerr_is_refused(inc):
    with pytest.raises(ValueError, match="away from the jet axis"):
        jf.JetModel(**dict(CONFIGS["linear"], inc=inc), geometry="kerr")
