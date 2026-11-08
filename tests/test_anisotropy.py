"""
The anisotropy factors of the Faraday rotativities (jetfuncs/_anisotropy.py): the response
of a thin shell of anisotropic electrons against an independent, slow evaluation of the same
orbit integral, its limits, the average over a cell's electrons, the tables' lookup on both
back ends, and what the factors do to the rotativities.
"""

import math

import numpy as np
import pytest

import jetfuncs as jf
from jetfuncs import _anisotropy as an
from jetfuncs import _kernel as K
from jetfuncs import _kernel_pol as KP

# (eta, p_eta, gamma, theta in degrees, R_V, R_Q): anisotropic over isotropic rotation and
# conversion of a thin shell, from the slow orbit integral of the anisotropy experiments
# (jet_circular_pol/old/anisotropy_variants/conversion/faraday_response.py: 1400 x 360 directions
# about the wave vector, 1440 azimuths for eta = 1e-3), whose own accuracy is about 2e-5 on
# isotropic shells.
SLOW_REFERENCE = [
    (0.03, 2.0, 1.5, 20.0, 1.085295, 2.058596),
    (0.03, 2.0, 3.0, 45.0, 0.199724, 0.837279),
    (0.03, 2.0, 10.0, 5.0, 3.216923, 9.488733),
    (0.03, 2.0, 10.0, 30.0, -0.081478, 1.493271),
    (0.03, 2.0, 30.0, 60.0, -0.297277, 0.533357),
    (0.03, 2.0, 100.0, 89.0, -0.283392, 0.403966),
    (0.03, 2.0, 1000.0, 10.0, 1.009016, 6.815828),
    (0.001, 2.0, 10.0, 60.0, -0.327387, 0.322710),
    (0.001, 2.0, 100.0, 20.0, -1.445731, 2.045606),
    (4.0, 2.0, 10.0, 60.0, 2.226313, 0.951660),
    (0.03, 1.0, 100.0, 20.0, 0.706080, 1.861568),
    (0.03, 4.0, 10.0, 60.0, -0.559797, 0.098595),
]


def _f(theta, eta, p_eta):
    """the emission's anisotropy factor f(theta)"""
    from scipy.special import hyp2f1

    phi = hyp2f1(0.5, p_eta / 2.0, 1.5, 1.0 - eta)
    return (1.0 + (eta - 1.0) * np.cos(theta) ** 2) ** (-p_eta / 2.0) / phi


@pytest.mark.parametrize("eta,p_eta,gamma,theta,RV_ref,RQ_ref", SLOW_REFERENCE)
def test_shell_factors_match_the_slow_orbit_integral(eta, p_eta, gamma, theta, RV_ref, RQ_ref):
    """within 1e-3 of the slow evaluation (relative; for R_V, of max(|R_V|, 0.05))"""
    RV, RQ = an.shell_factors([gamma], [math.radians(theta)], eta, p_eta)
    assert abs(RV[0, 0] - RV_ref) <= 1e-3 * max(abs(RV_ref), 0.05)
    assert abs(RQ[0, 0] - RQ_ref) <= 1e-3 * abs(RQ_ref)


def test_isotropic_shells_reproduce_the_closed_forms():
    """with eta = 1 the shell's rotation and conversion are the closed forms
    arccosh(gamma)/(gamma |p|) and (2 gamma^2 - 1)/gamma times the cold values, from mildly
    relativistic to gamma = 1e5: the quadrature resolves the 1/gamma peak about the wave
    vector at every energy (the residual is the central difference in momentum)"""
    gammas = [1.2, 3.0, 30.0, 1.0e3, 1.0e5]
    thetas = np.radians([5.0, 45.0, 85.0])
    RV, RQ = an.shell_factors(gammas, thetas, 1.0, 2.0)
    assert np.max(np.abs(RV - 1.0)) < 3e-6
    assert np.max(np.abs(RQ - 1.0)) < 3e-6


def test_cold_electrons_are_not_anisotropic():
    """a slow electron's response does not depend on its direction of motion"""
    RV, RQ = an.shell_factors([1.0 + 1e-6], np.radians([5.0, 45.0, 85.0]), 0.03, 2.0)
    assert np.max(np.abs(RV - 1.0)) < 1e-5
    assert np.max(np.abs(RQ - 1.0)) < 1e-5


def test_conversion_counts_the_electrons_moving_along_the_photon():
    """for gamma >~ 30 the conversion is beamed within 1/gamma of the photon, so the
    anisotropic conversion is f(theta) times the isotropic one, the factor of emission"""
    thetas = np.radians([15.0, 30.0, 60.0, 89.0])
    for eta, p_eta in ((0.03, 2.0), (0.1, 3.0)):
        _, RQ = an.shell_factors([30.0, 300.0], thetas, eta, p_eta)
        assert np.max(np.abs(RQ / _f(thetas, eta, p_eta)[None, :] - 1.0)) < 0.01


def test_rotation_reverses_away_from_the_field():
    """electrons crowded along the field rotate more than isotropic ones near the field and
    the other way far from it (rotation is not beamed, so its factor is not f)"""
    RV, _ = an.shell_factors([10.0, 30.0], np.radians([5.0, 45.0, 89.0]), 0.03, 2.0)
    assert np.all(RV[:, 0] > 3.0)
    assert np.all(RV[:, 1] < -0.2)
    assert np.all(RV[:, 2] < -0.2)


def test_rows_are_uniform_in_the_angle_variable():
    for eta in (0.01, 0.3, 3.0):
        r0, L, flip = an.theta_map(eta)
        c = an.theta_rows(eta)
        r = c if flip else 1.0 - c
        u = np.log1p(r / r0) / L
        assert np.allclose(u, np.linspace(0.0, 1.0, an.N_THETA), atol=1e-13)


def test_isotropic_electrons_get_factors_of_exactly_one():
    tabs, _, _ = an.cell_tables(1.0, 2.23, 2.0, 10.0, 1.0e8, 2001)
    assert np.all(tabs == 1.0)
    iso = jf.JetModel(m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, eta=1.0,
                      stokes="IQV")
    iso._load_rotativity_tables()
    assert iso._atab_on == 0.0
    cot = np.array([-3.0, -0.2, 0.0, 0.7, 40.0])
    for branch in (0, 1, 2):
        fV, fQ = iso._aniso_factors_numpy(cot, np.full(cot.shape, 3.0), branch)
        assert np.all(fV == 1.0) and np.all(fQ == 1.0)


def _direct_cell_average(kind, theta, eta, p_eta, p, gm, gx, gamma_c):
    """the cell factor by direct quadrature over the electrons, with the shells computed at
    the quadrature points (independent of the tables' grids and interpolation)"""

    def segment(lo_g, hi_g, shape, n=160):
        lp = np.linspace(0.5 * math.log10((lo_g - 1.0) * (lo_g + 1.0)),
                         0.5 * math.log10((hi_g - 1.0) * (hi_g + 1.0)), n)
        mom = 10.0 ** lp
        g = np.sqrt(1.0 + mom * mom)
        RV, RQ = an.shell_factors(g, [theta], eta, p_eta)
        R = (RV if kind == "V" else RQ)[:, 0]
        w = np.arcsinh(mom) / (g * mom) if kind == "V" else (2.0 * g * g - 1.0) / g
        f = shape(g) * w * mom * mom / g * math.log(10.0)
        dl = np.diff(lp)
        num = np.sum(0.5 * ((f * R)[1:] + (f * R)[:-1]) * dl)
        den = np.sum(0.5 * (f[1:] + f[:-1]) * dl)
        return num, den

    def cooled(g):
        return g ** (-(p + 1.0)) * g * g / (g * g - 1.0)

    if gamma_c >= gx:
        return np.divide(*segment(gm, gx, lambda g: g ** (-p), n=1000))
    if gamma_c <= gm:
        g_low = 1.0 / math.tanh(1.0 / gamma_c)
        n1, d1 = segment(g_low, gm, lambda g: 1.0 / (g * g - 1.0), n=400)
        n2, d2 = segment(gm, gx, cooled, n=1000)
        a = gm ** (p - 1.0)
        return (n1 + a * n2) / (d1 + a * d2)
    n1, d1 = segment(gm, gamma_c, lambda g: g ** (-p), n=400)
    n2, d2 = segment(gamma_c, gx, cooled, n=1000)
    return (n1 + gamma_c * n2) / (d1 + gamma_c * d2)


@pytest.mark.parametrize("kind,gamma_c", [("V", 1.0e5), ("V", 300.0), ("V", 3.0), ("Q", 3.0), ("Q", 1.3)])
def test_cell_tables_are_the_average_over_the_cell_s_electrons(kind, gamma_c):
    """the tables against a direct average, between the rows the shells are computed on, to
    1e-3 (gamma_max is not a power of ten, so the fine grid's spacing is not a round number)"""
    eta, p_eta, p, gm, gx = 0.03, 2.0, 2.23, 10.0, 3.0e5
    tabs, lg0, inv_dlg = an.cell_tables(eta, p, p_eta, gm, gx, 1001)
    j = 80
    theta = math.acos(an.theta_rows(eta)[j])
    fi = (math.log10(gamma_c) - lg0) * inv_dlg
    i = min(int(fi), tabs.shape[2] - 2)
    t = fi - i
    T = tabs[0 if kind == "V" else 1, j]
    table = T[i] + t * (T[i + 1] - T[i])
    direct = _direct_cell_average(kind, theta, eta, p_eta, p, gm, gx, gamma_c)
    assert table == pytest.approx(direct, rel=1e-3, abs=1e-3 * 0.05)


@pytest.fixture(scope="module")
def aniso_model():
    model = jf.JetModel(m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, p=2.23,
                        eta=0.03, gamma_m=10.0, stokes="IQV")
    model._load_rotativity_tables()
    return model


def test_kernel_and_numpy_look_up_the_same_factors(aniso_model):
    rng = np.random.default_rng(3)
    cot = np.concatenate([rng.normal(0.0, 3.0, 200), [0.0, 1e12, -1e12]])
    gc = 10.0 ** rng.uniform(-3.0, 9.0, cot.size)
    P = K.pack_params(aniso_model)
    RTAB = KP.pack_rot_tables(aniso_model)
    for branch in (0, 1, 2):
        fV_np, fQ_np = aniso_model._aniso_factors_numpy(cot, gc, branch)
        for k in range(cot.size):
            fV, fQ = KP._aniso_factors(RTAB, cot[k], gc[k], branch, P)
            assert fV == pytest.approx(fV_np[k], rel=1e-14, abs=1e-15)
            assert fQ == pytest.approx(fQ_np[k], rel=1e-14, abs=1e-15)


@pytest.mark.parametrize("scheme", ["HS11", "JO77"])
def test_both_schemes_get_the_same_factors(scheme):
    """the factors are those of the electrons, not of a fitting formula: the anisotropic
    rotativities are the isotropic ones times the same factors in either scheme"""
    kw = dict(m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, p=2.23, gamma_m=10.0,
              stokes="IQV", rotativities=scheme)
    an_m = jf.JetModel(eta=0.03, **kw)
    iso = jf.JetModel(eta=1.0, **kw)
    nn = np.full(4, 3.0e4)
    cot = np.array([0.1, 0.6, 2.0, -2.0])
    Kap = np.full(4, 1.0)
    for branch, gc in ((0, 1e9), (1, 1.0e3), (2, 3.0)):
        rQa, rVa = an_m._rotativities_numpy(nn, cot, Kap, branch, np.full(4, gc))
        rQi, rVi = iso._rotativities_numpy(nn, cot, Kap, branch, np.full(4, gc))
        fV, fQ = an_m._aniso_factors_numpy(cot, np.full(4, gc), branch)
        assert np.allclose(rQa, fQ * rQi, rtol=1e-14, atol=0.0)
        assert np.allclose(rVa, fV * rVi, rtol=1e-14, atol=0.0)


def test_rotation_is_odd_and_conversion_even_in_the_field_direction(aniso_model):
    nn = np.full(3, 3.0e4)
    cot = np.array([0.05, 0.8, 4.0])
    for branch, gc in ((0, 1e9), (1, 1.0e3), (2, 3.0)):
        g = np.full(3, gc)
        rQp, rVp = aniso_model._rotativities_numpy(nn, cot, np.ones(3), branch, g)
        rQm, rVm = aniso_model._rotativities_numpy(nn, -cot, np.ones(3), branch, g)
        assert np.array_equal(rQp, rQm)
        assert np.array_equal(rVp, -rVm)


def test_the_core_s_rotation_reverses_at_large_angles(aniso_model):
    """the M87 model's electrons (eta = 0.03): beyond about 30 deg from the field the
    cell's rotation has the opposite sign to that of isotropic electrons"""
    cot = np.array([1.0 / math.tan(math.radians(t)) for t in (10.0, 45.0, 80.0)])
    for branch, gc in ((0, 1e9), (1, 1.0e3), (2, 5.0)):
        fV, fQ = aniso_model._aniso_factors_numpy(cot, np.full(3, gc), branch)
        assert fV[0] > 1.0 and fV[1] < 0.0 and fV[2] < 0.0
        assert np.all(fQ > 0.0)
