"""
Input validation, pole-on viewing geometry, and the pitch-angle normalization.
"""

import numpy as np
import pytest

import jetfuncs as jf

BASE = dict(
    m=6.2e9,
    a=0.9,
    inc=163.0,
    mdot=5.45e-5,
    s=0.6,
    p=2.14,
    h=0.001,
    eta=0.01,
    gamma_inf=4.0,
    Nx=32,
    Ny=32,
    Nz=200,
    xmin=0.1,
    xmax=1.0e5,
    ymin=0.1,
    ymax=1.0e5,
    zmin=0.1,
    zmax=1.0e7,
    use_log_xgrid=True,
    use_log_ygrid=True,
    use_log_zgrid=True,
)


def cfg(**over):
    c = dict(BASE)
    c.update(over)
    return c


def _flux(model, I):
    """Area-weighted total, using the same cell widths as convert_units."""
    def widths(v):
        edges = np.empty(v.size + 1)
        same = v[:-1] * v[1:] > 0.0
        edges[1:-1] = np.where(same, np.sign(v[:-1]) * np.sqrt(np.abs(v[:-1] * v[1:])), 0.0)
        edges[0] = v[0] * np.sqrt(np.abs(v[0] / v[1]))
        edges[-1] = v[-1] * np.sqrt(np.abs(v[-1] / v[-2]))
        return np.diff(edges)

    return float(np.sum(I * widths(model.x_im_1D)[None, :] * widths(model.y_im_1D)[:, None]))


# ---------------------------------------------------------------- input validation
@pytest.mark.parametrize(
    "over,match",
    [
        (dict(m=-1.0), "must be positive"),
        (dict(mdot=0.0), "must be positive"),
        (dict(s=1.0), "must be less than 1"),
        (dict(s=1.5), "must be less than 1"),
        (dict(s=0.0), "must be positive"),
        (dict(s=-0.5), "must be positive"),
        (dict(eta=0.0), "must be positive"),
        (dict(eta=-1.0), "must be positive"),
        (dict(gamma_m=0.5), "must exceed 1"),
        (dict(gamma_m=1.0e9), "must exceed gamma_m"),
        (dict(gamma_inf=0.5), "at least 1"),
        (dict(jet_cutout_fraction=1.0), r"\[0, 1\)"),
        (dict(jet_cutout_fraction=-0.1), r"\[0, 1\)"),
        (dict(p_eta=-1.0), "non-negative"),
    ],
)
def test_rejected_inputs(over, match):
    with pytest.raises((ValueError, Exception), match=match):
        jf.JetModel(**cfg(Nx=4, Ny=4, Nz=8, **over))


def test_accepted_inputs_span_the_useful_range():
    """The combinations the guards must *not* reject."""
    for a in (0.01, 0.5, 0.99):
        for s in (0.05, 0.6, 0.9):
            m = jf.JetModel(**cfg(a=a, s=s, Nx=4, Ny=4, Nz=8, n_stagnation=50))
            assert np.all(np.isfinite(m.rstag_arr))
            del m


# ---------------------------------------------------------------- pole-on geometry
@pytest.mark.parametrize("inc", [0.0, 180.0, 0.5, 179.5, 5.0])
def test_pole_on_inclinations_produce_finite_images(inc):
    """
    inc = 0 and 180 are ordinary inclinations.  Sampling each ray about its closest
    approach to the jet axis puts that point at |x_im| cot(i), which diverges as the line
    of sight approaches the axis; exactly pole-on the correct limit is to sample about the
    closest approach to the black hole instead.
    """
    model = jf.JetModel(**cfg(inc=inc))
    _, _, I = model.make_image(230.0)
    assert np.all(np.isfinite(I))
    assert np.all(I >= 0.0)
    assert I.sum() > 0.0


@pytest.mark.parametrize("inc", [0.5, 5.0, 45.0, 90.0])
def test_reflection_symmetry(inc):
    """psi, the fields and the velocity are symmetric under z -> -z, so F(i) = F(180-i)."""
    a = jf.JetModel(**cfg(inc=inc))
    b = jf.JetModel(**cfg(inc=180.0 - inc))
    fa = _flux(a, a.make_image(230.0)[2])
    fb = _flux(b, b.make_image(230.0)[2])
    assert fa > 0.0
    np.testing.assert_allclose(fa, fb, rtol=1e-10, atol=0.0)


def test_warns_when_the_sampling_window_misses_the_inner_jet():
    with pytest.warns(RuntimeWarning, match="from the jet axis"):
        jf.JetModel(**cfg(inc=179.5, zmax=1.0e4, Nz=100))


# ---------------------------------------------------------------- pitch-angle normalization
def _phi_norm_reference(eta, p_eta, h=0.01, N=800):
    """tanh-sinh quadrature; handles the mu -> 1 spike that appears as eta -> 0."""
    u = np.arange(-N, N + 1) * h
    with np.errstate(over="ignore", invalid="ignore"):
        s = 0.5 * np.pi * np.sinh(u)
        mu = 0.5 * (1.0 + np.tanh(s))
        w = 0.25 * np.pi * np.cosh(u) / np.cosh(s) ** 2
        f = (1.0 + (eta - 1.0) * mu * mu) ** (-p_eta / 2.0)
    good = np.isfinite(f) & np.isfinite(w)
    return float(np.sum((f * w)[good]) * h)


@pytest.mark.parametrize("eta", [1e-6, 1e-4, 1e-2, 1.0, 10.0, 1e3])
@pytest.mark.parametrize("p_eta", [1.0, 2.0, 3.0])
def test_phi_norm_matches_a_converged_quadrature(eta, p_eta):
    model = jf.JetModel(**cfg(eta=eta, p_eta=p_eta, Nx=4, Ny=4, Nz=8, n_stagnation=50))
    np.testing.assert_allclose(
        model.phi_norm, _phi_norm_reference(eta, p_eta), rtol=1e-7, atol=0.0
    )


def test_small_eta_is_usable():
    """Emission must stay finite and scale sensibly for the strongly anisotropic case."""
    prev = None
    for eta in (1e-1, 1e-3, 1e-5):
        model = jf.JetModel(**cfg(eta=eta))
        _, _, I = model.make_image(230.0)
        assert np.all(np.isfinite(I))
        f = _flux(model, I)
        assert f > 0.0
        if prev is not None:
            assert f < prev  # more beaming along B -> less flux at a generic angle
        prev = f
