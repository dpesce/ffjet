"""
Full-Stokes radiative transfer.

The tests fall into four groups:

  * the cell solver against the analytic solutions of the polarized transfer equation
    with constant coefficients (Dexter 2016, Appendix C) -- the same problems grtrans
    is verified on;
  * the synchrotron coefficients against the closed forms their ratios must take in
    the optically thin limit, which is what fixes the polarization fractions;
  * the conventions (EVPA relative to the projected field, sign of V, handedness),
    each pinned by a test so that changing one is a deliberate act;
  * the compiled kernel against the numpy reference, and the cached and tabulated
    paths against the exact one.
"""

import numpy as np
import pytest

import jetfuncs as jf
from jetfuncs._core import _Q_SIGN, _U_SIGN, _pol_transfer_step

numba_only = pytest.mark.skipif(not jf.numba_available(), reason="numba is not installed")

# an M87-like model on a linear grid, and a Sgr A*-like one on a log grid that also
# exercises the fast-cooling branch, a general anisotropy exponent and the jet cutout
CONFIGS = {
    "linear": dict(
        m=6.5e9, a=0.9, inc=163.0, mdot=5.45e-5, s=0.6, p=2.14, h=0.001, eta=0.03,
        gamma_inf=4.0, Nx=10, Ny=10, Nz=64,
        xmin=-60.0, xmax=60.0, ymin=-60.0, ymax=60.0, zmin=-240.0, zmax=240.0,
    ),
    "log": dict(
        m=4.1e6, a=0.5, inc=30.0, mdot=7e-7, s=0.7, p=2.8, h=0.003, eta=0.1, p_eta=2.3,
        gamma_inf=6.0, gamma_m=20.0, gamma_max=10**2.5, jet_cutout_fraction=0.3,
        Nx=10, Ny=10, Nz=64,
        xmin=0.3, xmax=1e4, ymin=0.3, ymax=1e4, zmin=0.3, zmax=1e6,
        use_log_xgrid=True, use_log_ygrid=True, use_log_zgrid=True,
    ),
}


def _arr(v):
    return np.full(1, float(v))


# --------------------------------------------------------------------------- solver
@pytest.mark.parametrize("ds", [1e-6, 1e-3, 0.1, 1.0, 5.0, 50.0])
def test_solver_matches_emission_absorption_solution(ds):
    """
    Emission and absorption in I and Q only (Dexter 2016, Appendix C, first problem).
    The system decouples in I + Q and I - Q, each an ordinary scalar transfer problem,
    which is the cleanest form of his Eqs. (C2) and (C3).
    """
    jI, jQ, aI, aQ = 0.7, -0.3, 1.3, 0.4
    I, Q, U, V = _pol_transfer_step(
        _arr(0), _arr(0), _arr(0), _arr(0),
        _arr(jI), _arr(jQ), _arr(0), _arr(0),
        _arr(aI), _arr(aQ), _arr(0), _arr(0),
        _arr(0), _arr(0), _arr(0), ds,
    )
    lp, lm = aI + aQ, aI - aQ
    u = (jI + jQ) / lp * (1.0 - np.exp(-lp * ds))
    v = (jI - jQ) / lm * (1.0 - np.exp(-lm * ds))
    assert I[0] == pytest.approx(0.5 * (u + v), rel=1e-9)
    assert Q[0] == pytest.approx(0.5 * (u - v), rel=1e-9)
    assert U[0] == 0.0 and V[0] == 0.0


@pytest.mark.parametrize("ds", [1e-6, 1e-3, 0.3, 2.0, 20.0])
def test_solver_matches_faraday_solution(ds):
    """
    Pure Faraday rotation and conversion with emission in Q and V (Dexter 2016,
    Appendix C, second problem).  With no absorption the equation is a rigid rotation
    of (Q, U, V) about omega = (rho_Q, 0, rho_V) driven by a constant source, whose
    solution is the Rodrigues form used here -- equivalent to his Eqs. (C5)-(C7).
    """
    jQ, jV, rQ, rV = 0.4, -0.25, 1.7, 2.9
    I, Q, U, V = _pol_transfer_step(
        _arr(0), _arr(0), _arr(0), _arr(0),
        _arr(0), _arr(jQ), _arr(0), _arr(jV),
        _arr(0), _arr(0), _arr(0), _arr(0),
        _arr(rQ), _arr(0), _arr(rV), ds,
    )
    w = np.array([rQ, 0.0, rV])
    rho = np.linalg.norm(w)
    wh = w / rho
    jv = np.array([jQ, 0.0, jV])
    expect = (
        jv * np.sin(rho * ds) / rho
        + np.cross(wh, jv) * (1.0 - np.cos(rho * ds)) / rho
        + wh * np.dot(wh, jv) * (ds - np.sin(rho * ds) / rho)
    )
    got = np.array([Q[0], U[0], V[0]])
    assert np.allclose(got, expect, rtol=1e-8, atol=1e-12 * max(1.0, np.abs(expect).max()))
    assert I[0] == 0.0


@pytest.mark.parametrize("ds", [1e-9, 1e-3, 1.0, 30.0])
def test_solver_reduces_to_the_scalar_transfer_without_polarized_structure(ds):
    """With no polarized absorption and no rotativities every component is scalar."""
    aI = 0.8
    j = (1.3, -0.4, 0.6, -0.2)
    start = (0.2, 0.05, -0.1, 0.03)
    out = _pol_transfer_step(
        *[_arr(v) for v in start], *[_arr(v) for v in j],
        _arr(aI), _arr(0), _arr(0), _arr(0), _arr(0), _arr(0), _arr(0), ds,
    )
    e = np.exp(-aI * ds)
    for got, s0, js in zip(out, start, j):
        assert got[0] == pytest.approx(s0 * e + (js / aI) * (1.0 - e), rel=1e-10, abs=1e-300)


def test_solver_is_exact_for_a_split_step():
    """
    Two half steps must reproduce one whole step: the solution is exact for constant
    coefficients, so there is no step-size error to accumulate.
    """
    rng = np.random.default_rng(3)
    for _ in range(20):
        j = [_arr(v) for v in rng.normal(size=4)]
        aI = _arr(3.0 + abs(rng.normal()))
        aQ, aU, aV = (_arr(v) for v in 0.4 * rng.normal(size=3))
        rQ, rU, rV = (_arr(v) for v in 2.0 * rng.normal(size=3))
        s0 = [_arr(v) for v in rng.normal(size=4)]
        ds = 0.7
        one = _pol_transfer_step(*s0, *j, aI, aQ, aU, aV, rQ, rU, rV, ds)
        half = _pol_transfer_step(*s0, *j, aI, aQ, aU, aV, rQ, rU, rV, 0.5 * ds)
        two = _pol_transfer_step(*half, *j, aI, aQ, aU, aV, rQ, rU, rV, 0.5 * ds)
        for a, b in zip(one, two):
            assert a[0] == pytest.approx(b[0], rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------- coefficients
@pytest.fixture(scope="module")
def pol_model():
    return jf.JetModel(
        m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, p=2.14, stokes="IQV"
    )


def test_thin_limit_linear_polarization_fractions(pol_model):
    """
    Optically thin power-law synchrotron has j_Q/j_I -> (p+1)/(p+7/3) and
    alpha_Q/alpha_I -> (p+2)/(p+10/3) (Dexter 2016, Eqs. A33-A34 and A47-A48).  Both
    ratios are properties of the tabulated integrals on their plateau, so this pins the
    Stokes Q tables against the Stokes I ones and, with them, the polarization fraction
    the model can produce.
    """
    p = pol_model.p
    x = 1.0e-18
    assert float(pol_model.GQx_p(x) / pol_model.GIx_p(x)) == pytest.approx(
        (p + 1.0) / (p + 7.0 / 3.0), rel=1e-4
    )
    assert float(pol_model.GaQx_p(x) / pol_model.GaIx_p(x)) == pytest.approx(
        (p + 2.0) / (p + 10.0 / 3.0), rel=1e-4
    )


def test_stokes_V_table_plateaus_match_the_closed_form(pol_model):
    """G_V(0) and Ga_V(0) in closed form (Dexter 2016, Eqs. A35 and A49)."""
    from scipy.special import gamma as G

    p = pol_model.p
    x = 1.0e-18
    gv0 = 2.0 ** (p / 2.0 - 1.0) * (p + 2.0) / p * G(p / 4.0 + 1.0 / 3.0) * G(p / 4.0 + 2.0 / 3.0)
    gav0 = (
        (p + 3.0) / (p + 1.0) * 2.0 ** ((p - 1.0) / 2.0)
        * G(p / 4.0 + 7.0 / 12.0) * G(p / 4.0 + 11.0 / 12.0)
    )
    assert float(pol_model.GVx_p(x)) == pytest.approx(gv0, rel=1e-4)
    assert float(pol_model.GaVx_p(x)) == pytest.approx(gav0, rel=1e-4)


def test_kernel_tail_constants():
    """
    The small-argument coefficient of each kernel, which the coefficients fall back on
    where a difference of tabulated integrals is destroyed by cancellation:

        F(z) = z int_z^inf K_{5/3}(y) dy        -> Gamma(2/3) 2^(2/3) z^(1/3)
        G(z) = z K_{2/3}(z)                     -> Gamma(2/3) 2^(-1/3) z^(1/3)
        H(z) = int_z^inf K_{1/3} + z K_{1/3}(z) -> int_0^inf K_{1/3} = pi/sqrt(3)

    So G's coefficient is exactly half of F's and H's integral has no plateau
    divergence at all -- the two facts the polarized branch of _G_diff rests on.
    Checked against scipy, not against the tables they are used to repair.
    """
    from scipy.integrate import quad
    from scipy.special import gamma as G
    from scipy.special import kv

    C_F, e_F = jf.JetModel._TAIL["F"]
    C_G, e_G = jf.JetModel._TAIL["G"]
    C_H, e_H = jf.JetModel._TAIL["H"]
    assert (e_F, e_G, e_H) == (4.0 / 3.0, 4.0 / 3.0, 1.0)

    assert C_F == pytest.approx(G(2.0 / 3.0) * 2.0 ** (2.0 / 3.0), rel=1e-12)
    assert C_G == pytest.approx(0.5 * C_F, rel=1e-12)
    # and directly from the Bessel function itself
    z = 1.0e-10
    assert z * kv(2.0 / 3.0, z) / z ** (1.0 / 3.0) == pytest.approx(C_G, rel=1e-6)
    assert quad(lambda y: kv(1.0 / 3.0, y), 0.0, np.inf)[0] == pytest.approx(C_H, rel=1e-6)


def test_circular_emissivity_follows_the_sign_of_the_pitch_angle_cosine(pol_model):
    """
    V > 0 when the fluid-frame magnetic field points towards the observer, i.e. when
    cos(theta_B') > 0 (Dexter 2016, Eqs. A14/A24).  This is the convention of
    grtrans/ipole and is what make_image_polarized documents.
    """
    n = np.full(3, 30.0)
    nup = np.full(3, 1.0)
    gc = np.full(3, 1.0e30)  # uncooled branch
    nA = np.full(3, 1.0e-4)
    cth = np.array([0.6, -0.6, 0.0])
    jI, jQ, jV, aI, aQ, aV, rQ, rV = pol_model._pol_coefficients_numpy(n, nup, gc, cth, nA)
    assert jV[0] > 0.0 and jV[1] < 0.0
    assert jV[0] == pytest.approx(-jV[1], rel=1e-12)
    assert abs(jV[2]) < 1e-12 * jI[2]
    assert aV[0] > 0.0 and aV[1] < 0.0
    assert rV[0] > 0.0 and rV[1] < 0.0
    # and the linear coefficients are even in cos(theta_B')
    assert jQ[0] == pytest.approx(jQ[1], rel=1e-12)
    assert jI[0] > 0.0 and aI[0] > 0.0


def test_isotropic_distribution_leaves_the_circular_correction_alone():
    """
    The extra factor the circular coefficients pick up from an anisotropic pitch-angle
    distribution must vanish when the distribution is isotropic (eta = 1).
    """
    iso = jf.JetModel(m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, eta=1.0,
                      stokes="IQV")
    n = np.full(2, 30.0)
    args = (n, np.full(2, 1.0), np.full(2, 1e30), np.array([0.6, -0.6]), np.full(2, 1e-4))
    jI, jQ, jV, aI, aQ, aV, rQ, rV = iso._pol_coefficients_numpy(*args)
    # with eta = 1 the anisotropy factor is 1/phi_norm = 1 and g_eta = 0
    assert iso.phi_norm == pytest.approx(1.0, rel=1e-12)
    assert np.all(np.isfinite(jV)) and jV[0] > 0.0


# --------------------------------------------------------------------------- conventions
def test_sky_basis_is_orthonormal_after_the_boost():
    """
    The polarization basis is boosted into the fluid frame and shifted along the photon
    momentum to remove its time component.  Four-dimensional inner products survive
    both steps, so the result must come out orthonormal and perpendicular to the
    aberrated photon direction with no renormalization -- the property the kernel
    relies on when it skips one.
    """
    rng = np.random.default_rng(11)
    model = jf.JetModel(m=10.0, a=0.5, inc=47.0, mdot=1e-3, Nx=2, Ny=2, Nz=2)
    n = np.array([model.nx, model.ny, model.nz])
    a_hat = np.array([model.cos_i, 0.0, -model.sin_i])
    b_hat = np.array([0.0, -1.0, 0.0])
    for _ in range(25):
        v = rng.normal(size=3)
        vhat = v / np.linalg.norm(v)
        beta = rng.uniform(0.05, 0.98)
        gamma = 1.0 / np.sqrt(1.0 - beta * beta)
        # aberration, exactly as the model computes it
        k_par = float(vhat @ n)
        one_m = 1.0 - beta * k_par
        k_perp = n - k_par * vhat
        kp = k_perp / (gamma * one_m) + ((k_par - beta) / one_m) * vhat
        khat = kp / np.linalg.norm(kp)
        out = []
        for u in (a_hat, b_hat):
            s = float(vhat @ u)
            out.append(u + (gamma - 1.0) * s * vhat + gamma * beta * s * khat)
        A, B = out
        assert A @ A == pytest.approx(1.0, abs=1e-12)
        assert B @ B == pytest.approx(1.0, abs=1e-12)
        assert A @ B == pytest.approx(0.0, abs=1e-12)
        assert A @ khat == pytest.approx(0.0, abs=1e-12)
        assert B @ khat == pytest.approx(0.0, abs=1e-12)


def test_evpa_is_perpendicular_to_the_projected_field_east_of_north():
    """
    The defining convention: for optically thin emission the electric-vector position
    angle is perpendicular to the projected fluid-frame magnetic field, measured East
    of North in the radio convention.

    In model terms the observer sits on the -z_im side, so the un-mirrored sky view
    has -x_im to the right and +y_im up, putting North along +y_im and East along
    +x_im (sky_view may rotate by 180 degrees, which flips both and leaves Q and U
    alone).  Checked with the boost switched off, where the projection is unambiguous.
    """
    model = jf.JetModel(m=10.0, a=0.5, inc=55.0, mdot=1e-3, Nx=2, Ny=2, Nz=2)
    north = np.array([0.0, 1.0, 0.0])  # +y_im in jet coordinates
    east = np.array([model.cos_i, 0.0, -model.sin_i])  # +x_im
    khat = np.array([model.nx, model.ny, model.nz])
    # the sky triad must be right handed about the propagation direction
    assert np.dot(np.cross(north, east), khat) == pytest.approx(1.0, abs=1e-12)
    rng = np.random.default_rng(5)
    for _ in range(30):
        B = rng.normal(size=3)
        Bmag = np.linalg.norm(B)
        one = np.ones(1)
        cos2chi, sin2chi = model._sky_basis_numpy(
            one, 0.0 * one,
            (0.0 * one, 0.0 * one, 0.0 * one),
            tuple(k * one for k in khat),
            tuple(b * one for b in B),
            Bmag * one,
        )
        # optically thin emission has j_Q > 0, so the Stokes pair is the rotation alone
        chi = 0.5 * np.arctan2(_U_SIGN * sin2chi[0], _Q_SIGN * cos2chi[0])
        pa_B = np.arctan2(B @ east, B @ north)  # position angle of B, East of North
        delta = (chi - (pa_B + 0.5 * np.pi)) % np.pi
        assert min(delta, np.pi - delta) == pytest.approx(0.0, abs=1e-9)


def test_cos2chi_and_sin2chi_are_a_rotation():
    model = jf.JetModel(m=10.0, a=0.5, inc=55.0, mdot=1e-3, Nx=2, Ny=2, Nz=2)
    rng = np.random.default_rng(7)
    B = rng.normal(size=(3, 40))
    Bmag = np.linalg.norm(B, axis=0)
    khat = np.array([model.nx, model.ny, model.nz])
    ones = np.ones(40)
    c2, s2 = model._sky_basis_numpy(
        ones, 0.0 * ones, (0.0 * ones,) * 3, tuple(k * ones for k in khat),
        (B[0], B[1], B[2]), Bmag,
    )
    assert np.allclose(c2 * c2 + s2 * s2, 1.0, atol=1e-12)


@numba_only
def test_module_level_sign_conventions_agree():
    from jetfuncs import _kernel_pol as kp

    assert kp._U_SIGN == _U_SIGN
    assert kp._Q_SIGN == _Q_SIGN
    assert kp._V_PREFAC == pytest.approx(4.0 / 3.0)


# --------------------------------------------------------------------------- images
def _compare(A, B, rtol, ref=None):
    ref = A[0] if ref is None else ref
    good = ref > ref.max() * 1e-9
    assert good.sum() > 0
    for k in range(4):
        scale = np.max(np.abs(A[k][good]))
        if scale == 0.0:
            continue
        assert np.max(np.abs(A[k][good] - B[k][good])) <= rtol * scale, f"Stokes {'IQUV'[k]}"


@numba_only
@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("frequency", [8.0, 230.0, 3.0e4])
def test_numba_matches_numpy(name, frequency):
    model = jf.JetModel(**CONFIGS[name])
    _, _, S_np = model.make_image_polarized(frequency, backend="numpy")
    _, _, S_nb = model.make_image_polarized(frequency, backend="numba")
    assert S_nb.shape == (4, model.Ny, model.Nx)
    _compare(S_np, S_nb, rtol=1e-9)


@numba_only
@pytest.mark.parametrize("name", list(CONFIGS))
def test_polarized_state_matches_the_full_kernel(name):
    model = jf.JetModel(**CONFIGS[name])
    freqs = (2.0, 86.0, 1.0e5)
    full = [model.make_image_polarized(f)[2] for f in freqs]
    nbytes = model.precompute_state(polarized=True)
    assert nbytes == 76 * model.n_jet_cells
    for f, S_full in zip(freqs, full):
        _compare(S_full, model.make_image_polarized(f)[2], rtol=1e-11, ref=S_full[0])
    model.clear_state()
    assert model._state is None


@numba_only
def test_polarized_state_also_serves_the_unpolarized_transfer():
    """The polarized state is a superset; make_image() must read it and be unchanged."""
    model = jf.JetModel(**CONFIGS["linear"])
    _, _, I_ref = model.make_image(86.0)
    model.precompute_state(polarized=True)
    _, _, I_cached = model.make_image(86.0)
    good = I_ref > I_ref.max() * 1e-9
    np.testing.assert_allclose(I_cached[good], I_ref[good], rtol=1e-11)


@numba_only
def test_field_table_path_runs_and_stays_close():
    model = jf.JetModel(**CONFIGS["linear"])
    _, _, S_exact = model.make_image_polarized(230.0)
    model.build_field_table(points_per_decade=200, n_u=256)
    _, _, S_tab = model.make_image_polarized(230.0)
    # bilinear interpolation of the axisymmetric physics, not an exact path
    _compare(S_exact, S_tab, rtol=5e-3)


@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("frequency", [5.0, 43.0, 230.0, 1.0e4])
def test_polarized_images_are_physical(name, frequency):
    """I >= 0 everywhere, and the polarized intensity never exceeds the total."""
    model = jf.JetModel(**CONFIGS[name], backend="numpy" if not jf.numba_available() else "auto")
    _, _, S = model.make_image_polarized(frequency)
    assert np.all(np.isfinite(S))
    assert np.all(S[0] >= 0.0)
    pol = np.sqrt(S[1] ** 2 + S[2] ** 2 + S[3] ** 2)
    lit = S[0] > S[0].max() * 1e-12
    assert np.all(pol[lit] <= S[0][lit] * (1.0 + 1e-9))


def test_stokes_I_is_close_to_the_unpolarized_transfer():
    """
    Polarized absorption feeds Q, U and V back into I, so the two transfers do not give
    identical Stokes I -- but the difference must stay small, which is the regime in
    which the unpolarized model was published.
    """
    model = jf.JetModel(**CONFIGS["linear"])
    _, _, I_unpol = model.make_image(230.0)
    _, _, S = model.make_image_polarized(230.0)
    good = I_unpol > I_unpol.max() * 1e-6
    rel = np.abs(S[0][good] - I_unpol[good]) / I_unpol[good]
    assert rel.max() < 0.05
    tot = np.abs(S[0].sum() - I_unpol.sum()) / I_unpol.sum()
    assert tot < 0.01


def test_transfer_reduces_to_the_unpolarized_one_without_polarized_coefficients(monkeypatch):
    """
    With j_Q, j_V, alpha_Q, alpha_V, rho_Q and rho_V switched off, the full-Stokes
    transfer must reproduce the published unpolarized image exactly.  This isolates the
    transfer itself -- in particular that rays accumulate front to back, so that each
    cell's emission is attenuated by the material between it and the observer and not
    by the material behind it -- from the polarized physics layered on top of it.
    A low frequency is used so that much of the image is optically thick, where the
    direction of accumulation matters.
    """
    model = jf.JetModel(**CONFIGS["linear"], backend="numpy")
    original = type(model)._pol_coefficients_numpy

    def unpolarized(self, nu_nup, nup, gamma_c, costhetaB, nA):
        jI, jQ, jV, aI, aQ, aV, rQ, rV = original(self, nu_nup, nup, gamma_c, costhetaB, nA)
        z = np.zeros_like(jI)
        return jI, z, z, aI, z, z, z, z

    monkeypatch.setattr(type(model), "_pol_coefficients_numpy", unpolarized)
    _, _, S = model.make_image_polarized(5.0)
    _, _, I_unpol = model.make_image(5.0)
    good = I_unpol > I_unpol.max() * 1e-9
    np.testing.assert_allclose(S[0][good], I_unpol[good], rtol=1e-11)
    assert np.all(S[1] == 0.0) and np.all(S[2] == 0.0) and np.all(S[3] == 0.0)


def test_tau_stop_truncates_the_ray():
    model = jf.JetModel(**CONFIGS["linear"])
    _, _, S_all = model.make_image_polarized(8.0)
    _, _, S_cut = model.make_image_polarized(8.0, tau_stop=1.0)
    assert np.all(S_cut[0] <= S_all[0] * (1.0 + 1e-12))
    assert S_cut[0].sum() < S_all[0].sum()


def test_polarized_tables_are_built_on_demand():
    """A model made with the default stokes='I' can still be imaged in full Stokes."""
    model = jf.JetModel(**CONFIGS["linear"])
    assert model.stokes == "I"
    assert not hasattr(model, "GQ_p")
    _, _, S = model.make_image_polarized(230.0)
    assert model.stokes == "IQV"
    assert np.all(np.isfinite(S))


def test_make_image_polarized_rejects_a_bad_heating_prescription():
    model = jf.JetModel(**CONFIGS["linear"])
    with pytest.raises(ValueError):
        model.make_image_polarized(230.0, heating_prescription="poynting")


# --------------------------------------------------------------------------- rotativities
def _hs11_reference(nA, nuHz, nuB_sin, cth, p1, p2, g1, g2, g3, N=40001):
    """
    Huang & Shcherbakov (2011) Eqs. (55) and (56) integrated directly over the broken
    power law, with exact kinematics: the ground truth the tabulated scheme in
    _rotativities.py is an approximation to.
    """
    from jetfuncs import _rotativities as rm

    e, me, c = 4.8032045e-10, 9.10938e-28, 2.99792458e10
    XA = np.sqrt(np.sqrt(2) * (nuB_sin / nuHz) / 1e-4)
    sth = np.sqrt(max(1.0 - cth * cth, 1e-30))
    g = np.exp(np.linspace(np.log(g1), np.log(g3), N))
    mom = np.sqrt(np.maximum(g * g - 1.0, 0.0))
    S = np.where(g < g2, g ** (-p1), g2 ** (p2 - p1) * g ** (-p2))
    f = nA * S / (4 * np.pi * g * mom)
    L = np.log((1 + mom / g) / np.maximum(1 - mom / g, 1e-300))
    cQ = 4 * np.pi * (e * e / (me * c)) / nuHz
    cV = 4 * np.pi * (e * e / (me * c)) * (nuB_sin / sth) / nuHz**2
    rQ = np.trapz(f * cQ * XA * rm.H_X(XA * g, g) * g, np.log(g))
    rV = np.trapz(f * cV * L * rm.g_X(XA * g) * cth * g, np.log(g))
    # boundary term at gamma_min; the one at gamma_max is suppressed by gamma_max^-(p+2)
    m1 = np.sqrt(max(g1 * g1 - 1.0, 0.0))
    L1 = np.log((1 + m1 / g1) / max(1 - m1 / g1, 1e-300))
    ff = nA * g1 ** (-p1) / (4 * np.pi * g1 * m1)
    rQ += ff * cQ * float(rm.H_B(np.array(XA * g1), np.array(g1)))
    rV += ff * cV * (g1 * L1 - 2 * m1) * float(rm.g_B(np.array(XA * g1))) * cth
    return float(rQ), float(rV)


def _call_rotativities(model, branch, nu_nup, cth, nA_over_nup, gamma_c):
    """_rotativities_numpy on a single cell, with K_a/nu_p supplied directly."""
    arr = lambda v: np.full(1, float(v))
    sth = np.sqrt(1.0 - cth * cth)
    return model._rotativities_numpy(
        arr(nu_nup), arr(cth / sth), arr(nA_over_nup), branch, arr(gamma_c)
    )


# (branch, gamma_c) covering uncooled, slow and fast cooling
_ROT_CASES = [(0, 1e9), (1, 1.0e4), (2, 3.0)]


@pytest.mark.parametrize("branch,gamma_c", _ROT_CASES)
@pytest.mark.parametrize("nu_nup", [3.0e2, 3.0e4, 3.0e6])
def test_hs11_matches_direct_numerical_integration(branch, gamma_c, nu_nup):
    """
    The tabulated, factorised HS11 scheme against a brute-force integration of the
    same fitting formulae with exact kinematics.  The difference is the
    ultrarelativistic simplification that makes the integral separable, which the
    module docstring quotes as ~2% on rho_Q and ~0.1% on rho_V.
    """
    p, gm, gx = 2.23, 10.0, 1e8
    model = jf.JetModel(
        m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, p=p,
        gamma_m=gm, gamma_max=gx, rotativities="HS11",
    )
    cth = 0.5
    # K_a/nu_p is the scale the schemes are written in; any positive value will do
    prefac_absorp = model.prefac_absorp
    nA = 1.0e-4
    nup_ghz = 1.0
    Kap = prefac_absorp * nA / nup_ghz
    rQ, rV = _call_rotativities(model, branch, nu_nup, cth, Kap, gamma_c)

    if branch == 0:
        p1, p2, g1, g2, g3 = p, p, gm, gx, gx
    elif branch == 1:
        p1, p2, g1, g2, g3 = p, p + 1.0, gm, gamma_c, gx
    else:
        p1, p2, g1, g2, g3 = 2.0, p + 1.0, gamma_c, gm, gx
    nuHz = nu_nup * nup_ghz * 1e9
    nuB_sin = (2.0 / 3.0) * nup_ghz * 1e9
    eq, ev = _hs11_reference(nA, nuHz, nuB_sin, cth, p1, p2, g1, g2, g3)

    assert float(rQ[0]) == pytest.approx(eq, rel=0.05)
    assert float(rV[0]) == pytest.approx(ev, rel=0.02)


@pytest.mark.parametrize("nu_nup", [1.0e3, 1.0e5])
def test_jo77_reduces_to_the_single_power_law_form(nu_nup):
    """
    With no cooling break the finite-limit JO77 scheme must reproduce the textbook
    power-law expression, because the bracket of Dexter (2016) Eq. (B1) is identically
    2 int_{gamma_m}^{gamma_*} gamma^(1-p) dgamma.  This pins the integral identity the
    generalization to a broken power law rests on.
    """
    p, gm, gx = 2.23, 10.0, 1e8
    model = jf.JetModel(
        m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, p=p,
        gamma_m=gm, gamma_max=gx, rotativities="JO77",
    )
    cth = 0.5
    Kap = model.prefac_absorp * 1.0e-4 / 1.0
    rQ, _ = _call_rotativities(model, 0, nu_nup, cth, Kap, 1e9)
    # Dexter (2016) B1 with Sazonov's sign and bracket argument, as SHAKO writes it
    a = 0.5 * (p - 2.0)
    y = gm * gm / (1.5 * nu_nup)  # = (2/3)/x_1
    br = -np.expm1(a * np.log(y)) / a
    expect = (16.0 * np.sqrt(3.0) / 9.0) * Kap * gm ** (2.0 - p) * br / nu_nup**3
    assert float(rQ[0]) == pytest.approx(expect, rel=1e-10)


def test_hs11_is_the_better_approximation_for_faraday_conversion():
    """
    Swept across all three cooling branches and five decades of nu'/nu_p: HS11 tracks
    the exact integral of the fitting formulae everywhere, while the power-law form
    does not.  That is what makes HS11 the default.  rho_V, by contrast, is
    insensitive to the choice -- the two schemes differ only in Faraday conversion,
    which is what HS11 set out to fix.

    Stated as a sweep rather than at a single point on purpose: the two schemes happen
    to agree at particular combinations, so a point comparison proves nothing.

    The HS11 bound is split by gamma_1 on purpose.  The factorisation that makes the
    integral separable assumes ultrarelativistic kinematics, and its error tracks
    sqrt(1 - 1/gamma_1): measured, 1.9% at gamma_1 = 10, 2.1% at 9.9, 5.8% at 3 and
    9.4% at 1.5.  Fast cooling always has gamma_1 = gamma_c < gamma_m, so it is the
    branch that probes the mildly relativistic end -- where the ultrarelativistic
    synchrotron kernels the emission coefficients use are equally approximate.
    """
    p, gm, gx = 2.23, 10.0, 1e8
    kw = dict(m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, p=p,
              gamma_m=gm, gamma_max=gx)
    hs = jf.JetModel(**kw, rotativities="HS11")
    jo = jf.JetModel(**kw, rotativities="JO77")
    Kap = hs.prefac_absorp * 1.0e-4
    cth = 0.5
    err_q_hs, err_q_jo, err_v_hs, err_v_jo = [], [], [], []
    hot_q, hot_v = [], []  # the ultrarelativistic subset, gamma_1 >= 10
    for branch, gamma_c in _ROT_CASES:
        for nu_nup in (3.0e2, 3.0e3, 3.0e4, 3.0e5, 3.0e6):
            if branch == 0:
                p1, p2, g1, g2, g3 = p, p, gm, gx, gx
            elif branch == 1:
                p1, p2, g1, g2, g3 = p, p + 1.0, gm, gamma_c, gx
            else:
                p1, p2, g1, g2, g3 = 2.0, p + 1.0, gamma_c, gm, gx
            eq, ev = _hs11_reference(
                1.0e-4, nu_nup * 1e9, (2.0 / 3.0) * 1e9, cth, p1, p2, g1, g2, g3
            )
            qh, vh = _call_rotativities(hs, branch, nu_nup, cth, Kap, gamma_c)
            qj, vj = _call_rotativities(jo, branch, nu_nup, cth, Kap, gamma_c)
            eqh = abs(float(qh[0]) / eq - 1.0)
            evh = abs(float(vh[0]) / ev - 1.0)
            err_q_hs.append(eqh)
            err_q_jo.append(abs(float(qj[0]) / eq - 1.0))
            err_v_hs.append(evh)
            err_v_jo.append(abs(float(vj[0]) / ev - 1.0))
            if g1 >= 10.0:
                hot_q.append(eqh)
                hot_v.append(evh)

    # ultrarelativistic regime: the factorisation is good to a couple of per cent
    assert max(hot_q) < 0.025
    assert max(hot_v) < 0.005
    # mildly relativistic (fast cooling, gamma_1 = gamma_c = 3): degrades, but bounded
    assert max(err_q_hs) < 0.08
    assert max(err_v_hs) < 0.02
    # and HS11 is uniformly the better approximation, in the worst case and typically
    assert max(err_q_jo) > 5.0 * max(err_q_hs)
    assert np.median(err_q_jo) > 2.0 * np.median(err_q_hs)
    # rho_V is insensitive to the scheme
    assert max(err_v_jo) < 0.07


def test_rotativities_none_switches_them_off():
    model = jf.JetModel(**CONFIGS["linear"], rotativities="none")
    q, v = _call_rotativities(model, 1, 1.0e4, 0.5, 1.0e-20, 1.0e4)
    assert float(q[0]) == 0.0 and float(v[0]) == 0.0
    # and the image is still a valid polarized image
    _, _, S = model.make_image_polarized(230.0)
    assert np.all(np.isfinite(S))
    pol = np.sqrt(S[1] ** 2 + S[2] ** 2 + S[3] ** 2)
    lit = S[0] > S[0].max() * 1e-12
    assert np.all(pol[lit] <= S[0][lit] * (1.0 + 1e-9))


def test_rotativity_scheme_is_validated():
    with pytest.raises(ValueError, match="rotativities"):
        jf.JetModel(**CONFIGS["linear"], rotativities="Jones77")


def test_rotativity_tables_track_the_spectral_index():
    model = jf.JetModel(**CONFIGS["linear"])
    model._load_rotativity_tables()
    first = model._rtab
    model._load_rotativity_tables()
    assert model._rtab is first  # cached, not rebuilt
    assert model._rtab.shape[0] == len(jf._rotativities.ROT_ROWS)
    assert np.all(np.isfinite(model._rtab))


@numba_only
@pytest.mark.parametrize("scheme", ["HS11", "JO77", "none"])
def test_numba_matches_numpy_for_every_rotativity_scheme(scheme):
    model = jf.JetModel(**CONFIGS["linear"], rotativities=scheme)
    _, _, S_np = model.make_image_polarized(43.0, backend="numpy")
    _, _, S_nb = model.make_image_polarized(43.0, backend="numba")
    _compare(S_np, S_nb, rtol=1e-9)


# --------------------------------------------------------------------------- sky view
def test_sky_view_puts_the_approaching_jet_on_the_right():
    """
    sky_view re-labels the grid so that the picture is the observer's view: the
    approaching (brighter) jet points right, and no mirror image is taken.  It must
    not touch the Stokes values, which are referred to (North, East) and so survive
    the 180 degree rotation unchanged.
    """
    model = jf.JetModel(**CONFIGS["linear"])
    x, y, IQUV = model.make_image_polarized(230.0)
    X, Y, D = jf.sky_view(x, y, IQUV)
    assert D.shape == IQUV.shape
    assert np.all(np.diff(X) > 0.0) and np.all(np.diff(Y) > 0.0)
    assert np.nansum(D[0] * X[None, :]) > 0.0  # brighter jet on the right
    # a pure re-labelling: every Stokes value survives, with its sign
    for k in range(4):
        np.testing.assert_array_equal(np.sort(D[k].ravel()), np.sort(IQUV[k].ravel()))
    # a plain 2-D map is re-oriented the same way
    X2, Y2, T = jf.sky_view(x, y, IQUV[0])
    np.testing.assert_array_equal(T, D[0])
    np.testing.assert_allclose(X2, X)
    np.testing.assert_allclose(Y2, Y)


def test_sky_view_validates_its_input():
    model = jf.JetModel(**CONFIGS["linear"])
    x, y, IQUV = model.make_image_polarized(230.0)
    with pytest.raises(ValueError, match="Stokes axis"):
        jf.sky_view(x, y, IQUV[:3])
    with pytest.raises(ValueError, match="must end in"):
        jf.sky_view(x, y, np.zeros((len(y) + 1, len(x))))
    with pytest.raises(ValueError):
        jf.sky_view(x, y, np.zeros((2, 2, len(y), len(x))))


# --------------------------------------------------------------------------- helpers
def test_polarization_fractions_and_evpa():
    I = np.array([2.0, 1.0, 0.0])
    Q = np.array([1.0, 0.0, 0.0])
    U = np.array([0.0, 0.5, 0.0])
    V = np.array([-0.2, 0.1, 0.0])
    m_lin, m_circ, chi = jf.polarization_fractions(np.array([I, Q, U, V]))
    assert m_lin[0] == pytest.approx(0.5)
    assert m_circ[0] == pytest.approx(-0.1)
    assert np.isnan(m_lin[2]) and np.isnan(m_circ[2])
    assert chi[0] == pytest.approx(0.0)
    assert chi[1] == pytest.approx(np.pi / 4.0)
    # the four-argument form and degrees
    m2, c2, chi2 = jf.polarization_fractions(I, Q, U, V, degrees=True)
    assert np.allclose(m2[:2], m_lin[:2])
    assert chi2[1] == pytest.approx(45.0)
    assert jf.evpa(-1.0, 0.0, degrees=True) == pytest.approx(90.0)


def test_polarization_fractions_input_validation():
    with pytest.raises(ValueError):
        jf.polarization_fractions(np.zeros((3, 5)))
    with pytest.raises(ValueError):
        jf.polarization_fractions(np.zeros(5), np.zeros(5))


def test_get_quantity_polarized_entries():
    model = jf.JetModel(m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2)
    r = np.array([2.2, 3.0, 3.8])
    th = np.array([np.pi / 6.0, np.pi / 4.0, np.pi / 3.0])
    c2 = model.get_quantity(r, th, quantity="cos2chi")
    s2 = model.get_quantity(r, th, quantity="sin2chi")
    assert np.allclose(c2 * c2 + s2 * s2, 1.0, atol=1e-12)
    chi = model.get_quantity(r, th, quantity="EVPA")
    assert np.all(np.isfinite(chi))
    for q in ("jQ", "jV", "alphaQ", "alphaV", "rhoQ", "rhoV"):
        val = model.get_quantity(r, th, quantity=q, frequency=230.0)
        assert np.shape(val) == np.shape(r)
        assert np.all(np.isfinite(val))
    jI = model.get_quantity(r, th, quantity="jI", frequency=230.0)
    jQ = model.get_quantity(r, th, quantity="jQ", frequency=230.0)
    assert np.all(np.abs(jQ) <= jI)
    with pytest.raises(Exception, match="frequency"):
        model.get_quantity(r, th, quantity="rhoQ")


def test_export_fits_writes_a_stokes_cube(tmp_path):
    from astropy.io import fits

    model = jf.JetModel(**CONFIGS["linear"])
    x, y, S = model.make_image_polarized(230.0)
    x_new, y_new, I_new = jf.interp_to_regular_grid(S[0], x, y, nx=16, ny=16)
    cube = np.stack(
        [jf.interp_to_regular_grid(S[k], x, y, nx=16, ny=16)[2] for k in range(4)]
    )
    path = tmp_path / "cube.fits"
    jf.export_fits(str(path), cube, x_new * 1e-9, y_new * 1e-9, bunit="Jy/pix")
    with fits.open(path) as hdul:
        assert hdul[0].data.shape == (4, 16, 16)
        assert hdul[0].header["CTYPE3"] == "STOKES"
        assert hdul[0].header["CRVAL3"] == 1.0
