"""
Kerr null geodesics (_geodesics.py): the rays, the photon's local frame, and the
observer's Stokes basis carried along each ray.

Checks against results the code does not produce itself: the first integrals of the
geodesic equation, the post-Newtonian deflection of a Schwarzschild ray, Bardeen's
critical curve, the straight rays of flat space (M = a = 0), and a direct numerical
integration of the parallel-transport equation in Boyer-Lindquist coordinates.
"""

import math

import numpy as np
import pytest

import jetfuncs as jf

numba_only = pytest.mark.skipif(not jf.numba_available(), reason="numba is not installed")

if jf.numba_available():
    from jetfuncs import _geodesics as geo

NMAX = 200000


def _trace(alpha, beta, M, a, theta_o, r0, eps=None, wcap=None):
    out = np.empty((6, NMAX))
    eps = geo.EPS_STEP if eps is None else eps
    wcap = geo.WCAP_STEP if wcap is None else wcap
    n, status, lam, eta = geo.trace(alpha, beta, M, a, theta_o, r0, eps, wcap, out)
    return out[:, :n], n, status, lam, eta


@numba_only
@pytest.mark.parametrize("alpha,beta", [(8.0, 3.0), (-20.0, 15.0), (40.0, -60.0), (5.5, 0.5),
                                        (-3.0, 6.0)])
def test_first_integrals_and_the_null_condition_hold(alpha, beta):
    a, theta_o = 0.6, math.radians(17.0)
    traj, n, status, lam, eta = _trace(alpha, beta, 1.0, a, theta_o, 1.0e6)
    rH = 1.0 + math.sqrt(1.0 - a * a)
    for k in range(0, n, max(1, n // 300)):
        w, u, pw, pu = traj[0, k], traj[1, k], traj[2, k], traj[3, k]
        P, Th = geo.potentials(w, u, lam, eta, 1.0, a)
        assert abs(pw * pw - P) < 1e-8
        assert abs(pu * pu - Th) < 1e-8 * max(eta + lam * lam, 1.0)
        if 1.0 / w > 1.05 * rH:
            n0, n1, n2 = geo.zamo_frame(w, u, pw, pu, lam, 1.0, a)[:3]
            assert abs(math.sqrt(n0 * n0 + n1 * n1 + n2 * n2) - 1.0) < 1e-7


@numba_only
@pytest.mark.parametrize("alpha,beta", [(0.3, 0.3), (-0.3, 0.3), (0.1, -0.05)])
def test_vortical_rays_are_traced(alpha, beta):
    """
    Close to the axis of a spinning hole, rays with eta < 0 stay on one side of the
    equator, and eta + lambda^2 can be negative; they must still be traced to the
    horizon with their first integrals intact.
    """
    a, theta_o = 0.5, math.radians(30.0)
    lam, eta = geo.conserved_quantities(alpha, beta, a, theta_o)
    assert eta < 0.0 and eta + lam * lam < 0.0
    traj, n, status, lam, eta = _trace(alpha, beta, 1.0, a, theta_o, 1.0e9)
    assert status == geo.CAPTURED
    assert np.all(np.isfinite(traj))
    assert np.all(traj[1] * np.sign(math.cos(theta_o)) > 0.0)     # never crosses the equator
    for k in range(n):
        P, Th = geo.potentials(traj[0, k], traj[1, k], lam, eta, 1.0, a)
        assert abs(traj[2, k] ** 2 - P) < 1e-8
        assert abs(traj[3, k] ** 2 - Th) < 1e-8


@numba_only
@pytest.mark.parametrize("b,tol", [(100.0, 2e-6), (30.0, 3e-4)])
def test_schwarzschild_deflection_matches_the_post_newtonian_series(b, tol):
    """
    In the equatorial plane of a = 0 the azimuth advances at d phi / d tau = |lambda|,
    so the deflection is |lambda| tau_total - pi (plus the two asymptotic tails b/r).
    Series: 4/b + (15 pi/4)/b^2 + (128/3)/b^3; the next term is ~(3465 pi/64)/b^4.
    """
    r0 = 1.0e9
    traj, n, status, lam, eta = _trace(b, 0.0, 1.0, 0.0, math.pi / 2, r0)
    assert status == geo.ESCAPED
    r_end = 1.0 / traj[0, n - 1]
    deflection = abs(lam) * traj[5, n - 1] + math.asin(b / r0) + math.asin(b / r_end) - math.pi
    series = 4.0 / b + (15.0 * math.pi / 4.0) / b**2 + (128.0 / 3.0) / b**3
    assert deflection == pytest.approx(series, abs=tol)


@numba_only
def test_schwarzschild_capture_switches_on_at_three_root_three():
    b_crit = 3.0 * math.sqrt(3.0)
    for b, captured in ((b_crit - 0.003, True), (b_crit + 0.003, False)):
        status = _trace(b, 0.0, 1.0, 0.0, math.radians(30.0), 1.0e4)[2]
        assert (status == geo.CAPTURED) == captured


@numba_only
def test_capture_follows_bardeens_critical_curve():
    """
    The shadow edge for a = 0.6 seen at 17 deg: screen points of the spherical photon
    orbits (Bardeen 1973), slightly inside are captured and slightly outside escape.
    """
    a, theta_o = 0.6, math.radians(17.0)
    checked = 0
    for r in np.linspace(2.0, 4.0, 41):
        lam = (r * r * (3.0 - r) - a * a * (r + 1.0)) / (a * (r - 1.0))
        eta = r**3 * (4.0 * a * a - r * (r - 3.0) ** 2) / (a * a * (r - 1.0) ** 2)
        beta2 = eta + a * a * math.cos(theta_o) ** 2 - lam * lam / math.tan(theta_o) ** 2
        if eta < 0.0 or beta2 <= 0.0:
            continue
        alpha = -lam / math.sin(theta_o)
        for sign in (1.0, -1.0):
            beta = sign * math.sqrt(beta2)
            inner = _trace(0.995 * alpha, 0.995 * beta, 1.0, a, theta_o, 1.0e4)[2]
            outer = _trace(1.005 * alpha, 1.005 * beta, 1.0, a, theta_o, 1.0e4)[2]
            assert inner == geo.CAPTURED
            assert outer == geo.ESCAPED
            checked += 1
    assert checked > 10


def _flat_ray(x_im, y_im, inc_deg):
    """The flat model's straight ray: points X(z_im) and its constant direction."""
    i = math.pi - math.radians(inc_deg)
    ex = np.array([math.cos(i), 0.0, -math.sin(i)])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([math.sin(i), 0.0, math.cos(i)])
    return lambda z: x_im * ex + y_im * ey + z * ez, ex, ey, ez


def _z_on_line(w, pw, x_im, y_im):
    """
    The flat model's z_im of the stored points (w, pw) of a flat-limit ray: the line's
    point at radius r = 1/w on the leg the ray is on (moving in, pw > 0: z_im < 0).
    """
    r = 1.0 / np.asarray(w)
    return np.sqrt(np.maximum(r * r - x_im * x_im - y_im * y_im, 0.0)) * np.where(np.asarray(pw) > 0.0, -1.0, 1.0)


def _affine_tolerance(span):
    """
    Error bound of affine-parameter differences over a span of the ray.  The affine
    parameter is the Mino-time integral of 1/w^2 + a^2 u^2.  Far out, where each step
    changes w by the fraction wcap, fourth-order Runge-Kutta integrates it with a
    relative error of ~wcap^4/24 (Simpson's rule on 1/w^2), so differences err by up to
    (wcap^4/24) span, 2.6e-7 span for wcap = 0.05; 25% is added for the next order in
    wcap (test_affine_parameter_converges_at_fourth_order measures both).
    """
    return 1.25 * (geo.WCAP_STEP ** 4 / 24.0) * np.abs(span)


def _closest_approach_tolerance(r_J, inc_deg):
    """
    Precision of lambda_J, the minimum of R = r sin(theta) along the ray.  A minimum
    search locates a quadratic minimum to sqrt(2 dR / R''), with dR the noise in R.
    Positions carry a relative roundoff of machine epsilon, which, because the ray
    state holds cos(theta), gives R a noise of eps r^2 / R near the axis; with
    R'' = sin^2(i) / R for a straight ray crossing the axis at angle i, the precision is
    sqrt(2 eps) r_J / sin(i), with r_J the radius of the closest approach; times 4.
    """
    return 4.0 * math.sqrt(2.0 * np.finfo(float).eps) * r_J / math.sin(math.radians(inc_deg))


@numba_only
@pytest.mark.parametrize("x_im,y_im,r0", [(-60.5, 25.5, 1.0e6), (7.5, -3.5, 1.0e9),
                                          (120.5, 0.5, 1.0e6), (900.5, 200.5, 1.0e9)])
def test_flat_limit_rays_are_the_flat_models_straight_rays(x_im, y_im, r0):
    """
    With M = a = 0 the ray through pixel (x_im, y_im) is the flat model's straight line:

    * its track: every stored point (r, cos theta) lies on the line, to roundoff;
    * its closest approach to the jet axis, lambda_J, is the flat model's z_J;
    * its cells: the affine offset s = lambda - lambda_J equals the flat model's
      z_im - z_J, to the error bound of the affine parameter.

    The affine parameter from the start of the ray is not compared: its error grows with
    the distance traveled (0.26 r_g for r0 = 1e6, 260 r_g for 1e9), but it is common to
    every point near the hole and cancels in s, which is all the cells use.
    """
    inc = 17.0
    alpha, beta = geo.screen_coordinates(x_im, y_im)
    traj, n, status, lam, eta = _trace(float(alpha), float(beta), 0.0, 0.0, math.radians(inc), r0)
    assert status == geo.ESCAPED
    X, ex, ey, ez = _flat_ray(x_im, y_im, inc)
    w, u, pw, la = traj[0], traj[1], traj[2], traj[4]
    z = _z_on_line(w, pw, x_im, y_im)
    # away from the turning point in r, where z(r) is ill-conditioned
    good = np.abs(z) > 1.0e-3 / w
    assert good.sum() > 0.9 * n
    P = x_im * ex[None] + y_im * ey[None] + z[good][:, None] * ez[None]
    np.testing.assert_allclose(u[good], P[:, 2] * w[good], rtol=0.0, atol=1e-9)
    # closest approach to the axis: at z_J for the straight ray
    i = math.pi - math.radians(inc)
    z_J = -(x_im * math.cos(i) / math.sin(i))
    lam_J = geo.closest_approach_to_axis(traj, n, 0.0)
    wJ, uJ, pwJ, puJ = geo.state_at(traj, n, lam_J, 0.0)
    tol_J = _closest_approach_tolerance(np.linalg.norm(X(z_J)), inc)
    assert float(_z_on_line(wJ, pwJ, x_im, y_im)) == pytest.approx(z_J, abs=tol_J)
    # the cells' coordinate
    s = la[good] - lam_J
    np.testing.assert_array_less(np.abs(s - (z[good] - z_J)),
                                 _affine_tolerance(s) + tol_J + 8.0 * np.finfo(float).eps * np.abs(la[good]))


@numba_only
def test_affine_parameter_converges_at_fourth_order():
    """
    The affine parameter's error in the flat limit is the Runge-Kutta quadrature error of
    d lambda / d tau = 1/w^2 over the far-out steps, ~(wcap^4/24) per unit length: check
    its size at the default wcap, and that halving wcap divides it by 2^4 = 16.
    """
    inc = 17.0
    x_im, y_im, r0 = 40.5, -10.5, 1.0e6
    alpha, beta = geo.screen_coordinates(x_im, y_im)
    i = math.pi - math.radians(inc)
    z_J = -(x_im * math.cos(i) / math.sin(i))
    slopes = []
    for wcap in (geo.WCAP_STEP, 0.5 * geo.WCAP_STEP):
        traj, n, status, lam, eta = _trace(float(alpha), float(beta), 0.0, 0.0, math.radians(inc),
                                           r0, wcap=wcap)
        lam_J = geo.closest_approach_to_axis(traj, n, 0.0)
        z = _z_on_line(traj[0], traj[2], x_im, y_im)
        s = traj[4] - lam_J
        sel = (np.abs(s) > 1.0e4) & (np.abs(s) < 0.9 * r0)
        slope = np.polyfit(np.abs(s[sel]), np.abs(s[sel] - (z[sel] - z_J)), 1)[0]
        assert slope == pytest.approx(wcap ** 4 / 24.0, rel=0.25)
        slopes.append(slope)
    assert slopes[0] / slopes[1] == pytest.approx(16.0, rel=0.15)


@numba_only
def test_flat_limit_basis_is_the_fixed_sky_basis():
    """With M = a = 0 the carried Stokes basis is the flat model's (+x_im, -y_im) everywhere."""
    inc = 17.0
    x_im, y_im = 30.5, -12.5
    r0 = 1.0e6
    alpha, beta = geo.screen_coordinates(x_im, y_im)
    traj, n, status, lam, eta = _trace(float(alpha), float(beta), 0.0, 0.0, math.radians(inc), r0)
    Ka1, Ka2, Kb1, Kb2 = geo.observer_kappas(traj, float(alpha), float(beta), math.radians(inc), lam, 0.0, 0.0)
    X, ex, ey, ez = _flat_ray(x_im, y_im, inc)
    lam_J = geo.closest_approach_to_axis(traj, n, 0.0)
    checked = 0
    for target in lam_J + np.concatenate([-np.logspace(5, -1, 13), [0.0], np.logspace(-1, 5, 13)]):
        w, u, pw, pu = geo.state_at(traj, n, target, 0.0)
        n0, n1, n2, kt, al, om, r, sq_rr, sq_tt, sq_pp = geo.zamo_frame(w, u, pw, pu, lam, 0.0, 0.0)
        # the azimuth of the point, which the ray state does not carry, from its place
        # on the line
        P = X(float(_z_on_line(w, pw, x_im, y_im)))
        if abs(P[2] / np.linalg.norm(P) - u) > 1e-8:
            continue                                  # at the turning point in r
        phi = math.atan2(P[1], P[0])
        st, ct = math.sqrt(1.0 - u * u), u
        rhat = np.array([st * math.cos(phi), st * math.sin(phi), ct])
        that = np.array([ct * math.cos(phi), ct * math.sin(phi), -st])
        phat = np.array([-math.sin(phi), math.cos(phi), 0.0])
        for K1, K2, expected in ((Ka1, Ka2, ex), (Kb1, Kb2, -ey)):
            f = geo.carried_vector(K1, K2, n0, n1, n2, kt, al, om, r, u, 0.0, sq_rr, sq_tt, sq_pp)
            cart = f[0] * rhat + f[1] * that + f[2] * phat
            np.testing.assert_allclose(cart, expected, atol=1e-8)
        checked += 1
    assert checked >= 24


def _bl_metric(r, th, a):
    """Boyer-Lindquist metric (t, r, theta, phi), M = 1."""
    s2 = math.sin(th) ** 2
    sig = r * r + a * a * math.cos(th) ** 2
    delta = r * r - 2.0 * r + a * a
    g = np.zeros((4, 4))
    g[0, 0] = -(1.0 - 2.0 * r / sig)
    g[0, 3] = g[3, 0] = -2.0 * a * r * s2 / sig
    g[1, 1] = sig / delta
    g[2, 2] = sig
    g[3, 3] = (r * r + a * a + 2.0 * a * a * r * s2 / sig) * s2
    return g


def _christoffel(r, th, a, h=1.0e-6):
    """Gamma^mu_{alpha beta} from central differences of the metric."""
    g = _bl_metric(r, th, a)
    ginv = np.linalg.inv(g)
    dg = np.zeros((4, 4, 4))  # dg[c, i, j] = d g_ij / d x^c
    hr = h * r
    dg[1] = (_bl_metric(r + hr, th, a) - _bl_metric(r - hr, th, a)) / (2.0 * hr)
    dg[2] = (_bl_metric(r, th + h, a) - _bl_metric(r, th - h, a)) / (2.0 * h)
    term = dg.transpose(1, 0, 2) + dg.transpose(1, 2, 0) - dg  # [nu, alpha, beta]
    return 0.5 * np.einsum("mn,nab->mab", ginv, term)


@numba_only
def test_walker_penrose_basis_matches_direct_parallel_transport():
    """
    Along a ray that passes ~7 r_g from a = 0.6 hole, carry the observer's basis vector
    by integrating the parallel-transport equation directly (with the geodesic equation,
    in Boyer-Lindquist coordinates and the affine parameter), and compare it with the
    vector the Walker-Penrose constant gives at the same points.
    """
    a, theta_o = 0.6, math.radians(17.0)
    alpha, beta = 6.0, 4.0
    traj, n, status, lam, eta = _trace(alpha, beta, 1.0, a, theta_o, 1.0e6)
    Ka1, Ka2, Kb1, Kb2 = geo.observer_kappas(traj, alpha, beta, theta_o, lam, 1.0, a)

    def local(target):
        w, u, pw, pu = geo.state_at(traj, n, target, a)
        fr = geo.zamo_frame(w, u, pw, pu, lam, 1.0, a)
        return w, u, fr

    def coord(fr, f=None):
        n0, n1, n2, kt, al, om, r, sq_rr, sq_tt, sq_pp = fr
        if f is None:   # the photon's contravariant momentum
            return np.array([kt / al, kt * n0 / sq_rr, kt * n1 / sq_tt,
                             kt * (om / al + n2 / sq_pp)])
        return np.array([0.0, f[0] / sq_rr, f[1] / sq_tt, f[2] / sq_pp])

    # start where the ray is at r ~ 300, well before the strong field
    rr = 1.0 / traj[0, :n]
    k0 = int(np.argmax(rr < 300.0))
    lam0 = traj[4, k0]
    w, u, fr = local(lam0)
    f0 = geo.carried_vector(Ka1, Ka2, *fr[:3], fr[3], fr[4], fr[5], fr[6], u, a, *fr[7:])
    x = np.array([0.0, 1.0 / w, math.acos(u), 0.0])
    # the module's photon momentum is the forward photon; going away from the observer
    # means integrating the forward geodesic backward, i.e. along -k
    k = -coord(fr)
    f = coord(fr, f0)

    def deriv(state):
        G = _christoffel(state[1], state[2], a)
        kk, ff = state[4:8], state[8:12]
        return np.concatenate([kk, -np.einsum("mab,a,b->m", G, kk, kk),
                               -np.einsum("mab,a,b->m", G, kk, ff)])

    state = np.concatenate([x, k, f])
    s = lam0
    checks = 0
    for target in (lam0 + 150.0, lam0 + 280.0, lam0 + 295.0, lam0 + 305.0, lam0 + 320.0):
        while s < target - 1e-12:
            h = min(0.002 * max(state[1], 2.0), target - s)
            k1 = deriv(state)
            k2 = deriv(state + 0.5 * h * k1)
            k3 = deriv(state + 0.5 * h * k2)
            k4 = deriv(state + h * k3)
            state = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            s += h
        r_d, th_d = state[1], state[2]
        u_d = math.cos(th_d)
        kb = state[4:8]
        # the same ray: the tracer's point at the same affine parameter, to the error
        # bound of affine-parameter differences (see test_flat_limit_rays_...)
        w, u, fr = local(target)
        tol = _affine_tolerance(target - lam0)
        assert abs(1.0 / w - r_d) <= abs(kb[1]) * tol + 1e-7 * r_d
        assert abs(u - u_d) <= abs(math.sin(th_d) * kb[2]) * tol + 1e-7
        # the Walker-Penrose vector at the direct integration's own point, for its own
        # photon momentum (Mino-time rates of the backward ray: dr/dtau = Sigma k^r)
        sig = r_d * r_d + a * a * u_d * u_d
        pw_d = -sig * kb[1] / (r_d * r_d)
        pu_d = -math.sin(th_d) * sig * kb[2]
        fr = geo.zamo_frame(1.0 / r_d, u_d, pw_d, pu_d, lam, 1.0, a)
        n0, n1, n2, kt, al, om, r, sq_rr, sq_tt, sq_pp = fr
        # the directly transported vector, in the ZAMO frame, with its component along
        # the photon removed (f -> f + c k is the same polarization)
        fz_t = al * state[8]
        fz = np.array([sq_rr * state[9], sq_tt * state[10], sq_pp * (state[11] - om * state[8])])
        nvec = np.array([n0, n1, n2])
        fz = fz - (fz_t / kt) * (kt * nvec)
        fz -= np.dot(fz, nvec) * nvec
        fz /= np.linalg.norm(fz)
        carried = np.array(geo.carried_vector(Ka1, Ka2, n0, n1, n2, kt, al, om, r, u_d, a,
                                              sq_rr, sq_tt, sq_pp))
        np.testing.assert_allclose(fz, carried, atol=1e-6)
        checks += 1
    assert checks == 5
