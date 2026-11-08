"""
Null geodesics of the Kerr metric for JetModel(geometry="kerr"): the rays, the photon's
direction and energy in the local zero-angular-momentum (ZAMO) frame, and the observer's
Stokes basis carried along each ray.

COORDINATES AND UNITS.  Boyer-Lindquist (t, r, theta, phi), the coordinates the jet
model is written in, with G = c = M = 1 (lengths in r_g); photons have energy E = 1 at
infinity, and lambda = L/E, eta = Q/E^2 are their conserved angular momentum and Carter
constant.  The mass M and spin a of the ray geometry are arguments throughout, so that
M = a = 0 gives the straight rays of the flat model.

RAYS.  A ray is traced backward from the observer in Mino time tau
(d tau = d lambda_affine / Sigma), in which the radial and polar motions decouple.  With
w = 1/r and u = cos(theta),

    (dw/dtau)^2 = P(w) = 1 + (a^2 - eta - lambda^2) w^2 + 2 M (eta + (lambda - a)^2) w^3
                         - a^2 eta w^4,
    (du/dtau)^2 = Theta(u) = eta - (eta + lambda^2 - a^2) u^2 - a^2 u^4,

and the code integrates the second-order forms w'' = P'(w)/2, u'' = Theta'(u)/2, which
are polynomials, pass through turning points without special handling, and keep the
first integrals to ~1e-9.  The variable must be w rather than r: the first integral
(dr/dtau)^2 - R(r) scales as r^4, so roundoff far from the hole grows by (r0/r)^4 on the
way in, whereas every term of P(w) is of order unity and the observer sits at w = 0.
The affine parameter follows from d lambda_affine / d tau = Sigma = r^2 + a^2 u^2.

Steps are fourth-order Runge-Kutta in tau, of length eps / max(b, 1) with
b^2 = eta + lambda^2 (the scale of both motions; it is negative for vortical rays), shortened near the horizon (where phi
would wind up as 1/Delta, were it integrated) and, far out, to a fractional change wcap
of w, which makes them geometric in r there.  Positions in between are interpolated by
cubic Hermite polynomials in tau, which use the stored derivatives.

CAMERA.  The flat model's image plane has the observer on the -z_im side, at polar angle
`inc` from the jet axis and azimuth pi in the model's own frame.  The ray through pixel
(x_im, y_im) arrives at the screen point (alpha, beta) = (-y_im, -x_im) in Bardeen's
coordinates, lambda = -alpha sin(theta_o) and eta = (alpha^2 - a^2) cos^2(theta_o) +
beta^2.  With M = a = 0 these rays are exactly the flat model's straight rays.

STOKES BASIS.  The observer's basis vectors (+x_im, -y_im), which the flat model keeps
fixed along each straight ray, are theta-hat and phi-hat at the observer.  Along a Kerr
ray each is carried by parallel transport, which conserves its Walker-Penrose constant

    kappa = kappa_1 + i kappa_2 = (A - i B)(r - i a cos(theta)),
    A = (k^t f^r - k^r f^t) + a sin^2(theta) (k^r f^phi - k^phi f^r),
    B = [(r^2 + a^2)(k^phi f^theta - k^theta f^phi) - a (k^t f^theta - k^theta f^t)] sin(theta)

(Chandrasekhar 1983; Connors, Piran & Stark 1980).  At any point, the carried vector is
the combination of two unit vectors perpendicular to the photon's ZAMO-frame direction
whose kappa equals the one it had at the observer: a 2 x 2 linear solve.
"""

import math

import numpy as np
from numba import njit

# default integration parameters; see JetModel(geometry="kerr").  The flat-limit test
# (a Kerr image with M = a = 0 must reproduce the flat model) needs eps <= 0.005 and
# wcap <= 0.05: with 0.02 and 0.2 the cell positions err by ~1e-4 r_g.
EPS_STEP = 0.005
WCAP_STEP = 0.05
MAX_STEPS = 40000

# status codes returned by trace()
CAPTURED = 1
ESCAPED = 2
OUT_OF_STEPS = 3


# ----------------------------------------------------------------------------- the camera
def screen_coordinates(x_im, y_im):
    """Bardeen's screen coordinates (alpha, beta) of the ray through pixel (x_im, y_im)."""
    return -np.asarray(y_im, dtype=float), -np.asarray(x_im, dtype=float)


@njit(cache=True, inline="always")
def conserved_quantities(alpha, beta, a, theta_o):
    """(lambda, eta) of the photon that reaches the observer at screen point (alpha, beta)."""
    so = math.sin(theta_o)
    co = math.cos(theta_o)
    return -alpha * so, (alpha * alpha - a * a) * co * co + beta * beta


# ----------------------------------------------------------------------------- the rays
@njit(cache=True, inline="always")
def potentials(w, u, lam, eta, M, a):
    """P(w) = w^4 R(1/w) and Theta(u): the squared Mino-time rates of w = 1/r and u."""
    P = 1.0 + (a * a - eta - lam * lam) * w * w + 2.0 * M * (eta + (lam - a) ** 2) * w ** 3 \
        - a * a * eta * w ** 4
    Th = eta - (eta + lam * lam - a * a) * u * u - a * a * u ** 4
    return P, Th


@njit(cache=True, inline="always")
def _rhs(w, u, pw, pu, lam, eta, M, a):
    """Mino-time derivatives of (w, u, pw, pu, affine parameter) along a backward ray."""
    a2 = a * a
    aw = (a2 - eta - lam * lam) * w + 3.0 * M * (eta + (lam - a) ** 2) * w * w \
        - 2.0 * a2 * eta * w ** 3
    au = -(eta + lam * lam - a2) * u - 2.0 * a2 * u ** 3
    return pw, pu, aw, au, 1.0 / (w * w) + a2 * u * u


@njit(cache=True)
def trace(alpha, beta, M, a, theta_o, r0, eps, wcap, out):
    """
    Trace the ray that reaches the observer at screen point (alpha, beta) backward from
    radius r0, storing (w, u, dw/dtau, du/dtau, affine parameter, tau) at every step in
    out[:, k] (out has shape (6, nmax)).  Returns (number of stored points, status,
    lambda, eta), with status CAPTURED (the ray ends on the horizon), ESCAPED (it has
    left beyond r0 again) or OUT_OF_STEPS.

    The affine parameter is measured from the start, increasing away from the observer,
    which is the direction in which the cells of a ray are laid out.
    """
    nmax = out.shape[1]
    lam, eta = conserved_quantities(alpha, beta, a, theta_o)
    # the scale of both motions; eta + lambda^2 is negative for the vortical rays of a
    # spinning hole (eta < 0, confined to one side of the equator), seen close to the axis
    b2 = eta + lam * lam
    bscale = math.sqrt(b2) if b2 > 1.0 else 1.0
    if M > 0.0:
        wH = 1.0 / ((M + math.sqrt(M * M - a * a)) * (1.0 + 1.0e-3))
    else:
        wH = 1.0e300
    # start on the ray itself: far from the hole it is the straight line displaced by
    # (alpha, beta) from the line of sight, whose point at radius r0 lies at
    # d n + alpha e_alpha + beta e_beta, with d = sqrt(r0^2 - alpha^2 - beta^2) along the
    # line of sight n and e_beta = (cos theta_o, 0, sin theta_o) (see _start_point)
    w0 = 1.0 / r0
    so = math.sin(theta_o)
    co = math.cos(theta_o)
    d = math.sqrt(r0 * r0 - alpha * alpha - beta * beta)
    u0 = (d * co + beta * so) / r0
    P0, Th0 = potentials(w0, u0, lam, eta, M, a)
    w = w0
    u = u0
    pw = math.sqrt(P0 if P0 > 0.0 else 0.0)                     # moving in
    # du/dtau has the sign of du/dt on the straight line at r0 moving in, which is
    # (beta sin(theta_o) d - cos(theta_o) (alpha^2 + beta^2)) / r0^3: the sign of beta,
    # except for |beta| <~ (alpha^2 / r0) cot(theta_o).  This keeps the rays of an
    # observer at pi - theta_o exact mirror images (u -> -u) of those at theta_o.
    du_sign = beta * so * d - co * (alpha * alpha + beta * beta)
    pu = (1.0 if du_sign >= 0.0 else -1.0) * math.sqrt(Th0 if Th0 > 0.0 else 0.0)
    la = 0.0
    tau = 0.0
    k = 0
    status = OUT_OF_STEPS
    while k < nmax:
        out[0, k] = w
        out[1, k] = u
        out[2, k] = pw
        out[3, k] = pu
        out[4, k] = la
        out[5, k] = tau
        if w > wH:
            status = CAPTURED
            break
        if pw < 0.0 and w < w0:
            status = ESCAPED
            break
        fac = 2.0 * (1.0 - 2.0 * M * w + a * a * w * w)
        if fac > 1.0:
            fac = 1.0
        if fac < 0.01:
            fac = 0.01
        h = fac * eps / bscale
        if pw != 0.0:
            hw = wcap * w / abs(pw)
            if hw < h:
                h = hw
        k1 = _rhs(w, u, pw, pu, lam, eta, M, a)
        k2 = _rhs(w + 0.5 * h * k1[0], u + 0.5 * h * k1[1], pw + 0.5 * h * k1[2],
                  pu + 0.5 * h * k1[3], lam, eta, M, a)
        k3 = _rhs(w + 0.5 * h * k2[0], u + 0.5 * h * k2[1], pw + 0.5 * h * k2[2],
                  pu + 0.5 * h * k2[3], lam, eta, M, a)
        k4 = _rhs(w + h * k3[0], u + h * k3[1], pw + h * k3[2], pu + h * k3[3], lam, eta, M, a)
        w += h * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        u += h * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        pw += h * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        pu += h * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
        la += h * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0
        tau += h
        k += 1
    n = k + 1 if status != OUT_OF_STEPS else k
    return n, status, lam, eta


# ----------------------------------------------------------------------------- along a ray
@njit(cache=True, inline="always")
def _hermite(t, h, f0, d0, f1, d1):
    """Cubic Hermite value and derivative (per unit tau) at fraction t of a step of length h."""
    t2 = t * t
    t3 = t2 * t
    val = (2.0 * t3 - 3.0 * t2 + 1.0) * f0 + (t3 - 2.0 * t2 + t) * h * d0 \
        + (-2.0 * t3 + 3.0 * t2) * f1 + (t3 - t2) * h * d1
    der = ((6.0 * t2 - 6.0 * t) * f0 + (3.0 * t2 - 4.0 * t + 1.0) * h * d0
           + (-6.0 * t2 + 6.0 * t) * f1 + (3.0 * t2 - 2.0 * t) * h * d1) / h
    return val, der


@njit(cache=True, inline="always")
def state_in_step(traj, k, t, a):
    """(w, u, dw/dtau, du/dtau, affine parameter) at fraction t of step k."""
    h = traj[5, k + 1] - traj[5, k]
    w, pw = _hermite(t, h, traj[0, k], traj[2, k], traj[0, k + 1], traj[2, k + 1])
    u, pu = _hermite(t, h, traj[1, k], traj[3, k], traj[1, k + 1], traj[3, k + 1])
    s0 = 1.0 / (traj[0, k] ** 2) + a * a * traj[1, k] ** 2
    s1 = 1.0 / (traj[0, k + 1] ** 2) + a * a * traj[1, k + 1] ** 2
    la, _ = _hermite(t, h, traj[4, k], s0, traj[4, k + 1], s1)
    return w, u, pw, pu, la


@njit(cache=True, inline="always")
def wu_in_step(traj, k, t):
    """(w, u) at fraction t of step k: the values of state_in_step, without the rest."""
    h = traj[5, k + 1] - traj[5, k]
    t2 = t * t
    t3 = t2 * t
    c00 = 2.0 * t3 - 3.0 * t2 + 1.0
    c10 = t3 - 2.0 * t2 + t
    c01 = -2.0 * t3 + 3.0 * t2
    c11 = t3 - t2
    w = c00 * traj[0, k] + c10 * h * traj[2, k] + c01 * traj[0, k + 1] + c11 * h * traj[2, k + 1]
    u = c00 * traj[1, k] + c10 * h * traj[3, k] + c01 * traj[1, k + 1] + c11 * h * traj[3, k + 1]
    return w, u


@njit(cache=True, inline="always")
def step_affine(traj, k, a):
    """
    What the interpolation of the affine parameter over step k needs: its values at the
    ends, and d lambda/d tau = Sigma at the ends times the step's Mino-time length.
    """
    h = traj[5, k + 1] - traj[5, k]
    s0 = 1.0 / (traj[0, k] ** 2) + a * a * traj[1, k] ** 2
    s1 = 1.0 / (traj[0, k + 1] ** 2) + a * a * traj[1, k + 1] ** 2
    return traj[4, k], traj[4, k + 1], h * s0, h * s1


@njit(cache=True, inline="always")
def newton_t(target, l0, l1, s0h, s1h, t):
    """
    The fraction t of a step at which the Hermite interpolant of the affine parameter
    equals target, by Newton's method from the guess t; returns t and d lambda/dt there.
    The affine parameter grows to ~r_start along a ray, so its residual cannot fall below
    a few units in the last place of target (1.2e-7 rg near 1e9); iterating stops there,
    after 1-3 iterations.
    """
    tol = 1.0e-15 * abs(target)
    dla = 0.0
    for _ in range(8):
        t2 = t * t
        t3 = t2 * t
        err = ((2.0 * t3 - 3.0 * t2 + 1.0) * l0 + (t3 - 2.0 * t2 + t) * s0h
               + (-2.0 * t3 + 3.0 * t2) * l1 + (t3 - t2) * s1h) - target
        dla = ((6.0 * t2 - 6.0 * t) * l0 + (3.0 * t2 - 4.0 * t + 1.0) * s0h
               + (-6.0 * t2 + 6.0 * t) * l1 + (3.0 * t2 - 2.0 * t) * s1h)
        if abs(err) <= tol or not dla > 0.0:
            break
        t -= err / dla
        if t < 0.0:
            t = 0.0
        if t > 1.0:
            t = 1.0
    return t, dla


@njit(cache=True, inline="always")
def _cubic(t, f0, d0, f1, d1):
    """The cubic Hermite value at fraction t, with end slopes d0, d1 per unit t."""
    t2 = t * t
    t3 = t2 * t
    return ((2.0 * t3 - 3.0 * t2 + 1.0) * f0 + (t3 - 2.0 * t2 + t) * d0
            + (-2.0 * t3 + 3.0 * t2) * f1 + (t3 - t2) * d1)


@njit(cache=True, inline="always")
def w_max_in_step(traj, k):
    """
    The largest w = 1/r on step k: the maximum of the cubic Hermite interpolant
    (wu_in_step) over the step, at an end or where its derivative vanishes.
    """
    h = traj[5, k + 1] - traj[5, k]
    w0 = traj[0, k]
    w1 = traj[0, k + 1]
    d0 = h * traj[2, k]
    d1 = h * traj[2, k + 1]
    wm = w0 if w0 > w1 else w1
    # dw/dt = A t^2 + B t + C
    A = 6.0 * w0 + 3.0 * d0 - 6.0 * w1 + 3.0 * d1
    B = -6.0 * w0 - 4.0 * d0 + 6.0 * w1 - 2.0 * d1
    C = d0
    if A != 0.0:
        disc = B * B - 4.0 * A * C
        if disc >= 0.0:
            sq = math.sqrt(disc)
            for root in ((-B + sq) / (2.0 * A), (-B - sq) / (2.0 * A)):
                if 0.0 < root < 1.0:
                    wm = max(wm, _cubic(root, w0, d0, w1, d1))
    elif B != 0.0:
        root = -C / B
        if 0.0 < root < 1.0:
            wm = max(wm, _cubic(root, w0, d0, w1, d1))
    return wm


@njit(cache=True, inline="always")
def linear_t(target, l0, l1):
    """The linear guess of the fraction of a step at which the affine parameter is target."""
    t = (target - l0) / (l1 - l0) if l1 > l0 else 0.0
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    return t


@njit(cache=True, inline="always")
def _in_step(traj, k, target, a):
    """The state at affine parameter target, known to lie in step k (Newton on Hermite)."""
    l0, l1, s0h, s1h = step_affine(traj, k, a)
    t, _ = newton_t(target, l0, l1, s0h, s1h, linear_t(target, l0, l1))
    w, u, pw, pu, _ = state_in_step(traj, k, t, a)
    return w, u, pw, pu


@njit(cache=True)
def locate(traj, n, kk, target, a):
    """
    The state (w, u, dw/dtau, du/dtau) at affine parameter target, searching forward
    from step kk; for targets visited in increasing order.  Returns (kk, w, u, pw, pu).
    """
    while kk < n - 2 and traj[4, kk + 1] < target:
        kk += 1
    w, u, pw, pu = _in_step(traj, kk, target, a)
    return kk, w, u, pw, pu


@njit(cache=True)
def state_at(traj, n, target, a):
    """The state (w, u, dw/dtau, du/dtau) at an arbitrary affine parameter (bisection on steps)."""
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if traj[4, mid] <= target:
            lo = mid
        else:
            hi = mid
    return _in_step(traj, lo, target, a)


@njit(cache=True)
def closest_approach_to_axis(traj, n, a):
    """
    Affine parameter of the ray's closest approach to the jet axis: the minimum of
    r sin(theta), refined by golden-section search on the interpolant over the two steps
    around the coarse minimum.  This is the point the cells of the ray are laid out
    about, as z_J is for a straight ray.
    """
    kbest = 0
    rbest = 1.0e300
    for k in range(n):
        s2 = 1.0 - traj[1, k] * traj[1, k]
        R = math.sqrt(s2 if s2 > 0.0 else 0.0) / traj[0, k]
        if R < rbest:
            rbest = R
            kbest = k
    if kbest == 0 or kbest >= n - 1:
        return traj[4, kbest]
    lo = kbest - 1.0
    hi = kbest + 1.0
    gr = 0.5 * (math.sqrt(5.0) - 1.0)
    for _ in range(60):
        x1 = hi - gr * (hi - lo)
        x2 = lo + gr * (hi - lo)
        k1 = min(int(x1), n - 2)
        w1, u1, _, _, _ = state_in_step(traj, k1, x1 - k1, a)
        k2 = min(int(x2), n - 2)
        w2, u2, _, _, _ = state_in_step(traj, k2, x2 - k2, a)
        R1 = math.sqrt(max(1.0 - u1 * u1, 0.0)) / w1
        R2 = math.sqrt(max(1.0 - u2 * u2, 0.0)) / w2
        if R1 < R2:
            hi = x2
        else:
            lo = x1
    x = 0.5 * (lo + hi)
    kx = min(int(x), n - 2)
    _, _, _, _, la = state_in_step(traj, kx, x - kx, a)
    return la


# ----------------------------------------------------------------------------- local frame
@njit(cache=True, inline="always")
def zamo_frame(w, u, pw, pu, lam, M, a):
    """
    The forward photon's direction (r, theta, phi components) and energy
    k^(t) = (1 - omega lambda)/alpha in the local ZAMO frame, from the state of the
    backward ray, together with alpha, omega, r and the metric factors sqrt(g_rr),
    sqrt(g_thth), sqrt(g_phph) that convert between orthonormal and coordinate
    components.  (k_r = -(dr/dtau)/Delta = (dw/dtau)/(1 - 2 M w + a^2 w^2) for the
    backward ray, k_theta = (du/dtau)/sin(theta), k_phi = lambda, k_t = -1.)
    """
    r = 1.0 / w
    a2 = a * a
    s2 = 1.0 - u * u
    if s2 < 1.0e-300:
        s2 = 1.0e-300
    st = math.sqrt(s2)
    sig = r * r + a2 * u * u
    delta = r * r - 2.0 * M * r + a2
    xi = (r * r + a2) ** 2 - a2 * delta * s2
    alpha = math.sqrt(delta * sig / xi)
    omega = 2.0 * M * a * r / xi
    k_r = pw / (1.0 - 2.0 * M * w + a2 * w * w)
    k_th = pu / st
    kt = (1.0 - omega * lam) / alpha
    sq_rr = math.sqrt(sig / delta)
    sq_tt = math.sqrt(sig)
    sq_pp = math.sqrt(xi * s2 / sig)
    return (k_r / sq_rr / kt, k_th / sq_tt / kt, lam / sq_pp / kt, kt, alpha, omega,
            r, sq_rr, sq_tt, sq_pp)


@njit(cache=True, inline="always")
def _wp_kappa(r, u, a, kt_c, kr_c, kth_c, kph_c, fr, fth, fph):
    """Walker-Penrose constant (kappa_1, kappa_2) of a vector f with f^t = 0."""
    st = math.sqrt(max(1.0 - u * u, 0.0))
    A = kt_c * fr + a * st * st * (kr_c * fph - kph_c * fr)
    B = ((r * r + a * a) * (kph_c * fth - kth_c * fph) - a * kt_c * fth) * st
    return A * r - B * a * u, -(A * a * u + B * r)


@njit(cache=True, inline="always")
def kappa_of(f0, f1, f2, n0, n1, n2, kt, alpha, omega, r, u, a, sq_rr, sq_tt, sq_pp):
    """kappa of the ZAMO-frame spatial vector f for the photon with ZAMO direction n."""
    kt_c = kt / alpha
    kr_c = kt * n0 / sq_rr
    kth_c = kt * n1 / sq_tt
    kph_c = kt * (omega / alpha + n2 / sq_pp)
    return _wp_kappa(r, u, a, kt_c, kr_c, kth_c, kph_c, f0 / sq_rr, f1 / sq_tt, f2 / sq_pp)


@njit(cache=True, inline="always")
def _perpendicular_pair(n0, n1, n2):
    """Two unit vectors completing n to a right-handed orthonormal triad (e, g, n)."""
    a0 = abs(n0)
    a1 = abs(n1)
    a2 = abs(n2)
    if a0 <= a1 and a0 <= a2:
        s0, s1, s2 = 1.0, 0.0, 0.0
    elif a1 <= a2:
        s0, s1, s2 = 0.0, 1.0, 0.0
    else:
        s0, s1, s2 = 0.0, 0.0, 1.0
    d = s0 * n0 + s1 * n1 + s2 * n2
    e0 = s0 - d * n0
    e1 = s1 - d * n1
    e2 = s2 - d * n2
    m = math.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
    e0 /= m
    e1 /= m
    e2 /= m
    return e0, e1, e2, n1 * e2 - n2 * e1, n2 * e0 - n0 * e2, n0 * e1 - n1 * e0


@njit(cache=True, inline="always")
def carried_vector(K1, K2, n0, n1, n2, kt, alpha, omega, r, u, a, sq_rr, sq_tt, sq_pp):
    """The vector perpendicular to n (ZAMO frame) whose Walker-Penrose constant is K1 + i K2."""
    e0, e1, e2, g0, g1, g2 = _perpendicular_pair(n0, n1, n2)
    k11, k12 = kappa_of(e0, e1, e2, n0, n1, n2, kt, alpha, omega, r, u, a, sq_rr, sq_tt, sq_pp)
    k21, k22 = kappa_of(g0, g1, g2, n0, n1, n2, kt, alpha, omega, r, u, a, sq_rr, sq_tt, sq_pp)
    det = k11 * k22 - k12 * k21
    c1 = (K1 * k22 - K2 * k21) / det
    c2 = (K2 * k11 - K1 * k12) / det
    return c1 * e0 + c2 * g0, c1 * e1 + c2 * g1, c1 * e2 + c2 * g2


@njit(cache=True, inline="always")
def _start_point(alpha, beta, theta_o, r0):
    """
    Cartesian position (model frame) of the ray's point at radius r0, far from the hole:
    d n + alpha e_alpha + beta e_beta with n = (-sin theta_o, 0, cos theta_o) the line of
    sight, e_alpha = (0, -1, 0), e_beta = (cos theta_o, 0, sin theta_o) and
    d = sqrt(r0^2 - alpha^2 - beta^2).
    """
    so = math.sin(theta_o)
    co = math.cos(theta_o)
    d = math.sqrt(r0 * r0 - alpha * alpha - beta * beta)
    return -d * so + beta * co, -alpha, d * co + beta * so


@njit(cache=True)
def observer_kappas(traj, alpha, beta, theta_o, lam, M, a):
    """
    Walker-Penrose constants of the observer's Stokes basis vectors, the flat model's
    (+x_im, -y_im) = (-e_beta, e_alpha) -- theta-hat and phi-hat at the observer --
    expressed at the start of the ray and made exactly perpendicular to the photon there.
    Returns (Ka1, Ka2, Kb1, Kb2).
    """
    n0, n1, n2, kt, al, om, rr, sq_rr, sq_tt, sq_pp = zamo_frame(
        traj[0, 0], traj[1, 0], traj[2, 0], traj[3, 0], lam, M, a)
    so = math.sin(theta_o)
    co = math.cos(theta_o)
    X0, X1, X2 = _start_point(alpha, beta, theta_o, rr)
    phi0 = math.atan2(X1, X0)
    cp = math.cos(phi0)
    sp = math.sin(phi0)
    st0 = math.sqrt(X0 * X0 + X1 * X1) / rr
    ct0 = X2 / rr
    # +x_im = (-cos theta_o, 0, -sin theta_o) in the local (r, theta, phi) basis
    fa0 = -co * st0 * cp - so * ct0
    fa1 = -co * ct0 * cp + so * st0
    fa2 = co * sp
    d = fa0 * n0 + fa1 * n1 + fa2 * n2
    fa0 -= d * n0
    fa1 -= d * n1
    fa2 -= d * n2
    m = math.sqrt(fa0 * fa0 + fa1 * fa1 + fa2 * fa2)
    fa0 /= m
    fa1 /= m
    fa2 /= m
    fb0 = n1 * fa2 - n2 * fa1
    fb1 = n2 * fa0 - n0 * fa2
    fb2 = n0 * fa1 - n1 * fa0
    u0 = traj[1, 0]
    Ka1, Ka2 = kappa_of(fa0, fa1, fa2, n0, n1, n2, kt, al, om, rr, u0, a, sq_rr, sq_tt, sq_pp)
    Kb1, Kb2 = kappa_of(fb0, fb1, fb2, n0, n1, n2, kt, al, om, rr, u0, a, sq_rr, sq_tt, sq_pp)
    return Ka1, Ka2, Kb1, Kb2
