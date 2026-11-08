"""
Faraday rotation and conversion of the jet model's anisotropic electrons.

The electrons' pitch angles follow f(mu) = [1 + (eta - 1) mu^2]^(-p_eta/2) / Phi(eta, p_eta)
(mu the cosine of the pitch angle, f = 1 for isotropic electrons), which ffjet applies to
emission and absorption.  The rotativities of _rotativities.py (HS11, JO77) are those of
isotropic electrons of the same density.  This module supplies the factors that make them
those of the model's electrons:

    rho_V = R_V(gamma_c, theta_B') rho_V,isotropic,
    rho_Q = R_Q(gamma_c, theta_B') rho_Q,isotropic,

with theta_B' the fluid-frame angle between the photon and the field.

ONE ELECTRON.  The response of a thin shell of momentum p (density 1, pitch angles f) to a
transverse wave follows from the linearized Vlasov equation integrated along the unperturbed
helical orbits, expanded in the gyrofrequency: the first order is Faraday rotation (the
antisymmetric part of the transverse response tensor), the second order Faraday conversion
(its symmetric, traceless part).  The orbit-time integrals are done analytically, term by
term; what remains is an integral over momentum directions, with terms carrying 1/D^(m+1) to
1/D^(2m+1), D = 1 - beta cos(psi), psi the angle between the momentum and the wave vector.
The expansion is the high-frequency limit, omega >> gamma^2 Omega_0 sin(theta).  The shell's
radial profile enters through d/dp of its momentum dependence (a central difference, relative
step 1e-3) and the pitch-angle gradient through f'(mu).  (The physics and the slow reference
integral: jet_circular_pol/old/anisotropy_variants, rotation/ and conversion/.)

  R_V(gamma, theta) and R_Q(gamma, theta) are the anisotropic shell's rotation and conversion
  over those of an isotropic shell of the same momentum, which are known in closed form:
  -4 pi cos(theta) arccosh(gamma)/(gamma |p|) and 2 pi sin^2(theta) (2 gamma^2 - 1)/gamma in
  the units of the integral (cold electrons: -4 pi cos(theta) and 2 pi sin^2(theta)).

  The integral over directions is done about the wave vector: psi through xi = ln D, which
  turns the 1/D^n peak within 1/gamma of the wave vector into a smooth function, with
  Gauss-Legendre panels of width 2 in xi, broken also at the field ridge of f (psi = theta and
  pi - theta, and offsets growing by factors of 2 from 0.5 sqrt(eta), which follow its
  1/sin^p_eta tail); the azimuth about the wave vector by the periodic midpoint rule, over half
  the circle (the four components needed are even under reflection through the plane of the
  wave vector and the field), with max(48, 8/sqrt(eta)) nodes (8 sqrt(eta) for eta > 1).
  Against the slow reference (1400 x 360 directions, 20 million evaluations per point), for
  eta = 1e-3, 0.03 and 4 and p_eta = 1, 2 and 4, gamma from 1.2 to 1000 and theta from 2 to 89
  deg: within 4e-4 of R_V and 1.1e-4 of R_Q, the accuracy of the reference itself; doubling the
  nodes changes them by at most 2e-5.  Isotropic shells come out within 1e-6 of the closed
  forms from gamma = 1.2 to 1e5 (the central difference).  About 2.5 ms per point.

THE MODEL'S ELECTRONS.  A cell's factor is the average of the shell's over its electrons,
weighted by how much each one rotates (converts) in the high-frequency limit:

    R_V,cell = int N(g) w_V(g) R_V(g) dg / int N(g) w_V(g) dg,   w_V = arccosh(g) / (g |p|),
    R_Q,cell = the same with w_Q = (2 g^2 - 1) / g,

with N the cell's cooled broken power law exactly as the HS11 rotativities sum it
(_rotativities.py): g^-p from gamma_m (to gamma_c when the electrons cool slowly, then
gamma_c g^-(p+1) g^2/(g^2 - 1)); in fast cooling 1/(g^2 - 1) from the momentum
p_c = 1/sinh(1/gamma_c) to gamma_m, then gamma_m^(p-1) g^-(p+1) g^2/(g^2 - 1).  Since p,
gamma_m and gamma_max are model constants, a cell enters only through gamma_c (and the
uncooled cells are the column gamma_c = gamma_max).  The shells are computed on a grid of 55
momenta (log-spaced from 1e-3 to 1e6; cold below, where the factors are 1 to O(p^2), and
held at the top value above, where the weights have fallen away) and 41 angles, and
interpolated (cubic) onto 400 points per decade of momentum and the 241 angles of the
tables; the averages are cumulative integrals in ln p, read at each cell's segment edges.
Against direct averages over the electrons (shells computed at every quadrature point),
the tables are within about 1e-4 of the factors, at the rows and between them.

THE TABLES.  R_V,cell and R_Q,cell are stored as 2 x 241 rows (angle) of as many columns
(log10 gamma_c, from 1e-2 to gamma_max) as the rotativity tables have, packed below them for
the kernels.  The rows are uniform in u = ln(1 + r/r0)/ln(1 + 1/r0), r = 1 - |cos(theta_B')|
and r0 = eta/2 when eta < 1 (r = |cos(theta_B')|, r0 = 1/(2 sqrt(eta)) when eta > 1), which
puts a third of them inside the ridge (8 MB for 2001 columns).  Isotropic electrons (eta = 1)
need no tables: the factors are exactly 1, and the images are those of the isotropic
rotativities bit for bit.
"""

import math

import numpy as np
from numba import njit, prange
from scipy.interpolate import CubicSpline
from scipy.special import hyp2f1

from . import _rotativities as _rotmod

# quadrature of the shell response
GL_NODES = 20                  # Gauss-Legendre nodes per panel in xi = ln D
PANEL = 2.0                    # panel width in xi
RIDGE_FACTOR = 2.0             # growth of the breakpoint offsets about the field ridge
CHI_FLOOR = 48                 # azimuth nodes on half the circle, at least
CHI_SCALE = 8.0                # ... and 8 / sqrt(eta) (8 sqrt(eta) for eta > 1)
DP_REL = 1.0e-3                # relative momentum step of the radial derivative

# grids
LOG10P_SHELL = (-3.0, 6.0)     # momenta of the shells (log10 |p|, m_e c units)
N_SHELL_P = 55
N_THETA = 241                  # rows (angles) of the cell tables
THETA_STRIDE = 6               # the shells are computed on every sixth row
LOG10P_FINE = (-8.0, None)     # fine momentum grid of the averages, to gamma_max
FINE_PER_DEX = 400
LOG10_GC_MIN = -2.0            # lowest gamma_c of the tables (colder cells are clamped)
THETA_EDGE = 1.0e-4            # the rows at theta = 0 and 90 deg are evaluated this close

_SHELL_CACHE = {}              # (eta, p_eta) -> per-electron factors


# ----------------------------------------------------------------------------- one shell
@njit(cache=True, inline="always")
def _node(p, part, sth, cth, px, py, pz, mu, g, dg, s2h, w, acc):
    """add one direction's contribution to acc = [T1_e1e2, T1_e2e1, T2_e1e1, T2_e2e2] for
    momentum p; part 0 is the radial gradient g phat, part 1 the angular (g'/p)(z - mu phat)"""
    gam = math.sqrt(1.0 + p * p)
    beta = p / gam
    D = 1.0 / (gam * (gam + p)) + 2.0 * beta * s2h          # 1 - beta cos(psi), without cancellation
    if part == 0:
        ax = g * px
        ay = g * py
        az = g * pz
    else:
        c = dg / p
        ax = c * (-sth - mu * px)
        ay = -c * mu * py
        az = c * (cth - mu * pz)
    vx = beta * px
    vy = beta * py
    vz = beta * pz
    # zhat x w, with zhat = (-sin theta, 0, cos theta) in (e1, e2, khat)
    Zvx = -cth * vy
    Zvy = cth * vx + sth * vz
    Zvz = -sth * vy
    Zax = -cth * ay
    Zay = cth * ax + sth * az
    Zaz = -sth * ay
    Z2vx = -cth * Zvy
    Z2vy = cth * Zvx + sth * Zvz
    Z2vz = -sth * Zvy
    Z2ax = -cth * Zay
    Z2ay = cth * Zax + sth * Zaz
    Z2az = -sth * Zay
    G0x = D * ax + az * vx
    G0y = D * ay + az * vy
    G1x = Zvz * ax - D * Zax - az * Zvx - Zaz * vx
    G1y = Zvz * ay - D * Zay - az * Zvy - Zaz * vy
    G2x = -0.5 * Z2vz * ax - Zvz * Zax + 0.5 * D * Z2ax + 0.5 * az * Z2vx + Zaz * Zvx + 0.5 * Z2az * vx
    G2y = -0.5 * Z2vz * ay - Zvz * Zay + 0.5 * D * Z2ay + 0.5 * az * Z2vy + Zaz * Zvy + 0.5 * Z2az * vy
    a1 = 0.5 * Zvz
    a2 = -Z2vz / 6.0
    iD = 1.0 / D
    iD2 = iD * iD
    iD3 = iD2 * iD
    iD4 = iD3 * iD
    c1 = 2.0 * a1 * iD3
    c2 = 6.0 * a1 * iD4
    c3 = 6.0 * (a2 - 2.0 * a1 * a1 * iD) * iD4
    w1 = w / gam
    w2 = w1 / gam
    acc[0] += w1 * vx * (-G1y * iD2 + c1 * G0y)
    acc[1] += w1 * vy * (-G1x * iD2 + c1 * G0x)
    acc[2] += w2 * vx * (-2.0 * G2x * iD3 + c2 * G1x + c3 * G0x)
    acc[3] += w2 * vy * (-2.0 * G2y * iD3 + c2 * G1y + c3 * G0y)


@njit(cache=True)
def _xi_breaks(p0, theta, eta):
    """the panel breakpoints in xi = ln(1 - beta cos psi), sorted"""
    gam0 = math.sqrt(1.0 + p0 * p0)
    beta0 = p0 / gam0
    omb0 = 1.0 / (gam0 * (gam0 + p0))
    xmin = math.log(omb0)
    xmax = math.log(1.0 + beta0)
    br = np.empty(int((xmax - xmin) / PANEL) + 160)
    nb = 0
    x = xmin
    while x < xmax:
        br[nb] = x
        nb += 1
        x += PANEL
    br[nb] = xmax
    nb += 1
    if eta < 1.0:
        for cidx in range(2):
            cen = theta if cidx == 0 else math.pi - theta
            if 0.0 < cen < math.pi:
                s = math.sin(0.5 * cen)
                br[nb] = math.log(omb0 + 2.0 * beta0 * s * s)
                nb += 1
            d = 0.5 * math.sqrt(eta)
            while d < math.pi:
                for sgn in (-1.0, 1.0):
                    ps = cen + sgn * d
                    if 0.0 < ps < math.pi:
                        s = math.sin(0.5 * ps)
                        br[nb] = math.log(omb0 + 2.0 * beta0 * s * s)
                        nb += 1
                d *= RIDGE_FACTOR
    return np.sort(br[:nb])


@njit(cache=True)
def shell_response(p0, theta, eta, p_eta, inv_phi, xg, wg, n_chi_half):
    """(rotation, conversion) of a thin shell of momentum p0 with pitch angles f"""
    gam0 = math.sqrt(1.0 + p0 * p0)
    beta0 = p0 / gam0
    omb0 = 1.0 / (gam0 * (gam0 + p0))
    sth = math.sin(theta)
    cth = math.cos(theta)
    brs = _xi_breaks(p0, theta, eta)
    h = DP_REL * p0
    pm = p0 - h
    pp = p0 + h
    acc_lo = np.zeros(4)
    acc_hi = np.zeros(4)
    acc_an = np.zeros(4)
    dchi = math.pi / n_chi_half
    half_peta = 0.5 * p_eta
    for ib in range(brs.size - 1):
        a = brs[ib]
        b = brs[ib + 1]
        if b - a < 1.0e-12:
            continue
        for ig in range(xg.size):
            xi = 0.5 * (b - a) * xg[ig] + 0.5 * (a + b)
            Dn = math.exp(xi)
            s2h = (Dn - omb0) / (2.0 * beta0)
            if s2h < 0.0:
                s2h = 0.0
            elif s2h > 1.0:
                s2h = 1.0
            sps = 2.0 * math.sqrt(s2h * (1.0 - s2h))
            cps = 1.0 - 2.0 * s2h
            w = 0.5 * (b - a) * wg[ig] * (Dn / beta0) * (2.0 * dchi)
            for ic in range(n_chi_half):
                chi = (ic + 0.5) * dchi
                px = sps * math.cos(chi)
                py = sps * math.sin(chi)
                pz = cps
                mu = -sth * px + cth * pz
                q = 1.0 + (eta - 1.0) * mu * mu
                if p_eta == 2.0:
                    fq = 1.0 / q
                else:
                    fq = q ** (-half_peta)
                g = fq * inv_phi
                dg = -p_eta * (eta - 1.0) * mu * (fq / q) * inv_phi
                _node(pm, 0, sth, cth, px, py, pz, mu, g, dg, s2h, w, acc_lo)
                _node(pp, 0, sth, cth, px, py, pz, mu, g, dg, s2h, w, acc_hi)
                _node(p0, 1, sth, cth, px, py, pz, mu, g, dg, s2h, w, acc_an)
    T = np.empty(4)
    for k in range(4):
        T[k] = -((pp * pp * acc_hi[k] - pm * pm * acc_lo[k]) / (2.0 * h)) / (p0 * p0) + acc_an[k]
    return 0.5 * (T[0] - T[1]), 0.5 * (T[2] - T[3])


@njit(cache=True, parallel=True)
def _shell_table(lnp, theta, eta, p_eta, inv_phi, xg, wg, n_chi_half):
    """R_V and R_Q of shells on the grid lnp x theta (closed-form isotropic denominators)"""
    n2 = theta.size
    RV = np.empty((lnp.size, n2))
    RQ = np.empty((lnp.size, n2))
    for idx in prange(lnp.size * n2):
        i = idx // n2
        j = idx % n2
        p0 = math.exp(lnp[i])
        gam = math.sqrt(1.0 + p0 * p0)
        rA, cA = shell_response(p0, theta[j], eta, p_eta, inv_phi, xg, wg, n_chi_half)
        st = math.sin(theta[j])
        RV[i, j] = rA / (-4.0 * math.pi * math.cos(theta[j]) * math.asinh(p0) / (gam * p0))
        RQ[i, j] = cA / (2.0 * math.pi * st * st * (2.0 * gam * gam - 1.0) / gam)
    return RV, RQ


def n_chi_half(eta):
    """azimuth nodes on half the circle"""
    s = CHI_SCALE / math.sqrt(eta) if eta < 1.0 else CHI_SCALE * math.sqrt(eta)
    return max(CHI_FLOOR, int(math.ceil(s)))


def shell_factors(gamma, theta, eta, p_eta):
    """R_V and R_Q of shells of Lorentz factors `gamma` (> 1) at angles `theta` (radians, in
    (0, pi/2)): arrays of shape (len(gamma), len(theta))"""
    gamma = np.atleast_1d(np.asarray(gamma, dtype=float))
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    xg, wg = np.polynomial.legendre.leggauss(GL_NODES)
    inv_phi = 1.0 / float(hyp2f1(0.5, p_eta / 2.0, 1.5, 1.0 - eta))
    lnp = np.log(np.sqrt((gamma - 1.0) * (gamma + 1.0)))
    return _shell_table(lnp, theta, float(eta), float(p_eta), inv_phi, xg, wg, n_chi_half(eta))


# ----------------------------------------------------------------------------- angle rows
def theta_map(eta):
    """(r0, L, flip) of the rows: u = ln(1 + r/r0)/L, r = 1 - |cos theta| (flip 0) or |cos theta|
    (flip 1)"""
    if eta < 1.0:
        r0, flip = 0.5 * eta, 0
    else:
        r0, flip = 0.5 / math.sqrt(eta), 1
    return r0, math.log1p(1.0 / r0), flip


def theta_rows(eta, n=N_THETA):
    """|cos theta| of the n rows"""
    r0, L, flip = theta_map(eta)
    u = np.linspace(0.0, 1.0, n)
    r = r0 * np.expm1(u * L)
    r[-1] = 1.0
    return r if flip else 1.0 - r


# ----------------------------------------------------------------------------- the tables
def _shells(eta, p_eta):
    """the shells' factors on the shell grid (memoized per (eta, p_eta))"""
    key = (float(eta), float(p_eta))
    if key not in _SHELL_CACHE:
        lp = np.linspace(LOG10P_SHELL[0], LOG10P_SHELL[1], N_SHELL_P)
        mom = 10.0 ** lp
        gamma = np.sqrt(1.0 + mom * mom)
        c = theta_rows(eta)[::THETA_STRIDE]
        theta = np.clip(np.arccos(np.clip(c, 0.0, 1.0)), THETA_EDGE, 0.5 * np.pi - THETA_EDGE)
        RV, RQ = shell_factors(gamma, theta, eta, p_eta)
        _SHELL_CACHE[key] = (lp, RV, RQ)
    return _SHELL_CACHE[key]


def cell_tables(eta, p, p_eta, gamma_m, gamma_max, n_cols):
    """
    The cell factors R_V,cell and R_Q,cell: array (2, N_THETA, n_cols), rows the angles of
    theta_rows(eta), columns log10 gamma_c from LOG10_GC_MIN to log10 gamma_max; and the
    grid (log10 gamma_c0, inverse step).  For eta = 1 the factors are exactly 1.
    """
    lg_top = math.log10(gamma_max)
    lgc = np.linspace(LOG10_GC_MIN, lg_top, n_cols)
    inv_dlg = (n_cols - 1) / (lg_top - LOG10_GC_MIN)
    if eta == 1.0:
        return np.ones((2, N_THETA, n_cols)), LOG10_GC_MIN, inv_dlg
    lp_sh, RV_sh, RQ_sh = _shells(eta, p_eta)
    rows = N_THETA
    u_rows = np.linspace(0.0, 1.0, rows)
    u_sh = u_rows[::THETA_STRIDE]
    # fine momentum grid; the shells interpolated onto it (cubic in u, then in log10 p)
    top = math.log10(math.sqrt((gamma_max - 1.0) * (gamma_max + 1.0)))
    n_fine = int(math.ceil((top - LOG10P_FINE[0]) * FINE_PER_DEX)) + 1
    lpf = np.linspace(LOG10P_FINE[0], top, n_fine)
    R = {}
    for name, tab in (("V", RV_sh), ("Q", RQ_sh)):
        full = CubicSpline(u_sh, tab, axis=1)(u_rows)                        # (N_SHELL_P, rows)
        R[name] = CubicSpline(lp_sh, full, axis=0)(np.clip(lpf, lp_sh[0], lp_sh[-1]))   # (n_fine, rows)
    mom = 10.0 ** lpf
    g = np.sqrt(1.0 + mom * mom)
    ratio = np.where(mom > 1.0e-6, np.arcsinh(mom) / np.where(mom > 1.0e-6, mom, 1.0), 1.0 - mom * mom / 6.0)
    w = {"V": ratio / g, "Q": (2.0 * g * g - 1.0) / g}
    # integrands per ln p: N(g) w(g) dg/dln p, dg/dln p = p^2/g; the shapes of _rotativities._shape
    jac = mom * mom / g
    shapes = {
        "p": g ** (-p) * jac,
        "pp1": g ** (-(p + 1.0)) * g * g / g,        # g^-(p+1) g^2/p^2 times p^2/g
        "2": 1.0 / g,                                # 1/p^2 times p^2/g
    }
    du = (lpf[1] - lpf[0]) * math.log(10.0)

    def cumulative_from_top(f):
        seg = 0.5 * (f[1:] + f[:-1]) * du
        return np.concatenate([np.cumsum(seg[::-1], axis=0)[::-1], np.zeros((1,) + f.shape[1:])], axis=0)

    C = {}
    for s, S in shapes.items():
        for k in ("V", "Q"):
            C[(s, k, "num")] = cumulative_from_top((S * w[k])[:, None] * R[k])
            C[(s, k, "den")] = cumulative_from_top(S * w[k])

    inv_dlp = (n_fine - 1) / (top - LOG10P_FINE[0])

    def at(Cs, lp):
        """the cumulative integral at log10 p = lp (one per column), linear in log10 p; zero above
        the grid; held at the bottom value below it"""
        f = np.clip((lp - lpf[0]) * inv_dlp, 0.0, n_fine - 1.000001)
        i = f.astype(np.intp)
        t = f - i
        if Cs.ndim == 1:
            return Cs[i] + t * (Cs[i + 1] - Cs[i])
        return Cs[i] + t[:, None] * (Cs[i + 1] - Cs[i])

    def lmom(gg):
        return 0.5 * np.log10((gg - 1.0) * (gg + 1.0))

    gc = 10.0 ** lgc
    fast = gc <= gamma_m
    out = np.empty((2, rows, n_cols))
    lp_m = float(lmom(np.array(gamma_m)))
    for kk, k in enumerate(("V", "Q")):
        num = np.empty((n_cols, rows))
        den = np.empty(n_cols)
        # fast cooling: 1/(g^2 - 1) from p_c to gamma_m, then gamma_m^(p-1) times the cooled shape
        if np.any(fast):
            _, lp_c = _rotmod.fast_cooling_lower_edge(gc[fast])
            lo = np.minimum(lp_c, lp_m)
            n2 = at(C[("2", k, "num")], lo) - at(C[("2", k, "num")], np.full(lo.shape, lp_m))
            d2 = at(C[("2", k, "den")], lo) - at(C[("2", k, "den")], np.full(lo.shape, lp_m))
            # below the fine grid the electrons are cold: factor 1, integrand 1 per unit ln p
            tail = np.maximum(LOG10P_FINE[0] - lp_c, 0.0) * math.log(10.0)
            amp = gamma_m ** (p - 1.0)
            num[fast] = n2 + tail[:, None] + amp * at(C[("pp1", k, "num")], np.full(lo.shape, lp_m))
            den[fast] = d2 + tail + amp * at(C[("pp1", k, "den")], np.full(lo.shape, lp_m))
        # slow cooling (and uncooled at gamma_c = gamma_max): g^-p from gamma_m to gamma_c,
        # then gamma_c times the cooled shape
        slow = ~fast
        if np.any(slow):
            gcs = gc[slow]
            lp_c = lmom(gcs)
            lm = np.full(gcs.shape, lp_m)
            num[slow] = (at(C[("p", k, "num")], lm) - at(C[("p", k, "num")], lp_c)
                         + gcs[:, None] * at(C[("pp1", k, "num")], lp_c))
            den[slow] = (at(C[("p", k, "den")], lm) - at(C[("p", k, "den")], lp_c)
                         + gcs * at(C[("pp1", k, "den")], lp_c))
        out[kk] = (num / den[:, None]).T
    return np.ascontiguousarray(out), LOG10_GC_MIN, inv_dlg
