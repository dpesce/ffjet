"""
Compiled (numba) full-Stokes radiative-transfer kernel for JetModel.make_image_polarized.

This module is the polarized counterpart of _kernel.py.  It reuses that module's
geometry and fluid chain verbatim (`_rtheta_chain`, the field table, the parameter
packing) and adds

  * the polarized synchrotron coefficients j_{I,Q,V} and alpha_{I,Q,V} of an
    anisotropic double power-law electron distribution (Dexter 2016, Appendix A;
    Tsunetoe et al. 2025);
  * the Faraday rotativities rho_Q and rho_V, in either of two schemes -- see
    _rotativities.py for the formulae, the measured accuracy of each, and why the
    HS11 scheme reduces to table lookups;
  * the local Stokes basis: the observer's sky basis boosted into the fluid frame
    and projected perpendicular to the aberrated photon direction, which fixes the
    rotation angle chi between the fluid-frame magnetic field and the image axes;
  * the exact constant-coefficient evolution operator of the 4x4 transfer equation,
    from the closed form of Landi Degl'Innocenti & Landi Degl'Innocenti (1985)
    reproduced in Dexter (2016) Appendix D.

Rays are marched from the observer outwards, in the same order and over the same
cells as the unpolarized kernel, and the emission of each cell is attenuated by the
material in front of it.  The scalar attenuation exp(-tau) of the Stokes I path
becomes the cumulative 4x4 operator O_cum = prod_{k in front of i} exp(-K_k ds_k), so
that the ray integral is sum_i O_cum,i c_i with c_i the emission of cell i.

ROTATIVITY PREFACTORS.  Both schemes are written relative to K_a/nu_p, the scale
alpha_I is built on, so that they reuse the stored per-cell state.  With
K_a = prefac_absorp * n_e A_norm, prefac_absorp = e^2/(4 sqrt(3) m_e c 10^9) and
frequencies in GHz, e^2/(m_e c) = 4 sqrt(3) 10^9 prefac_absorp, and with
nu_B sin(theta) = (2/3) nu_p and x = nu'/nu_p:

    HS11:  rho_Q = 4 sqrt(3)   (K_a/nu_p) x^-1 [ X_A J_Q + boundary ]
           rho_V = 8 sqrt(3)/3 (K_a/nu_p) cot(theta_B) x^-2 [ J_V + boundary ]
    JO77:  rho_Q = 32 sqrt(3)/9 (K_a/nu_p) x^-3 sum_seg int gamma^(1-q) dgamma
           rho_V = 16 sqrt(3)/3 (K_a/nu_p) cot(theta_B) x^-2 [ int + boundary ]

CONVENTIONS  (see also JetModel.make_image_polarized and jetfuncs.sky_view)
--------------------------------------------------------------------------
Stokes Q and U are returned in the radio convention: referred to (North, East), so
that the EVPA is 0.5*atan2(U, Q) measured East of North, and perpendicular to the
projected fluid-frame magnetic field for optically thin emission.  V > 0 when the
fluid-frame magnetic field has a component pointing towards the observer
(cos(theta_B') > 0), the sign convention of Dexter (2016) Eqs. (A14)/(A24) and of
grtrans/ipole.

Where North is, in model terms: jetfuncs places the observer on the -z_im side of
the image plane, so the un-mirrored view of the sky has -x_im to the right and
+y_im up (jetfuncs.sky_view, and sky_view() in the paper's figure scripts), giving
North along +y_im and East along +x_im.  sky_view may rotate the picture by 180
degrees so that the approaching jet points right, which flips both North and East
and therefore leaves Q, U and V unchanged -- so the convention is well defined
without having to know which way the jet points.

The triad (x_im, y_im, n) -- with n the propagation direction towards the observer
-- is LEFT handed for this grid, so a raw pcolormesh(x, y, I) is a MIRROR of the
sky rather than a rotation of it.  Dexter's transfer matrix assumes a right-handed
triad, so the kernel works internally in the right-handed basis (a, b) =
(+x_im, -y_im, n).  Going from that basis to (North, East) is a rotation by -90
degrees, which negates both Q and U and leaves V alone; `_Q_SIGN` and `_U_SIGN`
below are the only place that enters.
"""

import math

import numpy as np
from numba import njit, prange

from ._kernel import (  # noqa: F401  (re-exported for the polarized drivers)
    N_TAB,
    P_COS_I,
    P_ETA,
    P_GAMMA_M,
    P_GAMMA_MAX,
    P_GM_M3,
    P_GM_MP,
    P_GM_MPM1,
    P_GM_MPP1,
    P_GM_MPP2,
    P_GM_PM1,
    P_GM_2MP,
    P_GX_MPP1,
    P_GX_MPP2,
    P_LN_GM,
    P_NX,
    P_NY,
    P_NZ,
    P_P,
    P_PETA,
    P_PHI_NORM,
    P_RH,
    P_ROT_SCHEME,
    P_RTAB_INVDLOGX,
    P_RTAB_KMAX,
    P_RTAB_LOGX0,
    P_S,
    P_SIN_I,
    P_TAB_INVDLOGX,
    P_TAB_KMAX,
    P_TAB_LOGX0,
    P_TAIL_C,
    P_X_TAIL,
    _G_bracket,
    _cont,
    _rtheta_chain,
)
from ._rotativities import (
    HS11_Q_PREFAC,
    HS11_V_PREFAC,
    JO77_Q_PREFAC,
    JO77_V_PREFAC,
    N_ROT_ROWS,
    ROT_ROWS,
    XA_CONST,
)

# ----------------------------------------------------------------------------- conventions
# Rotation from the internal right-handed basis (a, b) = (+x_im, -y_im) onto
# (North, East) = (+y_im, +x_im), applied once at the end of each ray.  A -90 degree
# rotation of the Stokes basis sends (Q, U) -> (-Q, -U) and leaves V unchanged; see
# the module docstring.
_Q_SIGN = -1.0
_U_SIGN = -1.0

# rotativity schemes, as packed into P[P_ROT_SCHEME]
ROT_NONE = 0
ROT_JO77 = 1
ROT_HS11 = 2

# ----------------------------------------------------------------------------- tables
#
# The eighteen synchrotron integrals are passed as one contiguous (18, n) array so
# that a kernel signature stays readable; every row shares the same log10(x) grid.
(
    T_GI_2,
    T_GI_P,
    T_GI_PP1,
    T_GaI_2,
    T_GaI_P,
    T_GaI_PP1,
    T_GQ_2,
    T_GQ_P,
    T_GQ_PP1,
    T_GaQ_2,
    T_GaQ_P,
    T_GaQ_PP1,
    T_GV_2,
    T_GV_P,
    T_GV_PP1,
    T_GaV_2,
    T_GaV_P,
    T_GaV_PP1,
    N_TAB_ROWS,
) = range(19)

_TAB_ORDER = (
    "GI_2", "GI_p", "GI_pp1", "GaI_2", "GaI_p", "GaI_pp1",
    "GQ_2", "GQ_p", "GQ_pp1", "GaQ_2", "GaQ_p", "GaQ_pp1",
    "GV_2", "GV_p", "GV_pp1", "GaV_2", "GaV_p", "GaV_pp1",
)

# rows of the rotativity table array, in the order _rotativities.ROT_ROWS defines
(R_Q_P, R_Q_PP1, R_Q_2, R_V1_P, R_V1_PP1, R_V1_2, R_V2_P, R_V2_PP1, R_V2_2) = range(N_ROT_ROWS)

# small-argument coefficient of each synchrotron kernel; see JetModel._TAIL
_H0 = math.pi / math.sqrt(3.0)

# prefactor of j_V and alpha_V relative to j_I and alpha_I (Dexter A24/A43 over A22/A41)
_V_PREFAC = 4.0 / 3.0

# per-cell polarized state: g, nu_p, gamma_c, C_j, C_a, cos(theta_B'), K_a/nu_p, cos2chi, sin2chi
N_POL_STATE = 9


def pack_tables_pol(model):
    """The eighteen log-tables of the polarized coefficients, as one (18, n) array."""
    rows = []
    for name in _TAB_ORDER:
        tab = getattr(model, "logG_" + name, None)
        if tab is None:
            raise AttributeError(
                f"table {name!r} was not built; the polarized transfer needs the Stokes Q and V "
                f"tables (construct the model with stokes='IQV', or call "
                f"JetModel.make_image_polarized, which builds them on demand)."
            )
        rows.append(tab)
    return np.ascontiguousarray(np.vstack(rows))


def pack_rot_tables(model):
    """The nine rotativity tables, or a (9, 1) placeholder when none are needed."""
    tab = getattr(model, "_rtab", None)
    if tab is None:
        # never read: the polarized entry points build the tables first.  Two columns
        # so that an unguarded lookup would still be in bounds.
        return np.zeros((N_ROT_ROWS, 2))
    return np.ascontiguousarray(tab)


# ----------------------------------------------------------------------------- table lookup
@njit(cache=True, inline="always")
def _G_eval_row(k, t, flag, TAB, row):
    """_G_eval for one row of the packed table array."""
    if flag == 0:
        a = TAB[row, k]
        return 10.0 ** (a + t * (TAB[row, k + 1] - a))
    if flag == -1:
        return 10.0 ** TAB[row, 0]
    if flag == 1:
        return 0.0
    return math.nan


@njit(cache=True, inline="always")
def _G_diff_row(x_lo, x_hi, k_lo, t_lo, f_lo, k_hi, t_hi, f_hi, TAB, row, m, tailC, tail_e0, xtail):
    """G(x_lo) - G(x_hi), with the cancellation-free small-argument form below xtail."""
    if x_hi <= xtail:
        e = m + tail_e0
        return (tailC / e) * ((x_hi**e) - (x_lo**e))
    return _G_eval_row(k_lo, t_lo, f_lo, TAB, row) - _G_eval_row(k_hi, t_hi, f_hi, TAB, row)


@njit(cache=True, inline="always")
def _rot_V(RTAB, row, x, logx0, invdlogx, kmax):
    """
    The scaled rotativity integral x^(q+1) W_q(x), linearly interpolated in log10(x).

    Stored scaled because W_q itself spans many decades and changes sign, whereas the
    scaled form is smooth and of order unity everywhere.  Outside the tabulated range
    the value is clamped to the edge: the grid spans 20 decades and the contribution
    at the upper edge is suppressed by gamma_max^-(q+1) ~ 1e-26, so neither end is
    reachable in practice.
    """
    if x <= 0.0:
        return RTAB[row, 0]
    f = (math.log10(x) - logx0) * invdlogx
    if f < 0.0:
        f = 0.0
    elif f > kmax:
        f = kmax
    k = int(f)
    t = f - k
    a = RTAB[row, k]
    return a + t * (RTAB[row, k + 1] - a)


# ----------------------------------------------------------------------------- per-cell physics
@njit(cache=True)
def _ray_part_pol(x, y, z, r, R, vr, vt, vp, gamma, alphalapse, Bpr, Bpt, Bpp, gamma_c, Kj, Ka, P):
    """
    Polarized counterpart of _kernel._ray_part.

    Reproduces that function's arithmetic exactly for the five quantities they share --
    the two are checked against each other, bit for bit, in the test suite -- and adds
    the three quantities the polarized coefficients need:

        cos(theta_B')  fluid-frame pitch angle cosine (signed; fixes j_V, alpha_V, rho_V)
        K_a / nu_p     the absorption scale *without* the anisotropy factor, which is
                       what the Faraday rotativities are built on
        cos2chi, sin2chi  rotation from the fluid-frame Stokes basis (aligned with the
                       projected B') to the observer's, in the right-handed sky triad
                       (x_im, -y_im, n); Dexter (2016) Eqs. (44) and (45)
    """
    nx = P[P_NX]
    ny = P[P_NY]
    nz = P[P_NZ]
    eta = P[P_ETA]
    p_eta = P[P_PETA]
    phi_norm = P[P_PHI_NORM]
    cos_i = P[P_COS_I]
    sin_i = P[P_SIN_I]

    b2 = 1.0 - 1.0 / (gamma * gamma)
    beta = math.sqrt(b2 if b2 > 0.0 else 0.0)
    vx = ((x * vr) / r) + ((x * z * vt) / (r * R)) - ((y * vp) / R)
    vy = ((y * vr) / r) + ((y * z * vt) / (r * R)) + ((x * vp) / R)
    vz = ((z * vr) / r) - ((R * vt) / r)
    vmag = math.sqrt(vx * vx + vy * vy + vz * vz)
    if vmag > 1.0e-8:
        vhat_x = vx / vmag
        vhat_y = vy / vmag
        vhat_z = vz / vmag
    else:
        vhat_x = 0.0
        vhat_y = 0.0
        vhat_z = 0.0

    # redshift factor
    k_par = (vhat_x * nx) + (vhat_y * ny) + (vhat_z * nz)
    one_m_betak = 1.0 - (beta * k_par)
    g = alphalapse / (gamma * one_m_betak)
    if not math.isfinite(g):
        g = 1.0

    # photon direction in the comoving frame
    k_perp_x = nx - k_par * vhat_x
    k_perp_y = ny - k_par * vhat_y
    k_perp_z = nz - k_par * vhat_z
    gamma_one_m_betak = gamma * one_m_betak
    k_par_prime = (k_par - beta) / one_m_betak
    k_x_prime = (k_perp_x / gamma_one_m_betak) + k_par_prime * vhat_x
    k_y_prime = (k_perp_y / gamma_one_m_betak) + k_par_prime * vhat_y
    k_z_prime = (k_perp_z / gamma_one_m_betak) + k_par_prime * vhat_z
    k_prime_mag = math.sqrt(k_x_prime * k_x_prime + k_y_prime * k_y_prime + k_z_prime * k_z_prime)
    khat_x_prime = k_x_prime / k_prime_mag
    khat_y_prime = k_y_prime / k_prime_mag
    khat_z_prime = k_z_prime / k_prime_mag

    # fluid-frame B in Cartesian components, and the pitch angle
    Bpx = ((x * Bpr) / r) + ((x * z * Bpt) / (r * R)) - ((y * Bpp) / R)
    Bpy = ((y * Bpr) / r) + ((y * z * Bpt) / (r * R)) + ((x * Bpp) / R)
    Bpz = ((z * Bpr) / r) - ((R * Bpt) / r)
    Bpm = math.sqrt(Bpx * Bpx + Bpy * Bpy + Bpz * Bpz)
    costhetaB = ((khat_x_prime * Bpx) + (khat_y_prime * Bpy) + (khat_z_prime * Bpz)) / Bpm
    sinthetaB = math.sqrt(1.0 - (costhetaB * costhetaB))  # NaN if cos^2 > 1, as in numpy

    anisotropy_term = 1.0 + ((eta - 1.0) * (costhetaB * costhetaB))
    if p_eta == 2.0:
        aniso_fac = (1.0 / anisotropy_term) / phi_norm
    else:
        aniso_fac = (anisotropy_term ** (-p_eta / 2.0)) / phi_norm
    nup = (4.1987e-3) * Bpm * sinthetaB

    Cj = Kj * nup * aniso_fac
    Ca = (Ka / nup) * aniso_fac

    # ---------------- polarization basis
    #
    # The observer's sky basis vectors are, in jet-frame Cartesian components,
    #     a = ( cos i, 0, -sin i)      (the +x_im direction)
    #     b = ( 0,    -1,  0      )    (the -y_im direction; (a, b, n) is right handed)
    # As four-vectors with vanishing time component they Lorentz transform like any
    # other vector; the boost gives a time component -gamma*beta*(vhat.a), which is
    # removed by adding the multiple -(a'^0/k'^0) k'^mu of the (null) photon momentum
    # that the gauge freedom f -> f + c k allows.  The spatial part of the result is
    #
    #     a'' = a + (gamma - 1)(vhat.a) vhat + gamma beta (vhat.a) khat'.
    #
    # Four-dimensional inner products are unchanged by both steps, so a'' and b'' come
    # out exactly orthonormal in the fluid frame with no renormalization (checked in
    # the tests); only roundoff separates them from unit length.
    gm1 = gamma - 1.0
    gb = gamma * beta
    sa = (vhat_x * cos_i) - (vhat_z * sin_i)
    sb = -vhat_y
    ax = cos_i + gm1 * sa * vhat_x + gb * sa * khat_x_prime
    ay = 0.0 + gm1 * sa * vhat_y + gb * sa * khat_y_prime
    az = -sin_i + gm1 * sa * vhat_z + gb * sa * khat_z_prime
    bx = 0.0 + gm1 * sb * vhat_x + gb * sb * khat_x_prime
    by = -1.0 + gm1 * sb * vhat_y + gb * sb * khat_y_prime
    bz = 0.0 + gm1 * sb * vhat_z + gb * sb * khat_z_prime

    # Dexter (2016) Eqs. (44), (45).  The denominator is |B'_perp|^2 / |B'|^2, i.e.
    # sin^2(theta_B'), because (a'', b'') spans the plane perpendicular to khat'.
    ca = (ax * Bpx + ay * Bpy + az * Bpz) / Bpm
    cb = (bx * Bpx + by * Bpy + bz * Bpz) / Bpm
    den = ca * ca + cb * cb
    if den > 1.0e-300:
        cos2chi = (cb * cb - ca * ca) / den
        sin2chi = -2.0 * ca * cb / den
    else:
        # B' along the line of sight: there is no linear polarization to orient
        # (j_Q vanishes with nu_p there), so any basis will do
        cos2chi = -1.0
        sin2chi = 0.0

    return g, nup, gamma_c, Cj, Ca, costhetaB, Ka / nup, cos2chi, sin2chi


@njit(cache=True)
def _cell_state_pol(xi, yi, zJ, zi, P, xs_tab, rs_tab, ts_tab, as_tab):
    """Frequency-independent polarized state of one cell, evaluated exactly."""
    cos_i = P[P_COS_I]
    sin_i = P[P_SIN_I]
    z_im_now = zi + zJ
    x = (xi * cos_i) + (z_im_now * sin_i)
    y = yi
    z = (z_im_now * cos_i) - (xi * sin_i)
    R2 = (x * x) + (y * y)
    r = math.sqrt(R2 + (z * z))
    R = math.sqrt(R2)
    az = abs(z)
    sgn = 1.0 if z >= 0.0 else -1.0
    omc = R2 / (r * (r + az))
    vr, vt, vp, gamma, alpha, Bpr, Bpt, Bpp, Bpm, gamma_c, Kj, Ka = _rtheta_chain(
        r, omc, sgn, P, xs_tab, rs_tab, ts_tab, as_tab
    )
    return _ray_part_pol(
        x, y, z, r, R, vr, vt, vp, gamma, alpha, Bpr, Bpt, Bpp, gamma_c, Kj, Ka, P
    )


@njit(cache=True)
def _cell_state_tab_pol(xi, yi, zJ, zi, P, Tup, Tlo, logr0, invdlogr, nr, invdu, nu_):
    """As _cell_state_pol, with the (r,theta)-only part interpolated from the field table."""
    cos_i = P[P_COS_I]
    sin_i = P[P_SIN_I]
    rH = P[P_RH]
    s = P[P_S]
    z_im_now = zi + zJ
    x = (xi * cos_i) + (z_im_now * sin_i)
    y = yi
    z = (z_im_now * cos_i) - (xi * sin_i)
    R2 = (x * x) + (y * y)
    r = math.sqrt(R2 + (z * z))
    R = math.sqrt(R2)
    omc = R2 / (r * (r + abs(z)))
    r_rH_1_s = (r / rH) ** (1.0 - s)
    u = r_rH_1_s * math.sqrt(omc)
    T = Tup if z >= 0.0 else Tlo
    fr = (math.log10(r - rH) - logr0) * invdlogr
    if fr < 0.0:
        fr = 0.0
    if fr > nr - 1.0000001:
        fr = nr - 1.0000001
    ir = int(fr)
    tr = fr - ir
    fu = u * invdu
    if fu < 0.0:
        fu = 0.0
    if fu > nu_ - 1.0000001:
        fu = nu_ - 1.0000001
    iu = int(fu)
    tu = fu - iu
    w00 = (1.0 - tr) * (1.0 - tu)
    w01 = (1.0 - tr) * tu
    w10 = tr * (1.0 - tu)
    w11 = tr * tu
    q = np.empty(N_TAB)
    for k in range(N_TAB):
        q[k] = (
            w00 * T[ir, iu, k]
            + w01 * T[ir, iu + 1, k]
            + w10 * T[ir + 1, iu, k]
            + w11 * T[ir + 1, iu + 1, k]
        )
    inv_cont = 1.0 / _cont(q[9], P)
    return _ray_part_pol(
        x, y, z, r, R, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[9],
        q[10] * inv_cont, q[11] * inv_cont, P
    )


# ----------------------------------------------------------------------------- rotativities
@njit(cache=True, inline="always")
def _H_B(x, gamma):
    """HS11 Eq. (54) H_B, evaluated pointwise at a distribution edge."""
    if x < 40.0:
        kin = 1.0 - 1.0 / gamma
        if kin < 0.0:
            kin = 0.0
        return 4.67e-9 * (kin**1.5) * (x**3.84)
    L = math.log(x)
    L2 = L * L
    L4 = L2 * L2
    return (
        0.864
        - 0.2082 * L2
        + 0.0175 * L4
        - 0.000626 * L4 * L2
        + 1.0175e-5 * L4 * L4
        - 7.686e-8 * L4 * L4 * L2
        - 0.01 * math.exp(-((L - 4.0755) ** 2) / 0.0763)
    )


@njit(cache=True, inline="always")
def _g_B(x):
    """HS11 Eq. (54) g_B."""
    return 1.0 - 0.0045 * (x**0.52)


@njit(cache=True, inline="always")
def _pow_int(a, b, q):
    """int_a^b gamma^(1-q) dgamma, with the logarithmic case at q = 2."""
    if b <= a:
        return 0.0
    m = 2.0 - q
    if abs(m) < 1.0e-12:
        return math.log(b / a)
    return (b**m - a**m) / m


@njit(cache=True, inline="always")
def _log_int(a, b, q):
    """int_a^b gamma^(-q-2) ln(2 gamma) dgamma."""
    if b <= a:
        return 0.0
    n1 = -(q + 1.0)  # exponent + 1
    fa = (a**n1) * (math.log(2.0 * a) / n1 - 1.0 / (n1 * n1))
    fb = (b**n1) * (math.log(2.0 * b) / n1 - 1.0 / (n1 * n1))
    return fb - fa


@njit(cache=True)
def _rot_jo77(nu_nup, cot, Kap, p1, p2, g1, g2, g3, amp2):
    """
    Jones & Odell (1977) / Sazonov (1969) power-law rotativities, with each segment of
    the broken power law integrated over its own range of Lorentz factors.

    The bracket of Dexter (2016) Eq. (B1) is, identically,
    gamma_1^(2-p) [1 - (gamma_1/gamma_*)^(p-2)]/((p-2)/2) = 2 int_{gamma_1}^{gamma_*}
    gamma^(1-p) dgamma with gamma_* = sqrt(nu'/(nu_B sin theta)) = sqrt(1.5 nu'/nu_p),
    so the generalization to a broken power law is just a matter of limits.  rho_V
    carries the boundary term of HS11 Eq. (56), which is where the factor (p+2) of
    Dexter's Eq. (B2) comes from.
    """
    gstar = math.sqrt(1.5 * nu_nup)
    hi1 = g2 if g2 < gstar else gstar
    acc = _pow_int(g1, hi1, p1)
    if amp2 > 0.0:
        lo2 = g2
        hi2 = g3 if g3 < gstar else gstar
        acc += amp2 * _pow_int(lo2, hi2, p2)
    rQ = JO77_Q_PREFAC * Kap * acc / (nu_nup * nu_nup * nu_nup)

    integ = _log_int(g1, g2, p1)
    if amp2 > 0.0:
        integ += amp2 * _log_int(g2, g3, p2)
    s1 = g1 ** (-p1)
    s3 = amp2 * g3 ** (-p2) if amp2 > 0.0 else g2 ** (-p1)
    bnd = s1 * (math.log(2.0 * g1) - 1.0) / g1
    top = g3 if amp2 > 0.0 else g2
    bnd -= s3 * (math.log(2.0 * top) - 1.0) / top
    rV = JO77_V_PREFAC * Kap * cot * (integ + bnd) / (nu_nup * nu_nup)
    return rQ, rV


@njit(cache=True)
def _rot_hs11(nu_nup, cot, Kap, p1, p2, g1, g2, g3, amp2,
              rowQ1, rowV1_1, rowV2_1, rowQ2, rowV1_2, rowV2_2,
              w1_lo, w1_hi, w2_lo, w2_hi, s1, RTAB, P):
    """
    Huang & Shcherbakov (2011) rotativities: their monoenergetic fitting formulae
    (Eq. 55) integrated over the distribution (Eq. 56).

    The integral over each power-law segment is a difference of the tabulated
    x^(q+1) W_q(x) at the segment edges, weighted by gamma^-(q+1) there -- the
    X_A^(q+1) prefactor cancels against the table's scaling.  `w*_lo/hi` carry those
    gamma^-(q+1) factors, which the caller has because they are model constants
    everywhere except at gamma_c.  The boundary term at gamma_min is evaluated
    pointwise with the exact kinematics; the one at gamma_max is suppressed by
    gamma_max^-(p+2) and is dropped, as HS11 themselves do for a power law.
    """
    logx0 = P[P_RTAB_LOGX0]
    invdlogx = P[P_RTAB_INVDLOGX]
    kmax = P[P_RTAB_KMAX]
    XA = math.sqrt(XA_CONST / nu_nup)
    lnXA = math.log(XA)

    x1 = XA * g1
    x2 = XA * g2
    JQ = (
        _rot_V(RTAB, rowQ1, x1, logx0, invdlogx, kmax) * w1_lo
        - _rot_V(RTAB, rowQ1, x2, logx0, invdlogx, kmax) * w1_hi
    )
    dV1 = (
        _rot_V(RTAB, rowV1_1, x1, logx0, invdlogx, kmax) * w1_lo
        - _rot_V(RTAB, rowV1_1, x2, logx0, invdlogx, kmax) * w1_hi
    )
    dV2 = (
        _rot_V(RTAB, rowV2_1, x1, logx0, invdlogx, kmax) * w1_lo
        - _rot_V(RTAB, rowV2_1, x2, logx0, invdlogx, kmax) * w1_hi
    )
    JV = 2.0 * dV2 - 2.0 * lnXA * dV1
    if amp2 > 0.0:
        x3 = XA * g3
        JQ += amp2 * (
            _rot_V(RTAB, rowQ2, x2, logx0, invdlogx, kmax) * w2_lo
            - _rot_V(RTAB, rowQ2, x3, logx0, invdlogx, kmax) * w2_hi
        )
        e1 = (
            _rot_V(RTAB, rowV1_2, x2, logx0, invdlogx, kmax) * w2_lo
            - _rot_V(RTAB, rowV1_2, x3, logx0, invdlogx, kmax) * w2_hi
        )
        e2 = (
            _rot_V(RTAB, rowV2_2, x2, logx0, invdlogx, kmax) * w2_lo
            - _rot_V(RTAB, rowV2_2, x3, logx0, invdlogx, kmax) * w2_hi
        )
        JV += amp2 * (2.0 * e2 - 2.0 * lnXA * e1)

    # boundary term at gamma_min (HS11 Eq. 56), with exact kinematics
    mom = math.sqrt(g1 * g1 - 1.0) if g1 > 1.0 else 1.0e-12
    L1 = math.log((1.0 + mom / g1) / max(1.0 - mom / g1, 1.0e-300))
    bq = s1 * _H_B(x1, g1) / (g1 * mom)
    bv = s1 * (g1 * L1 - 2.0 * mom) * _g_B(x1) / (g1 * mom)

    rQ = HS11_Q_PREFAC * Kap * (XA * JQ + bq) / nu_nup
    rV = HS11_V_PREFAC * Kap * cot * (JV + bv) / (nu_nup * nu_nup)
    return rQ, rV


@njit(cache=True)
def _rotativities(scheme, nu_nup, cot, Kap, P, RTAB, branch, gamma_c):
    """
    Dispatch to the selected rotativity scheme.  `branch` is 0 for uncooled, 1 for
    slow cooling and 2 for fast cooling, which fixes the segment indices and edges.
    """
    if scheme == ROT_NONE:
        return 0.0, 0.0
    p = P[P_P]
    gamma_m = P[P_GAMMA_M]
    gamma_max = P[P_GAMMA_MAX]
    if branch == 0:
        # one segment of index p from gamma_m to gamma_max
        p1 = p
        p2 = p
        g1 = gamma_m
        g2 = gamma_max
        g3 = gamma_max
        amp2 = 0.0
        rowQ1, rowV1_1, rowV2_1 = R_Q_P, R_V1_P, R_V2_P
        rowQ2, rowV1_2, rowV2_2 = R_Q_P, R_V1_P, R_V2_P
        w1_lo = P[P_GM_MPP1]
        w1_hi = P[P_GX_MPP1]
        w2_lo = 0.0
        w2_hi = 0.0
        s1 = P[P_GM_MP]
    elif branch == 1:
        # index p from gamma_m to gamma_c, then p + 1 to gamma_max
        p1 = p
        p2 = p + 1.0
        g1 = gamma_m
        g2 = gamma_c
        g3 = gamma_max
        amp2 = gamma_c
        rowQ1, rowV1_1, rowV2_1 = R_Q_P, R_V1_P, R_V2_P
        rowQ2, rowV1_2, rowV2_2 = R_Q_PP1, R_V1_PP1, R_V2_PP1
        gc_mpp1 = gamma_c ** (-(p + 1.0))
        w1_lo = P[P_GM_MPP1]
        w1_hi = gc_mpp1
        w2_lo = gc_mpp1 / gamma_c
        w2_hi = P[P_GX_MPP2]
        s1 = P[P_GM_MP]
    else:
        # index 2 from gamma_c to gamma_m, then p + 1 to gamma_max
        p1 = 2.0
        p2 = p + 1.0
        g1 = gamma_c
        g2 = gamma_m
        g3 = gamma_max
        amp2 = P[P_GM_PM1]
        rowQ1, rowV1_1, rowV2_1 = R_Q_2, R_V1_2, R_V2_2
        rowQ2, rowV1_2, rowV2_2 = R_Q_PP1, R_V1_PP1, R_V2_PP1
        w1_lo = 1.0 / (gamma_c * gamma_c * gamma_c)
        w1_hi = P[P_GM_M3]
        w2_lo = P[P_GM_MPP2]
        w2_hi = P[P_GX_MPP2]
        s1 = 1.0 / (gamma_c * gamma_c)
    if scheme == ROT_JO77:
        return _rot_jo77(nu_nup, cot, Kap, p1, p2, g1, g2, g3, amp2)
    return _rot_hs11(
        nu_nup, cot, Kap, p1, p2, g1, g2, g3, amp2,
        rowQ1, rowV1_1, rowV2_1, rowQ2, rowV1_2, rowV2_2,
        w1_lo, w1_hi, w2_lo, w2_hi, s1, RTAB, P,
    )


# ----------------------------------------------------------------------------- coefficients
@njit(cache=True)
def _cell_emis_pol(frequency, g, nup, gamma_c, Cj, Ca, cthB, Kap, P, TAB, RTAB):
    """
    Frequency-dependent polarized synchrotron coefficients of one cell, in the fluid
    frame and in the fluid Stokes basis (B' along the Q axis, so j_U = alpha_U = 0).

    Returns (j_I, j_Q, j_V, alpha_I, alpha_Q, alpha_V, rho_Q, rho_V).

    Emission and absorption follow Dexter (2016) Eqs. (A22)-(A24) and (A41)-(A43),
    summed over the two branches of the cooled double power law exactly as the Stokes
    I path does; the anisotropy of the electron distribution enters through the factor
    already folded into C_j and C_a, plus the extra factor 1 + g_eta/(p_i + 2) that a
    pitch-angle-dependent distribution contributes to the circular coefficients
    (Tsunetoe et al. 2025).  The rotativities come from _rotativities.py.
    """
    p = P[P_P]
    gamma_m = P[P_GAMMA_M]
    gamma_max = P[P_GAMMA_MAX]
    eta = P[P_ETA]
    p_eta = P[P_PETA]
    logx0 = P[P_TAB_LOGX0]
    invdlogx = P[P_TAB_INVDLOGX]
    kmax = P[P_TAB_KMAX]
    tailF = P[P_TAIL_C]
    tailG = 0.5 * P[P_TAIL_C]
    xtail = P[P_X_TAIL]
    e_FG = 4.0 / 3.0
    e_H = 1.0

    nu_nup = (frequency / g) / nup
    if math.isnan(nu_nup):
        return math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, 0.0, 0.0
    l10nn = math.log10(nu_nup) if nu_nup > 0.0 else -math.inf
    sq = math.sqrt(nu_nup)

    # pitch angle: cot(theta_B') and the anisotropy correction to the circular terms
    cth2 = cthB * cthB
    if cth2 > 1.0:
        cth2 = 1.0
    sthB = math.sqrt(1.0 - cth2)
    if sthB < 1.0e-12:
        sthB = 1.0e-12
    cot = cthB / sthB
    aniso_term = 1.0 + ((eta - 1.0) * cth2)
    g_eta = p_eta * (eta - 1.0) * (1.0 - cth2) / aniso_term

    if gamma_c >= gamma_max:
        branch = 0
        # ---- uncooled: a single power law, p_1 = p from gamma_m to gamma_max
        p1 = p
        x1 = nu_nup / (gamma_m * gamma_m)
        x2 = nu_nup / (gamma_max * gamma_max)
        k1, t1, f1 = _G_bracket(x1, logx0, invdlogx, kmax)
        k2, t2, f2 = _G_bracket(x2, logx0, invdlogx, kmax)
        pw = 10.0 ** (0.5 * (1.0 - p) * l10nn)
        pwV = pw / sq
        pwa = pw / (nu_nup * nu_nup * sq)
        pwaV = pwa / sq
        mI = 0.5 * (p - 3.0)
        maI = 0.5 * (p - 2.0)
        mV = 0.5 * (p - 2.0)
        maV = 0.5 * (p - 1.0)
        fV = 1.0 + g_eta / (p1 + 2.0)
        jI = Cj * pw * _G_diff_row(
            x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GI_P, mI, tailF, e_FG, xtail
        )
        jQ = Cj * pw * _G_diff_row(
            x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GQ_P, mI, tailG, e_FG, xtail
        )
        jV = _V_PREFAC * Cj * cot * fV * pwV * _G_diff_row(
            x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GV_P, mV, _H0, e_H, xtail
        )
        aI = Ca * (p1 + 2.0) * pwa * _G_diff_row(
            x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GaI_P, maI, tailF, e_FG, xtail
        )
        aQ = Ca * (p1 + 2.0) * pwa * _G_diff_row(
            x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GaQ_P, maI, tailG, e_FG, xtail
        )
        aV = _V_PREFAC * Ca * (p1 + 2.0) * cot * fV * pwaV * _G_diff_row(
            x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GaV_P, maV, _H0, e_H, xtail
        )
    elif gamma_c > gamma_m:
        branch = 1
        # ---- slow cooling: p_1 = p to gamma_c, p_2 = p+1 to gamma_max
        p1 = p
        p2 = p + 1.0
        x1 = nu_nup / (gamma_m * gamma_m)
        x2 = nu_nup / (gamma_c * gamma_c)
        x3 = nu_nup / (gamma_max * gamma_max)
        k1, t1, f1 = _G_bracket(x1, logx0, invdlogx, kmax)
        k2, t2, f2 = _G_bracket(x2, logx0, invdlogx, kmax)
        k3, t3, f3 = _G_bracket(x3, logx0, invdlogx, kmax)
        pw1 = 10.0 ** (0.5 * (1.0 - p) * l10nn)
        pw2 = pw1 / sq
        pwV1 = pw1 / sq
        pwV2 = pwV1 / sq
        pwa1 = pw1 / (nu_nup * nu_nup * sq)
        pwa2 = pwa1 / sq
        pwaV1 = pwa1 / sq
        pwaV2 = pwaV1 / sq
        m1 = 0.5 * (p1 - 3.0)
        m2 = 0.5 * (p2 - 3.0)
        ma1 = 0.5 * (p1 - 2.0)
        ma2 = 0.5 * (p2 - 2.0)
        mV1 = 0.5 * (p1 - 2.0)
        mV2 = 0.5 * (p2 - 2.0)
        maV1 = 0.5 * (p1 - 1.0)
        maV2 = 0.5 * (p2 - 1.0)
        fV1 = 1.0 + g_eta / (p1 + 2.0)
        fV2 = 1.0 + g_eta / (p2 + 2.0)
        cont = gamma_c
        jI = Cj * (
            pw1 * _G_diff_row(x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GI_P, m1, tailF, e_FG, xtail)
            + cont * pw2
            * _G_diff_row(x3, x2, k3, t3, f3, k2, t2, f2, TAB, T_GI_PP1, m2, tailF, e_FG, xtail)
        )
        jQ = Cj * (
            pw1 * _G_diff_row(x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GQ_P, m1, tailG, e_FG, xtail)
            + cont * pw2
            * _G_diff_row(x3, x2, k3, t3, f3, k2, t2, f2, TAB, T_GQ_PP1, m2, tailG, e_FG, xtail)
        )
        jV = _V_PREFAC * Cj * cot * (
            fV1 * pwV1
            * _G_diff_row(x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GV_P, mV1, _H0, e_H, xtail)
            + fV2 * cont * pwV2
            * _G_diff_row(x3, x2, k3, t3, f3, k2, t2, f2, TAB, T_GV_PP1, mV2, _H0, e_H, xtail)
        )
        aI = Ca * (
            (p1 + 2.0) * pwa1
            * _G_diff_row(x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GaI_P, ma1, tailF, e_FG, xtail)
            + (p2 + 2.0) * cont * pwa2
            * _G_diff_row(x3, x2, k3, t3, f3, k2, t2, f2, TAB, T_GaI_PP1, ma2, tailF, e_FG, xtail)
        )
        aQ = Ca * (
            (p1 + 2.0) * pwa1
            * _G_diff_row(x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GaQ_P, ma1, tailG, e_FG, xtail)
            + (p2 + 2.0) * cont * pwa2
            * _G_diff_row(x3, x2, k3, t3, f3, k2, t2, f2, TAB, T_GaQ_PP1, ma2, tailG, e_FG, xtail)
        )
        aV = _V_PREFAC * Ca * cot * (
            (p1 + 2.0) * fV1 * pwaV1
            * _G_diff_row(x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GaV_P, maV1, _H0, e_H, xtail)
            + (p2 + 2.0) * fV2 * cont * pwaV2
            * _G_diff_row(x3, x2, k3, t3, f3, k2, t2, f2, TAB, T_GaV_PP1, maV2, _H0, e_H, xtail)
        )
    else:
        branch = 2
        # ---- fast cooling: p_1 = 2 from gamma_c, p_2 = p+1 above gamma_m
        p1 = 2.0
        p2 = p + 1.0
        x1 = nu_nup / (gamma_c * gamma_c)
        x2 = nu_nup / (gamma_m * gamma_m)
        x3 = nu_nup / (gamma_max * gamma_max)
        k1, t1, f1 = _G_bracket(x1, logx0, invdlogx, kmax)
        k2, t2, f2 = _G_bracket(x2, logx0, invdlogx, kmax)
        k3, t3, f3 = _G_bracket(x3, logx0, invdlogx, kmax)
        cont = P[P_GM_PM1]
        pw1 = 1.0 / sq
        pw2 = 10.0 ** (-0.5 * p * l10nn)
        pwV1 = pw1 / sq
        pwV2 = pw2 / sq
        pwa1 = 1.0 / (nu_nup * nu_nup * nu_nup)
        pwa2 = pw2 / (nu_nup * nu_nup * sq)
        pwaV1 = pwa1 / sq
        pwaV2 = pwa2 / sq
        m1 = -0.5
        m2 = 0.5 * (p2 - 3.0)
        ma1 = 0.0
        ma2 = 0.5 * (p2 - 2.0)
        mV1 = 0.0
        mV2 = 0.5 * (p2 - 2.0)
        maV1 = 0.5
        maV2 = 0.5 * (p2 - 1.0)
        fV1 = 1.0 + g_eta / (p1 + 2.0)
        fV2 = 1.0 + g_eta / (p2 + 2.0)
        jI = Cj * (
            pw1 * _G_diff_row(x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GI_2, m1, tailF, e_FG, xtail)
            + cont * pw2
            * _G_diff_row(x3, x2, k3, t3, f3, k2, t2, f2, TAB, T_GI_PP1, m2, tailF, e_FG, xtail)
        )
        jQ = Cj * (
            pw1 * _G_diff_row(x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GQ_2, m1, tailG, e_FG, xtail)
            + cont * pw2
            * _G_diff_row(x3, x2, k3, t3, f3, k2, t2, f2, TAB, T_GQ_PP1, m2, tailG, e_FG, xtail)
        )
        jV = _V_PREFAC * Cj * cot * (
            fV1 * pwV1
            * _G_diff_row(x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GV_2, mV1, _H0, e_H, xtail)
            + fV2 * cont * pwV2
            * _G_diff_row(x3, x2, k3, t3, f3, k2, t2, f2, TAB, T_GV_PP1, mV2, _H0, e_H, xtail)
        )
        aI = Ca * (
            (p1 + 2.0) * pwa1
            * _G_diff_row(x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GaI_2, ma1, tailF, e_FG, xtail)
            + (p2 + 2.0) * cont * pwa2
            * _G_diff_row(x3, x2, k3, t3, f3, k2, t2, f2, TAB, T_GaI_PP1, ma2, tailF, e_FG, xtail)
        )
        aQ = Ca * (
            (p1 + 2.0) * pwa1
            * _G_diff_row(x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GaQ_2, ma1, tailG, e_FG, xtail)
            + (p2 + 2.0) * cont * pwa2
            * _G_diff_row(x3, x2, k3, t3, f3, k2, t2, f2, TAB, T_GaQ_PP1, ma2, tailG, e_FG, xtail)
        )
        aV = _V_PREFAC * Ca * cot * (
            (p1 + 2.0) * fV1 * pwaV1
            * _G_diff_row(x2, x1, k2, t2, f2, k1, t1, f1, TAB, T_GaV_2, maV1, _H0, e_H, xtail)
            + (p2 + 2.0) * fV2 * cont * pwaV2
            * _G_diff_row(x3, x2, k3, t3, f3, k2, t2, f2, TAB, T_GaV_PP1, maV2, _H0, e_H, xtail)
        )

    rQ, rV = _rotativities(
        int(P[P_ROT_SCHEME]), nu_nup, cot, Kap, P, RTAB, branch, gamma_c
    )
    return jI, jQ, jV, aI, aQ, aV, rQ, rV


@njit(cache=True, inline="always")
def _clamp_polarized(sI, sQ, sV):
    """
    Keep the polarized part of a coefficient vector below its total: the fitting
    functions for j and alpha are independent approximations, so their ratio can drift
    above unity where each is individually poor (very low frequency, or a pitch angle
    close to zero).  Rescaling Q and V leaves I -- the quantity the published,
    unpolarized model computes -- untouched, whereas the reference implementation
    zeroes the whole cell.
    """
    if not math.isfinite(sI) or sI <= 0.0:
        return 0.0, 0.0
    pol = math.sqrt(sQ * sQ + sV * sV)
    if pol > sI:
        f = sI / pol
        return sQ * f, sV * f
    return sQ, sV


# ----------------------------------------------------------------------------- transfer
@njit(cache=True, inline="always")
def _expm1_ratio(a, ds):
    """(1 - exp(-a ds)) / a, continuous and exact in the limit a ds -> 0."""
    z = a * ds
    if abs(z) < 1.0e-8:
        return ds * (1.0 - 0.5 * z * (1.0 - z / 3.0))
    return -math.expm1(-z) / a


@njit(cache=True)
def _pol_cell_operator(jI, jQ, jU, jV, aI, aQ, aU, aV, rQ, rU, rV, ds, O, cv):
    """
    Write into O and cv the evolution operator and the emission of one cell of
    constant coefficients:

        O = exp(-K ds),        cv = [int_0^ds exp(-K u) du] j,

    so that a ray is the sum of O_cum,i cv_i over its cells, with O_cum the product of
    the operators of everything in front of cell i.

    Landi Degl'Innocenti & Landi Degl'Innocenti (1985) give exp(-K ds) in closed form
    as a combination of four fixed matrices M1..M4 whose scalar coefficients are built
    from cosh(L1 ds), sinh(L1 ds), cos(L2 ds), sin(L2 ds) (Dexter 2016, Eqs. D2-D12).
    The same four matrices carry the emission once those four scalars are replaced by
    their integrals over the cell, which is how this routine avoids inverting K -- the
    inverse is what fails in the optically thin, strongly Faraday-rotating limit that
    the outer jet sits in.  Both the pure-absorption limit (Theta -> 0, where the
    polarized structure of K vanishes and the result reduces to the scalar update of
    the unpolarized kernel) and the geometrically thin limit (ds -> 0) are reached
    continuously, the latter through _expm1_ratio.
    """
    a2 = aQ * aQ + aU * aU + aV * aV
    r2 = rQ * rQ + rU * rU + rV * rV
    adr = aQ * rQ + aU * rU + aV * rV

    e0 = math.exp(-aI * ds)
    A_e = _expm1_ratio(aI, ds)

    d = 0.5 * (a2 - r2)
    disc = math.sqrt(d * d + adr * adr)
    Theta = 2.0 * disc
    if (a2 + r2) <= 0.0 or Theta <= 1.0e-290:
        for i in range(4):
            for j in range(4):
                O[i, j] = 0.0
            O[i, i] = e0
        cv[0] = A_e * jI
        cv[1] = A_e * jQ
        cv[2] = A_e * jU
        cv[3] = A_e * jV
        return

    l1sq = disc + d
    l2sq = disc - d
    L1 = math.sqrt(l1sq if l1sq > 0.0 else 0.0)
    L2 = math.sqrt(l2sq if l2sq > 0.0 else 0.0)
    sg = 1.0 if adr >= 0.0 else -1.0

    Em = math.exp(-(aI - L1) * ds)
    Ep = math.exp(-(aI + L1) * ds)
    ch = 0.5 * (Em + Ep)
    sh = 0.5 * (Em - Ep)
    csl = math.cos(L2 * ds)
    snl = math.sin(L2 * ds)
    co = e0 * csl
    si = e0 * snl

    A_ch = 0.5 * (_expm1_ratio(aI - L1, ds) + _expm1_ratio(aI + L1, ds))
    A_sh = 0.5 * (_expm1_ratio(aI - L1, ds) - _expm1_ratio(aI + L1, ds))
    den = aI * aI + L2 * L2
    if den > 0.0:
        A_co = (aI - e0 * (aI * csl - L2 * snl)) / den
        A_si = (L2 - e0 * (L2 * csl + aI * snl)) / den
    else:
        A_co = ds
        A_si = 0.0

    c1 = 0.5 * (ch + co)
    c2 = -si
    c3 = -sh
    c4 = 0.5 * (ch - co)
    C1 = 0.5 * (A_ch + A_co)
    C2 = -A_si
    C3 = -A_sh
    C4 = 0.5 * (A_ch - A_co)

    invT = 1.0 / Theta
    m2_01 = (L2 * aQ - sg * L1 * rQ) * invT
    m2_02 = (L2 * aU - sg * L1 * rU) * invT
    m2_03 = (L2 * aV - sg * L1 * rV) * invT
    m2_12 = (sg * L1 * aV + L2 * rV) * invT
    m2_13 = (-sg * L1 * aU - L2 * rU) * invT
    m2_23 = (sg * L1 * aQ + L2 * rQ) * invT
    m3_01 = (L1 * aQ + sg * L2 * rQ) * invT
    m3_02 = (L1 * aU + sg * L2 * rU) * invT
    m3_03 = (L1 * aV + sg * L2 * rV) * invT
    m3_12 = (-sg * L2 * aV + L1 * rV) * invT
    m3_13 = (sg * L2 * aU - L1 * rU) * invT
    m3_23 = (-sg * L2 * aQ + L1 * rQ) * invT
    hh = 0.5 * (a2 + r2)
    t2 = 2.0 * invT
    m4_00 = hh * t2
    m4_01 = (aV * rU - aU * rV) * t2
    m4_02 = (aQ * rV - aV * rQ) * t2
    m4_03 = (aU * rQ - aQ * rU) * t2
    m4_11 = (aQ * aQ + rQ * rQ - hh) * t2
    m4_12 = (aQ * aU + rQ * rU) * t2
    m4_13 = (aV * aQ + rV * rQ) * t2
    m4_22 = (aU * aU + rU * rU - hh) * t2
    m4_23 = (aU * aV + rU * rV) * t2
    m4_33 = (aV * aV + rV * rV - hh) * t2

    O[0, 0] = c1 + c4 * m4_00
    O[0, 1] = c2 * m2_01 + c3 * m3_01 + c4 * m4_01
    O[0, 2] = c2 * m2_02 + c3 * m3_02 + c4 * m4_02
    O[0, 3] = c2 * m2_03 + c3 * m3_03 + c4 * m4_03
    O[1, 0] = c2 * m2_01 + c3 * m3_01 - c4 * m4_01
    O[1, 1] = c1 + c4 * m4_11
    O[1, 2] = c2 * m2_12 + c3 * m3_12 + c4 * m4_12
    O[1, 3] = c2 * m2_13 + c3 * m3_13 + c4 * m4_13
    O[2, 0] = c2 * m2_02 + c3 * m3_02 - c4 * m4_02
    O[2, 1] = -c2 * m2_12 - c3 * m3_12 + c4 * m4_12
    O[2, 2] = c1 + c4 * m4_22
    O[2, 3] = c2 * m2_23 + c3 * m3_23 + c4 * m4_23
    O[3, 0] = c2 * m2_03 + c3 * m3_03 - c4 * m4_03
    O[3, 1] = -c2 * m2_13 - c3 * m3_13 + c4 * m4_13
    O[3, 2] = -c2 * m2_23 - c3 * m3_23 + c4 * m4_23
    O[3, 3] = c1 + c4 * m4_33

    p2_0 = m2_01 * jQ + m2_02 * jU + m2_03 * jV
    p2_1 = m2_01 * jI + m2_12 * jU + m2_13 * jV
    p2_2 = m2_02 * jI - m2_12 * jQ + m2_23 * jV
    p2_3 = m2_03 * jI - m2_13 * jQ - m2_23 * jU
    p3_0 = m3_01 * jQ + m3_02 * jU + m3_03 * jV
    p3_1 = m3_01 * jI + m3_12 * jU + m3_13 * jV
    p3_2 = m3_02 * jI - m3_12 * jQ + m3_23 * jV
    p3_3 = m3_03 * jI - m3_13 * jQ - m3_23 * jU
    p4_0 = m4_00 * jI + m4_01 * jQ + m4_02 * jU + m4_03 * jV
    p4_1 = -m4_01 * jI + m4_11 * jQ + m4_12 * jU + m4_13 * jV
    p4_2 = -m4_02 * jI + m4_12 * jQ + m4_22 * jU + m4_23 * jV
    p4_3 = -m4_03 * jI + m4_13 * jQ + m4_23 * jU + m4_33 * jV
    cv[0] = C1 * jI + C2 * p2_0 + C3 * p3_0 + C4 * p4_0
    cv[1] = C1 * jQ + C2 * p2_1 + C3 * p3_1 + C4 * p4_1
    cv[2] = C1 * jU + C2 * p2_2 + C3 * p3_2 + C4 * p4_2
    cv[3] = C1 * jV + C2 * p2_3 + C3 * p3_3 + C4 * p4_3

    if not math.isfinite(O[0, 0] + cv[0]):
        # the M matrices are 0/0 on the measure-zero set |alpha| = |rho|, alpha.rho = 0
        for i in range(4):
            for j in range(4):
                O[i, j] = 0.0
            O[i, i] = e0
        cv[0] = A_e * jI
        cv[1] = A_e * jQ
        cv[2] = A_e * jU
        cv[3] = A_e * jV


@njit(cache=True, inline="always")
def _pol_cell(jI, jQ, jV, aI, aQ, aV, rQ, rV, c2, s2, g, dz, O, cv):
    """
    Guard and clamp the fluid-frame coefficients, rotate the fluid Stokes basis onto
    the sky basis through chi, apply the invariant scalings j -> g^2 j and K -> K/g,
    and build the cell operator.  Returns the Stokes I optical depth of the cell.
    """
    if not math.isfinite(aI) or aI < 0.0:
        aI = 0.0
    if not math.isfinite(jI):
        jI = 0.0
    if not math.isfinite(jQ):
        jQ = 0.0
    if not math.isfinite(jV):
        jV = 0.0
    if not math.isfinite(aQ):
        aQ = 0.0
    if not math.isfinite(aV):
        aV = 0.0
    if not math.isfinite(rQ):
        rQ = 0.0
    if not math.isfinite(rV):
        rV = 0.0
    jQ, jV = _clamp_polarized(jI, jQ, jV)
    aQ, aV = _clamp_polarized(aI, aQ, aV)

    g2 = g * g
    inv_g = 1.0 / g
    AI = aI * inv_g
    _pol_cell_operator(
        g2 * jI, g2 * jQ * c2, g2 * jQ * s2, g2 * jV,
        AI, aQ * c2 * inv_g, aQ * s2 * inv_g, aV * inv_g,
        rQ * c2 * inv_g, rQ * s2 * inv_g, rV * inv_g,
        dz, O, cv,
    )
    return AI * dz


@njit(cache=True, inline="always")
def _pol_accumulate(Itot, Ocum, O, cv, tmp):
    """I_total += O_cum . c;  O_cum <- O_cum . O."""
    for i in range(4):
        s = 0.0
        for k in range(4):
            s += Ocum[i, k] * cv[k]
        Itot[i] += s
    for i in range(4):
        for j in range(4):
            s = 0.0
            for k in range(4):
                s += Ocum[i, k] * O[k, j]
            tmp[i, j] = s
    for i in range(4):
        for j in range(4):
            Ocum[i, j] = tmp[i, j]


@njit(cache=True, inline="always")
def _pol_ray_init(Itot, Ocum):
    for i in range(4):
        Itot[i] = 0.0
        for j in range(4):
            Ocum[i, j] = 0.0
        Ocum[i, i] = 1.0


@njit(cache=True, inline="always")
def _pol_ray_store(out, pix, Itot):
    out[0, pix] = Itot[0]
    out[1, pix] = _Q_SIGN * Itot[1]
    out[2, pix] = _U_SIGN * Itot[2]
    out[3, pix] = Itot[3]


# ----------------------------------------------------------------------------- drivers
@njit(cache=True, parallel=True)
def rt_kernel_pol(
    frequency, order, x_im_f, y_im_f, z_J_f, z_mid_1D, dz_1D, starts, ends, nint,
    P, xs_tab, rs_tab, ts_tab, as_tab, TAB, RTAB, tau_stop, out,
):
    """Full polarized computation for one frequency; state and coefficients on the fly."""
    Npix = order.shape[0]
    for q in prange(Npix):
        pix = order[q]
        Itot = np.zeros(4)
        Ocum = np.zeros((4, 4))
        O = np.zeros((4, 4))
        cv = np.zeros(4)
        tmp = np.zeros((4, 4))
        _pol_ray_init(Itot, Ocum)
        tau = 0.0
        xi = x_im_f[pix]
        yi = y_im_f[pix]
        zJ = z_J_f[pix]
        done = False
        for k in range(nint[pix]):
            if done:
                break
            for i in range(starts[pix, k], ends[pix, k]):
                g, nup, gc, Cj, Ca, cth, Kap, c2, s2 = _cell_state_pol(
                    xi, yi, zJ, z_mid_1D[i], P, xs_tab, rs_tab, ts_tab, as_tab
                )
                jI, jQ, jV, aI, aQ, aV, rQ, rV = _cell_emis_pol(
                    frequency, g, nup, gc, Cj, Ca, cth, Kap, P, TAB, RTAB
                )
                tau += _pol_cell(jI, jQ, jV, aI, aQ, aV, rQ, rV, c2, s2, g, dz_1D[i], O, cv)
                _pol_accumulate(Itot, Ocum, O, cv, tmp)
                if tau_stop > 0.0 and tau >= tau_stop:
                    done = True
                    break
        _pol_ray_store(out, pix, Itot)


@njit(cache=True, parallel=True)
def rt_kernel_tab_pol(
    frequency, order, x_im_f, y_im_f, z_J_f, z_mid_1D, dz_1D, starts, ends, nint,
    P, Tup, Tlo, logr0, invdlogr, nr, invdu, nu_, TAB, RTAB, tau_stop, out,
):
    """As rt_kernel_pol, with the (r,theta)-only physics from the field table."""
    Npix = order.shape[0]
    for q in prange(Npix):
        pix = order[q]
        Itot = np.zeros(4)
        Ocum = np.zeros((4, 4))
        O = np.zeros((4, 4))
        cv = np.zeros(4)
        tmp = np.zeros((4, 4))
        _pol_ray_init(Itot, Ocum)
        tau = 0.0
        xi = x_im_f[pix]
        yi = y_im_f[pix]
        zJ = z_J_f[pix]
        done = False
        for k in range(nint[pix]):
            if done:
                break
            for i in range(starts[pix, k], ends[pix, k]):
                g, nup, gc, Cj, Ca, cth, Kap, c2, s2 = _cell_state_tab_pol(
                    xi, yi, zJ, z_mid_1D[i], P, Tup, Tlo, logr0, invdlogr, nr, invdu, nu_
                )
                jI, jQ, jV, aI, aQ, aV, rQ, rV = _cell_emis_pol(
                    frequency, g, nup, gc, Cj, Ca, cth, Kap, P, TAB, RTAB
                )
                tau += _pol_cell(jI, jQ, jV, aI, aQ, aV, rQ, rV, c2, s2, g, dz_1D[i], O, cv)
                _pol_accumulate(Itot, Ocum, O, cv, tmp)
                if tau_stop > 0.0 and tau >= tau_stop:
                    done = True
                    break
        _pol_ray_store(out, pix, Itot)


@njit(cache=True, parallel=True)
def precompute_state_pol(
    order, offsets, x_im_f, y_im_f, z_J_f, z_mid_1D, starts, ends, nint,
    P, xs_tab, rs_tab, ts_tab, as_tab, st, st_iz,
):
    """Store the frequency-independent polarized state of every jet cell, ray by ray."""
    Npix = order.shape[0]
    for q in prange(Npix):
        pix = order[q]
        o = offsets[q]
        xi = x_im_f[pix]
        yi = y_im_f[pix]
        zJ = z_J_f[pix]
        for k in range(nint[pix]):
            for i in range(starts[pix, k], ends[pix, k]):
                g, nup, gc, Cj, Ca, cth, Kap, c2, s2 = _cell_state_pol(
                    xi, yi, zJ, z_mid_1D[i], P, xs_tab, rs_tab, ts_tab, as_tab
                )
                st[0, o] = g
                st[1, o] = nup
                st[2, o] = gc
                st[3, o] = Cj
                st[4, o] = Ca
                st[5, o] = cth
                st[6, o] = Kap
                st[7, o] = c2
                st[8, o] = s2
                st_iz[o] = i
                o += 1


@njit(cache=True, parallel=True)
def precompute_state_tab_pol(
    order, offsets, x_im_f, y_im_f, z_J_f, z_mid_1D, starts, ends, nint,
    P, Tup, Tlo, logr0, invdlogr, nr, invdu, nu_, st, st_iz,
):
    """As precompute_state_pol, with the (r,theta)-only physics from the field table."""
    Npix = order.shape[0]
    for q in prange(Npix):
        pix = order[q]
        o = offsets[q]
        xi = x_im_f[pix]
        yi = y_im_f[pix]
        zJ = z_J_f[pix]
        for k in range(nint[pix]):
            for i in range(starts[pix, k], ends[pix, k]):
                g, nup, gc, Cj, Ca, cth, Kap, c2, s2 = _cell_state_tab_pol(
                    xi, yi, zJ, z_mid_1D[i], P, Tup, Tlo, logr0, invdlogr, nr, invdu, nu_
                )
                st[0, o] = g
                st[1, o] = nup
                st[2, o] = gc
                st[3, o] = Cj
                st[4, o] = Ca
                st[5, o] = cth
                st[6, o] = Kap
                st[7, o] = c2
                st[8, o] = s2
                st_iz[o] = i
                o += 1


@njit(cache=True, parallel=True)
def rt_from_state_pol(frequency, order, offsets, dz_1D, P, TAB, RTAB, tau_stop, st, st_iz, out):
    """Polarized transfer for one frequency from the stored state."""
    Npix = order.shape[0]
    for q in prange(Npix):
        pix = order[q]
        Itot = np.zeros(4)
        Ocum = np.zeros((4, 4))
        O = np.zeros((4, 4))
        cv = np.zeros(4)
        tmp = np.zeros((4, 4))
        _pol_ray_init(Itot, Ocum)
        tau = 0.0
        for o in range(offsets[q], offsets[q + 1]):
            g = st[0, o]
            jI, jQ, jV, aI, aQ, aV, rQ, rV = _cell_emis_pol(
                frequency, g, st[1, o], st[2, o], st[3, o], st[4, o], st[5, o], st[6, o],
                P, TAB, RTAB,
            )
            tau += _pol_cell(
                jI, jQ, jV, aI, aQ, aV, rQ, rV, st[7, o], st[8, o], g, dz_1D[st_iz[o]], O, cv
            )
            _pol_accumulate(Itot, Ocum, O, cv, tmp)
            if tau_stop > 0.0 and tau >= tau_stop:
                break
        _pol_ray_store(out, pix, Itot)
