"""
Compiled (numba) radiative-transfer kernel for JetModel.make_image.

This module reproduces the numpy implementation in _core.py cell for cell -- same
formulas, same table lookups, same radiative-transfer update -- but organized as a
loop over image pixels (rays) rather than a loop over depth slices:

  * the set of z-cells that lie inside the jet is found once per model
    (`jet_intervals`), so each call does work proportional to the number of jet
    cells rather than the number of grid cells (typically a ~10x difference);
  * each ray is marched sequentially through its jet cells inside one compiled
    function, so no large temporary arrays are created and the rays run in parallel
    across cores;
  * the frequency-independent physics of a cell (geometry -> fields -> velocity ->
    Doppler factor -> fluid-frame B -> pitch angle -> cooling -> electron density)
    is separated from the frequency-dependent synchrotron coefficients, which allows
    an optional cached mode for loops over frequency (SEDs).

The frequency-independent state of a cell is five numbers:

    g        redshift factor  nu_observed / nu_fluid
    nu_p     synchrotron characteristic frequency  nu_c / gamma^2   [GHz]
    gamma_c  cooling Lorentz factor
    C_j      everything multiplying the bracket in Eq. (C.4) of the paper
    C_a      everything multiplying the bracket in Eq. (C.9) of the paper

Agreement with the numpy path is at the level of floating-point roundoff (relative
differences ~1e-13); the only intentional arithmetic difference is that the powers of
nu/nu_p in Eqs. (C.4) and (C.9) are derived from one call to 10**() rather than
several calls to pow().

Threads: the kernels use numba's thread pool (all cores by default); control it with
jetfuncs.set_num_threads().  The first call on a given machine compiles the kernels
(~5-10 s); the result is cached on disk, so later Python processes start in <1 s.
"""

import math

import numpy as np
from numba import njit, prange

# physical constants (cgs); identical to _core.py
m_e = 9.10938e-28
c = 2.99792458e10
sigma_T = 6.65246e-25


# ----------------------------------------------------------------------------- parameters
# The scalar model parameters are passed to the kernels packed in one float64 array;
# these are the indices into it.
(
    P_COS_I,
    P_SIN_I,
    P_NX,
    P_NY,
    P_NZ,
    P_RH,
    P_NU,
    P_A,
    P_S,
    P_P,
    P_PETA,
    P_H,
    P_ETA,
    P_RG,
    P_SCALING,
    P_SQRT_SCALING,
    P_PREF_EMIS,
    P_PREF_ABS,
    P_PHI_NORM,
    P_GAMMA_INF,
    P_GB_SUPP,
    P_GAMMA_M,
    P_GAMMA_MAX,
    P_HEAT_POYNTING,
    P_TAB_LOGX0,
    P_TAB_INVDLOGX,
    P_TAB_KMAX,
    P_JET_CUTOUT,
    N_PARAMS,
) = range(29)


def pack_params(model, heating_prescription="Poynting"):
    """Collect the scalar parameters of a JetModel into the array the kernels expect."""
    if heating_prescription not in ("Poynting", "magnetic"):
        raise ValueError(
            f"unrecognized heating_prescription {heating_prescription!r}; "
            f"expected 'Poynting' or 'magnetic'"
        )
    P = np.zeros(N_PARAMS, dtype=np.float64)
    P[P_COS_I] = model.cos_i
    P[P_SIN_I] = model.sin_i
    P[P_NX] = model.nx
    P[P_NY] = model.ny
    P[P_NZ] = model.nz
    P[P_RH] = model.rH
    P[P_NU] = model.nu
    P[P_A] = model.a
    P[P_S] = model.s
    P[P_P] = model.p
    P[P_PETA] = model.p_eta
    P[P_H] = model.h
    P[P_ETA] = model.eta
    P[P_RG] = model.rg
    P[P_SCALING] = model.scaling
    P[P_SQRT_SCALING] = model.sqrt_scaling
    P[P_PREF_EMIS] = model.prefac_emis
    P[P_PREF_ABS] = model.prefac_absorp
    P[P_PHI_NORM] = model.phi_norm
    P[P_GAMMA_INF] = model.gamma_inf
    P[P_GB_SUPP] = model.gammabeta_suppression
    P[P_GAMMA_M] = model.gamma_m
    P[P_GAMMA_MAX] = model.gamma_max
    P[P_HEAT_POYNTING] = 1.0 if heating_prescription == "Poynting" else 0.0
    P[P_TAB_LOGX0] = model._tab_logx0
    P[P_TAB_INVDLOGX] = model._tab_inv_dlogx
    P[P_TAB_KMAX] = model._tab_kmax
    P[P_JET_CUTOUT] = model.jet_cutout_fraction
    return P


def pack_tables(model):
    """The six Stokes-I log-tables, in the order the kernels expect."""
    return (
        model.logG_GI_2,
        model.logG_GI_p,
        model.logG_GI_pp1,
        model.logG_GaI_2,
        model.logG_GaI_p,
        model.logG_GaI_pp1,
    )


# ----------------------------------------------------------------------------- small helpers
@njit(cache=True, inline="always")
def _interp_lin(x, xs, ys):
    """np.interp(x, xs, ys) for ascending xs (same arithmetic as numpy's C routine)."""
    n = xs.shape[0]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[n - 1]:
        return ys[n - 1]
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    slope = (ys[hi] - ys[lo]) / (xs[hi] - xs[lo])
    return slope * (x - xs[lo]) + ys[lo]


@njit(cache=True, inline="always")
def _G_bracket(x, logx0, invdlogx, kmax):
    """
    Position of x on the uniform log10(x) table grid, shared by every table evaluated
    at the same x.  Returns (k, t, flag): flag = -1 below the table (plateau), +1 above
    it (zero), 2 for NaN input, 0 otherwise.
    """
    if math.isnan(x):
        return 0, 0.0, 2
    xx = x if x > 1.0e-300 else 1.0e-300
    L = math.log10(xx)
    f = (L - logx0) * invdlogx
    if f < 0.0:
        return 0, 0.0, -1
    if f > kmax + 1.0:
        return 0, 0.0, 1
    fk = f if f < kmax else kmax
    k = int(fk)
    return k, f - k, 0


@njit(cache=True, inline="always")
def _G_eval(k, t, flag, logG):
    """Same arithmetic as JetModel._G_lookup: log-log interpolation at a bracket."""
    if flag == 0:
        return 10.0 ** (logG[k] + t * (logG[k + 1] - logG[k]))
    if flag == -1:
        return 10.0 ** logG[0]
    if flag == 1:
        return 0.0
    return math.nan


# ----------------------------------------------------------------------------- jet mask
@njit(cache=True, parallel=True)
def jet_intervals(x_im_f, y_im_f, z_J_f, z_mid_1D, P, max_int):
    """
    For every pixel, the half-open ranges [start, end) of z-slice indices whose cell
    centres lie inside the jet.  Uses exactly the same criterion and arithmetic as the
    numpy path in make_image, so the set of cells integrated is identical.

    Returns (starts, ends, nint, ncell); nint[pix] may exceed max_int, in which case the
    caller must retry with a larger max_int.
    """
    cos_i = P[P_COS_I]
    sin_i = P[P_SIN_I]
    rH = P[P_RH]
    nu = P[P_NU]
    s = P[P_S]
    cut = P[P_JET_CUTOUT]
    r_min = rH * (1.0 + 1.0e-2)
    Npix = x_im_f.shape[0]
    Nz = z_mid_1D.shape[0]
    starts = np.full((Npix, max_int), -1, dtype=np.int32)
    ends = np.full((Npix, max_int), -1, dtype=np.int32)
    nint = np.zeros(Npix, dtype=np.int32)
    ncell = np.zeros(Npix, dtype=np.int64)
    if cut > 0.0:
        theta_fp_cut = 2.0 * math.asin(cut / math.sqrt(2.0))
    else:
        theta_fp_cut = -1.0
    for pix in prange(Npix):
        inside = False
        k = 0
        cnt = 0
        for i in range(Nz):
            z_im_now = z_mid_1D[i] + z_J_f[pix]
            x = (x_im_f[pix] * cos_i) + (z_im_now * sin_i)
            y = y_im_f[pix]
            z = (z_im_now * cos_i) - (x_im_f[pix] * sin_i)
            r = math.sqrt((x * x) + (y * y) + (z * z))
            costheta = z / r
            omc = 1.0 - costheta
            opc = 1.0 + costheta
            rjet1 = rH * ((1.0 / omc) ** (1.0 / nu))
            rjet2 = rH * ((1.0 / opc) ** (1.0 / nu))
            inj = ((r <= rjet1) or (r <= rjet2)) and (r > r_min)
            if inj and theta_fp_cut > 0.0:
                r_rH_1_s = (r / rH) ** (1.0 - s)
                sin_half = math.sqrt(0.5 * omc)
                cos_half = math.sqrt(0.5 * opc)
                arg_arcsin = r_rH_1_s * sin_half
                if arg_arcsin < 0.7071067811865476:
                    theta_fp = 2.0 * math.asin(arg_arcsin)
                else:
                    theta_fp = 2.0 * math.acos(r_rH_1_s * cos_half)
                if (theta_fp < theta_fp_cut) or (theta_fp > (math.pi - theta_fp_cut)):
                    inj = False
            if inj:
                cnt += 1
                if not inside:
                    inside = True
                    if k < max_int:
                        starts[pix, k] = i
            elif inside:
                inside = False
                if k < max_int:
                    ends[pix, k] = i
                k += 1
        if inside:
            if k < max_int:
                ends[pix, k] = Nz
            k += 1
        nint[pix] = k
        ncell[pix] = cnt
    return starts, ends, nint, ncell


# ----------------------------------------------------------------------------- per-cell physics
#
# The frequency-independent physics of a cell is evaluated in two steps:
#
#   _rtheta_chain : everything that depends on (r, theta) only -- force-free fields,
#                   drift-frame velocity, velocity regularization, fluid-frame field
#                   strength, cooling and electron normalization; returns quantities in
#                   the spherical orthonormal basis
#   _ray_part     : the part that depends on the line of sight -- rotation to Cartesian
#                   components, Doppler factor, aberration, pitch angle -- and the final
#                   assembly of the emissivity/absorption prefactors
#
# _cell_state chains the two exactly (agreement with _core.make_image at roundoff).
# The split also allows the (r,theta)-only part to be tabulated once per model on a
# 2-D grid and interpolated (build_field_table / _cell_state_tab), which is what the
# optional "field table" mode of JetModel uses.
# -----------------------------------------------------------------------------


@njit(cache=True)
def _rtheta_chain(r, omc, sgn, P, th_tab, rs_tab, ts_tab):
    """
    (r, theta)-only physics of one point.  omc = 1 - |cos theta| (passed exactly, so that
    points very close to the axis keep their precision) and sgn = sign(cos theta).

    Returns (vr, vt, vp, gamma, alphalapse, Bpr, Bpt, Bpp, Bpm, gamma_c, Kj, Ka) with
      (vr, vt, vp)     regularized fluid velocity, spherical orthonormal basis
      gamma            corresponding Lorentz factor
      alphalapse       lapse
      (Bpr, Bpt, Bpp)  fluid-frame magnetic field, spherical orthonormal basis; Bpm = |B'|
      gamma_c          cooling Lorentz factor
      Kj, Ka           prefac_emis/absorp * n_e * A_norm * gamma_2^(p_2 - p_1); the last
                       factor makes this continuous across the fast/slow cooling boundary
                       (n_e*A_norm alone jumps there by gamma_m^(p-2), compensated by the
                       bracket in Eqs. C.4/C.9)
    """
    rH = P[P_RH]
    nu = P[P_NU]
    a = P[P_A]
    s = P[P_S]
    p = P[P_P]
    h = P[P_H]
    rg = P[P_RG]
    scaling = P[P_SCALING]
    sqrt_scaling = P[P_SQRT_SCALING]
    prefac_emis = P[P_PREF_EMIS]
    prefac_absorp = P[P_PREF_ABS]
    gamma_inf = P[P_GAMMA_INF]
    gammabeta_suppression = P[P_GB_SUPP]
    gamma_m = P[P_GAMMA_M]
    gamma_max = P[P_GAMMA_MAX]
    heating_is_poynting = P[P_HEAT_POYNTING] > 0.5

    # ---------------- geometry
    r2 = r * r
    costheta = sgn * (1.0 - omc)
    z = r * costheta
    half_small = math.sqrt(0.5 * omc)
    half_big = math.sqrt(0.5 * (2.0 - omc))
    if sgn > 0.0:
        sin_half = half_small
        cos_half = half_big
    else:
        sin_half = half_big
        cos_half = half_small

    # footpoint theta
    r_rH_1_s = (r / rH) ** (1.0 - s)
    arg_arcsin = r_rH_1_s * sin_half
    if arg_arcsin < 0.7071067811865476:
        theta_fp = 2.0 * math.asin(arg_arcsin)
    else:
        theta_fp = 2.0 * math.acos(r_rH_1_s * cos_half)

    # horizon buffer
    r_min = rH * (1.0 + 1.0e-2)

    # stream function, Omega, stagnation surface
    rHnu = rH**nu
    psi = rHnu * (1.0 - abs(math.cos(theta_fp)))
    cthhorizon = 1.0 - psi * (rH ** (-nu))
    Omega = a / (4.0 + 8.0 / (1.0 + cthhorizon))
    if theta_fp <= 0.5 * math.pi:
        rstag = _interp_lin(theta_fp, th_tab, rs_tab)
        tstag = _interp_lin(theta_fp, th_tab, ts_tab)
    else:
        tf = math.pi - theta_fp
        rstag = _interp_lin(tf, th_tab, rs_tab)
        tstag = _interp_lin(tf, th_tab, ts_tab)

    # metric quantities
    sth2 = omc * (2.0 - omc)  # = 1 - cos^2(theta), without cancellation
    sintheta = math.sqrt(sth2)
    costh = costheta
    cth2 = 1.0 - sth2
    a2 = a * a
    r2pa2 = r2 + a2
    rho2 = r2 + (a2 * cth2)
    Delta = r2pa2 - (2.0 * r)
    Delta_min = (r_min * r_min) - 2.0 * r_min + a2
    if Delta < Delta_min:
        Delta = Delta_min
    Sigma = (r2pa2 * r2pa2) - (a2 * Delta * sth2)
    alphalapse = math.sqrt(Delta * rho2 / Sigma)
    sth2_rho2 = sth2 / rho2
    g03 = -2.0 * a * r * sth2_rho2
    g11 = rho2 / Delta
    g22 = rho2
    g33 = Sigma * sth2_rho2
    gdet = sintheta * rho2

    # EM field
    r_nu = r**nu
    signcostheta = 1.0 if costh > 0.0 else (-1.0 if costh < 0.0 else 0.0)
    dpsidtheta = signcostheta * sintheta * r_nu
    dpsidr = nu * psi / r
    Ipol = -4.0 * math.pi * psi * Omega * signcostheta
    B1 = dpsidtheta / gdet
    B2 = -dpsidr / gdet
    B3 = Ipol / (2.0 * math.pi * Delta * sth2)
    sq_g11 = math.sqrt(g11)
    sq_g22 = math.sqrt(g22)
    sq_g33 = math.sqrt(g33)
    Br = B1 * sq_g11
    Btheta = B2 * sq_g22
    Bphi = B3 * sq_g33
    omegaz = 2.0 * a * r / Sigma
    E1 = (Omega - omegaz) * Sigma * sintheta * B2 / rho2
    E2 = -(Omega - omegaz) * Sigma * sintheta * B1 / (rho2 * Delta)

    # ---------------- u_driftframe (fast path)
    B1_cov = g11 * B1
    B2_cov = g22 * B2
    B3_cov = g33 * B3
    E1_cov = g11 * E1
    E2_cov = g22 * E2
    Bsq = (B1_cov * B1) + (B2_cov * B2) + (B3_cov * B3)
    Esq = (E1_cov * E1) + (E2_cov * E2)
    eps_EB = 1.0e-8
    ratio_raw = Esq / Bsq
    if ratio_raw >= (1.0 - eps_EB):
        scl = math.sqrt((1.0 - eps_EB) / ratio_raw)
        E1 = E1 * scl
        E2 = E2 * scl
        E1_cov = E1_cov * scl
        E2_cov = E2_cov * scl
        Esq = (E1_cov * E1) + (E2_cov * E2)
    ratio = Esq / Bsq
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0 - eps_EB:
        ratio = 1.0 - eps_EB

    cth2_stag = math.cos(tstag) ** 2.0
    sth2_stag = math.sin(tstag) ** 2.0
    r2_stag = rstag * rstag
    rho2_stag = r2_stag + (a2 * cth2_stag)
    Delta_stag = r2_stag - (2.0 * rstag) + a2
    Sigma_stag = ((r2_stag + a2) ** 2.0) - (a2 * Delta_stag * sth2_stag)
    g00_stag = ((a2 * sth2_stag) - Delta_stag) / rho2_stag
    g03_stag = -2.0 * a * rstag * sth2_stag / rho2_stag
    g33_stag = Sigma_stag * sth2_stag / rho2_stag
    ginv_denom_stag = (g00_stag * g33_stag) - (g03_stag * g03_stag)
    ginv00_stag = g33_stag / ginv_denom_stag
    ginv03_stag = -g03_stag / ginv_denom_stag
    ginv33_stag = g00_stag / ginv_denom_stag
    gtphifac = g03_stag + (g33_stag * Omega)
    gttfac = g00_stag + (g03_stag * Omega)
    coef0 = (g00_stag + (Omega * (2.0 * g03_stag + g33_stag * Omega))) ** 2.0
    coef1 = (ginv33_stag * (gtphifac * gtphifac)) + (
        2.0 * ginv03_stag * gtphifac * gttfac + ginv00_stag * (gttfac * gttfac)
    )
    Aconst = math.sqrt(-coef0 / coef1)
    Aconst2 = Aconst * Aconst

    vphiupper = (alphalapse / (Bsq * gdet)) * (E1_cov * B2_cov - B1_cov * E2_cov)
    gammap = 1.0 / math.sqrt(1.0 - ratio)
    Bhatphi = B3 / math.sqrt(Bsq)
    ffunc = gammap * (alphalapse - ((g03 + g33 * Omega) * vphiupper))
    bred = Bhatphi * (g03 + (g33 * Omega))
    bred2 = bred * bred
    disc = Aconst2 - (ffunc * ffunc) + bred2
    if disc < 0.0:
        disc = 0.0
    root = math.sqrt(disc)
    sgn_r = 1.0 if r > rstag else (-1.0 if r < rstag else 0.0)
    nu_parallel = ((ffunc * bred) + signcostheta * sgn_r * Aconst * root) / (Aconst2 + bred2)

    pref = alphalapse / (Bsq * gdet)
    vperp1 = pref * (E2_cov * B3_cov)  # E3 = 0
    vperp2 = pref * (-B3_cov * E1_cov)
    vperp3 = pref * (E1_cov * B2_cov - B1_cov * E2_cov)
    vpar_max = math.sqrt(1.0 - ratio)
    prefac_here = nu_parallel * vpar_max / math.sqrt(Bsq)
    v1 = vperp1 + prefac_here * B1
    v2 = vperp2 + prefac_here * B2
    v3 = vperp3 + prefac_here * B3
    vsq = g11 * v1 * v1 + g22 * v2 * v2 + g33 * v3 * v3
    if vsq >= (1.0 - 1.0e-12):
        fac = math.sqrt((1.0 - 1.0e-12) / vsq)
        v1 *= fac
        v2 *= fac
        v3 *= fac
        vsq = 1.0 - 1.0e-12
    gamma = 1.0 / math.sqrt(1.0 - vsq)
    # ZAMO-frame velocity (no frame-dragging shift; see _core.u_driftframe)
    u1 = gamma * v1
    u2 = gamma * v2
    u3 = gamma * (v3 + 0.0)
    Bdotv = (g11 * v1 * B1) + (g22 * v2 * B2) + (g33 * v3 * B3)
    Bdotv_Bsq = Bdotv / Bsq
    v1perp = v1 - (B1 * Bdotv_Bsq)
    v2perp = v2 - (B2 * Bdotv_Bsq)
    v3perp = v3 - (B3 * Bdotv_Bsq)
    vperpmag = math.sqrt(
        (g11 * v1perp * v1perp) + (g22 * v2perp * v2perp) + (g33 * v3perp * v3perp)
    )

    # ---------------- ZAMO-frame Poynting flux, B rescale
    B1Zamo = alphalapse * Br
    B2Zamo = alphalapse * Btheta
    B3Zamo = alphalapse * Bphi
    BsqZ = (B1Zamo * B1Zamo) + (B2Zamo * B2Zamo) + (B3Zamo * B3Zamo)
    poyntingmag = BsqZ * vperpmag * (c / (4.0 * math.pi))
    if math.isnan(poyntingmag):
        poyntingmag = 0.0
    S = abs(poyntingmag) * scaling
    Br *= alphalapse * sqrt_scaling
    Btheta *= alphalapse * sqrt_scaling
    Bphi *= alphalapse * sqrt_scaling

    # ---------------- velocity rescaling
    vr_orig = u1 * sq_g11 / gamma
    vtheta_orig = u2 * sq_g22 / gamma
    vphi_orig = u3 * sq_g33 / gamma
    b2 = 1.0 - 1.0 / (gamma * gamma)
    beta_orig = math.sqrt(b2 if b2 > 0.0 else 0.0)
    gammabeta = gammabeta_suppression * beta_orig * gamma
    gamma = math.sqrt(1.0 + (gammabeta * gammabeta))
    if gamma > gamma_inf:
        gamma = gamma_inf
    b2 = 1.0 - 1.0 / (gamma * gamma)
    beta = math.sqrt(b2 if b2 > 0.0 else 0.0)
    velscale = (beta / beta_orig) if beta_orig > 0.0 else 0.0
    vr = velscale * vr_orig
    vt = velscale * vtheta_orig
    vp = velscale * vphi_orig

    # ---------------- fluid-frame B, still in the spherical basis (the boost commutes
    # with the rotation to Cartesian components applied later)
    vmag = math.sqrt(vr * vr + vt * vt + vp * vp)
    if vmag > 1.0e-8:
        vhr = vr / vmag
        vht = vt / vmag
        vhp = vp / vmag
        B_par = Br * vhr + Btheta * vht + Bphi * vhp
        Bpr = ((Br - B_par * vhr) / gamma) + (B_par * vhr)
        Bpt = ((Btheta - B_par * vht) / gamma) + (B_par * vht)
        Bpp = ((Bphi - B_par * vhp) / gamma) + (B_par * vhp)
    else:
        Bpr = Br
        Bpt = Btheta
        Bpp = Bphi
    Bpm = math.sqrt(Bpr * Bpr + Bpt * Bpt + Bpp * Bpp)

    # ---------------- electrons
    t_c = abs(z * rg) / (c * gamma)
    gamma_c = (6.0 * math.pi * m_e * c) / (sigma_T * (Bpm * Bpm) * t_c)
    if heating_is_poynting:
        u_pl = h * S / (c * gamma)
    else:
        u_pl = h * ((Bpm * Bpm) / (8.0 * math.pi))
    n_m = (((p - 2.0) * u_pl) / ((gamma_m**p) * m_e * (c * c))) * (
        1.0 / ((gamma_m ** (2.0 - p)) - (gamma_max ** (2.0 - p)))
    )

    # n_e * A_norm for the cooling branch this point is in, times gamma_2^(p_2 - p_1)
    if gamma_c >= gamma_max:
        # uncooled: single power law between gamma_m and gamma_max
        A_norm = 1.0 / (((gamma_max ** (1.0 - p)) - (gamma_m ** (1.0 - p))) / (1.0 - p))
        n = n_m * (gamma_m**p) * (((gamma_max ** (1.0 - p)) - (gamma_m ** (1.0 - p))) / (1.0 - p))
        cont = gamma_max
    elif gamma_c > gamma_m:
        # slow cooling: p1 = p, p2 = p+1, break at gamma_c
        p2 = p + 1.0
        A_norm = 1.0 / (
            (((gamma_c ** (1.0 - p)) - (gamma_m ** (1.0 - p))) / (1.0 - p))
            + (
                (gamma_c ** (p2 - p))
                * (((gamma_max ** (1.0 - p2)) - (gamma_c ** (1.0 - p2))) / (1.0 - p2))
            )
        )
        n = (
            n_m
            * (gamma_m**p)
            * (
                (((gamma_c ** (1.0 - p)) - (gamma_m ** (1.0 - p))) / (1.0 - p))
                - (gamma_c * (((gamma_max ** (-p)) - (gamma_c ** (-p))) / p))
            )
        )
        cont = gamma_c
    else:
        # fast cooling: p1 = 2, p2 = p+1, break at gamma_m
        p2 = p + 1.0
        A_norm = 1.0 / (
            (((gamma_m ** (1.0 - 2.0)) - (gamma_c ** (1.0 - 2.0))) / (1.0 - 2.0))
            + (
                (gamma_m ** (p2 - 2.0))
                * (((gamma_max ** (1.0 - p2)) - (gamma_m ** (1.0 - p2))) / (1.0 - p2))
            )
        )
        n = (
            n_m
            * gamma_c
            * (
                (gamma_m * ((gamma_c ** (-1.0)) - (gamma_m ** (-1.0))))
                + ((gamma_m**p) * (((gamma_m ** (-p)) - (gamma_max ** (-p))) / p))
            )
        )
        cont = gamma_m ** (p - 1.0)
    Kj = prefac_emis * n * A_norm * cont
    Ka = prefac_absorp * n * A_norm * cont
    return vr, vt, vp, gamma, alphalapse, Bpr, Bpt, Bpp, Bpm, gamma_c, Kj, Ka


@njit(cache=True)
def _ray_part(x, y, z, r, R, vr, vt, vp, gamma, alphalapse, Bpr, Bpt, Bpp, gamma_c, Kj, Ka, P):
    """
    Line-of-sight-dependent part of the cell physics: rotate the spherical-basis
    velocity and fluid-frame field to Cartesian components, then Doppler factor,
    aberration and pitch angle, and assemble the emissivity/absorption prefactors.

    Returns (g, nu_p, gamma_c, C_j, C_a) as consumed by _cell_emis.
    """
    nx = P[P_NX]
    ny = P[P_NY]
    nz = P[P_NZ]
    eta = P[P_ETA]
    p_eta = P[P_PETA]
    phi_norm = P[P_PHI_NORM]
    p = P[P_P]
    gamma_m = P[P_GAMMA_M]
    gamma_max = P[P_GAMMA_MAX]

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

    # undo the gamma_2^(p_2 - p_1) factor of the stored prefactors (see _rtheta_chain)
    if gamma_c >= gamma_max:
        cont = gamma_max
    elif gamma_c > gamma_m:
        cont = gamma_c
    else:
        cont = gamma_m ** (p - 1.0)
    Cj = (Kj / cont) * nup * aniso_fac
    Ca = ((Ka / cont) / nup) * aniso_fac
    return g, nup, gamma_c, Cj, Ca


@njit(cache=True)
def _cell_state(xi, yi, zJ, zi, P, th_tab, rs_tab, ts_tab):
    """
    Frequency-independent physics of one cell, evaluated exactly.
    Returns (g, nu_p, gamma_c, C_j, C_a).
    """
    cos_i = P[P_COS_I]
    sin_i = P[P_SIN_I]
    z_im_now = zi + zJ
    x = (xi * cos_i) + (z_im_now * sin_i)
    y = yi
    z = (z_im_now * cos_i) - (xi * sin_i)
    R2 = (x * x) + (y * y)
    r = math.sqrt(R2 + (z * z))
    R = math.sqrt(R2)
    costheta = z / r
    sgn = 1.0 if costheta >= 0.0 else -1.0
    vr, vt, vp, gamma, alpha, Bpr, Bpt, Bpp, Bpm, gamma_c, Kj, Ka = _rtheta_chain(
        r, 1.0 - abs(costheta), sgn, P, th_tab, rs_tab, ts_tab
    )
    return _ray_part(x, y, z, r, R, vr, vt, vp, gamma, alpha, Bpr, Bpt, Bpp, gamma_c, Kj, Ka, P)


# ----------------------------------------------------------------------------- field table
#
# Optional acceleration: the (r,theta)-only quantities are tabulated on a 2-D grid that
# is uniform in log10(r - r_H) and in u = sqrt(psi/psi_edge) (u is proportional to the
# polar angle near the axis, which keeps v_phi and B_phi linear there), one table per
# hemisphere, and bilinearly interpolated per cell.  The interpolation error decreases
# as the square of the grid spacing; see JetModel.build_field_table for measured values.

N_TAB = 12  # stored quantities per node


@njit(cache=True, parallel=True)
def build_field_table(logr, ugrid, P, th_tab, rs_tab, ts_tab, hemisphere):
    """
    Table T[i_r, i_u, q] of the (r,theta)-only quantities for one hemisphere (+1/-1):
      0-2 (vr, vt, vp)   3 gamma   4 lapse   5-7 (Bpr, Bpt, Bpp)   8 |B'|
      9 gamma_c   10 Kj   11 Ka
    logr is log10(r - r_H); ugrid spans [0, 1].
    """
    rH = P[P_RH]
    nu = P[P_NU]
    nr = logr.shape[0]
    nu_ = ugrid.shape[0]
    T = np.empty((nr, nu_, N_TAB))
    for i in prange(nr):
        r = rH + 10.0 ** logr[i]
        t = (r / rH) ** nu
        for j in range(nu_):
            w = ugrid[j] * ugrid[j]
            if w < 1.0e-12:
                w = 1.0e-12
            if w > 1.0 - 1.0e-12:  # exactly w=1 falls on the arcsin/arccos branch boundary
                w = 1.0 - 1.0e-12
            omc = w / t  # 1 - |cos theta|, exact
            vr, vt, vp, gamma, alpha, Bpr, Bpt, Bpp, Bpm, gc, Kj, Ka = _rtheta_chain(
                r, omc, hemisphere, P, th_tab, rs_tab, ts_tab
            )
            T[i, j, 0] = vr
            T[i, j, 1] = vt
            T[i, j, 2] = vp
            T[i, j, 3] = gamma
            T[i, j, 4] = alpha
            T[i, j, 5] = Bpr
            T[i, j, 6] = Bpt
            T[i, j, 7] = Bpp
            T[i, j, 8] = Bpm
            T[i, j, 9] = gc
            T[i, j, 10] = Kj
            T[i, j, 11] = Ka
    return T


@njit(cache=True)
def _cell_state_tab(xi, yi, zJ, zi, P, Tup, Tlo, logr0, invdlogr, nr, invdu, nu_):
    """Same as _cell_state, with the (r,theta)-only part interpolated from the field table."""
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
    costheta = z / r
    r_rH_1_s = (r / rH) ** (1.0 - s)
    if costheta >= 0.0:
        u = r_rH_1_s * math.sqrt(1.0 - costheta)  # sqrt(psi/psi_edge)
        T = Tup
    else:
        u = r_rH_1_s * math.sqrt(1.0 + costheta)
        T = Tlo
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
    return _ray_part(
        x, y, z, r, R, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[9], q[10], q[11], P
    )


@njit(cache=True)
def _cell_emis(frequency, g, nup, gamma_c, Cj, Ca, P, T0, T1, T2, T3, T4, T5):
    """
    Frequency-dependent synchrotron coefficients (j_I, alpha_I) of one cell from its
    stored state.  T0..T5 are the log-tables GI_2, GI_p, GI_pp1, GaI_2, GaI_p, GaI_pp1.
    Same formulas as _core.make_image (paper Eqs. C.4 and C.9).
    """
    p = P[P_P]
    gamma_m = P[P_GAMMA_M]
    gamma_max = P[P_GAMMA_MAX]
    logx0 = P[P_TAB_LOGX0]
    invdlogx = P[P_TAB_INVDLOGX]
    kmax = P[P_TAB_KMAX]

    nu_nup = (frequency / g) / nup
    if math.isnan(nu_nup):
        return math.nan, math.nan  # propagates like the numpy path; zeroed in _rt_step
    l10nn = math.log10(nu_nup) if nu_nup > 0.0 else -math.inf
    sq = math.sqrt(nu_nup)

    if gamma_c >= gamma_max:
        # uncooled: p1 = p; g1 = gamma_m, g2 = gamma_max
        k1, t1, f1 = _G_bracket(nu_nup / (gamma_m * gamma_m), logx0, invdlogx, kmax)
        k2, t2, f2 = _G_bracket(nu_nup / (gamma_max * gamma_max), logx0, invdlogx, kmax)
        pw = 10.0 ** (0.5 * (1.0 - p) * l10nn)  # nu_nup**((1-p)/2)
        jI = Cj * pw * (_G_eval(k2, t2, f2, T1) - _G_eval(k1, t1, f1, T1))
        pwa = pw / (nu_nup * nu_nup * sq)  # nu_nup**(-(p+4)/2)
        alphaI = Ca * (p + 2.0) * pwa * (_G_eval(k2, t2, f2, T4) - _G_eval(k1, t1, f1, T4))
    elif gamma_c > gamma_m:
        # slow: p1 = p, p2 = p+1; g1 = gamma_m, g2 = gamma_c, g3 = gamma_max
        k1, t1, f1 = _G_bracket(nu_nup / (gamma_m * gamma_m), logx0, invdlogx, kmax)
        k2, t2, f2 = _G_bracket(nu_nup / (gamma_c * gamma_c), logx0, invdlogx, kmax)
        k3, t3, f3 = _G_bracket(nu_nup / (gamma_max * gamma_max), logx0, invdlogx, kmax)
        pw1 = 10.0 ** (0.5 * (1.0 - p) * l10nn)  # nu_nup**((1-p1)/2)
        pw2 = pw1 / sq  # nu_nup**((1-p2)/2)
        dG1 = _G_eval(k2, t2, f2, T1) - _G_eval(k1, t1, f1, T1)
        dG2 = _G_eval(k3, t3, f3, T2) - _G_eval(k2, t2, f2, T2)
        jI = Cj * (pw1 * dG1 + gamma_c * pw2 * dG2)  # g2**(p2-p1) = gamma_c
        pwa1 = pw1 / (nu_nup * nu_nup * sq)  # nu_nup**(-(p1+4)/2)
        pwa2 = pwa1 / sq  # nu_nup**(-(p2+4)/2)
        dGa1 = _G_eval(k2, t2, f2, T4) - _G_eval(k1, t1, f1, T4)
        dGa2 = _G_eval(k3, t3, f3, T5) - _G_eval(k2, t2, f2, T5)
        alphaI = Ca * ((p + 2.0) * pwa1 * dGa1 + (p + 3.0) * gamma_c * pwa2 * dGa2)
    else:
        # fast: p1 = 2, p2 = p+1; g1 = gamma_c, g2 = gamma_m, g3 = gamma_max
        k1, t1, f1 = _G_bracket(nu_nup / (gamma_c * gamma_c), logx0, invdlogx, kmax)
        k2, t2, f2 = _G_bracket(nu_nup / (gamma_m * gamma_m), logx0, invdlogx, kmax)
        k3, t3, f3 = _G_bracket(nu_nup / (gamma_max * gamma_max), logx0, invdlogx, kmax)
        gm_pm1 = gamma_m ** (p - 1.0)  # g2**(p2-p1)
        pw1 = 1.0 / sq  # nu_nup**(-1/2)
        pw2 = 10.0 ** (-0.5 * p * l10nn)  # nu_nup**((1-p2)/2) = nu_nup**(-p/2)
        dG1 = _G_eval(k2, t2, f2, T0) - _G_eval(k1, t1, f1, T0)
        dG2 = _G_eval(k3, t3, f3, T2) - _G_eval(k2, t2, f2, T2)
        jI = Cj * (pw1 * dG1 + gm_pm1 * pw2 * dG2)
        pwa1 = 1.0 / (nu_nup * nu_nup * nu_nup)  # nu_nup**(-3)
        pwa2 = pw2 / (nu_nup * nu_nup * sq)  # nu_nup**(-(p+5)/2)
        dGa1 = _G_eval(k2, t2, f2, T3) - _G_eval(k1, t1, f1, T3)
        dGa2 = _G_eval(k3, t3, f3, T5) - _G_eval(k2, t2, f2, T5)
        alphaI = Ca * (4.0 * pwa1 * dGa1 + (p + 3.0) * gm_pm1 * pwa2 * dGa2)
    return jI, alphaI


@njit(cache=True, inline="always")
def _rt_step(I_acc, tau_acc, jI, alphaI, g, dz):
    """One cell of the front-to-back transfer; same guards and update as make_image."""
    if not math.isfinite(alphaI) or alphaI < 0.0:
        alphaI = 0.0
    a0 = alphaI / g
    j0 = (g * g) * jI
    if not math.isfinite(a0) or a0 < 0.0:
        a0 = 0.0
    if not math.isfinite(j0):
        j0 = 0.0
    tau_step = a0 * dz
    atten = math.exp(-tau_acc)
    if a0 > 0.0:
        one_minus_e = -math.expm1(-tau_step)
        I_acc += atten * (j0 / a0) * one_minus_e
    else:
        I_acc += atten * j0 * dz
    tau_acc += tau_step
    return I_acc, tau_acc


# ----------------------------------------------------------------------------- drivers
@njit(cache=True, parallel=True)
def rt_kernel(
    frequency,
    order,
    x_im_f,
    y_im_f,
    z_J_f,
    z_mid_1D,
    dz_1D,
    starts,
    ends,
    nint,
    P,
    th_tab,
    rs_tab,
    ts_tab,
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    tau_stop,
    I_out,
):
    """Full computation for one frequency: state and coefficients evaluated on the fly."""
    Npix = order.shape[0]
    for q in prange(Npix):
        pix = order[q]
        I_acc = 0.0
        tau_acc = 0.0
        xi = x_im_f[pix]
        yi = y_im_f[pix]
        zJ = z_J_f[pix]
        done = False
        for k in range(nint[pix]):
            if done:
                break
            for i in range(starts[pix, k], ends[pix, k]):
                g, nup, gamma_c, Cj, Ca = _cell_state(
                    xi, yi, zJ, z_mid_1D[i], P, th_tab, rs_tab, ts_tab
                )
                jI, alphaI = _cell_emis(
                    frequency, g, nup, gamma_c, Cj, Ca, P, T0, T1, T2, T3, T4, T5
                )
                I_acc, tau_acc = _rt_step(I_acc, tau_acc, jI, alphaI, g, dz_1D[i])
                if tau_stop > 0.0 and tau_acc >= tau_stop:
                    done = True
                    break
        I_out[pix] = I_acc


@njit(cache=True, parallel=True)
def precompute_state(
    order,
    offsets,
    x_im_f,
    y_im_f,
    z_J_f,
    z_mid_1D,
    starts,
    ends,
    nint,
    P,
    th_tab,
    rs_tab,
    ts_tab,
    st_g,
    st_nup,
    st_gc,
    st_Cj,
    st_Ca,
    st_iz,
):
    """Store the frequency-independent state of every jet cell, ray by ray."""
    Npix = order.shape[0]
    for q in prange(Npix):
        pix = order[q]
        o = offsets[q]
        xi = x_im_f[pix]
        yi = y_im_f[pix]
        zJ = z_J_f[pix]
        for k in range(nint[pix]):
            for i in range(starts[pix, k], ends[pix, k]):
                g, nup, gamma_c, Cj, Ca = _cell_state(
                    xi, yi, zJ, z_mid_1D[i], P, th_tab, rs_tab, ts_tab
                )
                st_g[o] = g
                st_nup[o] = nup
                st_gc[o] = gamma_c
                st_Cj[o] = Cj
                st_Ca[o] = Ca
                st_iz[o] = i
                o += 1


@njit(cache=True, parallel=True)
def rt_from_state(
    frequency,
    order,
    offsets,
    dz_1D,
    P,
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    tau_stop,
    st_g,
    st_nup,
    st_gc,
    st_Cj,
    st_Ca,
    st_iz,
    I_out,
):
    """Transfer for one frequency from the stored state (frequency-dependent part only)."""
    Npix = order.shape[0]
    for q in prange(Npix):
        pix = order[q]
        I_acc = 0.0
        tau_acc = 0.0
        for o in range(offsets[q], offsets[q + 1]):
            g = st_g[o]
            jI, alphaI = _cell_emis(
                frequency, g, st_nup[o], st_gc[o], st_Cj[o], st_Ca[o], P, T0, T1, T2, T3, T4, T5
            )
            I_acc, tau_acc = _rt_step(I_acc, tau_acc, jI, alphaI, g, dz_1D[st_iz[o]])
            if tau_stop > 0.0 and tau_acc >= tau_stop:
                break
        I_out[pix] = I_acc


@njit(cache=True, parallel=True)
def rt_kernel_tab(
    frequency,
    order,
    x_im_f,
    y_im_f,
    z_J_f,
    z_mid_1D,
    dz_1D,
    starts,
    ends,
    nint,
    P,
    Tup,
    Tlo,
    logr0,
    invdlogr,
    nr,
    invdu,
    nu_,
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    tau_stop,
    I_out,
):
    """As rt_kernel, with the (r,theta)-only physics interpolated from the field table."""
    Npix = order.shape[0]
    for q in prange(Npix):
        pix = order[q]
        I_acc = 0.0
        tau_acc = 0.0
        xi = x_im_f[pix]
        yi = y_im_f[pix]
        zJ = z_J_f[pix]
        done = False
        for k in range(nint[pix]):
            if done:
                break
            for i in range(starts[pix, k], ends[pix, k]):
                g, nup, gamma_c, Cj, Ca = _cell_state_tab(
                    xi, yi, zJ, z_mid_1D[i], P, Tup, Tlo, logr0, invdlogr, nr, invdu, nu_
                )
                jI, alphaI = _cell_emis(
                    frequency, g, nup, gamma_c, Cj, Ca, P, T0, T1, T2, T3, T4, T5
                )
                I_acc, tau_acc = _rt_step(I_acc, tau_acc, jI, alphaI, g, dz_1D[i])
                if tau_stop > 0.0 and tau_acc >= tau_stop:
                    done = True
                    break
        I_out[pix] = I_acc


@njit(cache=True, parallel=True)
def precompute_state_tab(
    order,
    offsets,
    x_im_f,
    y_im_f,
    z_J_f,
    z_mid_1D,
    starts,
    ends,
    nint,
    P,
    Tup,
    Tlo,
    logr0,
    invdlogr,
    nr,
    invdu,
    nu_,
    st_g,
    st_nup,
    st_gc,
    st_Cj,
    st_Ca,
    st_iz,
):
    """As precompute_state, with the (r,theta)-only physics interpolated from the field table."""
    Npix = order.shape[0]
    for q in prange(Npix):
        pix = order[q]
        o = offsets[q]
        xi = x_im_f[pix]
        yi = y_im_f[pix]
        zJ = z_J_f[pix]
        for k in range(nint[pix]):
            for i in range(starts[pix, k], ends[pix, k]):
                g, nup, gamma_c, Cj, Ca = _cell_state_tab(
                    xi, yi, zJ, z_mid_1D[i], P, Tup, Tlo, logr0, invdlogr, nr, invdu, nu_
                )
                st_g[o] = g
                st_nup[o] = nup
                st_gc[o] = gamma_c
                st_Cj[o] = Cj
                st_Ca[o] = Ca
                st_iz[o] = i
                o += 1
