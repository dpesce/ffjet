"""
Compiled (numba) radiative-transfer kernels for JetModel(geometry="kerr"): the Kerr
counterparts of the drivers in _kernel.py (Stokes I) and _kernel_pol.py (full Stokes).

Everything that does not depend on the shape of the rays is shared with the flat
kernels and used unchanged: the axisymmetric physics of a point (_rtheta_chain, or
_rtheta_tab with the field table), the synchrotron coefficients (_cell_emis,
_cell_emis_pol), the transfer through a cell (_rt_step, _pol_cell), and the transfer
from stored state (rt_from_state, rt_from_state_pol), which needs no geometry at all.
What is new:

  * rays are the null geodesics of _geodesics.py, traced backward from the observer
    through the pixel's screen point.  Every kernel call traces its rays again (a few
    hundred to a few thousand Runge-Kutta steps, cheap next to the jet cells) into a
    trajectory buffer held per thread; the trace is deterministic, so every call sees
    the same trajectory;
  * the cells of a ray are laid out in its affine parameter about lambda_J, the
    geodesic's closest approach to the jet axis, on the model's z-grid, exactly as the
    flat model lays them out in z_im about z_J.  The segments of the ray inside the jet
    are found once per model by the same scan and bisection as in flat mode
    (jet_segments_kerr), so a cell cut by the jet boundary counts only its part inside;
  * the line-of-sight half of each cell (_ray_part_kerr, _ray_part_pol_kerr) works in
    the local ZAMO frame -- the frame _rtheta_chain gives the velocity and field in --
    with the photon's local direction n and energy factor 1 - omega lambda, so that the
    redshift g = alpha / [gamma (1 - omega lambda)(1 - beta n.v)] includes frame
    dragging, and with the observer's Stokes basis carried to the cell by its
    Walker-Penrose constant.  With n the flat line of sight, 1 - omega lambda = 1 and
    the fixed sky basis, these reduce to _ray_part and _ray_part_pol.

The ray geometry (mass, spin, observer, start radius, step parameters) is passed packed
in a small array G.  Its mass and spin are separate from the fluid model's, so that
M = a = 0 traces straight rays through the same jet: the flat-limit test.
"""

import math

import numba
import numpy as np
from numba import njit, prange

from . import _geodesics as geo
from ._kernel import (
    N_BISECT,
    N_TAB,
    P_ETA,
    P_INV_RHNU,
    P_JET_CUTOUT,
    P_NU,
    P_PETA,
    P_PHI_NORM,
    P_RG,
    P_RH,
    _cell_emis,
    _cell_overlap,
    _rt_step,
    _rtheta_chain,
    _rtheta_tab,
)
# numba caches each compiled function by its own source file only: the kernels below keep the
# copies of the _kernel_pol functions imported here (and of what those call) that they were
# compiled with until this file itself changes, so a change there must come with a change
# here.  Last: the end terms of the anisotropy in j_V (_kernel_pol._cell_emis_pol).
from ._kernel_pol import (
    _cell_emis_pol,
    _pol_accumulate,
    _pol_cell,
    _pol_ray_init,
    _pol_ray_store,
)

# ----------------------------------------------------------------------------- ray geometry
# indices into the packed geometry array G
(G_M, G_A, G_THETA_O, G_R0, G_EPS, G_WCAP, N_GEOM) = range(7)


def pack_geometry(M, a, theta_o, r0, eps=geo.EPS_STEP, wcap=geo.WCAP_STEP):
    """
    The ray geometry as the kernels expect it: mass M and spin a of the metric the rays
    follow, the observer's polar angle theta_o (radians), the start radius r0 of every
    ray, and the step parameters of the tracer (see _geodesics.trace).
    """
    G = np.zeros(N_GEOM)
    G[G_M] = M
    G[G_A] = a
    G[G_THETA_O] = theta_o
    G[G_R0] = r0
    G[G_EPS] = eps
    G[G_WCAP] = wcap
    return G


def trajectory_buffers(nmax=geo.MAX_STEPS):
    """One trajectory buffer of nmax points for every thread numba's pool can have."""
    return np.empty((numba.config.NUMBA_NUM_THREADS, 6, nmax))


def no_field_table():
    """
    Placeholder field-table arguments (Tup, Tlo, logr0, invdlogr, nr, invdu, nu_) for a
    driver called with use_table=False, which never reads them.
    """
    dummy = np.zeros((2, 2, N_TAB))
    return (dummy, dummy, 0.0, 1.0, 2, 1.0, 2)


@njit(cache=True, inline="always")
def _trace_pixel(pix, x_im_f, y_im_f, G, traj):
    """Trace the ray of pixel pix into traj; returns (alpha, beta, points, status, lambda)."""
    alpha = -y_im_f[pix]
    beta = -x_im_f[pix]
    n, status, lam, eta = geo.trace(
        alpha, beta, G[G_M], G[G_A], G[G_THETA_O], G[G_R0], G[G_EPS], G[G_WCAP], traj
    )
    return alpha, beta, n, status, lam


@njit(cache=True, inline="always")
def _local_frame(traj, n, kk, target, lam, G):
    """
    The ray's point at affine parameter target (targets in increasing order; kk is the
    step search's state): returns (kk, r, cos theta, 1 - |cos theta|, sign(cos theta),
    the photon's ZAMO direction n0, n1, n2, its ZAMO energy kt, the lapse, omega, and the
    metric factors sqrt(g_rr), sqrt(g_thth), sqrt(g_phph)).
    """
    kk, w, u, pw, pu = geo.locate(traj, n, kk, target, G[G_A])
    n0, n1, n2, kt, al, om, r, sq_rr, sq_tt, sq_pp = geo.zamo_frame(w, u, pw, pu, lam, G[G_M], G[G_A])
    omc = 1.0 - abs(u)
    sgn = 1.0 if u >= 0.0 else -1.0
    return kk, r, u, omc, sgn, n0, n1, n2, kt, al, om, sq_rr, sq_tt, sq_pp


@njit(cache=True, inline="always")
def _cell_basis(Ka1, Ka2, n0, n1, n2, kt, al, om, r, u, a_geo, sq_rr, sq_tt, sq_pp):
    """
    The observer's Stokes basis (a, b) at a cell: a is the vector perpendicular to n
    with the Walker-Penrose constant Ka1 + i Ka2 of the observer's +x_im, normalized,
    and b = n x a, so that (a, b, n) is exactly orthonormal and right handed, as the
    flat model's (+x_im, -y_im, n) is.
    """
    f0, f1, f2 = geo.carried_vector(Ka1, Ka2, n0, n1, n2, kt, al, om, r, u, a_geo, sq_rr, sq_tt, sq_pp)
    m = math.sqrt(f0 * f0 + f1 * f1 + f2 * f2)
    a0 = f0 / m
    a1 = f1 / m
    a2 = f2 / m
    return a0, a1, a2, n1 * a2 - n2 * a1, n2 * a0 - n0 * a2, n0 * a1 - n1 * a0


# ----------------------------------------------------------------------------- jet segments
@njit(cache=True, inline="always")
def _kerr_inside(w, u, r_min, nu, inv_rHnu, omc_fp_cut):
    """
    _kernel._flat_inside for a point given by w = 1/r and u = cos(theta): inside either
    lobe's edge field line, r^nu (1 -/+ cos theta) <= rH^nu, outside the horizon buffer,
    and not cut out.
    """
    r = 1.0 / w
    rnu = r**nu
    w_up = rnu * (1.0 - u) * inv_rHnu
    w_lo = rnu * (1.0 + u) * inv_rHnu
    inj = ((w_up <= 1.0) or (w_lo <= 1.0)) and (r > r_min)
    if inj and omc_fp_cut > 0.0:
        cos_fp = (1.0 - w_up) if w_up < 1.0 else (w_lo - 1.0)
        if (1.0 - abs(cos_fp)) < omc_fp_cut:
            inj = False
    return inj


@njit(cache=True, inline="always")
def _inside_at(traj, n, target, a_geo, r_min, nu, inv_rHnu, omc_fp_cut):
    """Whether the ray is inside the jet at an arbitrary affine parameter target."""
    if target < traj[4, 0] or target > traj[4, n - 1]:
        return False
    w, u, pw, pu = geo.state_at(traj, n, target, a_geo)
    return _kerr_inside(w, u, r_min, nu, inv_rHnu, omc_fp_cut)


@njit(cache=True, parallel=True)
def jet_segments_kerr(x_im_f, y_im_f, z_im_1D, P, G, max_int, bufs):
    """
    _kernel.jet_segments along the geodesics: for every pixel, the segments [s_in, s_out]
    of its ray inside the jet, in the affine offset s from the ray's closest approach to
    the jet axis, found by the same scan of every cell edge and center and the same
    bisection.  A ray ends where it is captured by the hole or leaves again beyond its
    start radius; the scan treats offsets beyond its ends as outside the jet.

    Returns (seg_in, seg_out, c_lo, c_hi, nint, ncell, lamJ, status, rmax): the first
    six as _kernel.jet_segments returns them, lamJ the affine parameter of the closest
    approach (from the start of the ray), status the ray's fate (geo.CAPTURED,
    geo.ESCAPED or geo.OUT_OF_STEPS) and rmax the largest radius inside the jet along
    the ray (0 if none), which sizes the field table: r falls to a single minimum along a
    ray and rises again, so over a segment it peaks at an end, which the scan evaluates.
    """
    rH = P[P_RH]
    nu = P[P_NU]
    cut = P[P_JET_CUTOUT]
    inv_rHnu = P[P_INV_RHNU]
    r_min = rH * (1.0 + 1.0e-2)
    omc_fp_cut = cut * cut if cut > 0.0 else -1.0
    a_geo = G[G_A]
    Npix = x_im_f.shape[0]
    Nz = z_im_1D.shape[0]
    nscan = 2 * Nz - 1
    seg_in = np.zeros((Npix, max_int))
    seg_out = np.zeros((Npix, max_int))
    c_lo = np.full((Npix, max_int), -1, dtype=np.int32)
    c_hi = np.full((Npix, max_int), -1, dtype=np.int32)
    nint = np.zeros(Npix, dtype=np.int32)
    ncell = np.zeros(Npix, dtype=np.int64)
    lamJ = np.zeros(Npix)
    status_out = np.zeros(Npix, dtype=np.int32)
    rmax = np.zeros(Npix)
    for pix in prange(Npix):
        traj = bufs[numba.get_thread_id()]
        alpha, beta, n, status, lam = _trace_pixel(pix, x_im_f, y_im_f, G, traj)
        lJ = geo.closest_approach_to_axis(traj, n, a_geo)
        lamJ[pix] = lJ
        status_out[pix] = status
        la_first = traj[4, 0]
        la_last = traj[4, n - 1]
        # The scan points come in increasing order, many to a step of the ray.  Per step,
        # the interpolation data are fetched once, and so is a bound below which no point
        # of the step can be inside the jet: inside means 1 - |cos theta| <= (rH w)^nu,
        # which is largest where w is, so a point with 1 - |cos theta| above (rH w_max)^nu
        # is outside, with no pow.  Within a step, Newton starts from the previous point's
        # solution, carried forward with the slope found there.
        kk = 0
        k_data = -1
        l0 = 0.0
        l1 = 0.0
        s0h = 0.0
        s1h = 0.0
        t = 0.0
        dla = 0.0
        target_prev = 0.0
        omc_out = 0.0
        k = 0
        cnt = 0
        rm = 0.0
        prev_in = False
        prev_s = z_im_1D[0]
        s_in = prev_s
        cell_in = 0
        for j in range(nscan):
            # scan point j: cell edge j/2 for even j, the center of cell (j-1)/2 for odd j
            if j % 2 == 0:
                s = z_im_1D[j // 2]
            else:
                s = 0.5 * (z_im_1D[j // 2] + z_im_1D[j // 2 + 1])
            target = lJ + s
            now_in = False
            if la_first <= target <= la_last:      # on the ray, not beyond its start or end
                while kk < n - 2 and traj[4, kk + 1] < target:
                    kk += 1
                if kk != k_data:
                    k_data = kk
                    l0, l1, s0h, s1h = geo.step_affine(traj, kk, a_geo)
                    t = geo.linear_t(target, l0, l1)
                    omc_out = (rH * geo.w_max_in_step(traj, kk)) ** nu * (1.0 + 1.0e-9)
                elif dla > 0.0:
                    t += (target - target_prev) / dla
                    if t > 1.0:
                        t = 1.0
                t, dla = geo.newton_t(target, l0, l1, s0h, s1h, t)
                target_prev = target
                w, u = geo.wu_in_step(traj, kk, t)
                if 1.0 - abs(u) <= omc_out:
                    now_in = _kerr_inside(w, u, r_min, nu, inv_rHnu, omc_fp_cut)
                    if now_in:
                        rm = max(rm, 1.0 / w)
            if j == 0:
                prev_in = now_in
                prev_s = s
                continue
            if now_in != prev_in:
                lo = prev_s
                hi = s
                for _ in range(N_BISECT):
                    mid = 0.5 * (lo + hi)
                    if mid <= lo or mid >= hi:
                        break
                    if _inside_at(traj, n, lJ + mid, a_geo, r_min, nu, inv_rHnu, omc_fp_cut) == prev_in:
                        lo = mid
                    else:
                        hi = mid
                crossing = 0.5 * (lo + hi)
                target = lJ + crossing
                if la_first <= target <= la_last:
                    rm = max(rm, 1.0 / geo.state_at(traj, n, target, a_geo)[0])
                cell = (j - 1) // 2  # the cell that holds [scan point j-1, scan point j]
                if now_in:
                    s_in = crossing
                    cell_in = cell
                else:
                    if k < max_int:
                        seg_in[pix, k] = s_in
                        seg_out[pix, k] = crossing
                        c_lo[pix, k] = cell_in
                        c_hi[pix, k] = cell
                    cnt += cell - cell_in + 1
                    k += 1
            prev_in = now_in
            prev_s = s
        if prev_in:
            if k < max_int:
                seg_in[pix, k] = s_in
                seg_out[pix, k] = z_im_1D[Nz - 1]
                c_lo[pix, k] = cell_in
                c_hi[pix, k] = Nz - 2
            cnt += (Nz - 2) - cell_in + 1
            k += 1
        nint[pix] = k
        ncell[pix] = cnt
        rmax[pix] = rm
    return seg_in, seg_out, c_lo, c_hi, nint, ncell, lamJ, status_out, rmax


# ----------------------------------------------------------------------------- per-cell physics
@njit(cache=True)
def _ray_part_kerr(vr, vt, vp, gamma, alphalapse, Bpr, Bpt, Bpp, gamma_c, Kj, Ka, P, n0, n1, n2, fd):
    """
    _kernel._ray_part in the ZAMO orthonormal basis (r, theta, phi), in which
    _rtheta_chain gives the velocity and the fluid-frame field: n is the photon's ZAMO
    direction and fd = 1 - omega lambda the frame-dragging factor of its ZAMO energy.
    Returns (g, nu_p, gamma_c, C_j, C_a) as consumed by _cell_emis.
    """
    eta = P[P_ETA]
    p_eta = P[P_PETA]
    phi_norm = P[P_PHI_NORM]

    b2 = 1.0 - 1.0 / (gamma * gamma)
    beta = math.sqrt(b2 if b2 > 0.0 else 0.0)
    vmag = math.sqrt(vr * vr + vt * vt + vp * vp)
    if vmag > 1.0e-8:
        vh0 = vr / vmag
        vh1 = vt / vmag
        vh2 = vp / vmag
    else:
        vh0 = 0.0
        vh1 = 0.0
        vh2 = 0.0

    # redshift factor, nu_observed / nu_fluid = 1 / (-k.u)
    k_par = (vh0 * n0) + (vh1 * n1) + (vh2 * n2)
    one_m_betak = 1.0 - (beta * k_par)
    g = alphalapse / (gamma * fd * one_m_betak)
    if not math.isfinite(g):
        g = 1.0

    # photon direction in the comoving frame
    gamma_one_m_betak = gamma * one_m_betak
    k_par_prime = (k_par - beta) / one_m_betak
    k0 = ((n0 - k_par * vh0) / gamma_one_m_betak) + k_par_prime * vh0
    k1 = ((n1 - k_par * vh1) / gamma_one_m_betak) + k_par_prime * vh1
    k2 = ((n2 - k_par * vh2) / gamma_one_m_betak) + k_par_prime * vh2
    km = math.sqrt(k0 * k0 + k1 * k1 + k2 * k2)
    k0 /= km
    k1 /= km
    k2 /= km

    # pitch angle
    Bpm = math.sqrt(Bpr * Bpr + Bpt * Bpt + Bpp * Bpp)
    costhetaB = ((k0 * Bpr) + (k1 * Bpt) + (k2 * Bpp)) / Bpm
    sinthetaB = math.sqrt(1.0 - (costhetaB * costhetaB))  # NaN if cos^2 > 1, as in _ray_part

    anisotropy_term = 1.0 + ((eta - 1.0) * (costhetaB * costhetaB))
    if p_eta == 2.0:
        aniso_fac = (1.0 / anisotropy_term) / phi_norm
    else:
        aniso_fac = (anisotropy_term ** (-p_eta / 2.0)) / phi_norm
    nup = (4.1987e-3) * Bpm * sinthetaB

    Cj = Kj * nup * aniso_fac
    Ca = (Ka / nup) * aniso_fac
    return g, nup, gamma_c, Cj, Ca


@njit(cache=True)
def _ray_part_pol_kerr(vr, vt, vp, gamma, alphalapse, Bpr, Bpt, Bpp, gamma_c, Kj, Ka, P,
                       n0, n1, n2, fd, a0, a1, a2, b0, b1, b2):
    """
    _kernel_pol._ray_part_pol in the ZAMO orthonormal basis: as _ray_part_kerr, plus
    the polarized state, with (a, b) the observer's Stokes basis at the cell (see
    _cell_basis) in place of the flat model's fixed sky vectors.  Returns the nine
    quantities of _ray_part_pol.
    """
    eta = P[P_ETA]
    p_eta = P[P_PETA]
    phi_norm = P[P_PHI_NORM]

    b2_ = 1.0 - 1.0 / (gamma * gamma)
    beta = math.sqrt(b2_ if b2_ > 0.0 else 0.0)
    vmag = math.sqrt(vr * vr + vt * vt + vp * vp)
    if vmag > 1.0e-8:
        vh0 = vr / vmag
        vh1 = vt / vmag
        vh2 = vp / vmag
    else:
        vh0 = 0.0
        vh1 = 0.0
        vh2 = 0.0

    # redshift factor
    k_par = (vh0 * n0) + (vh1 * n1) + (vh2 * n2)
    one_m_betak = 1.0 - (beta * k_par)
    g = alphalapse / (gamma * fd * one_m_betak)
    if not math.isfinite(g):
        g = 1.0

    # photon direction in the comoving frame
    gamma_one_m_betak = gamma * one_m_betak
    k_par_prime = (k_par - beta) / one_m_betak
    k0 = ((n0 - k_par * vh0) / gamma_one_m_betak) + k_par_prime * vh0
    k1 = ((n1 - k_par * vh1) / gamma_one_m_betak) + k_par_prime * vh1
    k2 = ((n2 - k_par * vh2) / gamma_one_m_betak) + k_par_prime * vh2
    km = math.sqrt(k0 * k0 + k1 * k1 + k2 * k2)
    k0 /= km
    k1 /= km
    k2 /= km

    # pitch angle
    Bpm = math.sqrt(Bpr * Bpr + Bpt * Bpt + Bpp * Bpp)
    costhetaB = ((k0 * Bpr) + (k1 * Bpt) + (k2 * Bpp)) / Bpm
    sinthetaB = math.sqrt(1.0 - (costhetaB * costhetaB))

    anisotropy_term = 1.0 + ((eta - 1.0) * (costhetaB * costhetaB))
    if p_eta == 2.0:
        aniso_fac = (1.0 / anisotropy_term) / phi_norm
    else:
        aniso_fac = (anisotropy_term ** (-p_eta / 2.0)) / phi_norm
    nup = (4.1987e-3) * Bpm * sinthetaB

    Cj = Kj * nup * aniso_fac
    Ca = (Ka / nup) * aniso_fac

    # the Stokes basis boosted into the fluid frame and projected perpendicular to the
    # aberrated photon direction; see _ray_part_pol
    gm1 = gamma - 1.0
    gb = gamma * beta
    sa = (vh0 * a0) + (vh1 * a1) + (vh2 * a2)
    sb = (vh0 * b0) + (vh1 * b1) + (vh2 * b2)
    ax = a0 + gm1 * sa * vh0 + gb * sa * k0
    ay = a1 + gm1 * sa * vh1 + gb * sa * k1
    az = a2 + gm1 * sa * vh2 + gb * sa * k2
    bx = b0 + gm1 * sb * vh0 + gb * sb * k0
    by = b1 + gm1 * sb * vh1 + gb * sb * k1
    bz = b2 + gm1 * sb * vh2 + gb * sb * k2

    # Dexter (2016) Eqs. (44), (45)
    ca = (ax * Bpr + ay * Bpt + az * Bpp) / Bpm
    cb = (bx * Bpr + by * Bpt + bz * Bpp) / Bpm
    den = ca * ca + cb * cb
    if den > 1.0e-300:
        cos2chi = (cb * cb - ca * ca) / den
        sin2chi = -2.0 * ca * cb / den
    else:
        cos2chi = -1.0
        sin2chi = 0.0

    return g, nup, gamma_c, Cj, Ca, costhetaB, Ka / nup, cos2chi, sin2chi


# ----------------------------------------------------------------------------- drivers
#
# As in _kernel.py, every driver walks a ray's jet segments front to back and, within
# each, the grid cells the segment overlaps, evaluating each cell on its overlap with the
# jet; the cell's point is the ray's point at affine parameter lambda_J + (overlap
# midpoint), and its path length the overlap's affine length.  Rays without jet
# segments are not traced.  Each driver takes both the exact chain's tables and the
# field table, and use_table selects between them (table_arguments supplies
# placeholders for the one not in use: no_field_table).


@njit(cache=True, parallel=True)
def rt_kernel_kerr(
    frequency, order, x_im_f, y_im_f, lamJ, z_im_1D, seg_in, seg_out, c_lo, c_hi, nint,
    P, G, bufs, use_table, xs_tab, rs_tab, ts_tab, as_tab, Tup, Tlo, logr0, invdlogr, nr,
    invdu, nu_, T0, T1, T2, T3, T4, T5, TK, tau_stop, I_out,
):
    """
    rt_kernel along the geodesics: state and coefficients evaluated on the fly, with
    the (r,theta)-only physics from the field table if use_table.
    """
    atten_stop = math.exp(-tau_stop) if tau_stop > 0.0 else -1.0
    rg = P[P_RG]
    Npix = order.shape[0]
    for q in prange(Npix):
        pix = order[q]
        I_acc = 0.0
        atten = 1.0
        if nint[pix] > 0:
            traj = bufs[numba.get_thread_id()]
            alpha, beta, n, status, lam = _trace_pixel(pix, x_im_f, y_im_f, G, traj)
            lJ = lamJ[pix]
            kk = 0
            done = False
            for k in range(nint[pix]):
                if done:
                    break
                s_in = seg_in[pix, k]
                s_out = seg_out[pix, k]
                for i in range(c_lo[pix, k], c_hi[pix, k] + 1):
                    zi, ds = _cell_overlap(z_im_1D, i, s_in, s_out, rg)
                    kk, r, u, omc, sgn, n0, n1, n2, kt, al, om, sq_rr, sq_tt, sq_pp = _local_frame(
                        traj, n, kk, lJ + zi, lam, G)
                    if use_table:
                        vr, vt, vp, gamma, alpha_f, Bpr, Bpt, Bpp, Bpm, gc, Kj, Ka = _rtheta_tab(
                            r, omc, sgn, P, Tup, Tlo, logr0, invdlogr, nr, invdu, nu_)
                    else:
                        vr, vt, vp, gamma, alpha_f, Bpr, Bpt, Bpp, Bpm, gc, Kj, Ka = _rtheta_chain(
                            r, omc, sgn, P, xs_tab, rs_tab, ts_tab, as_tab)
                    g, nup, gamma_c, Cj, Ca = _ray_part_kerr(
                        vr, vt, vp, gamma, alpha_f, Bpr, Bpt, Bpp, gc, Kj, Ka, P, n0, n1, n2,
                        1.0 - om * lam)
                    jI, alphaI = _cell_emis(frequency, g, nup, gamma_c, Cj, Ca, P, T0, T1, T2, T3, T4, T5, TK)
                    I_acc, atten = _rt_step(I_acc, atten, jI, alphaI, g, ds)
                    if atten <= atten_stop:
                        done = True
                        break
        I_out[pix] = I_acc


@njit(cache=True, parallel=True)
def precompute_state_kerr(
    order, offsets, x_im_f, y_im_f, lamJ, z_im_1D, seg_in, seg_out, c_lo, c_hi, nint,
    P, G, bufs, use_table, xs_tab, rs_tab, ts_tab, as_tab, Tup, Tlo, logr0, invdlogr, nr,
    invdu, nu_, st_g, st_nup, st_gc, st_Cj, st_Ca, st_ds,
):
    """
    Store the frequency-independent state and path length of every jet cell, ray by
    ray, for rt_from_state; with the (r,theta)-only physics from the field table if
    use_table.
    """
    rg = P[P_RG]
    Npix = order.shape[0]
    for q in prange(Npix):
        pix = order[q]
        o = offsets[q]
        if nint[pix] > 0:
            traj = bufs[numba.get_thread_id()]
            alpha, beta, n, status, lam = _trace_pixel(pix, x_im_f, y_im_f, G, traj)
            lJ = lamJ[pix]
            kk = 0
            for k in range(nint[pix]):
                s_in = seg_in[pix, k]
                s_out = seg_out[pix, k]
                for i in range(c_lo[pix, k], c_hi[pix, k] + 1):
                    zi, ds = _cell_overlap(z_im_1D, i, s_in, s_out, rg)
                    kk, r, u, omc, sgn, n0, n1, n2, kt, al, om, sq_rr, sq_tt, sq_pp = _local_frame(
                        traj, n, kk, lJ + zi, lam, G)
                    if use_table:
                        vr, vt, vp, gamma, alpha_f, Bpr, Bpt, Bpp, Bpm, gc, Kj, Ka = _rtheta_tab(
                            r, omc, sgn, P, Tup, Tlo, logr0, invdlogr, nr, invdu, nu_)
                    else:
                        vr, vt, vp, gamma, alpha_f, Bpr, Bpt, Bpp, Bpm, gc, Kj, Ka = _rtheta_chain(
                            r, omc, sgn, P, xs_tab, rs_tab, ts_tab, as_tab)
                    g, nup, gamma_c, Cj, Ca = _ray_part_kerr(
                        vr, vt, vp, gamma, alpha_f, Bpr, Bpt, Bpp, gc, Kj, Ka, P, n0, n1, n2,
                        1.0 - om * lam)
                    st_g[o] = g
                    st_nup[o] = nup
                    st_gc[o] = gamma_c
                    st_Cj[o] = Cj
                    st_Ca[o] = Ca
                    st_ds[o] = ds
                    o += 1


@njit(cache=True, parallel=True)
def rt_kernel_pol_kerr(
    frequency, order, x_im_f, y_im_f, lamJ, z_im_1D, seg_in, seg_out, c_lo, c_hi, nint,
    P, G, bufs, use_table, xs_tab, rs_tab, ts_tab, as_tab, Tup, Tlo, logr0, invdlogr, nr,
    invdu, nu_, TAB, RTAB, tau_stop, out,
):
    """
    rt_kernel_pol along the geodesics: the full polarized computation for one frequency,
    with the (r,theta)-only physics from the field table if use_table.
    """
    rg = P[P_RG]
    a_geo = G[G_A]
    Npix = order.shape[0]
    for q in prange(Npix):
        pix = order[q]
        Itot = np.zeros(4)
        Ocum = np.zeros((4, 4))
        O = np.zeros((4, 4))
        cv = np.zeros(4)
        tmp = np.zeros((4, 4))
        _pol_ray_init(Itot, Ocum)
        if nint[pix] > 0:
            traj = bufs[numba.get_thread_id()]
            alpha, beta, n, status, lam = _trace_pixel(pix, x_im_f, y_im_f, G, traj)
            Ka1, Ka2, Kb1, Kb2 = geo.observer_kappas(traj, alpha, beta, G[G_THETA_O], lam, G[G_M], a_geo)
            lJ = lamJ[pix]
            kk = 0
            tau = 0.0
            done = False
            for k in range(nint[pix]):
                if done:
                    break
                s_in = seg_in[pix, k]
                s_out = seg_out[pix, k]
                for i in range(c_lo[pix, k], c_hi[pix, k] + 1):
                    zi, ds = _cell_overlap(z_im_1D, i, s_in, s_out, rg)
                    kk, r, u, omc, sgn, n0, n1, n2, kt, al, om, sq_rr, sq_tt, sq_pp = _local_frame(
                        traj, n, kk, lJ + zi, lam, G)
                    if use_table:
                        vr, vt, vp, gamma, alpha_f, Bpr, Bpt, Bpp, Bpm, gc, Kj, Ka = _rtheta_tab(
                            r, omc, sgn, P, Tup, Tlo, logr0, invdlogr, nr, invdu, nu_)
                    else:
                        vr, vt, vp, gamma, alpha_f, Bpr, Bpt, Bpp, Bpm, gc, Kj, Ka = _rtheta_chain(
                            r, omc, sgn, P, xs_tab, rs_tab, ts_tab, as_tab)
                    a0, a1, a2, b0, b1, b2 = _cell_basis(
                        Ka1, Ka2, n0, n1, n2, kt, al, om, r, u, a_geo, sq_rr, sq_tt, sq_pp)
                    g, nup, gc2, Cj, Ca, cth, Kap, c2, s2 = _ray_part_pol_kerr(
                        vr, vt, vp, gamma, alpha_f, Bpr, Bpt, Bpp, gc, Kj, Ka, P, n0, n1, n2,
                        1.0 - om * lam, a0, a1, a2, b0, b1, b2)
                    jI, jQ, jV, aI, aQ, aV, rQ, rV = _cell_emis_pol(
                        frequency, g, nup, gc2, Cj, Ca, cth, Kap, P, TAB, RTAB)
                    tau += _pol_cell(jI, jQ, jV, aI, aQ, aV, rQ, rV, c2, s2, g, ds, O, cv)
                    _pol_accumulate(Itot, Ocum, O, cv, tmp)
                    if tau_stop > 0.0 and tau >= tau_stop:
                        done = True
                        break
        _pol_ray_store(out, pix, Itot)


@njit(cache=True, parallel=True)
def precompute_state_pol_kerr(
    order, offsets, x_im_f, y_im_f, lamJ, z_im_1D, seg_in, seg_out, c_lo, c_hi, nint,
    P, G, bufs, use_table, xs_tab, rs_tab, ts_tab, as_tab, Tup, Tlo, logr0, invdlogr, nr,
    invdu, nu_, st, st_ds,
):
    """
    Store the frequency-independent polarized state and path length of every jet cell,
    for rt_from_state_pol; with the (r,theta)-only physics from the field table if
    use_table.
    """
    rg = P[P_RG]
    a_geo = G[G_A]
    Npix = order.shape[0]
    for q in prange(Npix):
        pix = order[q]
        o = offsets[q]
        if nint[pix] > 0:
            traj = bufs[numba.get_thread_id()]
            alpha, beta, n, status, lam = _trace_pixel(pix, x_im_f, y_im_f, G, traj)
            Ka1, Ka2, Kb1, Kb2 = geo.observer_kappas(traj, alpha, beta, G[G_THETA_O], lam, G[G_M], a_geo)
            lJ = lamJ[pix]
            kk = 0
            for k in range(nint[pix]):
                s_in = seg_in[pix, k]
                s_out = seg_out[pix, k]
                for i in range(c_lo[pix, k], c_hi[pix, k] + 1):
                    zi, ds = _cell_overlap(z_im_1D, i, s_in, s_out, rg)
                    kk, r, u, omc, sgn, n0, n1, n2, kt, al, om, sq_rr, sq_tt, sq_pp = _local_frame(
                        traj, n, kk, lJ + zi, lam, G)
                    if use_table:
                        vr, vt, vp, gamma, alpha_f, Bpr, Bpt, Bpp, Bpm, gc, Kj, Ka = _rtheta_tab(
                            r, omc, sgn, P, Tup, Tlo, logr0, invdlogr, nr, invdu, nu_)
                    else:
                        vr, vt, vp, gamma, alpha_f, Bpr, Bpt, Bpp, Bpm, gc, Kj, Ka = _rtheta_chain(
                            r, omc, sgn, P, xs_tab, rs_tab, ts_tab, as_tab)
                    a0, a1, a2, b0, b1, b2 = _cell_basis(
                        Ka1, Ka2, n0, n1, n2, kt, al, om, r, u, a_geo, sq_rr, sq_tt, sq_pp)
                    g, nup, gc2, Cj, Ca, cth, Kap, c2, s2 = _ray_part_pol_kerr(
                        vr, vt, vp, gamma, alpha_f, Bpr, Bpt, Bpp, gc, Kj, Ka, P, n0, n1, n2,
                        1.0 - om * lam, a0, a1, a2, b0, b1, b2)
                    st[0, o] = g
                    st[1, o] = nup
                    st[2, o] = gc2
                    st[3, o] = Cj
                    st[4, o] = Ca
                    st[5, o] = cth
                    st[6, o] = Kap
                    st[7, o] = c2
                    st[8, o] = s2
                    st_ds[o] = ds
                    o += 1


# ----------------------------------------------------------------------------- cell geometry
# Per-cell geometry for the numpy reference path (JetModel(geometry="kerr",
# backend="numpy")), which takes the rays from the compiled tracer and does the physics
# and the transfer in numpy.
(
    KG_R,
    KG_U,
    KG_N0,
    KG_N1,
    KG_N2,
    KG_FD,
    KG_A0,
    KG_A1,
    KG_A2,
    KG_B0,
    KG_B1,
    KG_B2,
    KG_DS,
    N_KG,
) = range(14)


@njit(cache=True, parallel=True)
def kerr_cell_geometry(
    order, offsets, x_im_f, y_im_f, lamJ, z_im_1D, seg_in, seg_out, c_lo, c_hi, nint, P, G,
    bufs, out,
):
    """
    For every jet cell, ray by ray in the order of the stored state: r, cos(theta), the
    photon's ZAMO direction n, the frame-dragging factor 1 - omega lambda, the
    observer's Stokes basis (a, b) at the cell and the path length (cm), in the rows of
    out (shape (N_KG, cells)) indexed by the KG_* constants.
    """
    rg = P[P_RG]
    a_geo = G[G_A]
    Npix = order.shape[0]
    for q in prange(Npix):
        pix = order[q]
        o = offsets[q]
        if nint[pix] > 0:
            traj = bufs[numba.get_thread_id()]
            alpha, beta, n, status, lam = _trace_pixel(pix, x_im_f, y_im_f, G, traj)
            Ka1, Ka2, Kb1, Kb2 = geo.observer_kappas(traj, alpha, beta, G[G_THETA_O], lam, G[G_M], a_geo)
            lJ = lamJ[pix]
            kk = 0
            for k in range(nint[pix]):
                s_in = seg_in[pix, k]
                s_out = seg_out[pix, k]
                for i in range(c_lo[pix, k], c_hi[pix, k] + 1):
                    zi, ds = _cell_overlap(z_im_1D, i, s_in, s_out, rg)
                    kk, r, u, omc, sgn, n0, n1, n2, kt, al, om, sq_rr, sq_tt, sq_pp = _local_frame(
                        traj, n, kk, lJ + zi, lam, G)
                    a0, a1, a2, b0, b1, b2 = _cell_basis(
                        Ka1, Ka2, n0, n1, n2, kt, al, om, r, u, a_geo, sq_rr, sq_tt, sq_pp)
                    out[KG_R, o] = r
                    out[KG_U, o] = u
                    out[KG_N0, o] = n0
                    out[KG_N1, o] = n1
                    out[KG_N2, o] = n2
                    out[KG_FD, o] = 1.0 - om * lam
                    out[KG_A0, o] = a0
                    out[KG_A1, o] = a1
                    out[KG_A2, o] = a2
                    out[KG_B0, o] = b0
                    out[KG_B1, o] = b1
                    out[KG_B2, o] = b2
                    out[KG_DS, o] = ds
                    o += 1
