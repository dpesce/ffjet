###################################################
# imports and etc.

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time

# optional progress bar import
try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

# suppress some numpy warnings
np.seterr(divide="ignore", invalid="ignore")

###################################################
# physical constants (cgs)

m_e = 9.10938e-28
q_e = 4.8032045e-10
c = 2.99792458e10
G = 6.674e-8
sigma_T = 6.65246e-25

###################################################
# helper functions


# dot product
def _dot(a1, a2, a3, b1, b2, b3):
    return a1 * b1 + a2 * b2 + a3 * b3


# cross product
def _cross(ax, ay, az, bx, by, bz):
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


# function that takes in spherical coordinate
# components and outputs cartesian ones
def rtp_to_xyz(Vr, Vt, Vp, x, y, z, r, R):
    Vx = ((x * Vr) / r) + ((x * z * Vt) / (r * R)) - ((y * Vp) / R)
    Vy = ((y * Vr) / r) + ((y * z * Vt) / (r * R)) + ((x * Vp) / R)
    Vz = ((z * Vr) / r) - ((R * Vt) / r)
    return Vx, Vy, Vz


# function that takes in cartesian coordinate
# components and outputs spherical ones
def xyz_to_rtp(Vx, Vy, Vz, x, y, z, r, R):
    Vr = ((x * Vx) / r) + ((y * Vy) / r) + ((z * Vz) / r)
    Vt = ((x * z * Vx) / (r * R)) + ((y * z * Vy) / (r * R)) - ((R * Vz) / r)
    Vp = -((y * Vx) / R) + ((x * Vy) / R)
    return Vr, Vt, Vp


def _log_interp_with_extrap(x, lx, ly):
    # lx, ly are arrays of log10(x), log10(G)
    L = np.log10(np.maximum(x, 1e-300))
    y = np.interp(L, lx, ly, left=np.nan, right=np.nan)

    # left extrapolation: line through first two points
    left = L < lx[0]
    if np.any(left):
        mL = (ly[1] - ly[0]) / (lx[1] - lx[0])
        y[left] = ly[0] + mL * (L[left] - lx[0])

    # right extrapolation
    right = L > lx[-1]
    if np.any(right):
        mR = (ly[-1] - ly[-2]) / (lx[-1] - lx[-2])
        y[right] = ly[-1] + mR * (L[right] - lx[-1])

    return y


def psiBZpower(r, theta, p):
    return (r**p) * (1 - np.abs(np.cos(theta)))


# progress bar
def _progress(iterable, enabled: bool = False, **kwargs):
    if enabled and (_tqdm is not None):
        return _tqdm(iterable, **kwargs)
    return iterable


###################################################
# redefining some kgeo functions


# lightweight copy of Bfield class
class Bfield(object):
    def __init__(self, p=2.5):
        self.fieldtype = "power"
        self.fieldframe = "lab"
        self.pval = p

    def bfield_lab(self, a, r, th=np.pi / 2):
        (B1, B2, B3, omega) = Bfield_power(a, r, th, self.pval)
        b_components = (B1, B2, B3)
        return b_components

    def efield_lab(self, a, r, th=np.pi / 2):
        (B1, B2, B3, omega) = Bfield_power(a, r, th, self.pval)
        a2 = a**2
        r2 = r**2
        cth2 = np.cos(th) ** 2
        sth2 = np.sin(th) ** 2
        Delta = r2 - 2 * r + a2
        Sigma = r2 + a2 * cth2
        Pi = (r2 + a2) ** 2 - a2 * Delta * sth2
        omegaz = 2 * a * r / Pi
        E1 = (omega - omegaz) * Pi * np.sin(th) * B2 / Sigma
        E2 = -(omega - omegaz) * Pi * np.sin(th) * B1 / (Sigma * Delta)
        E3 = np.zeros_like(E2) if hasattr(E2, "__len__") else 0
        e_components = (E1, E2, E3)
        return e_components

    def omega_field(self, a, r, th=np.pi / 2):
        """fieldline angular speed"""
        (B1, B2, B3, omega) = Bfield_power(a, r, th, self.pval)
        return omega


def Bfield_power(a, r, th, p):
    """stream function of the form psi=r^p(1-costheta) with same Bphi as paraboloid"""

    if not (isinstance(a, float) and (0 <= np.abs(a) < 1)):
        raise Exception("|a| should be a float in range [0,1)")

    a2 = a**2
    r2 = r**2
    sth = np.sin(th)
    cth = np.cos(th)
    abscth = np.abs(cth)
    cth2 = cth**2
    sth2 = sth**2
    Delta = r2 - 2 * r + a2
    Sigma = r2 + a2 * cth2
    gdet = sth * Sigma

    # vector potential
    psi = r**p * (1 - abscth)
    dpsidtheta = np.sign(cth) * sth * (r**p)
    dpsidr = p * psi / r

    OmegaBZ = omega_BZpower(th, psi, a, p)

    # current
    if p > 0:
        Icur = -4 * np.pi * psi * OmegaBZ * np.sign(cth)
    else:
        Icur = -2 * np.pi * psi * (2 - psi) * OmegaBZ * np.sign(cth)

    # field components
    Br = dpsidtheta / gdet
    Bth = -dpsidr / gdet
    Bph = Icur / (2 * np.pi * Delta * sth2)

    return (Br, Bth, Bph, OmegaBZ)


def omega_BZpower(th, psi, a, p):
    if p == 0:
        return a / 8
    rp = 1 + np.sqrt(1 - a**2)
    cthhorizon = 1 - psi * rp ** (-p)
    denomfac = 8 / (1 + cthhorizon)
    return a / (4 + denomfac)


def Nderiv(r, theta, a, Omegaf, M, bf_here):
    cth = np.cos(theta)
    sth = np.sin(theta)
    Sigma = (r * r) + (a * a) * (cth * cth)
    dNdr = (
        a**4 * (Omegaf * Omegaf) * r * cth**4 * (sth * sth)
        + (r * r)
        * (
            -M
            + Omegaf
            * (sth * sth)
            * (2 * a * M + Omegaf * r**3 - (a * a) * M * Omegaf * (sth * sth))
        )
        + (a * a)
        * (cth * cth)
        * (
            M
            + Omegaf
            * (sth * sth)
            * (-2 * a * M + 2 * Omegaf * r**3 + (a * a) * M * Omegaf * (sth * sth))
        )
    )
    dNdr /= (Sigma * Sigma) / 2

    denomtheta = (a * a) + 2 * (r * r) + (a * a) * np.cos(2 * theta) ** 2
    dNdtheta = (Omegaf * Omegaf) * ((a * a) + r * (r - 2 * M)) + 8 * M * r * (
        a * (a * Omegaf - 1) + Omegaf * (r * r)
    ) ** 2 / (denomtheta * denomtheta)
    dNdtheta *= np.sin(2 * theta)

    bvec = bf_here.bfield_lab(a, r, th=theta)

    return bvec[0] * dNdr + bvec[1] * dNdtheta


def metric(r, a, theta, M=1.0):
    SigmaK = (r * r) + (a * a) * np.cos(theta) ** 2
    DeltaK = (r * r) - 2 * M * r + (a * a)
    gmunu = (
        np.zeros(
            (
                4,
                4,
            )
            + r.shape
        )
        if hasattr(r, "shape")
        else np.zeros((4, 4))
    )
    gmunu[0][0] = -(1 - 2 * M * r / SigmaK)
    gmunu[0][3] = gmunu[3][0] = -2 * a * M * r / SigmaK * np.sin(theta) ** 2
    gmunu[1][1] = SigmaK / DeltaK
    gmunu[2][2] = SigmaK
    gmunu[3][3] = ((r * r) + (a * a) + 2 * M * r * (a * a) / SigmaK * np.sin(theta) ** 2) * np.sin(
        theta
    ) ** 2
    return gmunu


def invmetric(r, a, theta, M=1.0):
    SigmaK = (r * r) + (a * a) * np.cos(theta) ** 2
    DeltaK = (r * r) - 2 * M * r + (a * a)
    ginvmunu = (
        np.zeros(
            (
                4,
                4,
            )
            + r.shape
        )
        if hasattr(r, "shape")
        else np.zeros((4, 4))
    )
    ginvmunu[0][0] = (
        -1 / DeltaK * ((r * r) + (a * a) + 2 * M * r * (a * a) / SigmaK * np.sin(theta) ** 2)
    )
    ginvmunu[0][3] = ginvmunu[3][0] = -2 * M * r * a / (SigmaK * DeltaK)
    ginvmunu[1][1] = DeltaK / SigmaK
    ginvmunu[2][2] = 1 / SigmaK
    ginvmunu[3][3] = (DeltaK - (a * a) * np.sin(theta) ** 2) / (
        SigmaK * DeltaK * np.sin(theta) ** 2
    )
    return ginvmunu


def getEco(r0, theta, Omegaf, spin, M=1.0):
    g = metric(r0, spin, theta, M)
    ginv = invmetric(r0, spin, theta, M)
    gtt = g[0, 0, :, :]
    gtphi = g[0, 3, :, :]
    ginvtt = ginv[0, 0, :, :]
    ginvtphi = ginv[0, 3, :, :]
    gphiphi = g[3, 3, :, :]
    ginvphiphi = ginv[3, 3, :, :]
    gtphifac = gtphi + gphiphi * Omegaf
    gttfac = gtt + gtphi * Omegaf
    coef0 = (gtt + Omegaf * (2 * gtphi + gphiphi * Omegaf)) ** 2
    coef1 = (
        ginvphiphi * (gtphifac * gtphifac) + 2 * ginvtphi * gtphifac * gttfac + ginvtt * gttfac**2
    )
    efac2 = -coef0 / coef1
    return np.sqrt(efac2)


def getnu_cons(bf_here, r, theta, r0, theta0, Omegaf, spin, M=1.0):
    Aconst = getEco(r0, theta0, Omegaf, spin, M)
    ghere = metric(r, spin, theta, M)
    (alpha, vphiupper, gammap, Bhatphi) = u_driftframe(
        spin, r, r0, theta0, Omegaf, bfield=bf_here, nu_parallel=0, th=theta, retbunit=True
    )  # get quantities from nu=0 case
    ffunc = gammap * (
        alpha - (ghere[0, 3, :, :] + ghere[3, 3, :, :] * Omegaf) * vphiupper
    )  # random function (useful for xi computation)
    bred = Bhatphi * (
        ghere[0, 3, :, :] + ghere[3, 3, :, :] * Omegaf
    )  # reduced Bphiunit (useful for xi computation)
    nunum = ffunc * bred + np.sign(np.cos(theta)) * np.sign(r - r0) * Aconst * np.sqrt(
        (Aconst * Aconst) - (ffunc * ffunc) + (bred * bred)
    )
    nudenom = (Aconst * Aconst) + (bred * bred)
    nutot = nunum / nudenom
    return np.real(nutot)


def u_driftframe(
    a,
    r,
    r0,
    theta0,
    omega,
    *fast_args,
    bfield=None,
    nu_parallel=0,
    th=np.pi / 2,
    gammamax=None,
    retbunit=False,
    retqty=False,
    eps=-1,
):
    """
    drift frame velocity for a given EM field in BL
      - If called with no extra positional args: uses the original (bfield-based) implementation from kgeo
      - If called with 21 extra positional args: uses a faster precomputed path and returns a 9-tuple
    """

    # ---------------------------------------------------------------------
    # Fast path
    # Signature expected (after omega): 21 positional args:
    #   r2,a2,cth2,sth2,Delta,rho2,Sigma,g00,g11,g22,g33,g03,alpha,gdet,B1,B2,B3,E1,E2,E3,signcostheta
    # ---------------------------------------------------------------------
    if fast_args:
        if len(fast_args) != 21:
            raise TypeError(
                f"u_driftframe fast-path expects 21 extra positional args; got {len(fast_args)}"
            )

        (
            r2,
            a2,
            cth2,
            sth2,
            Delta,
            rho2,
            Sigma,
            g00,
            g11,
            g22,
            g33,
            g03,
            alpha,
            gdet,
            B1,
            B2,
            B3,
            E1,
            E2,
            E3,
            signcostheta,
        ) = fast_args

        ########################
        # EM quantities

        B1_cov = g11 * B1
        B2_cov = g22 * B2
        B3_cov = g33 * B3
        E1_cov = g11 * E1
        E2_cov = g22 * E2
        E3_cov = g33 * E3
        Bsq = (B1_cov * B1) + (B2_cov * B2) + (B3_cov * B3)
        Esq = (E1_cov * E1) + (E2_cov * E2) + (E3_cov * E3)

        ########################
        # enforce magnetic dominance: Esq/Bsq < 1

        eps_EB = 1.0e-8
        ratio_raw = Esq / Bsq

        mask = ratio_raw >= (1.0 - eps_EB)
        if np.any(mask):
            # scale E -> E * sqrt((1-eps)/ratio_raw)
            scl = np.ones_like(ratio_raw)
            scl[mask] = np.sqrt((1.0 - eps_EB) / ratio_raw[mask])

            E1 = E1 * scl
            E2 = E2 * scl
            E3 = E3 * scl
            E1_cov = E1_cov * scl
            E2_cov = E2_cov * scl
            E3_cov = E3_cov * scl

            Esq = (E1_cov * E1) + (E2_cov * E2) + (E3_cov * E3)

        ratio = np.clip(Esq / Bsq, 0.0, 1.0 - eps_EB)

        ########################
        # compute the parallel boost

        # compute metric quantities associated with stagnation radius
        cth2_stag = np.cos(theta0) ** 2.0
        sth2_stag = np.sin(theta0) ** 2.0
        r2_stag = r0 * r0
        rho2_stag = r2_stag + (a2 * cth2_stag)
        Delta_stag = r2_stag - (2.0 * r0) + a2
        Sigma_stag = ((r2_stag + a2) ** 2.0) - (a2 * Delta_stag * sth2_stag)

        g00_stag = ((a2 * sth2_stag) - Delta_stag) / rho2_stag
        g03_stag = -2.0 * a * r0 * sth2_stag / rho2_stag
        g33_stag = Sigma_stag * sth2_stag / rho2_stag

        ginv_denom_stag = (g00_stag * g33_stag) - (g03_stag * g03_stag)
        ginv00_stag = g33_stag / ginv_denom_stag
        ginv03_stag = -g03_stag / ginv_denom_stag
        ginv33_stag = g00_stag / ginv_denom_stag

        # compute energy conservation factor
        gtphifac = g03_stag + (g33_stag * omega)
        gttfac = g00_stag + (g03_stag * omega)
        coef0 = (g00_stag + (omega * (2.0 * g03_stag + g33_stag * omega))) ** 2.0
        coef1 = (ginv33_stag * (gtphifac * gtphifac)) + (
            2.0 * ginv03_stag * gtphifac * gttfac + ginv00_stag * (gttfac * gttfac)
        )
        efac2 = -coef0 / coef1
        Aconst = np.sqrt(efac2)
        Aconst2 = Aconst * Aconst

        vphiupper = (alpha / (Bsq * gdet)) * (E1_cov * B2_cov - B1_cov * E2_cov)
        gammap = 1.0 / np.sqrt(1.0 - ratio)
        Bhatphi = B3 / np.sqrt(Bsq)

        ffunc = gammap * (alpha - ((g03 + g33 * omega) * vphiupper))
        bred = Bhatphi * (g03 + (g33 * omega))
        bred2 = bred * bred
        disc = Aconst2 - (ffunc * ffunc) + bred2
        disc = np.maximum(disc, 0.0)
        root = np.sqrt(disc)
        nunum = (ffunc * bred) + signcostheta * np.sign(r - r0) * Aconst * root
        nudenom = Aconst2 + bred2
        nutot = nunum / nudenom

        nu_parallel = np.real(nutot)

        ########################
        # velocities

        # perp velocity in the lnrf, vtilde_perp
        vperp1 = (alpha / (Bsq * gdet)) * (E2_cov * B3_cov - B2_cov * E3_cov)
        vperp2 = (alpha / (Bsq * gdet)) * (E3_cov * B1_cov - B3_cov * E1_cov)
        vperp3 = (alpha / (Bsq * gdet)) * (E1_cov * B2_cov - B1_cov * E2_cov)

        # parallel velocity in the lnrf, vtilde_perp
        vpar_max = np.sqrt(1 - ratio)
        prefac_here = nu_parallel * vpar_max / np.sqrt(Bsq)
        vpar1 = prefac_here * B1
        vpar2 = prefac_here * B2
        vpar3 = prefac_here * B3

        # convert to four-velocity
        v1 = vperp1 + vpar1
        v2 = vperp2 + vpar2
        v3 = vperp3 + vpar3

        eps_v = 1e-12
        vsq = g11 * v1 * v1 + g22 * v2 * v2 + g33 * v3 * v3

        mask = vsq >= (1.0 - eps_v)
        if np.any(mask):
            fac = np.sqrt((1.0 - eps_v) / vsq[mask])
            v1[mask] *= fac
            v2[mask] *= fac
            v3[mask] *= fac
            vsq[mask] = 1.0 - eps_v

        gamma = 1.0 / np.sqrt(1.0 - vsq)

        Xi = (r2 + a2) ** 2 - Delta * a2 * sth2
        eta3 = 2.0 * a * r / np.sqrt(Delta * Sigma * Xi)

        u0 = gamma / alpha
        u1 = gamma * v1
        u2 = gamma * v2
        u3 = gamma * (v3 + eta3)

        # subtract off vpar to get vperp
        Bdotv = (g11 * v1 * B1) + (g22 * v2 * B2) + (g33 * v3 * B3)
        Bdotv_Bsq = Bdotv / Bsq
        v1perp = v1 - (B1 * Bdotv_Bsq)
        v2perp = v2 - (B2 * Bdotv_Bsq)
        v3perp = v3 - (B3 * Bdotv_Bsq)
        vperpsq = (g11 * v1perp * v1perp) + (g22 * v2perp * v2perp) + (g33 * v3perp * v3perp)

        return (gamma, u0, u1, u2, u3, np.sqrt(vperpsq), v1perp, v2perp, v3perp)

    # ---------------------------------------------------------------------
    # Slow, original path
    # ---------------------------------------------------------------------

    # get boost from conservation of energy if requested
    if nu_parallel == "FF":
        # get the parallel boost
        nu_parallel = getnu_cons(bfield, r, th, r0, theta0, omega, a, 1.0)

    # checks
    nu_parallel = nu_parallel * np.ones_like(r)  # make sure that nu_parallel is appropriately sized
    if not (isinstance(a, float) and (0 <= np.abs(a) < 1)):
        raise Exception("|a| should be a float in range [0,1)")
    if np.any(np.logical_or(nu_parallel > 1, nu_parallel < -1)):
        raise Exception("nu_parallel should be in the range (-1,1)")
    if not isinstance(r, np.ndarray):
        r = np.array([r]).flatten()

    # metric
    a2 = a * a
    r2 = r * r
    cth2 = np.cos(th) ** 2
    sth2 = np.sin(th) ** 2

    Delta = r2 - 2 * r + a2
    Sigma = r2 + a2 * cth2
    Xi = (r2 + a2) ** 2 - Delta * a2 * sth2
    omegaz = 2 * a * r / Xi
    gdet = Sigma * np.sin(th)

    g11 = Sigma / Delta
    g22 = Sigma
    g33 = Xi * sth2 / Sigma
    g03 = -2 * r * a * np.sin(th) ** 2 / Sigma

    # lapse and shift
    alpha2 = Delta * Sigma / Xi
    alpha = np.sqrt(alpha2)  # lapse
    eta1 = 0
    eta2 = 0
    eta3 = 2 * a * r / np.sqrt(Delta * Sigma * Xi)

    # e and b field
    omega = bfield.omega_field(a, r, th=th)
    (B1, B2, B3) = bfield.bfield_lab(a, r, th=th)
    (E1, E2, E3) = bfield.efield_lab(a, r, th=th)

    E1 = (omega - omegaz) * Xi * np.sin(th) * B2 / Sigma
    E2 = -(omega - omegaz) * Xi * np.sin(th) * B1 / (Sigma * Delta)
    E3 = 0

    Bsq = g11 * B1 * B1 + g22 * B2 * B2 + g33 * B3 * B3
    Esq = g11 * E1 * E1 + g22 * E2 * E2 + g33 * E3 * E3

    B1_cov = g11 * B1
    B2_cov = g22 * B2
    B3_cov = g33 * B3

    E1_cov = g11 * E1
    E2_cov = g22 * E2
    E3_cov = g33 * E3

    # perp velocity in the lnrf, vtilde_perp
    vperp1 = (alpha / (Bsq * gdet)) * (E2_cov * B3_cov - B2_cov * E3_cov)
    vperp2 = (alpha / (Bsq * gdet)) * (E3_cov * B1_cov - B3_cov * E1_cov)
    vperp3 = (alpha / (Bsq * gdet)) * (E1_cov * B2_cov - B1_cov * E2_cov)

    # parallel velocity in the lnrf, vtilde_perp
    ratio = Esq / Bsq
    ratio = np.clip(ratio, 0.0, 1.0 - 1e-8)
    vpar_max = np.sqrt(1 - ratio)
    vpar1 = nu_parallel * vpar_max * B1 / np.sqrt(Bsq)
    vpar2 = nu_parallel * vpar_max * B2 / np.sqrt(Bsq)
    vpar3 = nu_parallel * vpar_max * B3 / np.sqrt(Bsq)

    # convert to four-velocity
    v1 = vperp1 + vpar1
    v2 = vperp2 + vpar2
    v3 = vperp3 + vpar3

    if retbunit:  # returns gammaperp and raised unit vector along B (useful for FF computations)
        return (alpha, v3, 1 / vpar_max, B3 / np.sqrt(Bsq))

    vsq = g11 * v1 * v1 + g22 * v2 * v2 + g33 * v3 * v3
    gamma = 1.0 / np.sqrt(1 - vsq)

    if gammamax:  # approximate MHD gamma by summing gamma_FF and gamma_max in series
        pval0 = 2.0
        gammamax = gammamax * np.ones_like(gamma)
        gammaeff = (1 / gammamax**pval0 + 1 / gamma**pval0) ** (-1 / pval0)

        gammamin = np.nanmin(gammaeff)
        argdiv = gammaeff == gammamin
        gammaeff0 = gammaeff * gamma[argdiv] / gammaeff[argdiv]

        vsqeff = 1 - 1 / (gammaeff0 * gammaeff0)  # convert
        v1new = v1 * np.sqrt(vsqeff / vsq)
        v2new = v2 * np.sqrt(vsqeff / vsq)
        v3new = v3 * np.sqrt(vsqeff / vsq)

        v1 = v1new
        v2 = v2new
        v3 = v3new
        gamma = np.real(gammaeff0)

    u0 = gamma / alpha
    u1 = gamma * (v1 + eta1)
    u2 = gamma * (v2 + eta2)
    u3 = gamma * (v3 + eta3)

    if retqty:
        Bdotv = g11 * v1 * B1 + g22 * v2 * B2 + g33 * v3 * B3
        v1perp = v1 - B1 * Bdotv / Bsq  # subtract off vpar to get vperp
        v2perp = v2 - B2 * Bdotv / Bsq
        v3perp = v3 - B3 * Bdotv / Bsq
        vperpsq = g11 * v1perp * v1perp + g22 * v2perp * v2perp + g33 * v3perp * v3perp
        return (np.sqrt(vperpsq), v1perp, v2perp, v3perp)  # returns magnitude of vperp

    return (gamma, u0, u1, u2, u3)


###################################################
# primary class


class JetModel:
    """
    A jet model that can be used to generate images or SEDs according to the
    prescription in Pesce et al. (2026).

    This class object caches all frequency-independent information, so that
    repeated calls over frequency only need to do the RT integration.

    Typical usage:
        model = JetModel(m=..., a=..., inc=..., mdot=..., Nx=..., Ny=..., Nz=..., s=..., p=..., ...)
        for freq in freqs:
            x, y, I = model.make_image(freq)
    """

    def __init__(
        self,
        m,
        a,
        inc,
        mdot,
        Nx=100,
        Ny=100,
        Nz=400,
        xmin=0.1,
        xmax=1000.0,
        ymin=0.1,
        ymax=1000.0,
        zmin=0.1,
        zmax=3000.0,
        use_log_xgrid=False,
        use_log_ygrid=False,
        use_log_zgrid=False,
        x_im_1D=None,
        y_im_1D=None,
        z_im_1D=None,
        s=0.6,
        p=2.5,
        h=0.0025,
        eta=0.01,
        jet_cutout_fraction=0.0,
        gammamax=6.0,
        betagamma_suppression=0.5,
        gamma_m=30.0,
        gamma_max=1.0e8,
        DTYPE=np.float64,
        pretab_dir=None,
    ):
        ####################
        # store inputs

        self.m = float(m)
        self.a = float(a)
        self.inc = float(inc)
        self.mdot = float(mdot)

        self.Nx, self.Ny, self.Nz = int(Nx), int(Ny), int(Nz)
        self.xmin, self.xmax = float(xmin), float(xmax)
        self.ymin, self.ymax = float(ymin), float(ymax)
        self.zmin, self.zmax = float(zmin), float(zmax)
        self.use_log_xgrid = use_log_xgrid
        self.use_log_ygrid = use_log_ygrid
        self.use_log_zgrid = use_log_zgrid
        self.x_im_1D_input = x_im_1D
        self.y_im_1D_input = y_im_1D
        self.z_im_1D_input = z_im_1D

        self.s = float(s)
        self.p = float(p)
        self.h = float(h)
        self.eta = float(eta)
        self.jet_cutout_fraction = float(jet_cutout_fraction)
        self.gammamax = float(gammamax)
        self.betagamma_suppression = float(betagamma_suppression)
        self.gamma_m = float(gamma_m)
        self.gamma_max = float(gamma_max)
        self.DTYPE = DTYPE

        self.pretab_dir = pretab_dir

        ####################
        # derived quantities

        self.nu = 2.0 - (2.0 * self.s)
        self.rH = 1.0 + np.sqrt(1.0 - (self.a * self.a))

        self.inc_rad = self.inc * np.pi / 180.0
        self.cos_i = np.cos(self.inc_rad)
        self.sin_i = np.sin(self.inc_rad)

        self.rg = (1.477e5) * self.m
        self.Mdot = self.mdot * self.m * (1.399e17)
        self.Pjet = 1.4 * (self.a * self.a) * self.Mdot * (c * c)

        # magnetic field object used in setup steps
        self.bf = Bfield(p=self.nu)

        # unit-bearing prefactors for emissivity and absorption
        self.prefac_emis = ((q_e**2.0) * (1.0e9)) / (2.0 * np.sqrt(3.0) * c)
        self.prefac_absorp = (q_e**2.0) / (4.0 * np.sqrt(3.0) * m_e * c * (1.0e9))

        # direction of photon propagation in jet frame
        self.nx = -self.sin_i
        self.ny = 0.0
        self.nz = -self.cos_i

        ####################
        # build stagnation-surface interpolator tables

        self._build_stagnation_surface()

        ####################
        # compute Poynting flux scaling

        self._build_poynting_scaling()

        ####################
        # load synchrotron lookup tables

        self._load_synchrotron_tables()

        ####################
        # anisotropy normalization

        self._compute_phi_norms()

        ####################
        # construct grids

        self._build_grids(x_im_1D, y_im_1D, z_im_1D)

    def _build_stagnation_surface(self):
        a = self.a
        nu = self.nu
        rH = self.rH
        bf = self.bf

        thetahorizon_arr = 10.0 ** np.linspace(-5.0, np.log10(np.pi / 2.0), 100)
        psi_arr = np.zeros_like(thetahorizon_arr)
        rstag_arr = np.zeros_like(thetahorizon_arr)
        tstag_arr = np.zeros_like(thetahorizon_arr)

        for i in range(len(thetahorizon_arr)):
            theta_a = 1.0e-10
            theta_b = thetahorizon_arr[i]

            r_a = rH * ((1.0 - np.cos(theta_b)) / (1.0 - np.cos(theta_a))) ** (1.0 / nu)
            r_b = rH

            psi_a = psiBZpower(rH, theta_a, nu)
            psi_b = psiBZpower(rH, theta_b, nu)

            Omega_a = omega_BZpower(0, psi_a, a, nu)
            Omega_b = omega_BZpower(0, psi_b, a, nu)

            Ndval_a = Nderiv(r_a, theta_a, a, Omega_a, 1.0, bf)
            Ndval_b = Nderiv(r_b, theta_b, a, Omega_b, 1.0, bf)

            for _ in range(30):
                theta_c = np.sqrt(theta_a * theta_b)
                r_c = rH * ((1.0 - np.cos(thetahorizon_arr[i])) / (1.0 - np.cos(theta_c))) ** (
                    1.0 / nu
                )
                psi_c = psiBZpower(rH, theta_c, nu)
                Omega_c = omega_BZpower(0, psi_c, a, nu)
                Ndval_c = Nderiv(r_c, theta_c, a, Omega_c, 1.0, bf)

                if Ndval_c > 0.0:
                    theta_a = theta_c
                    r_a = r_c
                    psi_a = psi_c
                    Omega_a = Omega_c
                    Ndval_a = Ndval_c
                else:
                    theta_b = theta_c
                    r_b = r_c
                    psi_b = psi_c
                    Omega_b = Omega_c
                    Ndval_b = Ndval_c

            psi_arr[i] = psi_c
            rstag_arr[i] = r_c
            tstag_arr[i] = theta_c

        self.thetahorizon_arr = thetahorizon_arr
        self.rstag_arr = rstag_arr
        self.tstag_arr = tstag_arr

    def stagnation(self, theta_fp):
        th = self.thetahorizon_arr
        rs = self.rstag_arr
        ts = self.tstag_arr

        rout = np.zeros_like(theta_fp)
        tout = np.zeros_like(theta_fp)

        ind1 = theta_fp <= (np.pi / 2.0)
        rout[ind1] = np.interp(theta_fp, th, rs)[ind1]
        tout[ind1] = np.interp(theta_fp, th, ts)[ind1]

        ind2 = theta_fp > (np.pi / 2.0)
        rout[ind2] = np.interp(theta_fp, (np.pi - th)[::-1], rs[::-1])[ind2]
        tout[ind2] = np.interp(theta_fp, (np.pi - th)[::-1], ts[::-1])[ind2]
        return rout, tout

    # determine the scaling factor necessary to ensure that the jet has the correct total power
    def _build_poynting_scaling(self):
        a = self.a
        s = self.s
        nu = self.nu
        rH = self.rH
        rg = self.rg
        Pjet = self.Pjet

        Nrescale = 500
        logzrescale = 8.0
        zhere = (10.0**logzrescale) * np.ones((Nrescale, 1))
        Rhere = (10.0 ** np.linspace(-1.0, logzrescale, Nrescale)).reshape((Nrescale, 1))

        rhere = np.sqrt((Rhere * Rhere) + (zhere * zhere))
        thetahere = np.arccos(zhere / rhere)

        theta_fp_1here = 2.0 * np.arcsin(((rhere / rH) ** (1.0 - s)) * np.sin(thetahere / 2.0))
        theta_fp_2here = 2.0 * np.arccos(((rhere / rH) ** (1.0 - s)) * np.cos(thetahere / 2.0))
        ind1 = theta_fp_1here < (np.pi / 2.0)
        ind2 = theta_fp_2here > (np.pi / 2.0)
        theta_fphere = np.zeros_like(theta_fp_1here)
        theta_fphere[ind1] = theta_fp_1here[ind1]
        theta_fphere[ind2] = theta_fp_2here[ind2]

        psihere = psiBZpower(rH, theta_fphere, nu)
        Omegahere = omega_BZpower(0, psihere, a, nu)

        r_jet1here = rH * ((1.0 / (1.0 - np.cos(thetahere))) ** (1.0 / nu))
        r_jet2here = rH * ((1.0 / (1.0 + np.cos(thetahere))) ** (1.0 / nu))
        ind_jet1here = rhere <= r_jet1here
        ind_jet2here = rhere <= r_jet2here
        ind_jethere = (ind_jet1here | ind_jet2here) & (rhere > rH)
        psihere[~ind_jethere] = np.nan
        Omegahere[~ind_jethere] = np.nan

        sig = (rhere * rhere) + (a * a) * np.cos(thetahere) ** 2
        delta = (rhere * rhere) - 2 * rhere + (a * a)
        pi = ((rhere * rhere) + (a * a)) ** 2 - (a * a) * delta * np.sin(thetahere) ** 2
        alphalapse_here = np.sqrt(delta * sig / pi)
        grr = sig / delta
        gthetatheta = sig
        gphiphi = pi * np.sin(thetahere) ** 2 / sig

        bf_here = Bfield(p=nu)
        (B1_here, B2_here, B3_here) = bf_here.bfield_lab(a, rhere, th=thetahere)
        B1Zamo_here = alphalapse_here * B1_here * np.sqrt(grr)
        B2Zamo_here = alphalapse_here * B2_here * np.sqrt(gthetatheta)
        B3Zamo_here = alphalapse_here * B3_here * np.sqrt(gphiphi)
        Bsq = (
            (B1Zamo_here * B1Zamo_here) + (B2Zamo_here * B2Zamo_here) + (B3Zamo_here * B3Zamo_here)
        )

        rstaghere, tstaghere = self.stagnation(theta_fphere)
        (vperpmag_here, _, _, _) = u_driftframe(
            a,
            rhere,
            rstaghere,
            tstaghere,
            Omegahere,
            bfield=bf_here,
            nu_parallel="FF",
            gammamax=None,
            th=thetahere,
            retqty=True,
        )

        poyntingmag = Bsq * vperpmag_here * (c / (4.0 * np.pi))
        poyntingmag = np.abs(np.nan_to_num(poyntingmag))

        S_outer = np.copy(poyntingmag[:, 0])
        R_outer = rg * np.copy(Rhere[:, 0])
        integrand = 2.0 * np.pi * R_outer * S_outer
        Pjet_comp = 2.0 * np.sum(
            0.5 * (integrand[1:] + integrand[0:-1]) * (R_outer[1:] - R_outer[0:-1])
        )

        self.scaling = Pjet / Pjet_comp
        self.sqrt_scaling = np.sqrt(self.scaling)

    # load the various tabulated integrals for synchrotron emission/absorption coefficients
    @staticmethod
    def _load_table_xy(path):
        logx, logy = np.loadtxt(path, unpack=True)
        return 10.0**logx, 10.0**logy

    def _load_synchrotron_tables(self):
        p = self.p

        if self.pretab_dir is None:
            from importlib.resources import files, as_file

            with as_file(files("jetfuncs") / "synchrotron_integrals") as dpath:
                d = str(dpath)
                self.x_GI_2, self.GI_2 = self._load_table_xy(
                    os.path.join(d, "GIx", f"GIx_p={np.round(2.0, 2)}.txt")
                )
                self.x_GI_p, self.GI_p = self._load_table_xy(
                    os.path.join(d, "GIx", f"GIx_p={np.round(p, 2)}.txt")
                )
                self.x_GI_pp1, self.GI_pp1 = self._load_table_xy(
                    os.path.join(d, "GIx", f"GIx_p={np.round(p + 1.0, 2)}.txt")
                )
                self.x_GaI_2, self.GaI_2 = self._load_table_xy(
                    os.path.join(d, "GaIx", f"GaIx_p={np.round(2.0, 2)}.txt")
                )
                self.x_GaI_p, self.GaI_p = self._load_table_xy(
                    os.path.join(d, "GaIx", f"GaIx_p={np.round(p, 2)}.txt")
                )
                self.x_GaI_pp1, self.GaI_pp1 = self._load_table_xy(
                    os.path.join(d, "GaIx", f"GaIx_p={np.round(p + 1.0, 2)}.txt")
                )

        else:
            d = self.pretab_dir
            self.x_GI_2, self.GI_2 = self._load_table_xy(
                os.path.join(d, "GIx", f"GIx_p={np.round(2.0, 2)}.txt")
            )
            self.x_GI_p, self.GI_p = self._load_table_xy(
                os.path.join(d, "GIx", f"GIx_p={np.round(p, 2)}.txt")
            )
            self.x_GI_pp1, self.GI_pp1 = self._load_table_xy(
                os.path.join(d, "GIx", f"GIx_p={np.round(p + 1.0, 2)}.txt")
            )
            self.x_GaI_2, self.GaI_2 = self._load_table_xy(
                os.path.join(d, "GaIx", f"GaIx_p={np.round(2.0, 2)}.txt")
            )
            self.x_GaI_p, self.GaI_p = self._load_table_xy(
                os.path.join(d, "GaIx", f"GaIx_p={np.round(p, 2)}.txt")
            )
            self.x_GaI_pp1, self.GaI_pp1 = self._load_table_xy(
                os.path.join(d, "GaIx", f"GaIx_p={np.round(p + 1.0, 2)}.txt")
            )

    # helper functions so the RT loop can use a simplified GIx_*/GaIx_* call style
    def GIx_2(self, x):
        return np.interp(x, self.x_GI_2, self.GI_2, left=self.GI_2[0], right=0.0)

    def GIx_p(self, x):
        return np.interp(x, self.x_GI_p, self.GI_p, left=self.GI_p[0], right=0.0)

    def GIx_pp1(self, x):
        return np.interp(x, self.x_GI_pp1, self.GI_pp1, left=self.GI_pp1[0], right=0.0)

    def GaIx_2(self, x):
        return np.interp(x, self.x_GaI_2, self.GaI_2, left=self.GaI_2[0], right=0.0)

    def GaIx_p(self, x):
        return np.interp(x, self.x_GaI_p, self.GaI_p, left=self.GaI_p[0], right=0.0)

    def GaIx_pp1(self, x):
        return np.interp(x, self.x_GaI_pp1, self.GaI_pp1, left=self.GaI_pp1[0], right=0.0)

    # compute normalizations for the anisotropic distributions
    def _compute_phi_norms(self):
        eta = self.eta
        p = self.p

        dummu = np.linspace(0.0, 1.0, 10000)
        integrand = (1.0 + ((eta - 1.0) * (dummu * dummu))) ** (-1.0)
        self.phi_norm_2 = np.sum(0.5 * (integrand[1:] + integrand[:-1]) * (dummu[1:] - dummu[:-1]))

        integrand = (1.0 + ((eta - 1.0) * (dummu * dummu))) ** (-p / 2.0)
        self.phi_norm_p = np.sum(0.5 * (integrand[1:] + integrand[:-1]) * (dummu[1:] - dummu[:-1]))

        integrand = (1.0 + ((eta - 1.0) * (dummu * dummu))) ** (-(p + 1.0) / 2.0)
        self.phi_norm_pp1 = np.sum(
            0.5 * (integrand[1:] + integrand[:-1]) * (dummu[1:] - dummu[:-1])
        )

    # construct the image and jet-frame grids
    @staticmethod
    def _make_grid_lin(Nv, vmax, DTYPE=np.float64):
        grid_1D = np.linspace(-vmax, vmax, Nv, dtype=DTYPE)
        return grid_1D

    @staticmethod
    def _make_grid_log(Nv, vmin, vmax, DTYPE=np.float64):
        grid_1D_pos = 10.0 ** np.linspace(np.log10(vmin), np.log10(vmax), int(Nv / 2), dtype=DTYPE)
        grid_1D_neg = -grid_1D_pos
        grid_1D = np.concatenate([grid_1D_neg[::-1], grid_1D_pos])
        return grid_1D

    def _build_grids(self, x_im_1D, y_im_1D, z_im_1D):
        # 1D arrays
        if x_im_1D is None:
            x_im_1D = (
                self._make_grid_log(self.Nx, self.xmin, self.xmax, DTYPE=self.DTYPE)
                if self.use_log_xgrid
                else self._make_grid_lin(self.Nx, self.xmax, DTYPE=self.DTYPE)
            )
        if y_im_1D is None:
            y_im_1D = (
                self._make_grid_log(self.Ny, self.ymin, self.ymax, DTYPE=self.DTYPE)
                if self.use_log_ygrid
                else self._make_grid_lin(self.Ny, self.ymax, DTYPE=self.DTYPE)
            )
        if z_im_1D is None:
            z_im_1D = (
                self._make_grid_log(self.Nz, self.zmin, self.zmax, DTYPE=self.DTYPE)
                if self.use_log_zgrid
                else self._make_grid_lin(self.Nz, self.zmax, DTYPE=self.DTYPE)
            )

        self.x_im_1D = x_im_1D
        self.y_im_1D = y_im_1D
        self.z_im_1D = z_im_1D

        self.x_im, self.y_im = np.meshgrid(self.x_im_1D, self.y_im_1D)
        self.z_J = -(self.x_im / np.tan(self.inc_rad))

        # flattened views used in the RT loop
        self.x_im_f = self.x_im.ravel()
        self.y_im_f = self.y_im.ravel()
        self.z_J_f = self.z_J.ravel()

        # path lengths between adjacent z-slices, used in the RT loop
        self.dz_1D = self.rg * np.abs(np.diff(self.z_im_1D))

    # primary image-generating function
    def make_image(
        self, frequency, *, tau_stop=None, show_progress=False, heating_prescription="Poynting"
    ):
        """
        Returns (x_im_1D, y_im_1D, I_nu)
        """

        # pull cached attributes into local variables
        rH = self.rH
        nu = self.nu
        a = self.a
        s = self.s
        p = self.p
        h = self.h
        eta = self.eta

        rg = self.rg
        cos_i = self.cos_i
        sin_i = self.sin_i
        nx, ny, nz = self.nx, self.ny, self.nz

        scaling = self.scaling
        sqrt_scaling = self.sqrt_scaling

        prefac_emis = self.prefac_emis
        prefac_absorp = self.prefac_absorp

        phi_norm_2 = self.phi_norm_2
        phi_norm_p = self.phi_norm_p
        phi_norm_pp1 = self.phi_norm_pp1

        GIx_2 = self.GIx_2
        GIx_p = self.GIx_p
        GIx_pp1 = self.GIx_pp1
        GaIx_2 = self.GaIx_2
        GaIx_p = self.GaIx_p
        GaIx_pp1 = self.GaIx_pp1

        stagnation = self.stagnation

        x_im_1D = self.x_im_1D
        y_im_1D = self.y_im_1D
        z_im_1D = self.z_im_1D

        x_im = self.x_im
        y_im = self.y_im
        z_J = self.z_J

        x_im_f = self.x_im_f
        y_im_f = self.y_im_f
        z_J_f = self.z_J_f

        dz_1D = self.dz_1D

        jet_cutout_fraction = self.jet_cutout_fraction
        gammamax = self.gammamax
        betagamma_suppression = self.betagamma_suppression
        gamma_m = self.gamma_m
        gamma_max = self.gamma_max

        ####################
        # initialize RT quantities

        I_nu = np.zeros_like(x_im)
        tau_acc = np.zeros_like(I_nu)  # τ accumulated FROM OBSERVER SIDE
        I_nu_f = I_nu.ravel()
        tau_acc_f = tau_acc.ravel()

        # interpret tau_stop
        if tau_stop is not None:
            tau_stop = float(tau_stop)
            if tau_stop <= 0.0:
                tau_stop = None

        working_f = np.ones_like(I_nu_f, dtype=bool)

        # Precompute full pixel index array (avoids flatnonzero when tau_stop is off)
        allpix = np.arange(I_nu_f.size, dtype=np.int64)

        ####################
        # RT loop

        for i in _progress(range(0, len(z_im_1D) - 1), enabled=show_progress):
            if tau_stop is not None:
                w = np.flatnonzero(working_f)
                if w.size == 0:
                    break
            else:
                w = allpix

            # observer z coordinate (same geometry as before)
            z_im_now = z_im_1D[i] + z_J_f[w]

            # jet coordinates
            x = (x_im_f[w] * cos_i) + (z_im_now * sin_i)
            y = y_im_f[w]
            z = (z_im_now * cos_i) - (x_im_f[w] * sin_i)
            R2 = (x * x) + (y * y)
            r2 = R2 + (z * z)
            r = np.sqrt(r2)

            costheta = z / r
            one_minus_costheta = 1.0 - costheta
            one_plus_costheta = 1.0 + costheta

            # footpoint theta
            r_rH_1_s = np.power(r / rH, 1.0 - s)
            sin_half = np.sqrt(0.5 * one_minus_costheta)
            cos_half = np.sqrt(0.5 * one_plus_costheta)
            arg_arcsin = r_rH_1_s * sin_half
            arg_arccos = r_rH_1_s * cos_half
            mask_sin = arg_arcsin < (1.0 / np.sqrt(2.0))
            theta_fp = np.empty_like(r)
            theta_fp[mask_sin] = 2.0 * np.arcsin(arg_arcsin[mask_sin])
            theta_fp[~mask_sin] = 2.0 * np.arccos(arg_arccos[~mask_sin])

            # jet region
            rjet1 = rH * np.power(1.0 / one_minus_costheta, 1.0 / nu)
            rjet2 = rH * np.power(1.0 / one_plus_costheta, 1.0 / nu)
            ind_jet = ((r <= rjet1) | (r <= rjet2)) & (r > rH)

            if jet_cutout_fraction > 0.0:
                theta_fp_cut = 2.0 * np.arcsin(jet_cutout_fraction / np.sqrt(2.0))
                ind_jet &= ~((theta_fp < theta_fp_cut) | (theta_fp > (np.pi - theta_fp_cut)))

            if not ind_jet.any():
                continue

            # local indices into arrays defined on w
            idx_loc = np.nonzero(ind_jet)[0]

            # global indices into flattened full image arrays (I_nu_f, tau_acc_f, working_f, etc.)
            idx = w[idx_loc]

            # stream function, Omega, stagnation surface
            psi = psiBZpower(rH, theta_fp[idx_loc], nu)
            Omega = omega_BZpower(0, psi, a, nu)
            rstag, tstag = stagnation(theta_fp[idx_loc])

            # metric quantities
            R = np.sqrt(R2[idx_loc])
            sintheta = R / r[idx_loc]
            costh = costheta[idx_loc]
            cth2 = costh * costh
            sth2 = 1.0 - cth2
            a2 = a * a
            r2pa2 = r2[idx_loc] + a2

            rho2 = r2[idx_loc] + (a2 * cth2)
            Delta = r2pa2 - (2.0 * r[idx_loc])
            Sigma = (r2pa2 * r2pa2) - (a2 * Delta * sth2)
            alphalapse = np.sqrt(Delta * rho2 / Sigma)

            sth2_rho2 = sth2 / rho2

            g00 = ((a2 * sth2) - Delta) / rho2
            g03 = -2.0 * a * r[idx_loc] * sth2_rho2
            g11 = rho2 / Delta
            g22 = rho2
            g33 = Sigma * sth2_rho2

            gdet = sintheta * rho2

            # EM field
            r_nu = r[idx_loc] ** nu
            signcostheta = np.sign(costh)
            dpsidtheta = signcostheta * sintheta * r_nu
            dpsidr = nu * psi / r[idx_loc]

            if nu > 0:
                Ipol = -4.0 * np.pi * psi * Omega * signcostheta
            else:
                Ipol = -2.0 * np.pi * psi * (2 - psi) * Omega * signcostheta

            B1 = dpsidtheta / gdet
            B2 = -dpsidr / gdet
            B3 = Ipol / (2 * np.pi * Delta * sth2)
            Br = B1 * np.sqrt(g11)
            Btheta = B2 * np.sqrt(g22)
            Bphi = B3 * np.sqrt(g33)

            omegaz = 2.0 * a * r[idx_loc] / Sigma
            E1 = (Omega - omegaz) * Sigma * sintheta * B2 / rho2
            E2 = -(Omega - omegaz) * Sigma * sintheta * B1 / (rho2 * Delta)
            E3 = 0.0

            # fluid velocity
            gamma, u0, u1, u2, u3, vperpmag, v1perp, v2perp, v3perp = u_driftframe(
                a,
                r[idx_loc],
                rstag,
                tstag,
                Omega,
                r2[idx_loc],
                a2,
                cth2,
                sth2,
                Delta,
                rho2,
                Sigma,
                g00,
                g11,
                g22,
                g33,
                g03,
                alphalapse,
                gdet,
                B1,
                B2,
                B3,
                E1,
                E2,
                E3,
                signcostheta,
            )

            # ZAMO-frame Poynting flux scaling
            B1Zamo = alphalapse * Br
            B2Zamo = alphalapse * Btheta
            B3Zamo = alphalapse * Bphi
            Bsq = (B1Zamo * B1Zamo) + (B2Zamo * B2Zamo) + (B3Zamo * B3Zamo)
            poyntingmag = Bsq * vperpmag * (c / (4.0 * np.pi))
            poyntingmag = np.abs(np.nan_to_num(poyntingmag))
            S = poyntingmag * scaling

            # rescale B fields
            Br *= sqrt_scaling
            Btheta *= sqrt_scaling
            Bphi *= sqrt_scaling
            Bx, By, Bz = rtp_to_xyz(
                Br, Btheta, Bphi, x[idx_loc], y[idx_loc], z[idx_loc], r[idx_loc], R
            )

            # velocity rescaling
            vr_orig = u1 * np.sqrt(g11) / gamma
            vtheta_orig = u2 * np.sqrt(g22) / gamma
            vphi_orig = u3 * np.sqrt(g33) / gamma

            beta_orig = np.sqrt(np.maximum(0.0, 1.0 - 1.0 / (gamma * gamma)))
            betagamma = betagamma_suppression * beta_orig * gamma
            gamma = np.sqrt(1.0 + (betagamma * betagamma))

            indgamma = gamma > gammamax
            gamma[indgamma] = gammamax

            beta = np.sqrt(np.maximum(0.0, 1.0 - 1.0 / (gamma * gamma)))
            velscale = np.zeros_like(beta)
            mask_boost = beta_orig > 0.0
            velscale[mask_boost] = beta[mask_boost] / beta_orig[mask_boost]

            vr = velscale * vr_orig
            vtheta = velscale * vtheta_orig
            vphi = velscale * vphi_orig
            vx, vy, vz = rtp_to_xyz(
                vr, vtheta, vphi, x[idx_loc], y[idx_loc], z[idx_loc], r[idx_loc], R
            )
            vmag = np.sqrt(vx * vx + vy * vy + vz * vz)

            # avoid v/|v| when |v|~0
            eps_v = 1.0e-8
            mask_v = vmag > eps_v

            vhat_x = np.zeros_like(vx)
            vhat_y = np.zeros_like(vy)
            vhat_z = np.zeros_like(vz)
            vhat_x[mask_v] = vx[mask_v] / vmag[mask_v]
            vhat_y[mask_v] = vy[mask_v] / vmag[mask_v]
            vhat_z[mask_v] = vz[mask_v] / vmag[mask_v]

            # redshift factor
            k_par = (vhat_x * nx) + (vhat_y * ny) + (vhat_z * nz)
            one_m_betak = 1.0 - (beta * k_par)
            g = 1.0 / (gamma * one_m_betak)
            g[~np.isfinite(g)] = 1.0

            # photon direction in comoving frame
            k_perp_x = nx - k_par * vhat_x
            k_perp_y = ny - k_par * vhat_y
            k_perp_z = nz - k_par * vhat_z

            gamma_one_m_betak = gamma * one_m_betak
            k_par_prime = (k_par - beta) / one_m_betak
            k_perp_x_prime = k_perp_x / gamma_one_m_betak
            k_perp_y_prime = k_perp_y / gamma_one_m_betak
            k_perp_z_prime = k_perp_z / gamma_one_m_betak

            k_x_prime = k_perp_x_prime + k_par_prime * vhat_x
            k_y_prime = k_perp_y_prime + k_par_prime * vhat_y
            k_z_prime = k_perp_z_prime + k_par_prime * vhat_z
            k_prime_mag = np.sqrt(
                k_x_prime * k_x_prime + k_y_prime * k_y_prime + k_z_prime * k_z_prime
            )

            khat_x_prime = k_x_prime / k_prime_mag
            khat_y_prime = k_y_prime / k_prime_mag
            khat_z_prime = k_z_prime / k_prime_mag

            # transform B-field
            B_par = _dot(Bx, By, Bz, vhat_x, vhat_y, vhat_z)
            B_perp_x = Bx - B_par * vhat_x
            B_perp_y = By - B_par * vhat_y
            B_perp_z = Bz - B_par * vhat_z
            Bprime_x = (B_perp_x / gamma) + (B_par * vhat_x)
            Bprime_y = (B_perp_y / gamma) + (B_par * vhat_y)
            Bprime_z = (B_perp_z / gamma) + (B_par * vhat_z)

            # revert to lab-frame B-field in small-v limit
            Bprime_x[~mask_v] = Bx[~mask_v]
            Bprime_y[~mask_v] = By[~mask_v]
            Bprime_z[~mask_v] = Bz[~mask_v]
            Bprime_mag = np.sqrt(Bprime_x * Bprime_x + Bprime_y * Bprime_y + Bprime_z * Bprime_z)

            costhetaB = (
                (khat_x_prime * Bprime_x) + (khat_y_prime * Bprime_y) + (khat_z_prime * Bprime_z)
            ) / Bprime_mag
            sinthetaB = np.sqrt(1.0 - (costhetaB * costhetaB))

            # synchrotron emissivity/absorption
            t_c = np.abs(z[idx_loc] * rg) / (c * gamma)
            gamma_c = (6.0 * np.pi * m_e * c) / (sigma_T * (Bprime_mag * Bprime_mag) * t_c)

            if heating_prescription == "Poynting":
                u_pl = h * S / c
            elif heating_prescription == "magnetic":
                u_pl = h * ((Bprime_mag * Bprime_mag) / (8.0 * np.pi))

            n_m = (((p - 2.0) * u_pl) / ((gamma_m**p) * m_e * (c * c))) * (
                1.0 / ((gamma_m ** (2.0 - p)) - (gamma_max ** (2.0 - p)))
            )

            ind_fast = gamma_c <= gamma_m
            ind_slow = gamma_c > gamma_m

            cosxi = costhetaB
            anisotropy_term = 1.0 + ((eta - 1.0) * (cosxi * cosxi))

            nup = (4.1987e-3) * Bprime_mag * sinthetaB
            nu_nup = (frequency / g) / nup

            jI = np.zeros_like(x[idx_loc])
            alphaI = np.zeros_like(x[idx_loc])

            # slow cooling
            if np.any(ind_slow):
                p1 = p
                p2 = p + 1.0
                g1 = gamma_m
                g2 = gamma_c[ind_slow]
                g3 = gamma_max
                x1 = nu_nup[ind_slow] / (g1 * g1)
                x2 = nu_nup[ind_slow] / (g2 * g2)
                x3 = nu_nup[ind_slow] / (g3 * g3)
                Pp1 = phi_norm_p
                Pp2 = phi_norm_pp1
                GIx_p1 = GIx_p
                GIx_p2 = GIx_pp1
                A_norm = 1.0 / (
                    (((g2 ** (1.0 - p1)) - (g1 ** (1.0 - p1))) / (1.0 - p1))
                    + ((g2 ** (p2 - p1)) * (((g3 ** (1.0 - p2)) - (g2 ** (1.0 - p2))) / (1.0 - p2)))
                )
                n = (
                    n_m[ind_slow]
                    * (gamma_m**p)
                    * (
                        (((gamma_c[ind_slow] ** (1.0 - p)) - (gamma_m ** (1.0 - p))) / (1.0 - p))
                        - (
                            gamma_c[ind_slow]
                            * (((gamma_max ** (-p)) - (gamma_c[ind_slow] ** (-p))) / p)
                        )
                    )
                )
                prefac_j = prefac_emis * n * A_norm * nup[ind_slow]
                term1 = (
                    ((anisotropy_term[ind_slow] ** (-p1 / 2.0)) / Pp1)
                    * (nu_nup[ind_slow] ** ((1.0 - p1) / 2.0))
                    * (GIx_p1(x2) - GIx_p1(x1))
                )
                term2 = (
                    (g2 ** (p2 - p1))
                    * ((anisotropy_term[ind_slow] ** (-p2 / 2.0)) / Pp2)
                    * (nu_nup[ind_slow] ** ((1.0 - p2) / 2.0))
                    * (GIx_p2(x3) - GIx_p2(x2))
                )
                jI[ind_slow] = prefac_j * (term1 + term2)

                GaIx_p1 = GaIx_p
                GaIx_p2 = GaIx_pp1
                prefac_a = prefac_absorp * n * A_norm / nup[ind_slow]
                term1 = (
                    ((p1 + 2.0) * ((anisotropy_term[ind_slow] ** (-p1 / 2.0)) / Pp1))
                    * (nu_nup[ind_slow] ** (-(p1 + 4.0) / 2.0))
                    * (GaIx_p1(x2) - GaIx_p1(x1))
                )
                term2 = (
                    (
                        (p2 + 2.0)
                        * (g2 ** (p2 - p1))
                        * ((anisotropy_term[ind_slow] ** (-p2 / 2.0)) / Pp2)
                    )
                    * (nu_nup[ind_slow] ** (-(p2 + 4.0) / 2.0))
                    * (GaIx_p2(x3) - GaIx_p2(x2))
                )
                alphaI[ind_slow] = prefac_a * (term1 + term2)

            # fast cooling
            if np.any(ind_fast):
                p1 = 2.0
                p2 = p + 1.0
                g1 = gamma_c[ind_fast]
                g2 = gamma_m
                g3 = gamma_max
                x1 = nu_nup[ind_fast] / (g1 * g1)
                x2 = nu_nup[ind_fast] / (g2 * g2)
                x3 = nu_nup[ind_fast] / (g3 * g3)
                Pp1 = phi_norm_2
                Pp2 = phi_norm_pp1
                GIx_p1 = GIx_2
                GIx_p2 = GIx_pp1
                A_norm = 1.0 / (
                    (((g2 ** (1.0 - p1)) - (g1 ** (1.0 - p1))) / (1.0 - p1))
                    + ((g2 ** (p2 - p1)) * (((g3 ** (1.0 - p2)) - (g2 ** (1.0 - p2))) / (1.0 - p2)))
                )
                n = (
                    n_m[ind_fast]
                    * (gamma_c[ind_fast])
                    * (
                        (gamma_m * ((gamma_c[ind_fast] ** (-1.0)) - (gamma_m ** (-1.0))))
                        + ((gamma_m**p) * (((gamma_m ** (-p)) - (gamma_max ** (-p))) / p))
                    )
                )
                prefac_j = prefac_emis * n * A_norm * nup[ind_fast]
                term1 = (
                    ((anisotropy_term[ind_fast] ** (-p1 / 2.0)) / Pp1)
                    * (nu_nup[ind_fast] ** ((1.0 - p1) / 2.0))
                    * (GIx_p1(x2) - GIx_p1(x1))
                )
                term2 = (
                    (g2 ** (p2 - p1))
                    * ((anisotropy_term[ind_fast] ** (-p2 / 2.0)) / Pp2)
                    * (nu_nup[ind_fast] ** ((1.0 - p2) / 2.0))
                    * (GIx_p2(x3) - GIx_p2(x2))
                )
                jI[ind_fast] = prefac_j * (term1 + term2)

                GaIx_p1 = GaIx_2
                GaIx_p2 = GaIx_pp1
                prefac_a = prefac_absorp * n * A_norm / nup[ind_fast]
                term1 = (
                    ((p1 + 2.0) * ((anisotropy_term[ind_fast] ** (-p1 / 2.0)) / Pp1))
                    * (nu_nup[ind_fast] ** (-(p1 + 4.0) / 2.0))
                    * (GaIx_p1(x2) - GaIx_p1(x1))
                )
                term2 = (
                    (
                        (p2 + 2.0)
                        * (g2 ** (p2 - p1))
                        * ((anisotropy_term[ind_fast] ** (-p2 / 2.0)) / Pp2)
                    )
                    * (nu_nup[ind_fast] ** (-(p2 + 4.0) / 2.0))
                    * (GaIx_p2(x3) - GaIx_p2(x2))
                )
                alphaI[ind_fast] = prefac_a * (term1 + term2)

            alphaI = np.nan_to_num(alphaI, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            alphaI = np.maximum(alphaI, 0.0)

            # radiative transfer
            dz = dz_1D[i]

            a0 = alphaI / g
            j0 = (g * g) * jI

            a0 = np.nan_to_num(a0, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            j0 = np.nan_to_num(j0, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            a0 = np.maximum(a0, 0.0)

            tau_step = a0 * dz
            atten = np.exp(-tau_acc_f[idx])  # attenuation through foreground material

            dI = np.empty_like(j0)

            has_abs = a0 > 0.0
            if np.any(has_abs):
                one_minus_e = -np.expm1(-tau_step[has_abs])
                S0 = j0[has_abs] / a0[has_abs]
                dI[has_abs] = atten[has_abs] * S0 * one_minus_e

            if np.any(~has_abs):
                # pure emission, no absorption: dI = j ds (still attenuated by foreground τ)
                dI[~has_abs] = atten[~has_abs] * j0[~has_abs] * dz

            I_nu_f[idx] += dI
            tau_acc_f[idx] += tau_step

            if tau_stop is not None:
                done = tau_acc_f[idx] >= tau_stop
                if np.any(done):
                    working_f[idx[done]] = False

        return x_im_1D, y_im_1D, I_nu

    # function to extract physical quantities at a given (r,theta)
    def get_quantity(
        self, r, theta, quantity="Bmag", frequency=None, heating_prescription="Poynting"
    ):
        """
        Returns the requested quantity at the input (r,theta)

        Understood quantities: Bmag, Bx, By, Bz, Br, Btheta, Bphi,
                               Bmag_prime, Bx_prime, By_prime, Bz_prime,
                               psi, Omega, Poynting, costhetaB, gamma, beta,
                               t_c, gamma_c, u_e, jI, alphaI

        For jI and alphaI, the frequency must be specified in GHz.

        """

        # check that the frequency is provided if necessary
        if quantity in ["jI", "alphaI"]:
            if frequency is None:
                raise Exception("For jI and alphaI, the frequency must be specified in GHz.")

        # pull cached attributes into local variables
        rH = self.rH
        nu = self.nu
        a = self.a
        s = self.s
        p = self.p
        h = self.h
        eta = self.eta

        rg = self.rg
        cos_i = self.cos_i
        sin_i = self.sin_i
        nx, ny, nz = self.nx, self.ny, self.nz

        scaling = self.scaling
        sqrt_scaling = self.sqrt_scaling

        prefac_emis = self.prefac_emis
        prefac_absorp = self.prefac_absorp

        phi_norm_2 = self.phi_norm_2
        phi_norm_p = self.phi_norm_p
        phi_norm_pp1 = self.phi_norm_pp1

        GIx_2 = self.GIx_2
        GIx_p = self.GIx_p
        GIx_pp1 = self.GIx_pp1
        GaIx_2 = self.GaIx_2
        GaIx_p = self.GaIx_p
        GaIx_pp1 = self.GaIx_pp1

        stagnation = self.stagnation

        jet_cutout_fraction = self.jet_cutout_fraction
        gammamax = self.gammamax
        betagamma_suppression = self.betagamma_suppression
        gamma_m = self.gamma_m
        gamma_max = self.gamma_max

        ####################
        # compute jet quantities

        # jet coordinates
        x = r * np.sin(theta)
        y = r * 0.0
        z = r * np.cos(theta)
        R2 = (x * x) + (y * y)
        r2 = R2 + (z * z)

        # polar angle
        costheta = np.cos(theta)
        one_minus_costheta = 1.0 - costheta
        one_plus_costheta = 1.0 + costheta

        # footpoint theta
        r_rH_1_s = np.power(r / rH, 1.0 - s)
        sin_half = np.sqrt(0.5 * one_minus_costheta)
        cos_half = np.sqrt(0.5 * one_plus_costheta)
        arg_arcsin = r_rH_1_s * sin_half
        arg_arccos = r_rH_1_s * cos_half
        mask_sin = arg_arcsin < (1.0 / np.sqrt(2.0))
        theta_fp = np.empty_like(r)
        theta_fp[mask_sin] = 2.0 * np.arcsin(arg_arcsin[mask_sin])
        theta_fp[~mask_sin] = 2.0 * np.arccos(arg_arccos[~mask_sin])

        # stream function, Omega, stagnation surface
        psi = psiBZpower(rH, theta_fp, nu)
        if quantity == "psi":
            return psi
        Omega = omega_BZpower(0, psi, a, nu)
        if quantity == "Omega":
            return Omega
        rstag, tstag = stagnation(theta_fp)

        # metric quantities
        R = np.sqrt(R2)
        sintheta = R / r
        costh = costheta
        cth2 = costh * costh
        sth2 = 1.0 - cth2
        a2 = a * a
        r2pa2 = r2 + a2

        rho2 = r2 + (a2 * cth2)
        Delta = r2pa2 - (2.0 * r)
        Sigma = (r2pa2 * r2pa2) - (a2 * Delta * sth2)
        alphalapse = np.sqrt(Delta * rho2 / Sigma)

        sth2_rho2 = sth2 / rho2

        g00 = ((a2 * sth2) - Delta) / rho2
        g03 = -2.0 * a * r * sth2_rho2
        g11 = rho2 / Delta
        g22 = rho2
        g33 = Sigma * sth2_rho2

        gdet = sintheta * rho2

        # EM field
        r_nu = r**nu
        signcostheta = np.sign(costh)
        dpsidtheta = signcostheta * sintheta * r_nu
        dpsidr = nu * psi / r

        if nu > 0:
            Ipol = -4.0 * np.pi * psi * Omega * signcostheta
        else:
            Ipol = -2.0 * np.pi * psi * (2 - psi) * Omega * signcostheta

        B1 = dpsidtheta / gdet
        B2 = -dpsidr / gdet
        B3 = Ipol / (2 * np.pi * Delta * sth2)
        Br = B1 * np.sqrt(g11)
        Btheta = B2 * np.sqrt(g22)
        Bphi = B3 * np.sqrt(g33)

        omegaz = 2.0 * a * r / Sigma
        E1 = (Omega - omegaz) * Sigma * sintheta * B2 / rho2
        E2 = -(Omega - omegaz) * Sigma * sintheta * B1 / (rho2 * Delta)
        E3 = 0.0

        # fluid velocity
        gamma, u0, u1, u2, u3, vperpmag, v1perp, v2perp, v3perp = u_driftframe(
            a,
            r,
            rstag,
            tstag,
            Omega,
            r2,
            a2,
            cth2,
            sth2,
            Delta,
            rho2,
            Sigma,
            g00,
            g11,
            g22,
            g33,
            g03,
            alphalapse,
            gdet,
            B1,
            B2,
            B3,
            E1,
            E2,
            E3,
            signcostheta,
        )

        # ZAMO-frame Poynting flux scaling
        B1Zamo = alphalapse * Br
        B2Zamo = alphalapse * Btheta
        B3Zamo = alphalapse * Bphi
        Bsq = (B1Zamo * B1Zamo) + (B2Zamo * B2Zamo) + (B3Zamo * B3Zamo)
        poyntingmag = Bsq * vperpmag * (c / (4.0 * np.pi))
        poyntingmag = np.abs(np.nan_to_num(poyntingmag))
        S = poyntingmag * scaling

        if quantity == "Poynting":
            return S

        # rescale B fields
        Br *= sqrt_scaling
        Btheta *= sqrt_scaling
        Bphi *= sqrt_scaling
        Bx, By, Bz = rtp_to_xyz(Br, Btheta, Bphi, x, y, z, r, R)
        B = np.sqrt(Bx * Bx + By * By + Bz * Bz)

        if quantity == "Bmag":
            return B
        if quantity == "Bx":
            return Bx
        if quantity == "By":
            return By
        if quantity == "Bz":
            return Bz
        if quantity == "Br":
            return Br
        if quantity == "Btheta":
            return Btheta
        if quantity == "Bphi":
            return Bphi

        # velocity rescaling
        vr_orig = u1 * np.sqrt(g11) / gamma
        vtheta_orig = u2 * np.sqrt(g22) / gamma
        vphi_orig = u3 * np.sqrt(g33) / gamma

        beta_orig = np.sqrt(np.maximum(0.0, 1.0 - 1.0 / (gamma * gamma)))
        betagamma = betagamma_suppression * beta_orig * gamma
        gamma = np.sqrt(1.0 + (betagamma * betagamma))

        indgamma = gamma > gammamax
        gamma[indgamma] = gammamax
        if quantity == "gamma":
            return gamma

        beta = np.sqrt(np.maximum(0.0, 1.0 - 1.0 / (gamma * gamma)))
        if quantity == "beta":
            return beta

        velscale = np.zeros_like(beta)
        mask_boost = beta_orig > 0.0
        velscale[mask_boost] = beta[mask_boost] / beta_orig[mask_boost]

        vr = velscale * vr_orig
        vtheta = velscale * vtheta_orig
        vphi = velscale * vphi_orig
        vx, vy, vz = rtp_to_xyz(vr, vtheta, vphi, x, y, z, r, R)
        vmag = np.sqrt(vx * vx + vy * vy + vz * vz)

        # avoid v/|v| when |v|~0
        eps_v = 1.0e-8
        mask_v = vmag > eps_v

        vhat_x = np.zeros_like(vx)
        vhat_y = np.zeros_like(vy)
        vhat_z = np.zeros_like(vz)
        vhat_x[mask_v] = vx[mask_v] / vmag[mask_v]
        vhat_y[mask_v] = vy[mask_v] / vmag[mask_v]
        vhat_z[mask_v] = vz[mask_v] / vmag[mask_v]

        # redshift factor
        k_par = (vhat_x * nx) + (vhat_y * ny) + (vhat_z * nz)
        one_m_betak = 1.0 - (beta * k_par)
        g = 1.0 / (gamma * one_m_betak)
        g[~np.isfinite(g)] = 1.0

        # photon direction in comoving frame
        k_perp_x = nx - k_par * vhat_x
        k_perp_y = ny - k_par * vhat_y
        k_perp_z = nz - k_par * vhat_z

        gamma_one_m_betak = gamma * one_m_betak
        k_par_prime = (k_par - beta) / one_m_betak
        k_perp_x_prime = k_perp_x / gamma_one_m_betak
        k_perp_y_prime = k_perp_y / gamma_one_m_betak
        k_perp_z_prime = k_perp_z / gamma_one_m_betak

        k_x_prime = k_perp_x_prime + k_par_prime * vhat_x
        k_y_prime = k_perp_y_prime + k_par_prime * vhat_y
        k_z_prime = k_perp_z_prime + k_par_prime * vhat_z
        k_prime_mag = np.sqrt(k_x_prime * k_x_prime + k_y_prime * k_y_prime + k_z_prime * k_z_prime)

        khat_x_prime = k_x_prime / k_prime_mag
        khat_y_prime = k_y_prime / k_prime_mag
        khat_z_prime = k_z_prime / k_prime_mag

        # transform B-field
        B_par = _dot(Bx, By, Bz, vhat_x, vhat_y, vhat_z)
        B_perp_x = Bx - B_par * vhat_x
        B_perp_y = By - B_par * vhat_y
        B_perp_z = Bz - B_par * vhat_z
        Bprime_x = (B_perp_x / gamma) + (B_par * vhat_x)
        Bprime_y = (B_perp_y / gamma) + (B_par * vhat_y)
        Bprime_z = (B_perp_z / gamma) + (B_par * vhat_z)

        # revert to lab-frame B-field in small-v limit
        Bprime_x[~mask_v] = Bx[~mask_v]
        Bprime_y[~mask_v] = By[~mask_v]
        Bprime_z[~mask_v] = Bz[~mask_v]
        Bprime_mag = np.sqrt(Bprime_x * Bprime_x + Bprime_y * Bprime_y + Bprime_z * Bprime_z)

        if quantity == "Bmag_prime":
            return Bprime_mag
        if quantity == "Bx_prime":
            return Bprime_x
        if quantity == "By_prime":
            return Bprime_y
        if quantity == "Bz_prime":
            return Bprime_z

        costhetaB = (
            (khat_x_prime * Bprime_x) + (khat_y_prime * Bprime_y) + (khat_z_prime * Bprime_z)
        ) / Bprime_mag
        sinthetaB = np.sqrt(1.0 - (costhetaB * costhetaB))

        if quantity == "costhetaB":
            return costhetaB

        # synchrotron emissivity/absorption
        t_c = np.abs(z * rg) / (c * gamma)
        if quantity == "t_c":
            return t_c

        gamma_c = (6.0 * np.pi * m_e * c) / (sigma_T * (Bprime_mag * Bprime_mag) * t_c)
        if quantity == "gamma_c":
            return gamma_c

        if heating_prescription == "Poynting":
            u_pl = h * S / c
        elif heating_prescription == "magnetic":
            u_pl = h * ((Bprime_mag * Bprime_mag) / (8.0 * np.pi))
        if quantity == "u_e":
            return u_pl

        n_m = (((p - 2.0) * u_pl) / ((gamma_m**p) * m_e * (c * c))) * (
            1.0 / ((gamma_m ** (2.0 - p)) - (gamma_max ** (2.0 - p)))
        )

        ind_fast = gamma_c <= gamma_m
        ind_slow = gamma_c > gamma_m

        cosxi = costhetaB
        anisotropy_term = 1.0 + ((eta - 1.0) * (cosxi * cosxi))

        if frequency is not None:
            nup = (4.1987e-3) * Bprime_mag * sinthetaB
            nu_nup = (frequency / g) / nup

            jI = np.zeros_like(x)
            alphaI = np.zeros_like(x)

            # slow cooling
            if np.any(ind_slow):
                p1 = p
                p2 = p + 1.0
                g1 = gamma_m
                g2 = gamma_c[ind_slow]
                g3 = gamma_max
                x1 = nu_nup[ind_slow] / (g1 * g1)
                x2 = nu_nup[ind_slow] / (g2 * g2)
                x3 = nu_nup[ind_slow] / (g3 * g3)
                Pp1 = phi_norm_p
                Pp2 = phi_norm_pp1
                GIx_p1 = GIx_p
                GIx_p2 = GIx_pp1
                A_norm = 1.0 / (
                    (((g2 ** (1.0 - p1)) - (g1 ** (1.0 - p1))) / (1.0 - p1))
                    + ((g2 ** (p2 - p1)) * (((g3 ** (1.0 - p2)) - (g2 ** (1.0 - p2))) / (1.0 - p2)))
                )
                n = (
                    n_m[ind_slow]
                    * (gamma_m**p)
                    * (
                        (((gamma_c[ind_slow] ** (1.0 - p)) - (gamma_m ** (1.0 - p))) / (1.0 - p))
                        - (
                            gamma_c[ind_slow]
                            * (((gamma_max ** (-p)) - (gamma_c[ind_slow] ** (-p))) / p)
                        )
                    )
                )
                prefac_j = prefac_emis * n * A_norm * nup[ind_slow]
                term1 = (
                    ((anisotropy_term[ind_slow] ** (-p1 / 2.0)) / Pp1)
                    * (nu_nup[ind_slow] ** ((1.0 - p1) / 2.0))
                    * (GIx_p1(x2) - GIx_p1(x1))
                )
                term2 = (
                    (g2 ** (p2 - p1))
                    * ((anisotropy_term[ind_slow] ** (-p2 / 2.0)) / Pp2)
                    * (nu_nup[ind_slow] ** ((1.0 - p2) / 2.0))
                    * (GIx_p2(x3) - GIx_p2(x2))
                )
                jI[ind_slow] = prefac_j * (term1 + term2)

                GaIx_p1 = GaIx_p
                GaIx_p2 = GaIx_pp1
                prefac_a = prefac_absorp * n * A_norm / nup[ind_slow]
                term1 = (
                    ((p1 + 2.0) * ((anisotropy_term[ind_slow] ** (-p1 / 2.0)) / Pp1))
                    * (nu_nup[ind_slow] ** (-(p1 + 4.0) / 2.0))
                    * (GaIx_p1(x2) - GaIx_p1(x1))
                )
                term2 = (
                    (
                        (p2 + 2.0)
                        * (g2 ** (p2 - p1))
                        * ((anisotropy_term[ind_slow] ** (-p2 / 2.0)) / Pp2)
                    )
                    * (nu_nup[ind_slow] ** (-(p2 + 4.0) / 2.0))
                    * (GaIx_p2(x3) - GaIx_p2(x2))
                )
                alphaI[ind_slow] = prefac_a * (term1 + term2)

            # fast cooling
            if np.any(ind_fast):
                p1 = 2.0
                p2 = p + 1.0
                g1 = gamma_c[ind_fast]
                g2 = gamma_m
                g3 = gamma_max
                x1 = nu_nup[ind_fast] / (g1 * g1)
                x2 = nu_nup[ind_fast] / (g2 * g2)
                x3 = nu_nup[ind_fast] / (g3 * g3)
                Pp1 = phi_norm_2
                Pp2 = phi_norm_pp1
                GIx_p1 = GIx_2
                GIx_p2 = GIx_pp1
                A_norm = 1.0 / (
                    (((g2 ** (1.0 - p1)) - (g1 ** (1.0 - p1))) / (1.0 - p1))
                    + ((g2 ** (p2 - p1)) * (((g3 ** (1.0 - p2)) - (g2 ** (1.0 - p2))) / (1.0 - p2)))
                )
                n = (
                    n_m[ind_fast]
                    * (gamma_c[ind_fast])
                    * (
                        (gamma_m * ((gamma_c[ind_fast] ** (-1.0)) - (gamma_m ** (-1.0))))
                        + ((gamma_m**p) * (((gamma_m ** (-p)) - (gamma_max ** (-p))) / p))
                    )
                )
                prefac_j = prefac_emis * n * A_norm * nup[ind_fast]
                term1 = (
                    ((anisotropy_term[ind_fast] ** (-p1 / 2.0)) / Pp1)
                    * (nu_nup[ind_fast] ** ((1.0 - p1) / 2.0))
                    * (GIx_p1(x2) - GIx_p1(x1))
                )
                term2 = (
                    (g2 ** (p2 - p1))
                    * ((anisotropy_term[ind_fast] ** (-p2 / 2.0)) / Pp2)
                    * (nu_nup[ind_fast] ** ((1.0 - p2) / 2.0))
                    * (GIx_p2(x3) - GIx_p2(x2))
                )
                jI[ind_fast] = prefac_j * (term1 + term2)

                GaIx_p1 = GaIx_2
                GaIx_p2 = GaIx_pp1
                prefac_a = prefac_absorp * n * A_norm / nup[ind_fast]
                term1 = (
                    ((p1 + 2.0) * ((anisotropy_term[ind_fast] ** (-p1 / 2.0)) / Pp1))
                    * (nu_nup[ind_fast] ** (-(p1 + 4.0) / 2.0))
                    * (GaIx_p1(x2) - GaIx_p1(x1))
                )
                term2 = (
                    (
                        (p2 + 2.0)
                        * (g2 ** (p2 - p1))
                        * ((anisotropy_term[ind_fast] ** (-p2 / 2.0)) / Pp2)
                    )
                    * (nu_nup[ind_fast] ** (-(p2 + 4.0) / 2.0))
                    * (GaIx_p2(x3) - GaIx_p2(x2))
                )
                alphaI[ind_fast] = prefac_a * (term1 + term2)

            alphaI = np.nan_to_num(alphaI, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            alphaI = np.maximum(alphaI, 0.0)

            if quantity == "jI":
                return jI
            if quantity == "alphaI":
                return alphaI

        return {}


###################################################
# post-processing functions


def convert_units(model, I_nu, *xy, output_units="luminosity", D=None, frequency=None):
    """
    Returns an image with pixel values provided in the requested units.
    The model argument must be an instance of the JetModel class.

    For flux units, the distance D to the source must be specified in Mpc.
    For brightness temperature (Tb) units, the frequency must be specified in GHz.

    Understood output units: luminosity, flux, Tb

    """

    # brightness temperature doesn't require any knowledge of pixel sizes
    if output_units == "Tb":
        # check that the frequency is provided
        if frequency is None:
            raise Exception(
                "For brightness temperature (Tb) units, the frequency must be specified in GHz."
            )

        # brightness temperature, in K
        Tb = (3.255e18) * I_nu / (frequency**2.0)

        return Tb

    # otherwise, pixel size matters
    if (
        (model.x_im_1D_input is not None)
        | (model.y_im_1D_input is not None)
        | (model.z_im_1D_input is not None)
    ):
        print("Warning: using custom input grids may cause issues with unit conversion.")

    # if the user doesn't pass a grid, use the one stored in the model
    if len(xy) == 0:
        # determine pixel sizes in x-direction
        if model.use_log_xgrid:
            dx = np.zeros_like(model.x_im_1D)
            for i in range(len(model.x_im_1D)):
                if i == 0:
                    xhi = np.sqrt(model.x_im_1D[i + 1] / model.x_im_1D[i]) * model.x_im_1D[i]
                    xlo = np.sqrt(model.x_im_1D[i] / model.x_im_1D[i + 1]) * model.x_im_1D[i]
                    dxhere = xhi - xlo
                elif (model.x_im_1D[i - 1] > 0.0) & (i < (len(model.x_im_1D) - 1)):
                    xhi = np.sqrt(model.x_im_1D[i + 1] / model.x_im_1D[i]) * model.x_im_1D[i]
                    xlo = np.sqrt(model.x_im_1D[i - 1] / model.x_im_1D[i]) * model.x_im_1D[i]
                    dxhere = xhi - xlo
                elif (model.x_im_1D[i - 1] > 0.0) & (i == (len(model.x_im_1D) - 1)):
                    xhi = np.sqrt(model.x_im_1D[i] / model.x_im_1D[i - 1]) * model.x_im_1D[i]
                    xlo = np.sqrt(model.x_im_1D[i - 1] / model.x_im_1D[i]) * model.x_im_1D[i]
                    dxhere = xhi - xlo
                elif (model.x_im_1D[i + 1] < 0.0) & (i > 0):
                    xhi = np.sqrt(model.x_im_1D[i + 1] / model.x_im_1D[i]) * model.x_im_1D[i]
                    xlo = np.sqrt(model.x_im_1D[i - 1] / model.x_im_1D[i]) * model.x_im_1D[i]
                    dxhere = xhi - xlo
                if model.x_im_1D[i] == model.xmin:
                    xhi = np.sqrt(model.x_im_1D[i + 1] / model.x_im_1D[i]) * model.x_im_1D[i]
                    xlo = 0.0
                    dxhere = xhi - xlo
                if model.x_im_1D[i] == -model.xmin:
                    xhi = 0.0
                    xlo = np.sqrt(model.x_im_1D[i - 1] / model.x_im_1D[i]) * model.x_im_1D[i]
                    dxhere = xhi - xlo
                dx[i] = dxhere

        else:
            dx = np.mean(np.diff(model.x_im_1D)) + np.zeros_like(model.x_im_1D)

        # determine pixel sizes in y-direction
        if model.use_log_ygrid:
            dy = np.zeros_like(model.y_im_1D)
            for i in range(len(model.y_im_1D)):
                if i == 0:
                    yhi = np.sqrt(model.y_im_1D[i + 1] / model.y_im_1D[i]) * model.y_im_1D[i]
                    ylo = np.sqrt(model.y_im_1D[i] / model.y_im_1D[i + 1]) * model.y_im_1D[i]
                    dyhere = yhi - ylo
                elif (model.y_im_1D[i - 1] > 0.0) & (i < (len(model.y_im_1D) - 1)):
                    yhi = np.sqrt(model.y_im_1D[i + 1] / model.y_im_1D[i]) * model.y_im_1D[i]
                    ylo = np.sqrt(model.y_im_1D[i - 1] / model.y_im_1D[i]) * model.y_im_1D[i]
                    dyhere = yhi - ylo
                elif (model.y_im_1D[i - 1] > 0.0) & (i == (len(model.y_im_1D) - 1)):
                    yhi = np.sqrt(model.y_im_1D[i] / model.y_im_1D[i - 1]) * model.y_im_1D[i]
                    ylo = np.sqrt(model.y_im_1D[i - 1] / model.y_im_1D[i]) * model.y_im_1D[i]
                    dyhere = yhi - ylo
                elif (model.y_im_1D[i + 1] < 0.0) & (i > 0):
                    yhi = np.sqrt(model.y_im_1D[i + 1] / model.y_im_1D[i]) * model.y_im_1D[i]
                    ylo = np.sqrt(model.y_im_1D[i - 1] / model.y_im_1D[i]) * model.y_im_1D[i]
                    dyhere = yhi - ylo
                if model.y_im_1D[i] == model.ymin:
                    yhi = np.sqrt(model.y_im_1D[i + 1] / model.y_im_1D[i]) * model.y_im_1D[i]
                    ylo = 0.0
                    dyhere = yhi - ylo
                if model.y_im_1D[i] == -model.ymin:
                    yhi = 0.0
                    ylo = np.sqrt(model.y_im_1D[i - 1] / model.y_im_1D[i]) * model.y_im_1D[i]
                    dyhere = yhi - ylo
                dy[i] = dyhere

        else:
            dy = np.mean(np.diff(model.y_im_1D)) + np.zeros_like(model.y_im_1D)

    # otherwise, unpack the user-specified coordinates
    else:
        (x_im_1D, y_im_1D) = xy
        dx = np.concatenate(([x_im_1D[1] - x_im_1D[0]], np.diff(x_im_1D)))
        dy = np.concatenate(([y_im_1D[1] - y_im_1D[0]], np.diff(y_im_1D)))

    # determine pixel areas, in rg^2
    dA = np.zeros_like(I_nu)
    for i in range(len(dx)):
        for j in range(len(dy)):
            dA[j, i] = dx[i] * dy[j]

    # convert to cgs
    dA *= model.rg * model.rg

    # luminosity density, in cgs
    Lnu = dA * I_nu

    if output_units == "luminosity":
        return Lnu

    # if user wants flux units
    if output_units == "flux":
        # check that the distance is provided
        if D is None:
            raise Exception(
                "For flux units, the distance D to the source must be specified in Mpc."
            )

        # convert distance from Mpc to cm
        D_cm = D * (3.086e24)

        # flux density, in cgs
        Snu = Lnu / (D_cm**2.0)

        return Snu


def interp_to_regular_grid(
    I_nu,
    x,
    y,
    *,
    x_new=None,
    y_new=None,
    nx=512,
    ny=512,
    method="linear",
    fill_value=np.nan,
    bounds_error=False,
):
    """
    Interpolate I_nu onto a regular rectangular grid.
    The method, fill_value, and bounds_error arguments get passed directly to
    scipy's RegularGridInterpolator.

    Assumes: I_nu.shape == (len(y), len(x)) and I_nu[j, i] is at (x[i], y[j]).

    Input method can be "linear" or "nearest"

    Provide either:
      - x_new and y_new (1D arrays of target grid centers), OR
      - nx and ny to auto-make uniform grids spanning [min(x), max(x)] and [min(y), max(y)].
    """

    I_nu = np.asarray(I_nu)
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if I_nu.shape != (len(y), len(x)):
        raise ValueError(f"I_nu shape {I_nu.shape} must be (len(y), len(x)) = {(len(y), len(x))}")

    # ensure ascending axes (required by RegularGridInterpolator)
    x_order = np.argsort(x)
    y_order = np.argsort(y)
    x_sorted = x[x_order]
    y_sorted = y[y_order]
    I_sorted = I_nu[np.ix_(y_order, x_order)]

    # build target grid if not provided
    if x_new is None:
        x_new = np.linspace(x_sorted.min(), x_sorted.max(), int(nx))
    else:
        x_new = np.asarray(x_new).ravel()

    if y_new is None:
        y_new = np.linspace(y_sorted.min(), y_sorted.max(), int(ny))
    else:
        y_new = np.asarray(y_new).ravel()

    # interpolator expects points to be in the same axis order as (y_sorted, x_sorted)
    interp = RegularGridInterpolator(
        (y_sorted, x_sorted),
        I_sorted,
        method=method,
        bounds_error=bounds_error,
        fill_value=fill_value,
    )

    # evaluate on the new grid
    Xn, Yn = np.meshgrid(x_new, y_new, indexing="xy")
    pts = np.column_stack([Yn.ravel(), Xn.ravel()])
    I_new = interp(pts).reshape(len(y_new), len(x_new))

    return x_new, y_new, I_new


def export_fits(
    filename,
    I_new,
    x_new,
    y_new,
    *,
    source_name=None,  # -> OBJECT
    ra_deg=None,  # -> OBJRA
    dec_deg=None,  # -> OBJDEC
    observing_frequency_hz=None,  # -> OBSFREQ (Hz)
    bunit=None,  # -> BUNIT (e.g. "Jy/beam", "K")
    extra_header=None,  # dict: {"KEY": value, ...}
    write_axes_extension=True,  # store x_new/y_new vectors in a table extension too
    overwrite=True,
):
    """
    Write a 2D image to FITS where x_new/y_new are angular offsets on a regular grid.
    Default assumption is that the offsets are specified in radians.

    Assumes:
      - I_new.shape == (len(y_new), len(x_new))
      - x_new and y_new are 1D arrays of pixel centers on a uniform grid
      - no rotation (i.e., the pixel grid is aligned with coordinate axes)

    """

    I_new = np.asarray(I_new)
    x_new = np.asarray(x_new).ravel()
    y_new = np.asarray(y_new).ravel()

    # convert from radians to degrees
    x_new *= 180.0 / np.pi
    y_new *= 180.0 / np.pi

    if I_new.ndim != 2:
        raise ValueError(f"I_new must be 2D, got shape {I_new.shape}")
    if I_new.shape != (len(y_new), len(x_new)):
        raise ValueError(
            f"I_new shape {I_new.shape} must be (len(y_new), len(x_new)) = {(len(y_new), len(x_new))}"
        )

    # ensure uniform spacing (required for CRVAL/CDELT WCS)
    def _uniform_step(arr, name, rtol=1e-7, atol=0.0):
        if len(arr) < 2:
            raise ValueError(f"{name} must have at least 2 points to define pixel scale.")
        d = np.diff(arr)
        step = float(np.median(d))
        if not np.allclose(d, step, rtol=rtol, atol=atol):
            raise ValueError(
                f"{name} is not uniformly spaced (required for simple CRVAL/CDELT WCS). "
                f"Resample to a uniform grid first."
            )
        return step

    dx = _uniform_step(x_new, "x_new")
    dy = _uniform_step(y_new, "y_new")

    ny, nx = I_new.shape

    # reference pixel at image center (FITS is 1-indexed)
    crpix1 = (nx + 1) / 2.0
    crpix2 = (ny + 1) / 2.0
    ix = int(np.round(crpix1 - 1))
    iy = int(np.round(crpix2 - 1))

    # build linear WCS: (x_offset, y_offset)
    w = WCS(naxis=2)
    w.wcs.crpix = [crpix1, crpix2]
    w.wcs.crval = [float(x_new[ix]), float(y_new[iy])]
    w.wcs.cdelt = [dx, dy]
    w.wcs.pc = np.eye(2)
    w.wcs.ctype = ["XOFFSET", "YOFFSET"]
    w.wcs.cunit = ["deg", "deg"]

    header = w.to_header()

    # header info
    header["DATE"] = Time.now().isot

    # object / coordinate metadata
    if source_name is not None:
        header["OBJECT"] = str(source_name)

    if ra_deg is not None:
        header["OBJRA"] = float(ra_deg)
        header["RADECSYS"] = "ICRS"
        header["EQUINOX"] = 2000.0

    if dec_deg is not None:
        header["OBJDEC"] = float(dec_deg)
        header["RADECSYS"] = "ICRS"
        header["EQUINOX"] = 2000.0

    # spectral / units
    if observing_frequency_hz is not None:
        header["OBSFREQ"] = float(observing_frequency_hz)  # Hz
    if bunit is not None:
        header["BUNIT"] = str(bunit)

    # other stats
    finite = np.isfinite(I_new)
    if np.any(finite):
        header["DATAMIN"] = float(np.nanmin(I_new))
        header["DATAMAX"] = float(np.nanmax(I_new))

    # extra header material
    if extra_header:
        for k, v in dict(extra_header).items():
            header[str(k).upper()] = v

    # primary image HDU
    hdus = [fits.PrimaryHDU(data=I_new.astype(np.float32, copy=False), header=header)]

    # store the exact axis vectors
    if write_axes_extension:
        col_x = fits.Column(
            name="x_offset_centers", format="D", unit="deg", array=x_new.astype(float)
        )
        col_y = fits.Column(
            name="y_offset_centers", format="D", unit="deg", array=y_new.astype(float)
        )
        axes_hdu = fits.BinTableHDU.from_columns([col_x, col_y], name="AXES")
        hdus.append(axes_hdu)

    fits.HDUList(hdus).writeto(filename, overwrite=overwrite)


###################################################
