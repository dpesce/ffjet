###################################################
# imports and etc.

import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import gamma as _gammafn, hyp2f1, kv
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time

# optional progress bar import
try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

# compiled radiative-transfer kernel (requires numba); the pure-numpy path in
# JetModel._make_image_numpy remains available as the reference implementation
try:
    from . import _kernel as _kern

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only when numba is absent
    _kern = None
    HAS_NUMBA = False

# suppress some numpy warnings
np.seterr(divide="ignore", invalid="ignore")


def numba_available():
    """True if the compiled (numba) radiative-transfer kernel can be used."""
    return HAS_NUMBA


def set_num_threads(n):
    """Set the number of threads used by the compiled kernel (default: all cores)."""
    if not HAS_NUMBA:
        raise ImportError("numba is not installed; there is no thread pool to configure")
    import numba

    numba.set_num_threads(int(n))


def get_num_threads():
    """Number of threads the compiled kernel will use."""
    if not HAS_NUMBA:
        return 1
    import numba

    return numba.get_num_threads()

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
# functions from Gelles+


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

    denomtheta = (a * a) + 2 * (r * r) + (a * a) * np.cos(2 * theta)
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


def _energy_const(a, r0, theta0, omega):
    """
    A = sqrt(-coef0/coef1) at the stagnation point: the constant that fixes the
    energy-conserving parallel boost.  A function of the field line alone, so it is
    tabulated alongside (r_stag, theta_stag); see JetModel._build_stagnation_surface.
    """
    a2 = a * a
    cth2 = np.cos(theta0) ** 2.0
    sth2 = np.sin(theta0) ** 2.0
    r2 = r0 * r0
    rho2 = r2 + (a2 * cth2)
    Delta = r2 - (2.0 * r0) + a2
    Sigma = ((r2 + a2) ** 2.0) - (a2 * Delta * sth2)
    g00 = ((a2 * sth2) - Delta) / rho2
    g03 = -2.0 * a * r0 * sth2 / rho2
    g33 = Sigma * sth2 / rho2
    ginv_denom = (g00 * g33) - (g03 * g03)
    ginv00 = g33 / ginv_denom
    ginv03 = -g03 / ginv_denom
    ginv33 = g00 / ginv_denom
    gtphifac = g03 + (g33 * omega)
    gttfac = g00 + (g03 * omega)
    coef0 = (g00 + (omega * (2.0 * g03 + g33 * omega))) ** 2.0
    coef1 = (ginv33 * (gtphifac * gtphifac)) + (
        2.0 * ginv03 * gtphifac * gttfac + ginv00 * (gttfac * gttfac)
    )
    return np.sqrt(-coef0 / coef1)


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
    gamma_inf=None,
    Aconst=None,
    retbunit=False,
    retqty=False,
    eps=-1,
):
    """
    drift frame velocity for a given EM field in BL
      - If called with no extra positional args: uses the original (bfield-based) implementation
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
        if Aconst is None:
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

        eta3 = np.zeros_like(r)

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
    eta1 = 0.0
    eta2 = 0.0
    eta3 = 0.0

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

    if gamma_inf:  # approximate MHD gamma by summing gamma_FF and gamma_max in series
        pval0 = 2.0
        gamma_inf = gamma_inf * np.ones_like(gamma)
        gammaeff = (1 / gamma_inf**pval0 + 1 / gamma**pval0) ** (-1 / pval0)

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

    All frequency-independent setup (stagnation surface, jet-power normalization,
    synchrotron integral tables, grids) is done once at construction.  Images are
    produced by make_image(), which has two interchangeable back ends:

      backend="numba"  compiled per-ray kernel (default when numba is installed);
                       parallel over image pixels, work proportional to the number
                       of grid cells inside the jet
      backend="numpy"  the original vectorized numpy loop over depth slices; slower
                       by an order of magnitude but dependency-free, and kept as
                       the reference implementation

    The two agree to floating-point roundoff.  For loops over frequency (SEDs), call
    precompute_state() once; subsequent make_image() calls then only evaluate the
    frequency-dependent synchrotron coefficients and the transfer integral.

    Typical usage:
        model = JetModel(m=..., a=..., inc=..., mdot=..., Nx=..., Ny=..., Nz=..., s=..., p=..., ...)
        model.precompute_state()          # optional; speeds up frequency loops
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
        gamma_inf=6.0,
        gamma_m=30.0,
        gamma_max=1.0e8,
        p_eta=2.0,
        gammabeta_suppression=0.5,
        DTYPE=np.float64,
        stokes="I",
        backend="auto",
        n_stagnation=1000,
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
        self.gamma_inf = float(gamma_inf)
        self.gamma_m = float(gamma_m)
        self.gamma_max = float(gamma_max)
        self.p_eta = float(p_eta)
        self.gammabeta_suppression = float(gammabeta_suppression)
        self.DTYPE = DTYPE

        self.stokes = str(stokes).upper()
        self.n_stagnation = int(n_stagnation)

        # radiative-transfer back end: "auto" (numba if installed, else numpy),
        # "numba", or "numpy"; can be overridden per call in make_image()
        if backend not in ("auto", "numba", "numpy"):
            raise ValueError(f"backend must be 'auto', 'numba' or 'numpy', got {backend!r}")
        self.backend = backend

        # lazily-built caches used by the compiled back end
        self._intervals = None  # per-ray ranges of z-cells inside the jet
        self._state = None  # per-cell frequency-independent state (see precompute_state)
        self._ftab = None  # 2-D table of the (r,theta)-only physics (see build_field_table)

        ####################
        # input validation
        #
        # Each of these otherwise fails silently or with an exception from deep inside the
        # setup, so they are checked here where the offending argument can be named.

        if not (self.m > 0.0):
            raise ValueError(f"m={self.m} must be positive (black hole mass, in solar masses)")
        if not (self.mdot > 0.0):
            raise ValueError(f"mdot={self.mdot} must be positive")
        # |a| < 1 is already enforced, with this same message, by omega_BZpower below
        if self.s >= 1.0:
            raise ValueError(
                f"s={self.s} must be less than 1: the field-line exponent nu = 2 - 2s would be "
                f"zero or negative, and the jet geometry r ~ (1 - |cos theta|)^(-1/nu) is undefined"
            )
        if self.s <= 0.0:
            raise ValueError(
                f"s={self.s} must be positive: nu = 2 - 2s >= 2 makes the jet boundary reach the "
                f"pole at finite radius and no stagnation surface exists on most field lines"
            )
        if not (self.eta > 0.0):
            raise ValueError(
                f"eta={self.eta} must be positive; the pitch-angle normalization "
                f"int_0^1 [1 + (eta - 1) mu^2]^(-p_eta/2) dmu diverges as eta -> 0 for p_eta >= 1"
            )
        if not (self.gamma_m > 1.0):
            raise ValueError(f"gamma_m={self.gamma_m} must exceed 1")
        if not (self.gamma_max > self.gamma_m):
            raise ValueError(
                f"gamma_max={self.gamma_max} must exceed gamma_m={self.gamma_m}; otherwise the "
                f"electron normalization changes sign and the emissivity comes out negative"
            )
        if not (self.gamma_inf >= 1.0):
            raise ValueError(f"gamma_inf={self.gamma_inf} must be at least 1")
        if not (0.0 <= self.jet_cutout_fraction < 1.0):
            raise ValueError(
                f"jet_cutout_fraction={self.jet_cutout_fraction} must be in [0, 1)"
            )
        if self.p_eta < 0.0:
            raise ValueError(f"p_eta={self.p_eta} must be non-negative")

        ####################
        # derived quantities

        self.nu = 2.0 - (2.0 * self.s)
        self.rH = 1.0 + np.sqrt(1.0 - (self.a * self.a))

        self.inc_rad = np.pi - (self.inc * np.pi / 180.0)
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
        """
        Stagnation surface, tabulated on n_stagnation field lines (labelled by the
        polar angle theta_H at which they thread the horizon, log-spaced in theta_H).

        The stagnation point is where the field-parallel derivative of the corotation
        energy N_co = -(g_tt + 2 g_tphi Omega_F + g_phiphi Omega_F^2) vanishes
        (Gelles et al. 2025).  Omega_F is the angular velocity of the field line being
        traced and is therefore held fixed along it.

        The field line is parameterized by *radius*, not by polar angle:

            1 - cos theta(r) = (1 - cos theta_H) (r_H / r)^nu,

        which is evaluated directly and so cannot overflow.  (Parameterizing by theta
        instead requires r = r_H [(1-cos theta_H)/(1-cos theta)]^(1/nu), which overflows
        to inf for small theta -- badly so when 1/nu is large.  Nderiv(inf, ...) is NaN,
        `NaN > 0` is False, and a bisection then walks towards the overflowing end and
        returns r_stag = inf with no error.)

        The root is bracketed on a coarse logarithmic radius grid and then bisected in
        log r.  All field lines are handled together as arrays, so a fine table costs
        ~30 ms.  If any field line has no bracketed root, a RuntimeError names it rather
        than letting a non-finite stagnation radius through.
        """
        a = self.a
        nu = self.nu
        rH = self.rH
        bf = self.bf
        n = int(self.n_stagnation)
        if n < 4:
            raise ValueError("n_stagnation must be at least 4")

        theta_H = 10.0 ** np.linspace(-5.0, np.log10(np.pi / 2.0), n)
        psi_H = psiBZpower(rH, theta_H, nu)  # stream function of each field line
        Omega_H = omega_BZpower(0, psi_H, a, nu)  # its (constant) angular velocity
        # 1 - cos(theta_H), by the half-angle form so that small theta_H keeps its digits
        omc_H = 2.0 * np.sin(0.5 * theta_H) ** 2

        def _theta_of_r(r):
            """Polar angle on each field line at radius r (r broadcast against theta_H)."""
            omc = np.clip(omc_H * (rH / r) ** nu, 0.0, 2.0)
            return 2.0 * np.arcsin(np.sqrt(0.5 * omc)), omc

        def _Nprime(r):
            th, _ = _theta_of_r(r)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                return Nderiv(r, th, a, Omega_H, 1.0, bf)

        # bracket: scan outwards from just outside the horizon and take the innermost
        # sign change of N' on each field line
        r_lo_scan = rH * (1.0 + 1.0e-9)
        lo, hi = np.log10(r_lo_scan), np.log10(self._STAG_R_MAX)
        n_scan = max(64, round((hi - lo) * self._STAG_SCAN_PER_DECADE))
        r_scan = (10.0 ** np.linspace(lo, hi, n_scan)).reshape(n_scan, 1)
        sgn = np.sign(_Nprime(r_scan))
        sgn[~np.isfinite(sgn)] = 0.0
        change = (sgn[:-1] * sgn[1:]) < 0.0
        found = change.any(axis=0)
        if not found.all():
            bad = np.flatnonzero(~found)
            raise RuntimeError(
                f"the stagnation surface has no bracketed root on {bad.size} of {n} field "
                f"lines (first at theta_H = {theta_H[bad[0]]:.4g} rad) for a={a}, s={self.s} "
                f"(nu={nu}); the force-free jet solution is not usable at these parameters"
            )
        k = np.argmax(change, axis=0)  # innermost sign change
        r_a = r_scan[:, 0][k]
        r_b = r_scan[:, 0][k + 1]

        # bisect in log r
        f_a = _Nprime(r_a)
        for _ in range(60):
            r_c = np.sqrt(r_a * r_b)
            f_c = _Nprime(r_c)
            same = np.sign(f_c) == np.sign(f_a)
            r_a = np.where(same, r_c, r_a)
            f_a = np.where(same, f_c, f_a)
            r_b = np.where(same, r_b, r_c)
        r_c = np.sqrt(r_a * r_b)
        theta_c, _ = _theta_of_r(r_c)

        self.thetahorizon_arr = theta_H
        self.rstag_arr = r_c
        self.tstag_arr = theta_c
        # The table is looked up by omc_fp = 1 - |cos(theta_fp)| rather than by theta_fp.
        # That quantity is psi / rH^nu, which the geometry already provides without any
        # inverse trigonometry, and it folds the two hemispheres together automatically
        # (it is symmetric about theta_fp = pi/2).  Ascending, like theta_H.
        self.omchorizon_arr = omc_H

        # Roots are found on field lines spaced logarithmically in theta_H, which is the
        # right distribution for the root find but leaves the lookup abscissa unevenly
        # spaced -- so a lookup has to bisect.  Resampling onto a grid uniform in
        # log10(omc) lets the kernel index it directly, with one log10 and no search, and
        # lets both stagnation columns share a single bracket.  The resampled grid is
        # finer than the original by _STAG_LUT_REFINE, so it reproduces the interpolant it
        # is built from rather than coarsening it.
        n_lut = int(self._STAG_LUT_REFINE * n)
        lut_lo = np.log10(omc_H[0])
        lut_hi = np.log10(omc_H[-1])
        lut_log = np.linspace(lut_lo, lut_hi, n_lut)
        lut_x = 10.0**lut_log
        self._stag_lut_x = lut_x
        self._stag_lut_r = np.interp(lut_x, omc_H, r_c)
        self._stag_lut_t = np.interp(lut_x, omc_H, theta_c)
        # The energy-conservation constant of the parallel boost depends only on the
        # field line -- through (r_stag, theta_stag, Omega_F) -- so it belongs in this
        # table too rather than being rebuilt, with a square root and a dozen divisions,
        # at every cell.  Omega_F on the lookup grid follows from the abscissa itself:
        # cos(theta_fp) = 1 - omc_fp.
        omega_lut = a / (4.0 + 8.0 / (2.0 - lut_x))
        self._stag_lut_a = _energy_const(a, self._stag_lut_r, self._stag_lut_t, omega_lut)
        self._stag_lut_logx0 = float(lut_lo)
        self._stag_lut_invdlog = 1.0 / (lut_log[1] - lut_log[0])
        self._stag_lut_kmax = float(n_lut - 2)

    def _stagnation_omc(self, omc_fp):
        """
        (r, theta) of the stagnation point, indexed by omc_fp = 1 - |cos(theta_fp)|.

        Direct index into the uniform-in-log10(omc) lookup grid built above -- the same
        arithmetic the compiled kernel uses, so the two back ends stay in lockstep.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            f = (np.log10(omc_fp) - self._stag_lut_logx0) * self._stag_lut_invdlog
        f = np.clip(np.nan_to_num(f, nan=0.0), 0.0, self._stag_lut_kmax)
        k = f.astype(np.intp)
        # the log grid supplies the bracket in O(1); the weight is taken linearly in omc,
        # which is the interpolation the table was tabulated for
        x_lut = self._stag_lut_x
        w = (np.clip(omc_fp, x_lut[0], x_lut[-1]) - x_lut[k]) / (x_lut[k + 1] - x_lut[k])
        r_lut, t_lut = self._stag_lut_r, self._stag_lut_t
        return (r_lut[k] + w * (r_lut[k + 1] - r_lut[k]),
                t_lut[k] + w * (t_lut[k + 1] - t_lut[k]))

    def _stagnation_aconst(self, omc_fp):
        """Energy-conservation constant of the parallel boost, off the same lookup grid."""
        with np.errstate(divide="ignore", invalid="ignore"):
            f = (np.log10(omc_fp) - self._stag_lut_logx0) * self._stag_lut_invdlog
        f = np.clip(np.nan_to_num(f, nan=0.0), 0.0, self._stag_lut_kmax)
        k = f.astype(np.intp)
        x_lut = self._stag_lut_x
        w = (np.clip(omc_fp, x_lut[0], x_lut[-1]) - x_lut[k]) / (x_lut[k + 1] - x_lut[k])
        a_lut = self._stag_lut_a
        return a_lut[k] + w * (a_lut[k + 1] - a_lut[k])

    def stagnation(self, theta_fp):
        """(r, theta) of the stagnation point on the field line with footpoint theta_fp."""
        return self._stagnation_omc(1.0 - np.abs(np.cos(theta_fp)))

    def _jet_edge_cylindrical(self, z, logR_lo=-1.0, logR_hi=10.0, n_bisect=80):
        """
        Cylindrical radius of the jet boundary at height z, by bisection in log R.

        A point (R, z) belongs to the jet when r = sqrt(R^2 + z^2) is inside the field line that
        threads the horizon at theta_H = pi/2, i.e. when r <= rH (1 - cos theta)^(-1/nu) with
        cos theta = z / r.  At fixed z that condition is monotonic in R, so a bisection brackets
        the boundary to machine precision; the bracket spans R = 0.1 to 10^10 r_g, which contains
        the boundary for every usable (a, s).
        """
        rH, nu = self.rH, self.nu

        def inside(logR):
            R = 10.0**logR
            r = np.sqrt((R * R) + (z * z))
            theta = np.arccos(z / r)
            with np.errstate(over="ignore", divide="ignore"):
                r_jet = rH * ((1.0 / (1.0 - np.cos(theta))) ** (1.0 / nu))
            return bool((r <= r_jet) and (r > rH))

        if not inside(logR_lo):
            raise RuntimeError(
                f"the jet boundary at z={z:.3g} r_g lies inside R=10^{logR_lo:g} r_g for a={self.a}, "
                f"s={self.s} (nu={self.nu}); the jet-power normalization cannot be computed"
            )
        if inside(logR_hi):
            raise RuntimeError(
                f"the jet boundary at z={z:.3g} r_g lies beyond R=10^{logR_hi:g} r_g for a={self.a}, "
                f"s={self.s} (nu={self.nu}); the jet-power normalization cannot be computed"
            )
        lo, hi = float(logR_lo), float(logR_hi)
        for _ in range(int(n_bisect)):
            mid = 0.5 * (lo + hi)
            if inside(mid):
                lo = mid
            else:
                hi = mid
        return 10.0**lo

    # determine the scaling factor necessary to ensure that the jet has the correct total power
    def _build_poynting_scaling(self):
        a = self.a
        s = self.s
        nu = self.nu
        rH = self.rH
        rg = self.rg
        Pjet = self.Pjet

        # The integration grid ends *at* the jet boundary rather than at a fixed outer radius.
        # The boundary radius grows as z^s, so with a fixed grid it moves across grid points as
        # the parameters change, and the integral -- hence this normalization, and with it every
        # flux the model produces -- jumps by several per cent each time a point crosses it: a
        # +-5% sawtooth in s with a period of ~0.0024, which turns a likelihood in s into a comb.
        Nrescale = self._POYNTING_N
        logzrescale = self._POYNTING_LOGZ
        zrescale = 10.0**logzrescale
        logR_edge = np.log10(self._jet_edge_cylindrical(zrescale))
        zhere = zrescale * np.ones((Nrescale, 1))
        Rhere = (10.0 ** np.linspace(-1.0, logR_edge, Nrescale)).reshape((Nrescale, 1))

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
            gamma_inf=None,
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

    # ------------------------------------------------------------------
    # synchrotron integrals
    #
    # These are computed at initialization rather than read from disk.  Following
    # Dexter (2016), the three kernels are
    #
    #     F(x) = x int_x^inf K_{5/3}(y) dy          (Stokes I)
    #     G(x) = x K_{2/3}(x)                       (Stokes Q)
    #     H(x) = int_x^inf K_{1/3}(y) dy + x K_{1/3}(x)   (Stokes V)
    #
    # and the tabulated quantities are, for emission (Dexter Eqs. A26-A28) and
    # absorption (Eqs. A44-A46),
    #
    #     G_I^(p)(x)  = int_x^inf z^((p-3)/2) F(z) dz
    #     G_Q^(p)(x)  = int_x^inf z^((p-3)/2) G(z) dz
    #     G_V^(p)(x)  = int_x^inf z^((p-2)/2) H(z) dz
    #     Ga_I^(p)(x) = int_x^inf z^((p-2)/2) F(z) dz = G_I^(p+1)(x)
    #     Ga_Q^(p)(x) = int_x^inf z^((p-2)/2) G(z) dz = G_Q^(p+1)(x)
    #     Ga_V^(p)(x) = int_x^inf z^((p-1)/2) H(z) dz = G_V^(p+1)(x)
    #
    # Every one of these is int_x^inf z^m kernel(z) dz for some m, and with the
    # substitution z = 10^u it becomes ln(10) int_u^inf z^(m+1) kernel(z) du, so a
    # single reverse cumulative trapezoid on a uniform u grid yields all tabulated x
    # at once.  Building the tables this way takes a few ms and is accurate to ~1e-6
    # against the closed form
    #
    #     int_0^inf x^m F(x) dx = 2^(m+1) Gamma(m/2+2/3) Gamma(m/2+7/3) / (m+2)
    #     int_0^inf x^m G(x) dx = 2^m Gamma(m/2+2/3) Gamma(m/2+4/3)
    #     int_0^inf x^m H(x) dx = 2^m Gamma(m/2+5/6) Gamma(m/2+7/6) (m+2)/(m+1)
    #
    # (the previously shipped tables were 0.5-1.3% low, because they were built by
    # integrating a coarsely tabulated F/G/H rather than the kernels themselves).
    # ------------------------------------------------------------------

    _TAB_DEX_FINE = 0.001            # internal resolution; plateau error ~1e-6
    _TAB_DEX_STORE = 0.01            # stored resolution; interpolation error ~5e-5
    _TAB_LOGX_FINE = (-22.0, 3.2)    # internal log10(x) range, padded on both ends
    _TAB_LOGX_STORE = (-20.0, 2.8)   # stored log10(x) range

    # stagnation-surface root search: how far out to look for a bracket, and how finely.
    # Near-axis field lines of a slowly spinning, wide (small-s) jet stagnate very far out
    # -- r_stag ~ 7e13 r_g at a=0.01, s=0.05 -- so the scan has to reach well beyond any
    # imaging domain.
    _STAG_LUT_REFINE = 4  # lookup-grid refinement over n_stagnation
    _STAG_R_MAX = 1.0e20
    _STAG_SCAN_PER_DECADE = 40

    # jet-power normalization (_build_poynting_scaling): the height at which the Poynting flux
    # is integrated across the jet, and the number of points of the integration grid -- which
    # ends exactly at the jet boundary, so that the result is a smooth function of the parameters
    _POYNTING_LOGZ = 8.0
    _POYNTING_N = 4000

    # pole-on handling: |sin(i)| below this counts as exactly along the jet axis, and a
    # ray-centring offset more than this multiple of zmax earns a warning
    _POLE_ON_SIN_TOL = 1.0e-12
    _POLE_ON_ZJ_WARN = 30.0

    # (Stokes parameter, kernel, exponent m) for each tabulated family
    _TAB_FAMILIES = {
        "GI": ("I", "F", lambda p: (p - 3.0) / 2.0),
        "GQ": ("Q", "G", lambda p: (p - 3.0) / 2.0),
        "GV": ("V", "H", lambda p: (p - 2.0) / 2.0),
        "GaI": ("I", "F", lambda p: (p - 2.0) / 2.0),
        "GaQ": ("Q", "G", lambda p: (p - 2.0) / 2.0),
        "GaV": ("V", "H", lambda p: (p - 1.0) / 2.0),
    }

    def _load_synchrotron_tables(self):
        p = self.p
        if p <= 2.0:
            raise ValueError(
                f"p={p} must exceed 2; the nonthermal electron normalization diverges as p -> 2."
            )

        stokes = self.stokes
        for s in stokes:
            if s not in ("I", "Q", "V"):
                raise ValueError(f"unrecognized Stokes parameter {s!r}; expected some of 'IQV'")
        if "I" not in stokes:
            raise ValueError("the Stokes I tables are always required")

        lo, hi = self._TAB_LOGX_FINE
        n = int(round((hi - lo) / self._TAB_DEX_FINE)) + 1
        u = np.linspace(lo, hi, n)
        du = u[1] - u[0]
        y = 10.0**u
        ln10 = np.log(10.0)

        def _cumulative_upper(f):
            """int_u^umax f du at every grid point, by reverse cumulative trapezoid."""
            f = np.where(np.isfinite(f), f, 0.0)
            seg = 0.5 * (f[1:] + f[:-1]) * du
            return np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])

        kernels = {"F": y * _cumulative_upper(ln10 * kv(5.0 / 3.0, y) * y)}
        if "Q" in stokes:
            kernels["G"] = y * kv(2.0 / 3.0, y)
        if "V" in stokes:
            kernels["H"] = _cumulative_upper(ln10 * kv(1.0 / 3.0, y) * y) + y * kv(1.0 / 3.0, y)

        # stored grid
        stride = int(round(self._TAB_DEX_STORE / self._TAB_DEX_FINE))
        i0 = int(round((self._TAB_LOGX_STORE[0] - lo) / self._TAB_DEX_FINE))
        i1 = int(round((self._TAB_LOGX_STORE[1] - lo) / self._TAB_DEX_FINE))
        idx = np.arange(i0, i1 + 1, stride)
        logx = u[idx]

        self._tab_logx0 = float(logx[0])
        self._tab_inv_dlogx = 1.0 / float(logx[1] - logx[0])
        self._tab_kmax = float(len(logx) - 2)

        cache = {}

        def _integral(kernel, m):
            """int_x^inf z^m kernel(z) dz, on the stored grid."""
            key = (kernel, round(m, 10))
            if key not in cache:
                integrand = ln10 * (y ** (m + 1.0)) * kernels[kernel]
                cache[key] = _cumulative_upper(integrand)[idx]
            return cache[key]

        for family, (stokes_label, kernel, exponent) in self._TAB_FAMILIES.items():
            if stokes_label not in stokes:
                continue
            for suffix, pval in (("_2", 2.0), ("_p", p), ("_pp1", p + 1.0)):
                name = family + suffix
                table = _integral(kernel, exponent(pval))
                setattr(self, "x_" + name, 10.0**logx)
                setattr(self, name, table)
                setattr(self, "logx_" + name, logx)
                setattr(self, "logG_" + name, np.log10(np.maximum(table, 1.0e-300)))

    # helper functions so the RT loop can use a simplified GIx_*/GaIx_* call style;
    # interpolation is linear in log(x)-log(G), with the small-x plateau extended to
    # the left and zero returned to the right of the tabulated range.  The stored grid
    # is uniform in log10(x), so the bracketing index is arithmetic rather than a binary
    # search and the cost does not depend on the table length.
    # --- small-argument tail of the synchrotron integrals -------------------------
    #
    # Every emissivity and absorption coefficient is a *difference* of the same tabulated
    # integral at two arguments, G(x_lo) - G(x_hi) with x_lo < x_hi.  As x -> 0 the
    # integral approaches a constant, so the difference is taken between two nearly equal
    # plateau values and is destroyed by cancellation -- and once both arguments round to
    # the same stored value it is exactly zero, which sets the coefficient to zero instead
    # of its correct small-x limit.  Measured against a cancellation-free reference, the
    # tabulated difference is wrong by 100% below x_hi ~ 1e-7 for GaI_(p+1), ~1e-8 for
    # GI_(p+1) and GaI_p, ~1e-9 for GaI_2 and ~1e-13 for GI_2 and GI_p.
    #
    # Below _X_TAIL the difference is evaluated analytically instead.  With
    # F(z) -> [4 pi / (sqrt(3) Gamma(1/3))] (z/2)^(1/3) as z -> 0,
    #
    #     G_m(x_lo) - G_m(x_hi) = int_{x_lo}^{x_hi} z^m F(z) dz
    #                           = C (x_hi^e - x_lo^e) / e,    e = m + 4/3,
    #
    # which has no cancellation and no table lookup.  Its own error is <= 7e-5 at
    # x_hi = 1e-6 and falls as x_hi^(2/3) below that -- more than thirty times smaller
    # than the grid-discretization error of any usable image.
    _X_TAIL = 1.0e-6
    _TAIL_C = 4.0 * np.pi / (np.sqrt(3.0) * _gammafn(1.0 / 3.0)) / 2.0 ** (1.0 / 3.0)

    def _G_diff(self, lookup, m, x_lo, x_hi):
        """G(x_lo) - G(x_hi) for x_lo <= x_hi, with the plateau handled analytically."""
        d = lookup(x_lo) - lookup(x_hi)
        small = x_hi <= self._X_TAIL
        if not np.any(small):
            return d
        e = m + (4.0 / 3.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            tail = (self._TAIL_C / e) * ((x_hi**e) - (x_lo**e))
        return np.where(small, tail, d)

    def _G_lookup(self, x, key):
        logG = getattr(self, "logG_" + key, None)
        if logG is None:
            raise AttributeError(
                f"table {key!r} was not built; pass stokes='IQV' (or the parameters you need) "
                f"to JetModel to include the polarized tables."
            )
        L = np.log10(np.maximum(x, 1.0e-300))
        f = (L - self._tab_logx0) * self._tab_inv_dlogx
        # NaN input must give NaN output (as np.interp did) rather than an index; the
        # NaN -> integer cast is platform dependent, so replace it before indexing
        k = np.clip(np.nan_to_num(f, nan=0.0), 0.0, self._tab_kmax).astype(np.intp)
        t = f - k
        out = 10.0 ** (logG[k] + t * (logG[k + 1] - logG[k]))
        return np.where(f < 0.0, 10.0**logG[0], np.where(f > self._tab_kmax + 1.0, 0.0, out))

    # Stokes I (used by the unpolarized radiative transfer)
    def GIx_2(self, x):
        return self._G_lookup(x, "GI_2")

    def GIx_p(self, x):
        return self._G_lookup(x, "GI_p")

    def GIx_pp1(self, x):
        return self._G_lookup(x, "GI_pp1")

    def GaIx_2(self, x):
        return self._G_lookup(x, "GaI_2")

    def GaIx_p(self, x):
        return self._G_lookup(x, "GaI_p")

    def GaIx_pp1(self, x):
        return self._G_lookup(x, "GaI_pp1")

    # Stokes Q (built when stokes includes "Q"; for future polarized transfer)
    def GQx_2(self, x):
        return self._G_lookup(x, "GQ_2")

    def GQx_p(self, x):
        return self._G_lookup(x, "GQ_p")

    def GQx_pp1(self, x):
        return self._G_lookup(x, "GQ_pp1")

    def GaQx_2(self, x):
        return self._G_lookup(x, "GaQ_2")

    def GaQx_p(self, x):
        return self._G_lookup(x, "GaQ_p")

    def GaQx_pp1(self, x):
        return self._G_lookup(x, "GaQ_pp1")

    # Stokes V (built when stokes includes "V"; for future polarized transfer)
    def GVx_2(self, x):
        return self._G_lookup(x, "GV_2")

    def GVx_p(self, x):
        return self._G_lookup(x, "GV_p")

    def GVx_pp1(self, x):
        return self._G_lookup(x, "GV_pp1")

    def GaVx_2(self, x):
        return self._G_lookup(x, "GaV_2")

    def GaVx_p(self, x):
        return self._G_lookup(x, "GaV_p")

    def GaVx_pp1(self, x):
        return self._G_lookup(x, "GaV_pp1")

    # compute normalizations for the anisotropic distributions
    def _compute_phi_norms(self):
        """
        Pitch-angle normalization

            Phi(eta, p_eta) = int_0^1 [1 + (eta - 1) mu^2]^(-p_eta/2) dmu
                            = 2F1(1/2, p_eta/2; 3/2; 1 - eta),

        in closed form.  The integrand develops a spike of height eta^(-p_eta/2) and width
        ~sqrt(eta) at mu = 1 as eta -> 0, which a uniform quadrature cannot follow: the
        10^4-point trapezoid this replaces was 2.6% high at eta = 1e-4 for p_eta = 2 and
        19% high for p_eta = 3, and Phi divides the emissivity, so that error goes
        straight into the flux.  The closed form matches a converged tanh-sinh quadrature
        to <= 5e-9 over p_eta in [0.5, 4] and eta in [1e-8, 1e4].
        """
        self.phi_norm = float(hyp2f1(0.5, self.p_eta / 2.0, 1.5, 1.0 - self.eta))

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

        # Each ray is sampled about z_J, the point at which it passes closest to the JET
        # AXIS -- the right place to spend resolution for an inclined view of a narrow
        # cone, and much better resolved than centring on the black hole (verified: at
        # i = 17 deg, axis centring is stable to 0.05% from zmax = 1e6 to 1e9, while
        # centring on the black hole is still drifting by tens of percent at 1e9).
        #
        # It is also cot(i), so it diverges as the line of sight approaches the jet axis.
        # Exactly pole-on, every ray is parallel to the axis, there is no unique closest
        # approach to it, and the correct limit is to sample about the closest approach to
        # the black hole instead.  (Left as -x/tan(i), inc = 180 gives tan(0) = 0 exactly,
        # z_J = -inf and a silently blank image, while inc = 0 gives tan(pi) = -1.2e-16 and
        # a ZeroDivisionError from the compiled kernel.)
        if abs(self.sin_i) < self._POLE_ON_SIN_TOL:
            self.z_J = np.zeros_like(self.x_im)
        else:
            self.z_J = -(self.x_im * self.cos_i / self.sin_i)
            zmax_sampled = float(np.abs(self.z_im_1D).max())
            zJ_max = float(np.abs(self.z_J).max())
            if zJ_max > self._POLE_ON_ZJ_WARN * zmax_sampled:
                from_axis = min(self.inc, 180.0 - self.inc)
                warnings.warn(
                    f"the line of sight is {from_axis:.3g} deg from the jet axis, so rays are "
                    f"sampled about |z_J| up to {zJ_max:.3g} r_g -- "
                    f"{zJ_max / zmax_sampled:.0f}x the sampling half-range zmax="
                    f"{zmax_sampled:.3g}, which leaves the inner jet under-sampled on the "
                    f"outer pixels.  Use zmax >~ xmax/tan(i) = "
                    f"{abs(self.x_im_1D).max() * abs(self.cos_i / self.sin_i):.3g}.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # flattened views used in the RT loop
        self.x_im_f = self.x_im.ravel()
        self.y_im_f = self.y_im.ravel()
        self.z_J_f = self.z_J.ravel()

        # path lengths between adjacent z-slices, used in the RT loop
        self.dz_1D = self.rg * np.abs(np.diff(self.z_im_1D))

        # cell-centre z for second-order RT quadrature
        self.z_mid_1D = 0.5 * (self.z_im_1D[1:] + self.z_im_1D[:-1])

    # primary image-generating function
    def make_image(
        self,
        frequency,
        *,
        tau_stop=None,
        show_progress=False,
        heating_prescription="Poynting",
        backend=None,
    ):
        """
        Compute the specific intensity image at the requested frequency (in GHz).

        Returns (x_im_1D, y_im_1D, I_nu), with I_nu in cgs units and shape (Ny, Nx).

        tau_stop:             stop integrating a ray once its accumulated optical depth
                              exceeds this value (None = never)
        show_progress:        show a progress bar (numpy back end only)
        heating_prescription: "Poynting" (u_e = h S / (c gamma)) or "magnetic"
                              (u_e = h B'^2 / 8 pi)
        backend:              "numba", "numpy", or None to use the model's default
        """
        if heating_prescription not in ("Poynting", "magnetic"):
            raise ValueError(
                f"unrecognized heating_prescription {heating_prescription!r}; "
                f"expected 'Poynting' or 'magnetic'"
            )
        if self._resolve_backend(backend) == "numba":
            return self._make_image_numba(frequency, tau_stop, heating_prescription)
        return self._make_image_numpy(
            frequency,
            tau_stop=tau_stop,
            show_progress=show_progress,
            heating_prescription=heating_prescription,
        )

    def _resolve_backend(self, backend):
        backend = self.backend if backend is None else backend
        if backend == "auto":
            return "numba" if HAS_NUMBA else "numpy"
        if backend == "numba" and not HAS_NUMBA:
            raise ImportError(
                "backend='numba' was requested but numba is not installed; "
                "install it (pip install numba) or use backend='numpy'"
            )
        if backend not in ("numba", "numpy"):
            raise ValueError(f"backend must be 'auto', 'numba' or 'numpy', got {backend!r}")
        return backend

    # ------------------------------------------------------------------
    # compiled back end
    # ------------------------------------------------------------------
    def _get_intervals(self):
        """
        Per-ray ranges of z-cells inside the jet, computed once per model.  Also fixes
        the (random) order in which rays are handed to the threads, which balances the
        work across cores; the result does not depend on that order.
        """
        if self._intervals is None:
            P = _kern.pack_params(self)
            max_int = 8
            while True:
                starts, ends, nint, ncell = _kern.jet_intervals(
                    self.x_im_f, self.y_im_f, self.z_J_f, self.z_mid_1D, P, max_int
                )
                if int(nint.max()) <= max_int:
                    break
                max_int *= 2
            order = np.random.default_rng(12345).permutation(nint.shape[0]).astype(np.int64)
            self._intervals = dict(starts=starts, ends=ends, nint=nint, ncell=ncell, order=order)
        return self._intervals

    @property
    def n_jet_cells(self):
        """Number of grid cells inside the jet (the work per make_image call)."""
        return int(self._get_intervals()["ncell"].sum())

    def _make_image_numba(self, frequency, tau_stop, heating_prescription):
        iv = self._get_intervals()
        P = _kern.pack_params(self, heating_prescription)
        T = _kern.pack_tables(self)
        tau = -1.0 if (tau_stop is None or float(tau_stop) <= 0.0) else float(tau_stop)
        I_out = np.zeros(self.x_im_f.shape[0], dtype=np.float64)

        st = self._state
        ft = self._ftab
        if st is not None and st["heating_prescription"] == heating_prescription:
            _kern.rt_from_state(
                float(frequency), iv["order"], st["offsets"], self.dz_1D, P, *T, tau,
                st["g"], st["nup"], st["gamma_c"], st["Cj"], st["Ca"], st["iz"], I_out,
            )
        elif ft is not None and ft["heating_prescription"] == heating_prescription:
            _kern.rt_kernel_tab(
                float(frequency), iv["order"],
                self.x_im_f, self.y_im_f, self.z_J_f, self.z_mid_1D, self.dz_1D,
                iv["starts"], iv["ends"], iv["nint"],
                P, *self._ftab_args(), *T, tau, I_out,
            )
        else:
            _kern.rt_kernel(
                float(frequency), iv["order"],
                self.x_im_f, self.y_im_f, self.z_J_f, self.z_mid_1D, self.dz_1D,
                iv["starts"], iv["ends"], iv["nint"],
                P, self._stag_lut_x, self._stag_lut_r, self._stag_lut_t, self._stag_lut_a, *T, tau, I_out,
            )
        return self.x_im_1D, self.y_im_1D, I_out.reshape(self.x_im.shape)

    # ------------------------------------------------------------------
    # optional field table (approximate, ~1.8x faster than the exact kernel)
    # ------------------------------------------------------------------
    def build_field_table(self, points_per_decade=200, n_u=256, heating_prescription="Poynting",
                          max_memory_gb=2.0):
        """
        Tabulate the (r,theta)-only physics of the jet -- fields, drift velocity, Lorentz
        factor, lapse, fluid-frame field, cooling Lorentz factor and electron
        normalization -- on a 2-D grid (uniform in log10(r - r_H) and in
        u = sqrt(psi/psi_edge), one table per hemisphere), so that make_image() only has
        to interpolate them and evaluate the line-of-sight-dependent part per cell.
        This makes the compiled kernel about 1.8x faster; it does not affect the
        cached-state mode (precompute_state), which is exact and faster still for loops
        over frequency, but it does speed up precompute_state() itself.

        The interpolation is bilinear and the error decreases as the square of the grid
        spacing.  Measured against the exact kernel at 230 GHz:

            points_per_decade x n_u  |  total flux  |  per-pixel 99th pct  |  memory
                 100 x 128           |   1e-5..1e-4 |     2e-4..6e-4       |  10-20 MB
                 200 x 256 (default) |   4e-6..3e-5 |     5e-5..2e-4       |  50-90 MB
                 400 x 512           |   6e-7..8e-6 |     1e-5..9e-5       | 200-350 MB

        (linear 100 r_g grid .. log grid to 1e5 r_g); SED errors are of the same order.
        Requires the numba back end.  Returns the table size in bytes; call
        clear_field_table() to go back to the exact kernel.
        """
        if self._resolve_backend(None) != "numba":
            raise ImportError("build_field_table() requires the numba back end")
        if heating_prescription not in ("Poynting", "magnetic"):
            raise ValueError(
                f"unrecognized heating_prescription {heating_prescription!r}; "
                f"expected 'Poynting' or 'magnetic'"
            )
        # radial extent: cover every cell that can lie in the jet, generously
        zmax = np.abs(self.z_mid_1D).max() + np.abs(self.z_J_f).max()
        r_max = np.sqrt(np.abs(self.x_im_f).max() ** 2 + np.abs(self.y_im_f).max() ** 2 + zmax**2)
        r_max = 1.001 * max(r_max, 2.0 * self.rH)
        dlog = 1.0 / float(points_per_decade)
        logr = np.arange(np.log10(0.01 * self.rH), np.log10(r_max - self.rH) + dlog, dlog)
        ugrid = np.linspace(0.0, 1.0, int(n_u))
        nbytes = 2 * logr.shape[0] * ugrid.shape[0] * _kern.N_TAB * 8
        if nbytes > max_memory_gb * 1.0e9:
            raise MemoryError(
                f"the field table needs {nbytes/1e9:.2f} GB, above max_memory_gb={max_memory_gb}; "
                f"use fewer points or raise the limit"
            )
        P = _kern.pack_params(self, heating_prescription)
        args = (P, self._stag_lut_x, self._stag_lut_r, self._stag_lut_t, self._stag_lut_a)
        Tup = _kern.build_field_table(logr, ugrid, *args, 1.0)
        Tlo = _kern.build_field_table(logr, ugrid, *args, -1.0)
        self._ftab = dict(
            heating_prescription=heating_prescription,
            Tup=Tup,
            Tlo=Tlo,
            logr0=float(logr[0]),
            invdlogr=1.0 / float(logr[1] - logr[0]),
            nr=int(logr.shape[0]),
            invdu=1.0 / float(ugrid[1] - ugrid[0]),
            nu=int(ugrid.shape[0]),
        )
        return nbytes

    def clear_field_table(self):
        """Discard the field table built by build_field_table()."""
        self._ftab = None

    def _ftab_args(self):
        ft = self._ftab
        return (ft["Tup"], ft["Tlo"], ft["logr0"], ft["invdlogr"], ft["nr"], ft["invdu"], ft["nu"])

    def precompute_state(self, heating_prescription="Poynting", max_memory_gb=8.0):
        """
        Evaluate and store the frequency-independent state of every jet cell (redshift
        factor, characteristic synchrotron frequency, cooling Lorentz factor, and the
        emissivity/absorption prefactors; 44 bytes per cell).  Subsequent make_image()
        calls with the same heating prescription then only compute the frequency-
        dependent synchrotron coefficients and the transfer integral, which is what
        makes loops over frequency (SEDs) fast.  Requires the numba back end.  If a
        field table has been built (build_field_table), the state is taken from it,
        otherwise it is computed exactly.

        Returns the memory used by the stored state, in bytes.  Raises MemoryError if
        that would exceed max_memory_gb; call clear_state() to release it.
        """
        if self._resolve_backend(None) != "numba":
            raise ImportError("precompute_state() requires the numba back end")
        if heating_prescription not in ("Poynting", "magnetic"):
            raise ValueError(
                f"unrecognized heating_prescription {heating_prescription!r}; "
                f"expected 'Poynting' or 'magnetic'"
            )
        iv = self._get_intervals()
        ncell_ordered = iv["ncell"][iv["order"]]
        offsets = np.zeros(ncell_ordered.shape[0] + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(ncell_ordered)
        N = int(offsets[-1])
        nbytes = N * (5 * 8 + 4)
        if nbytes > max_memory_gb * 1.0e9:
            raise MemoryError(
                f"the per-cell state for {N} jet cells needs {nbytes/1e9:.1f} GB, above the "
                f"max_memory_gb={max_memory_gb} limit; use a coarser grid or raise the limit"
            )
        st = dict(
            heating_prescription=heating_prescription,
            offsets=offsets,
            g=np.empty(N),
            nup=np.empty(N),
            gamma_c=np.empty(N),
            Cj=np.empty(N),
            Ca=np.empty(N),
            iz=np.empty(N, dtype=np.int32),
        )
        ft = self._ftab
        if ft is not None and ft["heating_prescription"] == heating_prescription:
            _kern.precompute_state_tab(
                iv["order"], offsets, self.x_im_f, self.y_im_f, self.z_J_f, self.z_mid_1D,
                iv["starts"], iv["ends"], iv["nint"],
                _kern.pack_params(self, heating_prescription), *self._ftab_args(),
                st["g"], st["nup"], st["gamma_c"], st["Cj"], st["Ca"], st["iz"],
            )
        else:
            _kern.precompute_state(
                iv["order"], offsets, self.x_im_f, self.y_im_f, self.z_J_f, self.z_mid_1D,
                iv["starts"], iv["ends"], iv["nint"],
                _kern.pack_params(self, heating_prescription),
                self._stag_lut_x, self._stag_lut_r, self._stag_lut_t, self._stag_lut_a,
                st["g"], st["nup"], st["gamma_c"], st["Cj"], st["Ca"], st["iz"],
            )
        self._state = st
        return nbytes

    def clear_state(self):
        """Release the per-cell state stored by precompute_state()."""
        self._state = None

    # ------------------------------------------------------------------
    # numpy back end (reference implementation)
    # ------------------------------------------------------------------
    def _make_image_numpy(
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
        p_eta = self.p_eta
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

        phi_norm = self.phi_norm

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
        z_mid_1D = self.z_mid_1D

        x_im = self.x_im
        y_im = self.y_im
        z_J = self.z_J

        x_im_f = self.x_im_f
        y_im_f = self.y_im_f
        z_J_f = self.z_J_f

        dz_1D = self.dz_1D

        jet_cutout_fraction = self.jet_cutout_fraction
        gamma_inf = self.gamma_inf
        gammabeta_suppression = self.gammabeta_suppression
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
            z_im_now = z_mid_1D[i] + z_J_f[w]

            # jet coordinates
            x = (x_im_f[w] * cos_i) + (z_im_now * sin_i)
            y = y_im_f[w]
            z = (z_im_now * cos_i) - (x_im_f[w] * sin_i)
            R2 = (x * x) + (y * y)
            r2 = R2 + (z * z)
            r = np.sqrt(r2)

            costheta = z / r
            # 1 -/+ cos(theta) without the cancellation that 1 - z/r suffers when the
            # point is close to the axis (|z| -> r); (r - |z|) = (x^2 + y^2)/(r + |z|)
            az = np.abs(z)
            big = (r + az) / r
            small = R2 / (r * (r + az))
            pos = z >= 0.0
            one_minus_costheta = np.where(pos, small, big)
            one_plus_costheta = np.where(pos, big, small)

            # Footpoint of the field line through this point.  sin(theta_fp/2) =
            # (r/rH)^(1-s) sin(theta/2) and nu = 2 - 2s give, without any inverse
            # trigonometry, cos(theta_fp) = 1 - (r/rH)^nu (1 - cos theta) on the branch
            # where that is >= -1, and (r/rH)^nu (1 + cos theta) - 1 on the other.  Only
            # omc_fp = 1 - |cos theta_fp| = psi / rH^nu is ever needed downstream.
            ratio_nu = np.power(r / rH, nu)
            w_fp = ratio_nu * one_minus_costheta
            cos_fp = np.where(w_fp < 1.0, 1.0 - w_fp, (ratio_nu * one_plus_costheta) - 1.0)
            omc_fp = 1.0 - np.abs(cos_fp)

            # horizon buffer (avoids numerical artifacts)
            eps_h = 1.0e-2
            r_min = rH * (1.0 + eps_h)

            # jet region: psi <= psi_edge in either lobe, i.e. (r/rH)^nu (1 -/+ cos
            # theta) <= 1.  Equivalent to r <= rH (1 -/+ cos theta)^(-1/nu) for nu > 0,
            # but reuses ratio_nu above and never forms 1/(1 - cos theta).
            ind_jet = ((w_fp <= 1.0) | ((ratio_nu * one_plus_costheta) <= 1.0)) & (r > r_min)

            if jet_cutout_fraction > 0.0:
                # both halves of the two-sided cut are the single condition
                # 1 - |cos theta_fp| < 1 - cos(theta_fp_cut)
                omc_fp_cut = jet_cutout_fraction * jet_cutout_fraction
                ind_jet &= ~(omc_fp < omc_fp_cut)

            if not ind_jet.any():
                continue

            # local indices into arrays defined on w
            idx_loc = np.nonzero(ind_jet)[0]

            # global indices into flattened full image arrays (I_nu_f, tau_acc_f, working_f, etc.)
            idx = w[idx_loc]

            # stream function, Omega, stagnation surface
            psi = (rH**nu) * omc_fp[idx_loc]
            Omega = omega_BZpower(0, psi, a, nu)
            rstag, tstag = self._stagnation_omc(omc_fp[idx_loc])
            Aconst_here = self._stagnation_aconst(omc_fp[idx_loc])

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

            # horizon buffer for Delta
            Delta_min = (r_min * r_min) - 2.0 * r_min + a2
            Delta = np.maximum(Delta, Delta_min)

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
                Aconst=Aconst_here,
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
            Br *= alphalapse * sqrt_scaling
            Btheta *= alphalapse * sqrt_scaling
            Bphi *= alphalapse * sqrt_scaling
            Bx, By, Bz = rtp_to_xyz(
                Br, Btheta, Bphi, x[idx_loc], y[idx_loc], z[idx_loc], r[idx_loc], R
            )

            # velocity rescaling
            vr_orig = u1 * np.sqrt(g11) / gamma
            vtheta_orig = u2 * np.sqrt(g22) / gamma
            vphi_orig = u3 * np.sqrt(g33) / gamma

            beta_orig = np.sqrt(np.maximum(0.0, 1.0 - 1.0 / (gamma * gamma)))
            gammabeta = gammabeta_suppression * beta_orig * gamma
            gamma = np.sqrt(1.0 + (gammabeta * gammabeta))

            indgamma = gamma > gamma_inf
            gamma[indgamma] = gamma_inf

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
            g = alphalapse / (gamma * one_m_betak)
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
                u_pl = h * S / (c * gamma)
            elif heating_prescription == "magnetic":
                u_pl = h * ((Bprime_mag * Bprime_mag) / (8.0 * np.pi))
            else:
                raise ValueError(
                    f"unrecognized heating_prescription {heating_prescription!r}; "
                    f"expected 'Poynting' or 'magnetic'"
                )

            n_m = (((p - 2.0) * u_pl) / ((gamma_m**p) * m_e * (c * c))) * (
                1.0 / ((gamma_m ** (2.0 - p)) - (gamma_max ** (2.0 - p)))
            )

            ind_fast = gamma_c <= gamma_m
            ind_slow = (gamma_c > gamma_m) & (gamma_c < gamma_max)
            ind_uncooled = gamma_c >= gamma_max

            cosxi = costhetaB
            anisotropy_term = 1.0 + ((eta - 1.0) * (cosxi * cosxi))

            nup = (4.1987e-3) * Bprime_mag * sinthetaB
            nu_nup = (frequency / g) / nup

            jI = np.zeros_like(x[idx_loc])
            alphaI = np.zeros_like(x[idx_loc])

            # uncooled branch
            if np.any(ind_uncooled):
                p1 = p
                g1 = gamma_m
                g2 = gamma_max
                x1 = nu_nup[ind_uncooled] / (g1 * g1)
                x2 = nu_nup[ind_uncooled] / (g2 * g2)
                Pp1 = phi_norm
                GIx_p1 = GIx_p

                A_norm = 1.0 / (((g2 ** (1.0 - p1)) - (g1 ** (1.0 - p1))) / (1.0 - p1))

                n = (
                    n_m[ind_uncooled]
                    * (gamma_m**p)
                    * (((g2 ** (1.0 - p1)) - (g1 ** (1.0 - p1))) / (1.0 - p1))
                )

                prefac_j = prefac_emis * n * A_norm * nup[ind_uncooled]

                term1 = (
                    ((anisotropy_term[ind_uncooled] ** (-p_eta / 2.0)) / Pp1)
                    * (nu_nup[ind_uncooled] ** ((1.0 - p1) / 2.0))
                    * self._G_diff(GIx_p1, (p1 - 3.0) / 2.0, x2, x1)
                )

                jI[ind_uncooled] = prefac_j * term1

                GaIx_p1 = GaIx_p
                prefac_a = prefac_absorp * n * A_norm / nup[ind_uncooled]

                term1 = (
                    ((p1 + 2.0) * ((anisotropy_term[ind_uncooled] ** (-p_eta / 2.0)) / Pp1))
                    * (nu_nup[ind_uncooled] ** (-(p1 + 4.0) / 2.0))
                    * self._G_diff(GaIx_p1, (p1 - 2.0) / 2.0, x2, x1)
                )

                alphaI[ind_uncooled] = prefac_a * term1

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
                Pp1 = phi_norm
                Pp2 = phi_norm
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
                    ((anisotropy_term[ind_slow] ** (-p_eta / 2.0)) / Pp1)
                    * (nu_nup[ind_slow] ** ((1.0 - p1) / 2.0))
                    * self._G_diff(GIx_p1, (p1 - 3.0) / 2.0, x2, x1)
                )
                term2 = (
                    (g2 ** (p2 - p1))
                    * ((anisotropy_term[ind_slow] ** (-p_eta / 2.0)) / Pp2)
                    * (nu_nup[ind_slow] ** ((1.0 - p2) / 2.0))
                    * self._G_diff(GIx_p2, (p2 - 3.0) / 2.0, x3, x2)
                )
                jI[ind_slow] = prefac_j * (term1 + term2)

                GaIx_p1 = GaIx_p
                GaIx_p2 = GaIx_pp1
                prefac_a = prefac_absorp * n * A_norm / nup[ind_slow]
                term1 = (
                    ((p1 + 2.0) * ((anisotropy_term[ind_slow] ** (-p_eta / 2.0)) / Pp1))
                    * (nu_nup[ind_slow] ** (-(p1 + 4.0) / 2.0))
                    * self._G_diff(GaIx_p1, (p1 - 2.0) / 2.0, x2, x1)
                )
                term2 = (
                    (
                        (p2 + 2.0)
                        * (g2 ** (p2 - p1))
                        * ((anisotropy_term[ind_slow] ** (-p_eta / 2.0)) / Pp2)
                    )
                    * (nu_nup[ind_slow] ** (-(p2 + 4.0) / 2.0))
                    * self._G_diff(GaIx_p2, (p2 - 2.0) / 2.0, x3, x2)
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
                Pp1 = phi_norm
                Pp2 = phi_norm
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
                    ((anisotropy_term[ind_fast] ** (-p_eta / 2.0)) / Pp1)
                    * (nu_nup[ind_fast] ** ((1.0 - p1) / 2.0))
                    * self._G_diff(GIx_p1, (p1 - 3.0) / 2.0, x2, x1)
                )
                term2 = (
                    (g2 ** (p2 - p1))
                    * ((anisotropy_term[ind_fast] ** (-p_eta / 2.0)) / Pp2)
                    * (nu_nup[ind_fast] ** ((1.0 - p2) / 2.0))
                    * self._G_diff(GIx_p2, (p2 - 3.0) / 2.0, x3, x2)
                )
                jI[ind_fast] = prefac_j * (term1 + term2)

                GaIx_p1 = GaIx_2
                GaIx_p2 = GaIx_pp1
                prefac_a = prefac_absorp * n * A_norm / nup[ind_fast]
                term1 = (
                    ((p1 + 2.0) * ((anisotropy_term[ind_fast] ** (-p_eta / 2.0)) / Pp1))
                    * (nu_nup[ind_fast] ** (-(p1 + 4.0) / 2.0))
                    * self._G_diff(GaIx_p1, (p1 - 2.0) / 2.0, x2, x1)
                )
                term2 = (
                    (
                        (p2 + 2.0)
                        * (g2 ** (p2 - p1))
                        * ((anisotropy_term[ind_fast] ** (-p_eta / 2.0)) / Pp2)
                    )
                    * (nu_nup[ind_fast] ** (-(p2 + 4.0) / 2.0))
                    * self._G_diff(GaIx_p2, (p2 - 2.0) / 2.0, x3, x2)
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
                               t_c, gamma_c, u_e, nu_p, jI, alphaI

        For jI and alphaI, the frequency must be specified in GHz.

        """

        # check that the frequency is provided if necessary
        if quantity in ["jI", "alphaI", "nu_p"]:
            if frequency is None:
                raise Exception("For jI, alphaI, or nu_p, the frequency must be specified in GHz.")

        # pull cached attributes into local variables
        rH = self.rH
        nu = self.nu
        a = self.a
        s = self.s
        p = self.p
        p_eta = self.p_eta
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

        phi_norm = self.phi_norm

        GIx_2 = self.GIx_2
        GIx_p = self.GIx_p
        GIx_pp1 = self.GIx_pp1
        GaIx_2 = self.GaIx_2
        GaIx_p = self.GaIx_p
        GaIx_pp1 = self.GaIx_pp1

        stagnation = self.stagnation

        jet_cutout_fraction = self.jet_cutout_fraction
        gamma_inf = self.gamma_inf
        gammabeta_suppression = self.gammabeta_suppression
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
        # half-angle forms: exact for theta near 0 or pi, where 1 -/+ cos(theta) cancels
        one_minus_costheta = 2.0 * np.sin(0.5 * theta) ** 2
        one_plus_costheta = 2.0 * np.cos(0.5 * theta) ** 2

        # footpoint of the field line, without inverse trigonometry (see _make_image_numpy)
        ratio_nu = np.power(r / rH, nu)
        w_fp = ratio_nu * one_minus_costheta
        cos_fp = np.where(w_fp < 1.0, 1.0 - w_fp, (ratio_nu * one_plus_costheta) - 1.0)
        omc_fp = 1.0 - np.abs(cos_fp)

        # stream function, Omega, stagnation surface
        psi = (rH**nu) * omc_fp
        if quantity == "psi":
            return psi
        Omega = omega_BZpower(0, psi, a, nu)
        if quantity == "Omega":
            return Omega
        rstag, tstag = self._stagnation_omc(omc_fp)
        Aconst_here = self._stagnation_aconst(omc_fp)

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

        # horizon buffer for Delta
        eps_h = 1.0e-2
        r_min = rH * (1.0 + eps_h)
        Delta_min = (r_min * r_min) - 2.0 * r_min + a2
        Delta = np.maximum(Delta, Delta_min)

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
            Aconst=Aconst_here,
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
        Br *= alphalapse * sqrt_scaling
        Btheta *= alphalapse * sqrt_scaling
        Bphi *= alphalapse * sqrt_scaling
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
        gammabeta = gammabeta_suppression * beta_orig * gamma
        gamma = np.sqrt(1.0 + (gammabeta * gammabeta))

        indgamma = gamma > gamma_inf
        gamma[indgamma] = gamma_inf
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
        g = alphalapse / (gamma * one_m_betak)
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
            u_pl = h * S / (c * gamma)
        elif heating_prescription == "magnetic":
            u_pl = h * ((Bprime_mag * Bprime_mag) / (8.0 * np.pi))
        else:
            raise ValueError(
                f"unrecognized heating_prescription {heating_prescription!r}; "
                f"expected 'Poynting' or 'magnetic'"
            )
        if quantity == "u_e":
            return u_pl

        n_m = (((p - 2.0) * u_pl) / ((gamma_m**p) * m_e * (c * c))) * (
            1.0 / ((gamma_m ** (2.0 - p)) - (gamma_max ** (2.0 - p)))
        )

        ind_fast = gamma_c <= gamma_m
        ind_slow = (gamma_c > gamma_m) & (gamma_c < gamma_max)
        ind_uncooled = gamma_c >= gamma_max

        cosxi = costhetaB
        anisotropy_term = 1.0 + ((eta - 1.0) * (cosxi * cosxi))

        if frequency is not None:
            nup = (4.1987e-3) * Bprime_mag * sinthetaB
            nu_nup = (frequency / g) / nup

            if quantity == "nu_p":
                return nup

            jI = np.zeros_like(x)
            alphaI = np.zeros_like(x)

            # uncooled branch
            if np.any(ind_uncooled):
                p1 = p
                g1 = gamma_m
                g2 = gamma_max
                x1 = nu_nup[ind_uncooled] / (g1 * g1)
                x2 = nu_nup[ind_uncooled] / (g2 * g2)
                Pp1 = phi_norm
                GIx_p1 = GIx_p

                A_norm = 1.0 / (((g2 ** (1.0 - p1)) - (g1 ** (1.0 - p1))) / (1.0 - p1))

                n = (
                    n_m[ind_uncooled]
                    * (gamma_m**p)
                    * (((g2 ** (1.0 - p1)) - (g1 ** (1.0 - p1))) / (1.0 - p1))
                )

                prefac_j = prefac_emis * n * A_norm * nup[ind_uncooled]

                term1 = (
                    ((anisotropy_term[ind_uncooled] ** (-p_eta / 2.0)) / Pp1)
                    * (nu_nup[ind_uncooled] ** ((1.0 - p1) / 2.0))
                    * self._G_diff(GIx_p1, (p1 - 3.0) / 2.0, x2, x1)
                )

                jI[ind_uncooled] = prefac_j * term1

                GaIx_p1 = GaIx_p
                prefac_a = prefac_absorp * n * A_norm / nup[ind_uncooled]

                term1 = (
                    ((p1 + 2.0) * ((anisotropy_term[ind_uncooled] ** (-p_eta / 2.0)) / Pp1))
                    * (nu_nup[ind_uncooled] ** (-(p1 + 4.0) / 2.0))
                    * self._G_diff(GaIx_p1, (p1 - 2.0) / 2.0, x2, x1)
                )

                alphaI[ind_uncooled] = prefac_a * term1

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
                Pp1 = phi_norm
                Pp2 = phi_norm
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
                    ((anisotropy_term[ind_slow] ** (-p_eta / 2.0)) / Pp1)
                    * (nu_nup[ind_slow] ** ((1.0 - p1) / 2.0))
                    * self._G_diff(GIx_p1, (p1 - 3.0) / 2.0, x2, x1)
                )
                term2 = (
                    (g2 ** (p2 - p1))
                    * ((anisotropy_term[ind_slow] ** (-p_eta / 2.0)) / Pp2)
                    * (nu_nup[ind_slow] ** ((1.0 - p2) / 2.0))
                    * self._G_diff(GIx_p2, (p2 - 3.0) / 2.0, x3, x2)
                )
                jI[ind_slow] = prefac_j * (term1 + term2)

                GaIx_p1 = GaIx_p
                GaIx_p2 = GaIx_pp1
                prefac_a = prefac_absorp * n * A_norm / nup[ind_slow]
                term1 = (
                    ((p1 + 2.0) * ((anisotropy_term[ind_slow] ** (-p_eta / 2.0)) / Pp1))
                    * (nu_nup[ind_slow] ** (-(p1 + 4.0) / 2.0))
                    * self._G_diff(GaIx_p1, (p1 - 2.0) / 2.0, x2, x1)
                )
                term2 = (
                    (
                        (p2 + 2.0)
                        * (g2 ** (p2 - p1))
                        * ((anisotropy_term[ind_slow] ** (-p_eta / 2.0)) / Pp2)
                    )
                    * (nu_nup[ind_slow] ** (-(p2 + 4.0) / 2.0))
                    * self._G_diff(GaIx_p2, (p2 - 2.0) / 2.0, x3, x2)
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
                Pp1 = phi_norm
                Pp2 = phi_norm
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
                    ((anisotropy_term[ind_fast] ** (-p_eta / 2.0)) / Pp1)
                    * (nu_nup[ind_fast] ** ((1.0 - p1) / 2.0))
                    * self._G_diff(GIx_p1, (p1 - 3.0) / 2.0, x2, x1)
                )
                term2 = (
                    (g2 ** (p2 - p1))
                    * ((anisotropy_term[ind_fast] ** (-p_eta / 2.0)) / Pp2)
                    * (nu_nup[ind_fast] ** ((1.0 - p2) / 2.0))
                    * self._G_diff(GIx_p2, (p2 - 3.0) / 2.0, x3, x2)
                )
                jI[ind_fast] = prefac_j * (term1 + term2)

                GaIx_p1 = GaIx_2
                GaIx_p2 = GaIx_pp1
                prefac_a = prefac_absorp * n * A_norm / nup[ind_fast]
                term1 = (
                    ((p1 + 2.0) * ((anisotropy_term[ind_fast] ** (-p_eta / 2.0)) / Pp1))
                    * (nu_nup[ind_fast] ** (-(p1 + 4.0) / 2.0))
                    * self._G_diff(GaIx_p1, (p1 - 2.0) / 2.0, x2, x1)
                )
                term2 = (
                    (
                        (p2 + 2.0)
                        * (g2 ** (p2 - p1))
                        * ((anisotropy_term[ind_fast] ** (-p_eta / 2.0)) / Pp2)
                    )
                    * (nu_nup[ind_fast] ** (-(p2 + 4.0) / 2.0))
                    * self._G_diff(GaIx_p2, (p2 - 2.0) / 2.0, x3, x2)
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

    # isotropic-equivalent luminosity density, L_nu = 4 pi D^2 S_nu, in cgs
    Lnu = 4.0 * np.pi * dA * I_nu

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
        Snu = Lnu / (4.0 * np.pi * (D_cm**2.0))

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

    I_new = np.asarray(I_new).copy()
    x_new = np.asarray(x_new).copy().ravel()
    y_new = np.asarray(y_new).copy().ravel()

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
