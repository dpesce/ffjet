"""
Faraday rotativities rho_Q (conversion) and rho_V (rotation) for the jet's cooled,
broken power-law electrons.

Two schemes are available, selected with JetModel(rotativities=...):

  "HS11"  Huang & Shcherbakov (2011).  Their Section 4.2 gives fitting formulae for
          the rotativities of a *monoenergetic* distribution (their Eq. 55), which are
          then integrated over any isotropic distribution (their Eq. 56).  This is the
          accurate option: the authors' stated goal was precisely that earlier
          power-law expressions "are inaccurate for high energies of electrons", and
          Faraday conversion in particular was not previously computed correctly.

  "JO77"  The power-law expressions of Jones & Odell (1977) / Sazonov (1969), as given
          in Dexter (2016) Appendix B1, generalized here so that each segment of the
          broken power law is integrated over its own range of Lorentz factors.

  "none"  rho_Q = rho_V = 0, i.e. emission and absorption only.

MEASURED ACCURACY, emission-weighted over the cells of an M87-like model, against
HS11 integrated numerically over the true broken power law:

                                        rho_Q                rho_V
    single segment, extended to inf     1.62 - 1.66 x        1.03 - 1.09 x
    "JO77" (this module)                0.80 - 0.84 x        1.006 x
    "HS11" (this module)                1.02 x               0.999 x

The residual 2% of "HS11" is the ultrarelativistic simplification below; the 20% of
"JO77" is the intrinsic inaccuracy of the power-law form for Faraday conversion.

That 2% is not uniform: it is set by the lowest Lorentz factor of the distribution and
tracks sqrt(1 - 1/gamma_1), measuring 1.9% at gamma_1 = 10, 5.8% at 3 and 9.4% at 1.5.
Fast cooling always has gamma_1 = gamma_c < gamma_m, so that branch is the one which
reaches the mildly relativistic end -- where the ultrarelativistic synchrotron kernels
used for the emission coefficients are equally approximate, so nothing is lost by being
consistent with them.  "JO77" is much less uniform over the same sweep: its rho_Q error
ranges from 2% to 45%.

WHY THIS TABULATES.  In HS11 the Lorentz factor enters the fitting functions only
through the product X_A*gamma, so with the ultrarelativistic kinematics that the rest
of this code already assumes (|p| -> gamma, and ln[(1+beta)/(1-beta)] -> 2 ln 2gamma,
whose logarithm separates), the integral over one power-law segment gamma^-q becomes

    int S(gamma)/(gamma |p|) Kernel(X_A gamma) dgamma
        = X_A^(q+1) [ W_q(X_A gamma_lo) - W_q(X_A gamma_hi) ],
    W_q(x) = int_x^inf u^-(q+2) Kernel(u) du,

which is the same structure as the synchrotron integrals in _core.  Three kernels
(one for rho_Q, two for rho_V, the second carrying the ln) and the three spectral
indices the cooling branches use (p, p+1, 2) give nine tables, built at construction
from closed-form fits in a few milliseconds.

The tables are stored SCALED as x^(q+1) W_q(x), which is smooth and O(1) across the
whole range -- W itself spans many decades and changes sign -- so linear interpolation
in log10(x) is accurate, and the X_A^(q+1) prefactor cancels against the scaling,
leaving powers of gamma that are model constants everywhere except at gamma_c.
"""

import numpy as np

# (kernel, spectral index) of each tabulated row; "Q" is H_X, "V1" is g_X and "V2" is
# g_X ln(2u).  The index suffixes match the synchrotron tables: _p, _pp1, _2.
ROT_ROWS = (
    ("Q", "p"), ("Q", "pp1"), ("Q", "2"),
    ("V1", "p"), ("V1", "pp1"), ("V1", "2"),
    ("V2", "p"), ("V2", "pp1"), ("V2", "2"),
)
N_ROT_ROWS = len(ROT_ROWS)

# log10(x) range and resolution of the rotativity tables.  X_A gamma_1 sits around
# 2-30 at the frequencies of interest and falls as 1/sqrt(nu'/nu_p'), reaching ~1e-3
# in the far X-ray tail of an SED; the range below is generous enough that a lookup
# never leaves it in practice, and the integrand falls as u^-3.8 at the top, so
# truncating there costs nothing.  Outside the range the lookup clamps to the edge.
ROT_LOGX = (-12.0, 8.0)
ROT_DEX = 0.01

# X_A^2 = sqrt(2) sin(theta) (Omega_0/omega) / 1e-4 and Omega_0 sin(theta)/omega
# = nu_B sin(theta)/nu' = (2/3)/(nu'/nu_p'), so X_A = sqrt(_XA_CONST / (nu'/nu_p')).
XA_CONST = np.sqrt(2.0) * (2.0 / 3.0) * 1.0e4

# prefactors relative to K_a/nu_p (the scale alpha_I is built on); see the derivation
# in the module docstring of _kernel_pol
HS11_Q_PREFAC = 4.0 * np.sqrt(3.0)
HS11_V_PREFAC = 8.0 * np.sqrt(3.0) / 3.0
JO77_Q_PREFAC = 32.0 * np.sqrt(3.0) / 9.0
JO77_V_PREFAC = 16.0 * np.sqrt(3.0) / 3.0


# --------------------------------------------------------------------------- fits
def H_X(x, gamma=None):
    """
    HS11 Eq. (54), the fit to -Re(alpha^11 - alpha^22) that carries rho_Q.

    `gamma` supplies the exact kinematic factor sqrt(1 - 1/gamma) of the low branch;
    omit it for the ultrarelativistic form that the tables are built on.
    """
    x = np.asarray(x, dtype=float)
    kin = 1.0 if gamma is None else np.sqrt(np.maximum(1.0 - 1.0 / np.asarray(gamma), 0.0))
    L = np.log(np.maximum(x, 1.0e-300))
    lo = 9.29e-9 * kin * x**3.036
    hi = (
        -0.000203 * x**0.4343
        - 0.0013 * np.cos(0.5646 * L - 4.03)
        + 0.002 * np.exp(-((L - 4.2137) ** 2) / 0.5429)
        + 0.00083 * np.exp(-((L - 4.2137) ** 2) / 0.2121)
    )
    return np.where(x < 40.0, lo, hi)


def H_B(x, gamma):
    """HS11 Eq. (54), the boundary counterpart of H_X (exact kinematics, pointwise)."""
    x = np.asarray(x, dtype=float)
    g = np.asarray(gamma, dtype=float)
    L = np.log(np.maximum(x, 1.0e-300))
    lo = 4.67e-9 * np.maximum(1.0 - 1.0 / g, 0.0) ** 1.5 * x**3.84
    hi = (
        0.864
        - 0.2082 * L**2
        + 0.0175 * L**4
        - 0.000626 * L**6
        + 1.0175e-5 * L**8
        - 7.686e-8 * L**10
        - 0.01 * np.exp(-((L - 4.0755) ** 2) / 0.0763)
    )
    return np.where(x < 40.0, lo, hi)


def g_X(x):
    """HS11 Eq. (54), the multiplier of rho_V."""
    L = np.log(np.maximum(np.asarray(x, dtype=float), 1.0e-300))
    return (
        1.0
        - 0.4 * np.exp(-((L - 9.21) ** 2) / 11.93)
        - 0.05 * np.exp(-((L - 5.76) ** 2) / 1.33)
        + 0.075 * np.exp(-((L - 4.03) ** 2) / 0.65)
    )


def g_B(x):
    """HS11 Eq. (54), the multiplier of the rho_V boundary term."""
    return 1.0 - 0.0045 * np.asarray(x, dtype=float) ** 0.52


def _kernel(name, x):
    if name == "Q":
        return H_X(x)
    if name == "V1":
        return g_X(x)
    if name == "V2":
        return g_X(x) * np.log(2.0 * x)
    raise ValueError(name)


def build_tables(p):
    """
    Build the nine rotativity tables for spectral index p.

    Returns (table, logx, logx0, inv_dlogx, kmax, exponents) where table has shape
    (9, n) and holds x^(q+1) int_x^inf u^-(q+2) Kernel(u) du, and `exponents` gives q
    for each row.
    """
    lo, hi = ROT_LOGX
    # integrate on a grid four times finer than the stored one, then decimate
    fine = ROT_DEX / 4.0
    n = int(round((hi - lo) / fine)) + 1
    u_log = np.linspace(lo, hi, n)
    u = 10.0**u_log
    ln10 = np.log(10.0)
    stride = int(round(ROT_DEX / fine))
    idx = np.arange(0, n, stride)
    logx = u_log[idx]

    qs = {"p": p, "pp1": p + 1.0, "2": 2.0}
    rows = np.empty((N_ROT_ROWS, idx.size))
    exponents = np.empty(N_ROT_ROWS)
    cache = {}
    for r, (kern, which) in enumerate(ROT_ROWS):
        q = qs[which]
        exponents[r] = q
        key = (kern, round(q, 10))
        if key not in cache:
            integrand = ln10 * (u ** (-(q + 2.0))) * _kernel(kern, u) * u
            seg = 0.5 * (integrand[1:] + integrand[:-1]) * (u_log[1] - u_log[0])
            W = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
            cache[key] = W
        rows[r] = (u[idx] ** (q + 1.0)) * cache[key][idx]
    return (
        np.ascontiguousarray(rows),
        logx,
        float(logx[0]),
        1.0 / float(logx[1] - logx[0]),
        float(logx.size - 2),
        exponents,
    )
