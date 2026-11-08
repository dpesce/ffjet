"""
Faraday rotativities rho_Q (conversion) and rho_V (rotation) for the jet's cooled,
broken power-law electrons.

Two schemes are available, selected with JetModel(rotativities=...):

  "HS11"  Huang & Shcherbakov (2011).  Their Section 4.2 gives fitting formulae for
          the rotativities of a *monoenergetic* distribution (their Eq. 55), which are
          summed here over the electrons of the distribution, with exact kinematics.
          This is the accurate option: the authors' stated goal was precisely that
          earlier power-law expressions "are inaccurate for high energies of
          electrons", and Faraday conversion in particular was not previously computed
          correctly.

  "JO77"  The power-law expressions of Jones & Odell (1977) / Sazonov (1969), as given
          in Dexter (2016) Appendix B1, generalized here so that each segment of the
          broken power law is integrated over its own range of Lorentz factors.  Its
          rho_V is the textbook rotation of one ultrarelativistic electron,
          ln(2 gamma)/gamma^2, summed over the distribution.

  "none"  rho_Q = rho_V = 0, i.e. emission and absorption only.

ONLY THE ELECTRONS THE DISTRIBUTION CONTAINS.  Kinetic theory gives each rotativity as
an integral over the slope of the distribution function, df/dp.  Integrating by parts
turns it into the sum, over the electrons, of the rotativity of one electron (HS11's
Eq. 55), plus end terms at the edges of the distribution.  The broken power law drops
from f(gamma_1) to zero at its lowest Lorentz factor gamma_1 (gamma_m, or gamma_c when
the electrons cool fast), so its slope has a spike there, which cancels the end term
exactly.  HS11's Eq. (56) keeps the end term but not the spike, and so does Dexter's
(B2) for rho_V, whose factor (p+2) comes from it.  That end term is exactly the
rotativity of electrons at the phase-space density f(gamma_1) filling all momenta below
gamma_1 -- 0.41 (slow cooling) and 0.36 (fast cooling, gamma_c = 3) times as many
electrons as the distribution holds, which it does not contain; in slow cooling it is
two to three times the rotation of the distribution's own electrons.  Both schemes
here leave the end terms out.  Electrons below gamma_1, if they are to be included,
must be added to the distribution, with the same care at their own lowest edge.  The
absorption coefficients carry the same kind of end term, in the factor p + 2 of the
power-law tables, and ffjet takes it off as well (_kernel._edge_absorption).

THE COOLED ELECTRONS.  The cooling break gamma_c (paper Eq. 25) is how far an electron
cools in the time t_c at the ultrarelativistic rate dgamma/dt = -C gamma^2.  The rate is
really -C (gamma^2 - 1) = -C |p|^2 (cyclo-synchrotron), which slows as the electrons
become mildly relativistic: in the time t_c = 1/(C gamma_c) they reach not gamma_c but the
momentum p_c = 1/sinh(1/gamma_c), i.e. gamma_c' = coth(1/gamma_c) = gamma_c + 1/(3 gamma_c)
+ ..., which is never below 1 (in the innermost cells of the M87 model gamma_c falls to
0.87, and the broken power law would reach below gamma = 1).  In the steady state the
cooled electrons follow 1/(gamma^2 - 1) times what they do above, the ultrarelativistic
gamma^-2 below gamma_m in fast cooling becoming 1/(gamma^2 - 1) from p_c up, with the same
amplitude, so the same number of electrons (n_m gamma_m) sits at somewhat higher energies.
The HS11 scheme sums the rotativities over this distribution (the "2" and "pp1"
correction tables below; fast_cooling_lower_edge); the emission and absorption
coefficients keep the ultrarelativistic shapes, since electrons this slow neither
radiate nor absorb at the frequencies modeled.  The JO77 scheme keeps its textbook form.

MEASURED ACCURACY, against HS11's Eq. (55) summed numerically with exact kinematics over
the electrons of each cell's distribution (as above), on the cells of the fiducial M87
model (inner jet, 43-230 GHz), weighted by emission (j_I ds) and by the rotativity
itself (|rho| ds):

                    rho_Q                         rho_V
                    emission      conversion      emission      rotation
    "HS11"          1.0005-1.0007 x               1.0000 x
    "JO77"          1.09-1.15 x   1.08-1.14 x     1.009-1.011 x 1.02 x

The residual of "HS11" is the resolution of the tables (worst cell: 3e-3 on rho_Q,
which crosses the jump of H_X at x = 40, and 2e-4 on rho_V).  "JO77" adds the
ultrarelativistic kinematics and the intrinsic inaccuracy of the power-law form for
Faraday conversion; its rho_Q error ranges from 7% to 260% over four decades of
nu'/nu_p (its rho_V stays within 1.3%).  HS11's fits are themselves accurate to about
10% in the authors' words, and H_X jumps by a third at x = 40 as printed (their Eq. 77
in the arXiv version).  Before the exact kinematics and the exact cooled shape were
included (2026-11-07), the scheme had rho_V 11% low and rho_Q 7% high from the
kinematics, and rho_V 14% high and rho_Q 2% low from the shape and lower edge of the
fast-cooled electrons, on the M87 model's cells (rotation- and conversion-weighted,
cells with gamma_c > 1); the two nearly cancel in rho_V.

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
from closed-form fits in a few milliseconds.  What the ultrarelativistic kinematics and
shapes leave out is small except at low gamma, and is tabulated separately in
(X_A, |p|) for each kind of segment (build_correction_tables, below).

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


def g_X(x):
    """HS11 Eq. (54), the multiplier of rho_V."""
    L = np.log(np.maximum(np.asarray(x, dtype=float), 1.0e-300))
    return (
        1.0
        - 0.4 * np.exp(-((L - 9.21) ** 2) / 11.93)
        - 0.05 * np.exp(-((L - 5.76) ** 2) / 1.33)
        + 0.075 * np.exp(-((L - 4.03) ** 2) / 0.65)
    )


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


# --------------------------------------------------------------------------- exact kinematics
# The tables above take |p| -> gamma.  The tables below hold what that leaves out, for
# each kind of power-law segment s and each rotativity K:
#
#     C_sK(X_A, p) = int_gamma(p)^inf [ S_s(g) w_K(X_A, g) - g^-q_s w_K,ur(X_A, g) ] dg,
#
# with HS11's rotativities of one electron, exactly (Eqs. 54-55: |p| = sqrt(g^2 - 1),
# the factor sqrt(1 - 1/g) of H_X's low branch, ln[(1 + beta)/(1 - beta)] = 2 arccosh g)
#
#     w_Q = H_X(X_A g; g) / (g |p|),          w_Q,ur = H_X(X_A g) / g^2,
#     w_V = 2 g_X(X_A g) arccosh(g) / (g |p|),  w_V,ur = 2 g_X(X_A g) ln(2 g) / g^2,
#
# and the exact shape S_s of each segment:
#
#     "p"    the injected electrons, S = g^-p (the shape the model gives them);
#     "pp1"  the cooled electrons above the break, S = g^-(p+1) g^2/(g^2 - 1);
#     "2"    the cooled electrons below gamma_m in fast cooling, S = 1/(g^2 - 1).
#
# The cooled shapes are the steady state of cyclo-synchrotron cooling, dg/dt ~ -(g^2 - 1)
# = -|p|^2, of which the ultrarelativistic g^-(q) is the limit.  A segment from g_lo to
# g_hi then contributes, on top of its ultrarelativistic table difference,
# C_sK(X_A, p_lo) - C_sK(X_A, p_hi).  The integrand differs from zero only by O(1/g^2),
# so C_sK vanishes above p = 10^3 to far better than the tables' accuracy.
#
# The tables are stored on a grid in (log10 X_A, log10 p) with as many points in p as the
# tables above have in x, so that both pack into one array for the kernels.  rho_Q's are
# stored divided by X_A^3.036, which makes them independent of X_A at small X_A (HS11's
# low branch), so clamping below the grid is exact there.  Below p = 10^-6 the cooled
# shape "2" integrates to C(p_min) + c ln(p_min/p), with c the integrand per ln p at
# p -> 0 (cold electrons; exact to O(p^2)); the kernels extrapolate linearly in log p.
CORR_SEGMENTS = ("p", "pp1", "2")
CORR_KERNELS = ("Q", "V")
N_CORR = len(CORR_SEGMENTS) * len(CORR_KERNELS)
CORR_LOGXA = (-4.0, 2.0)
CORR_DEX_XA = 0.05
CORR_LOGP = (-6.0, 3.0)
CORR_Q_SCALE = 3.036                     # rho_Q's tables are stored divided by X_A^this


def _shape(seg, g, mom, p):
    """the exact shape S_s(g) of a segment, and its ultrarelativistic counterpart g^-q;
    g^2 - 1 is formed as mom^2, which keeps its precision as g -> 1"""
    if seg == "p":
        s = g ** (-p)
        return s, s
    if seg == "pp1":
        ur = g ** (-(p + 1.0))
        return ur * g * g / (mom * mom), ur
    return 1.0 / (mom * mom), g ** (-2.0)


def rotativity_of_one_electron(kern, XA, g, exact=True, mom=None):
    """
    HS11's rotativity of one electron of Lorentz factor g (momentum mom = sqrt(g^2 - 1),
    in units of m_e c), w_Q or w_V above, in the normalization of the tables
    (exact=False: the ultrarelativistic form).  The kinematic factors are formed from the
    momentum, sqrt(1 - 1/g) = mom/sqrt(g (g + 1)) and arccosh(g) = arcsinh(mom), which keep
    their precision as g -> 1; w_Q of the low branch and w_V stay finite there.
    """
    g = np.asarray(g, dtype=float)
    x = XA * g
    if not exact:
        if kern == "Q":
            return H_X(x) / (g * g)
        return 2.0 * g_X(x) * np.log(2.0 * g) / (g * g)
    if mom is None:
        mom = np.sqrt(np.maximum(g * g - 1.0, 0.0))
    mom = np.asarray(mom, dtype=float)
    if kern == "Q":
        # low branch: the factor sqrt(1 - 1/g) cancels the 1/|p|
        with np.errstate(divide="ignore", invalid="ignore"):
            hi = H_X(x) / (g * mom)
        return np.where(x < 40.0, H_X(x) / (g * np.sqrt(g * (g + 1.0))), hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mom > 1.0e-6, np.arcsinh(mom) / np.where(mom > 1.0e-6, mom, 1.0), 1.0 - mom * mom / 6.0)
    return 2.0 * g_X(x) * ratio / g


def build_correction_tables(p, n_p):
    """
    The six tables C_sK(X_A, p) for spectral index p, with n_p points in log10 p.

    Returns (table, logxa0, inv_dlogxa, n_xa, logp0, inv_dlogp, n_p): table has shape
    (N_CORR, n_xa, n_p), ordered as (s, K) over CORR_SEGMENTS x CORR_KERNELS; rho_Q's
    are divided by X_A^CORR_Q_SCALE.
    """
    lxa = np.arange(CORR_LOGXA[0], CORR_LOGXA[1] + 0.5 * CORR_DEX_XA, CORR_DEX_XA)
    lo, hi = CORR_LOGP
    lp_out = np.linspace(lo, hi, n_p)
    # integrate on a grid four times finer than the stored one, then decimate
    lp = np.linspace(lo, hi, 4 * (n_p - 1) + 1)
    mom = 10.0 ** lp
    g = np.sqrt(1.0 + mom * mom)
    dg_dlnp = mom * mom / g
    XA = 10.0 ** lxa[:, None]
    ln10 = np.log(10.0)
    table = np.empty((N_CORR, lxa.size, n_p))
    w = {(k, e): rotativity_of_one_electron(k, XA, g[None, :], exact=e, mom=mom[None, :])
         for k in CORR_KERNELS for e in (True, False)}
    i = 0
    for seg in CORR_SEGMENTS:
        S, S_ur = _shape(seg, g, mom, p)
        for kern in CORR_KERNELS:
            integrand = (S * w[(kern, True)] - S_ur * w[(kern, False)]) * dg_dlnp
            segs = 0.5 * (integrand[:, 1:] + integrand[:, :-1]) * (ln10 * (lp[1] - lp[0]))
            C = np.concatenate([np.cumsum(segs[:, ::-1], axis=1)[:, ::-1], np.zeros((lxa.size, 1))], axis=1)
            C = C[:, ::4]
            if kern == "Q":
                C = C / XA**CORR_Q_SCALE
            table[i] = C
            i += 1
    return (
        np.ascontiguousarray(table),
        float(lxa[0]),
        1.0 / CORR_DEX_XA,
        lxa.size,
        float(lp_out[0]),
        1.0 / float(lp_out[1] - lp_out[0]),
        n_p,
    )


def fast_cooling_lower_edge(gamma_c):
    """
    The lowest Lorentz factor and log10 of the momentum of the cooled electrons in fast
    cooling: with dg/dt = -C (g^2 - 1), electrons injected at high energy a time t_c ago
    have reached p_c = 1/sinh(C t_c) = 1/sinh(1/gamma_c), where gamma_c = 1/(C t_c) is the
    ultrarelativistic cooling break (paper Eq. 25).  So gamma_c' = coth(1/gamma_c) >= 1,
    = gamma_c + 1/(3 gamma_c) + ... when gamma_c >> 1.  log10 p_c is computed without
    forming sinh for small gamma_c, where p_c underflows.
    """
    u = 1.0 / np.asarray(gamma_c, dtype=float)
    g_low = 1.0 / np.tanh(u)
    with np.errstate(over="ignore"):
        lp = np.where(u > 20.0, -(u - np.log(2.0)) / np.log(10.0), -np.log10(np.sinh(np.minimum(u, 20.0))))
    return g_low, lp
