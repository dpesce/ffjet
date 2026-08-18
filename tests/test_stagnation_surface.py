"""Vectorized stagnation-surface build (Omega_F held fixed along each field line)."""

import numpy as np

import jetfuncs as jf


# ---------------------------------------------------------------- stagnation surface
def _reference_stagnation(model, n):
    """Scalar bisection with Omega_F held fixed at the traced field line's value."""
    a, nu, rH, bf = model.a, model.nu, model.rH, model.bf
    th_H = 10.0 ** np.linspace(-5.0, np.log10(np.pi / 2.0), n)
    rs = np.zeros(n)
    ts = np.zeros(n)
    for i, tH in enumerate(th_H):
        Om = jf.omega_BZpower(0, jf.psiBZpower(rH, tH, nu), a, nu)
        ta, tb = 1e-10, tH
        for _ in range(30):
            tc = np.sqrt(ta * tb)
            rc = rH * ((1.0 - np.cos(tH)) / (1.0 - np.cos(tc))) ** (1.0 / nu)
            if jf.Nderiv(rc, tc, a, Om, 1.0, bf) > 0.0:
                ta = tc
            else:
                tb = tc
        rs[i], ts[i] = rc, tc
    return th_H, rs, ts


def test_stagnation_surface_matches_scalar_bisection():
    model = jf.JetModel(m=10.0, a=0.9, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, n_stagnation=25)
    th, rs, ts = _reference_stagnation(model, 25)
    assert np.array_equal(model.thetahorizon_arr, th)
    assert np.array_equal(model.rstag_arr, rs)
    assert np.array_equal(model.tstag_arr, ts)
    # the stagnation point lies between the horizon and the far field
    assert np.all(model.rstag_arr > model.rH)
    assert np.all((model.tstag_arr > 0.0) & (model.tstag_arr <= model.thetahorizon_arr))


def test_stagnation_surface_resolution_converged():
    base = dict(
        m=6.5e9,
        a=0.9,
        inc=163.0,
        mdot=5.45e-5,
        Nx=12,
        Ny=12,
        Nz=60,
        xmin=-30.0,
        xmax=30.0,
        ymin=-30.0,
        ymax=30.0,
        zmin=-120.0,
        zmax=120.0,
    )
    I_1k = jf.JetModel(n_stagnation=1000, **base).make_image(230.0, backend="numpy")[2]
    I_3k = jf.JetModel(n_stagnation=3000, **base).make_image(230.0, backend="numpy")[2]
    assert abs(I_1k.sum() / I_3k.sum() - 1.0) < 1e-4
