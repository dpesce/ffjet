"""Stagnation-surface solve (bracketed in log r, Omega_F held fixed along each field line)."""

import numpy as np
import pytest

import jetfuncs as jf


def _field_line_theta(rH, nu, th_H, r):
    """Polar angle at radius r on the field line whose footpoint is th_H."""
    omc = np.clip(2.0 * np.sin(0.5 * th_H) ** 2 * (rH / r) ** nu, 0.0, 2.0)
    return 2.0 * np.arcsin(np.sqrt(0.5 * omc))


@pytest.mark.parametrize("a,s", [(0.9, 0.6), (0.01, 0.3), (0.99, 0.9), (0.5, 0.05)])
def test_stagnation_surface_is_a_bracketed_root(a, s):
    """
    Implementation-independent check: every tabulated point must lie on its own field
    line and bracket a sign change of the field-parallel energy derivative.
    """
    model = jf.JetModel(m=10.0, a=a, inc=60.0, mdot=1e-3, s=s, Nx=2, Ny=2, Nz=2,
                        n_stagnation=200)
    rH, nu = model.rH, model.nu
    th_H, rs, ts = model.thetahorizon_arr, model.rstag_arr, model.tstag_arr
    assert np.all(np.isfinite(rs))
    assert np.all(rs > rH)
    assert np.all((ts > 0.0) & (ts <= th_H))
    # the tabulated (r, theta) lies on the field line
    np.testing.assert_allclose(ts, _field_line_theta(rH, nu, th_H, rs), rtol=1e-9, atol=0.0)
    # ... and N' changes sign across it.  Restricted to field lines that stagnate within
    # 1e4 r_g: further out the field line hugs the axis (theta < 1e-8), Nderiv is then a
    # difference of terms that cancel to ~1e-13, and the root is only located to order of
    # magnitude.  Those lines lie far outside any imaging domain -- replacing the whole
    # table changes images by <1e-12 -- but their sign structure is roundoff.
    Om = jf.omega_BZpower(0, jf.psiBZpower(rH, th_H, nu), a, nu)
    eps = 1.0e-4
    inner = jf.Nderiv(rs * (1 - eps), _field_line_theta(rH, nu, th_H, rs * (1 - eps)),
                      a, Om, 1.0, model.bf)
    outer = jf.Nderiv(rs * (1 + eps), _field_line_theta(rH, nu, th_H, rs * (1 + eps)),
                      a, Om, 1.0, model.bf)
    resolved = rs < 1.0e4
    assert resolved.sum() > 10
    assert np.all((np.sign(inner) * np.sign(outer))[resolved] <= 0.0)


def test_stagnation_surface_reports_when_no_root_is_bracketed():
    """A root outside the scanned radius range must raise, not come back as inf."""
    old = jf.JetModel._STAG_R_MAX
    try:
        jf.JetModel._STAG_R_MAX = 3.0  # far too small to contain the outer roots
        with pytest.raises(RuntimeError, match="no bracketed root"):
            jf.JetModel(m=10.0, a=0.9, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, n_stagnation=25)
    finally:
        jf.JetModel._STAG_R_MAX = old


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
