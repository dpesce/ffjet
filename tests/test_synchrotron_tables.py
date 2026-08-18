import numpy as np
import pytest
from scipy.special import gamma as _gamma

import jetfuncs as jf


def _plateau(kernel: str, m: float) -> float:
    """
    Closed form for int_0^inf x^m kernel(x) dx, i.e. the x -> 0 limit of the tabulated
    integrals.  Derived by swapping the order of integration and using
    int_0^inf x^(a-1) K_nu(x) dx = 2^(a-2) Gamma((a-nu)/2) Gamma((a+nu)/2).
    """
    if kernel == "F":  # F(x) = x int_x^inf K_{5/3}(y) dy
        return 2.0 ** (m + 1.0) / (m + 2.0) * _gamma(m / 2 + 2 / 3) * _gamma(m / 2 + 7 / 3)
    if kernel == "G":  # G(x) = x K_{2/3}(x)
        return 2.0**m * _gamma(m / 2 + 2 / 3) * _gamma(m / 2 + 4 / 3)
    if kernel == "H":  # H(x) = int_x^inf K_{1/3}(y) dy + x K_{1/3}(x)
        return 2.0**m * _gamma(m / 2 + 5 / 6) * _gamma(m / 2 + 7 / 6) * (m + 2.0) / (m + 1.0)
    raise ValueError(kernel)


@pytest.fixture(scope="module")
def model():
    return jf.JetModel(m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, p=2.14, stokes="IQV")


def test_all_stokes_tables_are_built(model):
    for family in jf.JetModel._TAB_FAMILIES:
        for suffix in ("_2", "_p", "_pp1"):
            assert hasattr(model, family + suffix), f"missing table {family + suffix}"


def test_tables_match_analytic_plateaus(model):
    """
    The x -> 0 limit of every tabulated integral has a closed form, which pins both the
    normalization and the kernel definitions.  (The tables shipped before v0.2 were
    0.5-1.3 percent below these values.)
    """
    p = model.p
    for family, (_stokes, kernel, exponent) in jf.JetModel._TAB_FAMILIES.items():
        for suffix, pval in (("_2", 2.0), ("_p", p), ("_pp1", p + 1.0)):
            table = getattr(model, family + suffix)
            expected = _plateau(kernel, exponent(pval))
            assert table[0] == pytest.approx(expected, rel=1e-4), family + suffix


def test_absorption_families_equal_shifted_emission_families(model):
    """Ga_X^(p)(x) = G_X^(p+1)(x) for X in I, Q, V."""
    for stokes in "IQV":
        npt_a = getattr(model, f"Ga{stokes}_p")
        npt_b = getattr(model, f"G{stokes}_pp1")
        assert np.array_equal(npt_a, npt_b)


def test_lookup_reproduces_the_stored_table(model):
    """The log-log direct-index lookup must return the tabulated values at the nodes."""
    x = model.x_GI_p[5:-5]
    assert np.allclose(model.GIx_p(x), model.GI_p[5:-5], rtol=1e-10)
    assert np.allclose(model.GQx_p(x), model.GQ_p[5:-5], rtol=1e-10)
    assert np.allclose(model.GVx_p(x), model.GV_p[5:-5], rtol=1e-10)


def test_lookup_edges(model):
    """Below the table the plateau is extended; above it the integral is zero."""
    assert model.GIx_p(1e-30) == pytest.approx(model.GI_p[0], rel=1e-12)
    assert float(model.GIx_p(1e6)) == 0.0


def test_polarized_tables_absent_unless_requested():
    m = jf.JetModel(m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, stokes="I")
    assert not hasattr(m, "GQ_p")
    with pytest.raises(AttributeError, match="stokes"):
        m.GQx_p(1.0)


def test_p_must_exceed_two():
    with pytest.raises(ValueError, match="must exceed 2"):
        jf.JetModel(m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, p=2.0)


def test_arbitrary_p_is_allowed():
    """p is no longer restricted to the 0.01-wide grid the shipped tables used."""
    m = jf.JetModel(m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, p=2.137)
    assert np.isfinite(float(m.GIx_p(1.0)))
