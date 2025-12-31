import numpy as np
import jetfuncs as jf


def _safe_r_theta():
    """
    pick (r, theta) pairs that avoid theta_fp NaNs for default s=0.6 (nu=0.8),
    i.e. keep the BZ footpoint trig arguments inside [-1, 1]
    """
    r = np.array([2.2, 3.0, 3.8], dtype=float)
    theta = np.array([np.pi / 6.0, np.pi / 4.0, np.pi / 3.0], dtype=float)
    return r, theta


def test_get_quantity_basic_fields_and_poynting():
    model = jf.JetModel(
        m=10.0,
        a=0.5,
        inc=60.0,
        mdot=1e-3,
        Nx=2,
        Ny=2,
        Nz=2,
        pretab_dir=None,
    )

    r, th = _safe_r_theta()

    psi = model.get_quantity(r, th, quantity="psi")
    Omega = model.get_quantity(r, th, quantity="Omega")
    Bmag = model.get_quantity(r, th, quantity="Bmag")
    S = model.get_quantity(r, th, quantity="Poynting")  # name is "Poynting"

    for arr in (psi, Omega, Bmag, S):
        assert np.shape(arr) == np.shape(r)
        assert np.all(np.isfinite(arr))

    # Poynting is constructed non-negative
    assert np.all(S >= 0.0)


def test_get_quantity_Bprime_reduces_to_lab_B_when_velocity_suppressed():
    # betagamma_suppression=0 drives beta->0 and triggers the small-v overwrite:
    # Bprime_*[~mask_v] = B*
    model = jf.JetModel(
        m=10.0,
        a=0.5,
        inc=60.0,
        mdot=1e-3,
        Nx=2,
        Ny=2,
        Nz=2,
        betagamma_suppression=0.0,
        pretab_dir=None,
    )

    r, th = _safe_r_theta()

    Bx = model.get_quantity(r, th, quantity="Bx")
    By = model.get_quantity(r, th, quantity="By")
    Bz = model.get_quantity(r, th, quantity="Bz")

    Bx_p = model.get_quantity(r, th, quantity="Bx_prime")
    By_p = model.get_quantity(r, th, quantity="By_prime")
    Bz_p = model.get_quantity(r, th, quantity="Bz_prime")

    for arr in (Bx_p, By_p, Bz_p):
        assert np.shape(arr) == np.shape(r)
        assert np.all(np.isfinite(arr))

    assert np.allclose(Bx_p, Bx, rtol=0.0, atol=0.0)
    assert np.allclose(By_p, By, rtol=0.0, atol=0.0)
    assert np.allclose(Bz_p, Bz, rtol=0.0, atol=0.0)


def test_get_quantity_unknown_returns_empty_dict_for_array_input():
    model = jf.JetModel(m=10.0, a=0.5, inc=60.0, mdot=1e-3, Nx=2, Ny=2, Nz=2, pretab_dir=None)

    # important: need to pass arrays (even length-1),
    # because the code does indexed assignment on gamma
    r = np.array([3.0], dtype=float)
    th = np.array([np.pi / 4.0], dtype=float)

    out = model.get_quantity(r, th, quantity="this_is_not_a_real_quantity")
    assert out == {}
