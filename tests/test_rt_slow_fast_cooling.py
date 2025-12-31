import numpy as np
import jetfuncs as jf


def _make_tiny_model(
    *, gamma_m: float, gamma_max: float, mdot: float, zmin: float, zmax: float
) -> jf.JetModel:
    return jf.JetModel(
        m=10.0,
        a=0.5,
        inc=60.0,
        mdot=mdot,
        Nx=2,
        Ny=2,
        Nz=6,
        xmin=-5.0,
        xmax=5.0,
        ymin=-5.0,
        ymax=5.0,
        zmin=zmin,
        zmax=zmax,
        use_log_xgrid=False,
        use_log_ygrid=False,
        use_log_zgrid=False,
        gamma_m=gamma_m,
        gamma_max=gamma_max,
        pretab_dir=None,
    )


def test_make_image_exercises_slow_cooling_path():
    """
    slow-cooling path sets GaIx_p1 = GaIx_p (slow-only), so GaIx_p must be called at least once
    If the default params don't hit slow cooling, try a few "more slow-cooling" configs
    """
    tried = []
    for mdot in (1e-3, 1e-6, 1e-9):
        for zmin, zmax in ((1e-3, 1e-1), (1e-2, 1.0), (1.0, 20.0)):
            model = _make_tiny_model(gamma_m=2.0, gamma_max=1e8, mdot=mdot, zmin=zmin, zmax=zmax)

            calls = {"GaIx_p": 0}
            original = model.GaIx_p

            def wrapped(x):
                calls["GaIx_p"] += 1
                return original(x)

            model.GaIx_p = wrapped

            _x, _y, I_nu = model.make_image(
                230.0, show_progress=False, heating_prescription="Poynting"
            )
            assert I_nu.shape == (len(_y), len(_x))
            assert np.all(np.isfinite(I_nu))

            tried.append((mdot, zmin, zmax, calls["GaIx_p"]))
            if calls["GaIx_p"] > 0:
                return

    # If we got here, none of the configs triggered slow cooling.
    raise AssertionError(f"Failed to trigger slow cooling (GaIx_p calls stayed 0). Tried: {tried}")


def test_make_image_exercises_fast_cooling_path():
    """
    fast-cooling path sets GaIx_p1 = GaIx_2,
    so wrap GaIx_2 and assert it gets called
    """
    model = _make_tiny_model(gamma_m=1.0e40, gamma_max=1.0e45, mdot=1e-3, zmin=1.0, zmax=20.0)

    calls = {"GaIx_2": 0}
    original = model.GaIx_2

    def wrapped(x):
        calls["GaIx_2"] += 1
        return original(x)

    model.GaIx_2 = wrapped

    _x, _y, I_nu = model.make_image(230.0, show_progress=False, heating_prescription="Poynting")
    assert I_nu.shape == (len(_y), len(_x))
    assert np.all(np.isfinite(I_nu))
    assert calls["GaIx_2"] > 0
