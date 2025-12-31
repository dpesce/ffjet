import numpy as np
import pytest

import jetfuncs as jf


@pytest.fixture(scope="module")
def tiny_model():
    # the idea is to keep this small so that CI stays fast
    # pretab_dir=None forces “package data” loading
    return jf.JetModel(
        m=10.0,
        a=0.5,
        inc=60.0,
        mdot=1e-3,
        Nx=4,
        Ny=4,
        Nz=20,
        xmin=0.2,
        xmax=5.0,
        ymin=0.2,
        ymax=5.0,
        zmin=0.2,
        zmax=10.0,
        pretab_dir=None,
    )


def test_synchrotron_tables_loaded(tiny_model):
    # Basic “tables exist and interpolator works” check
    val = np.asarray(tiny_model.GIx_2(1.0)).astype(float)
    assert np.all(np.isfinite(val))


def test_make_image_tiny_runs_and_returns_finite_output(tiny_model):
    x, y, I_nu = tiny_model.make_image(230.0)

    assert x.ndim == 1 and y.ndim == 1
    assert I_nu.shape == (len(y), len(x))

    # something should be finite
    assert np.isfinite(I_nu).sum() > 0

    # intensity should not go negative
    assert np.nanmin(I_nu) >= 0.0
