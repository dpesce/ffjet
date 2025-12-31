import numpy as np
import numpy.testing as npt
import pytest

import jetfuncs as jf


class _DummyModel:
    """Minimal object containing only what convert_units() needs in the xy-provided path."""

    def __init__(self, rg=1.0):
        self.rg = float(rg)

        # prevent warning about custom input grids
        self.x_im_1D_input = None
        self.y_im_1D_input = None
        self.z_im_1D_input = None

        # unused in xy-provided path, but harmless to define
        self.use_log_xgrid = False
        self.use_log_ygrid = False
        self.use_log_zgrid = False
        self.xmin = 0.0
        self.ymin = 0.0


def test_interp_to_regular_grid_raises_on_shape_mismatch():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([-1.0, 1.0])
    I_nu = np.zeros((3, 3))  # should be (len(y), len(x)) = (2, 3)

    with pytest.raises(ValueError):
        jf.interp_to_regular_grid(I_nu, x, y, nx=4, ny=5)


def test_interp_to_regular_grid_sorts_axes_and_returns_expected_shape():
    # Plane: I(x, y) = x + 2y, but feed x/y unsorted to exercise sorting logic
    x = np.array([2.0, 0.0, 1.0])  # unsorted
    y = np.array([1.0, -1.0, 0.0])  # unsorted

    X, Y = np.meshgrid(x, y, indexing="xy")
    I_nu = X + 2.0 * Y  # shape (len(y), len(x))

    x_new = np.linspace(0.0, 2.0, 5)
    y_new = np.linspace(-1.0, 1.0, 7)

    xg, yg, I_new = jf.interp_to_regular_grid(I_nu, x, y, x_new=x_new, y_new=y_new, method="linear")

    assert I_new.shape == (len(y_new), len(x_new))
    npt.assert_allclose(I_new[0, 0], x_new[0] + 2.0 * y_new[0], rtol=1e-12, atol=1e-12)
    npt.assert_allclose(I_new[-1, -1], x_new[-1] + 2.0 * y_new[-1], rtol=1e-12, atol=1e-12)


def test_convert_units_tb_requires_frequency():
    I_nu = np.ones((2, 2), dtype=float)
    with pytest.raises(Exception):
        jf.convert_units(None, I_nu, output_units="Tb")


def test_convert_units_flux_requires_distance():
    model = _DummyModel(rg=1.0)
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    I_nu = np.ones((len(y), len(x)), dtype=float)

    with pytest.raises(Exception):
        jf.convert_units(model, I_nu, x, y, output_units="flux", D=None)


def test_convert_units_luminosity_simple_uniform_grid():
    model = _DummyModel(rg=1.0)
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    I_nu = np.ones((len(y), len(x)), dtype=float)

    Lnu = jf.convert_units(model, I_nu, x, y, output_units="luminosity")
    assert Lnu.shape == I_nu.shape

    # dx = dy = 1, dA = 1; rg=1 -> Lnu == I_nu for this simple case
    npt.assert_allclose(Lnu, I_nu, rtol=0.0, atol=0.0)
