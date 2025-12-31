import numpy as np
import jetfuncs as jf


class DummyModel:
    def __init__(
        self,
        *,
        x_im_1D,
        y_im_1D,
        xmin,
        ymin,
        use_log_xgrid,
        use_log_ygrid,
        rg,
    ):
        self.x_im_1D = np.asarray(x_im_1D, dtype=float)
        self.y_im_1D = np.asarray(y_im_1D, dtype=float)
        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.use_log_xgrid = bool(use_log_xgrid)
        self.use_log_ygrid = bool(use_log_ygrid)
        self.rg = float(rg)

        # needed because convert_units() checks these early
        self.x_im_1D_input = None
        self.y_im_1D_input = None
        self.z_im_1D_input = None


def _log_pixel_widths_1d(arr: np.ndarray, amin: float) -> np.ndarray:
    """
    mirror the log-grid pixel-width logic in convert_units() for len(xy)==0.
    """
    out = np.zeros_like(arr, dtype=float)

    for i in range(len(arr)):
        dxhere = np.nan

        if i == 0:
            hi = np.sqrt(arr[i + 1] / arr[i]) * arr[i]
            lo = np.sqrt(arr[i] / arr[i + 1]) * arr[i]
            dxhere = hi - lo
        elif (arr[i - 1] > 0.0) and (i < (len(arr) - 1)):
            hi = np.sqrt(arr[i + 1] / arr[i]) * arr[i]
            lo = np.sqrt(arr[i - 1] / arr[i]) * arr[i]
            dxhere = hi - lo
        elif (arr[i - 1] > 0.0) and (i == (len(arr) - 1)):
            hi = np.sqrt(arr[i] / arr[i - 1]) * arr[i]
            lo = np.sqrt(arr[i - 1] / arr[i]) * arr[i]
            dxhere = hi - lo
        elif (arr[i + 1] < 0.0) and (i > 0):
            hi = np.sqrt(arr[i + 1] / arr[i]) * arr[i]
            lo = np.sqrt(arr[i - 1] / arr[i]) * arr[i]
            dxhere = hi - lo

        if arr[i] == amin:
            hi = np.sqrt(arr[i + 1] / arr[i]) * arr[i]
            lo = 0.0
            dxhere = hi - lo

        if arr[i] == -amin:
            hi = 0.0
            lo = np.sqrt(arr[i - 1] / arr[i]) * arr[i]
            dxhere = hi - lo

        out[i] = dxhere

    return out


def test_convert_units_lenxy0_log_grids_hits_edge_cases():
    # symmetric log-ish grids including exactly ±xmin and ±ymin
    x = np.array([-1.0, -0.1, 0.1, 1.0], dtype=float)
    y = np.array([-2.0, -0.2, 0.2, 2.0], dtype=float)

    m = DummyModel(
        x_im_1D=x,
        y_im_1D=y,
        xmin=0.1,
        ymin=0.2,
        use_log_xgrid=True,
        use_log_ygrid=True,
        rg=1.0,
    )

    I_nu = np.ones((len(y), len(x)), dtype=float)

    # no (x, y) args -> convert_units should go down the len(xy)==0 branch
    Lnu = jf.convert_units(m, I_nu, output_units="luminosity")

    assert Lnu.shape == I_nu.shape
    assert np.all(np.isfinite(Lnu))

    dx = _log_pixel_widths_1d(x, m.xmin)
    dy = _log_pixel_widths_1d(y, m.ymin)

    expected = np.outer(dy, dx)  # rg==1 and I_nu==1
    assert np.allclose(Lnu, expected, rtol=1e-12, atol=0.0)
