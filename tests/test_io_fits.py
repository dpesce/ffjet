import numpy as np
import pytest
from astropy.io import fits

import jetfuncs as jf


def test_export_fits_writes_file_and_axes_extension(tmp_path):
    # use a uniform grid in radians (note len(x) != len(y) on purpose)
    x = np.linspace(-1e-6, 1e-6, 5)
    y = np.linspace(-2e-6, 2e-6, 7)
    I_nu = np.arange(len(y) * len(x), dtype=float).reshape(len(y), len(x))

    out = tmp_path / "img.fits"
    jf.export_fits(
        out,
        I_nu,
        x.copy(),  # export_fits scales x/y in-place, so pass copies
        y.copy(),
        source_name="test",
        observing_frequency_hz=230.0e9,
        bunit="Jy/pix",
    )

    with fits.open(out) as hdul:
        assert hdul[0].data.shape == (len(y), len(x))

        hdr = hdul[0].header
        assert hdr["CTYPE1"] == "XOFFSET"
        assert hdr["CTYPE2"] == "YOFFSET"
        assert hdr["CUNIT1"].strip().lower() == "deg"
        assert hdr["CUNIT2"].strip().lower() == "deg"
        assert hdr["OBJECT"] == "test"
        assert hdr["OBSFREQ"] == pytest.approx(230.0e9)

        # Verify AXES extension exists
        assert "AXES" in [hdu.name for hdu in hdul]
        axes = hdul["AXES"].data

        deg = 180.0 / np.pi

        # Astropy/FITS tables require a single row count, so if len(x) != len(y),
        # one column can be padded; compare only the meaningful prefix
        xcol = np.asarray(axes["x_offset_centers"], dtype=float)
        ycol = np.asarray(axes["y_offset_centers"], dtype=float)

        assert np.allclose(xcol[: len(x)], x * deg)
        assert np.allclose(ycol[: len(y)], y * deg)

        # if there is padding, it should be zeros
        if len(xcol) > len(x):
            assert np.allclose(xcol[len(x) :], 0.0)
        if len(ycol) > len(y):
            assert np.allclose(ycol[len(y) :], 0.0)


def test_export_fits_rejects_nonuniform_grid(tmp_path):
    # non-uniform spacing in x should raise
    x = np.array([0.0, 1e-6, 3e-6])
    y = np.array([0.0, 1e-6, 2e-6])
    I_nu = np.zeros((len(y), len(x)), dtype=float)

    out = tmp_path / "bad.fits"
    with pytest.raises(ValueError, match="not uniformly spaced"):
        jf.export_fits(out, I_nu, x.copy(), y.copy())
