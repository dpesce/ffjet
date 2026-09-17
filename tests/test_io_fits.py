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


@pytest.mark.parametrize("cube", [False, True])
@pytest.mark.parametrize("nx, ny", [(5, 7), (6, 7), (5, 8), (6, 8)])
def test_export_fits_wcs_gives_every_pixel_its_own_centre(tmp_path, nx, ny, cube):
    # The reference pixel sits at the image centre, which for an even number of pixels
    # falls halfway between two pixel centres; the WCS must still map every pixel to its
    # own centre.  The grid is deliberately offset from zero and different along x and y.
    from astropy.wcs import WCS

    x = -3.0e-6 + (np.arange(nx) + 0.5) * 1.0e-6
    y = -2.0e-6 + (np.arange(ny) + 0.5) * 0.5e-6
    image = np.zeros((4, ny, nx)) if cube else np.zeros((ny, nx))

    out = tmp_path / "wcs.fits"
    jf.export_fits(out, image, x, y)

    with fits.open(out) as hdul:
        w = WCS(hdul[0].header)
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    if cube:
        wx, wy, _ = w.wcs_pix2world(i, j, np.zeros_like(i), 0)
    else:
        wx, wy = w.wcs_pix2world(i, j, 0)

    deg = 180.0 / np.pi
    assert np.allclose(wx, x[None, :] * deg, rtol=0.0, atol=1.0e-6 * 1.0e-6 * deg)
    assert np.allclose(wy, y[:, None] * deg, rtol=0.0, atol=1.0e-6 * 0.5e-6 * deg)
