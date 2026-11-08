"""
The jet segments: each line of sight is inside the jet along a few continuous segments,
whose ends are located by bisection, and a grid cell cut by the jet boundary is
evaluated only on its part inside the jet.  This removes the "staircase" that counting
whole cells by their centers produced: the path length inside the jet, and with it the
emission, jumped by up to a cell whenever the boundary crossed a cell edge.
"""

import numpy as np
import pytest

import jetfuncs as jf

numba_only = pytest.mark.skipif(not jf.numba_available(), reason="numba is not installed")

CFG = dict(
    m=6.5e9, a=0.9, inc=163.0, mdot=5.45e-5, s=0.6, p=2.14, h=0.001, eta=0.03, gamma_inf=4.0,
    Nx=12, Ny=10, Nz=96, xmin=-60.0, xmax=60.0, ymin=-60.0, ymax=60.0, zmin=-240.0, zmax=240.0,
)


def _scan_points(model):
    z = model.z_im_1D
    scan = np.empty(2 * z.shape[0] - 1)
    scan[0::2] = z
    scan[1::2] = 0.5 * (z[1:] + z[:-1])
    return scan


def _in_segments(s, seg_in, seg_out, k):
    return np.any((s >= seg_in[:k]) & (s <= seg_out[:k]))


@pytest.mark.parametrize("cutout", [0.0, 0.3])
def test_segments_agree_with_the_jet_criterion(cutout):
    """Every sampled point inside the jet lies in a segment, and every point outside in none."""
    model = jf.JetModel(**CFG, jet_cutout_fraction=cutout)
    seg_in, seg_out, nint = model._jet_segments_numpy()
    scan = _scan_points(model)
    for pix in range(model.x_im_f.shape[0]):
        inside = model._inside_numpy(scan, np.full(scan.shape, pix))
        listed = np.array([_in_segments(s, seg_in[pix], seg_out[pix], nint[pix]) for s in scan])
        np.testing.assert_array_equal(inside, listed)


def test_segment_ends_are_boundary_crossings():
    """Just inside each end of a segment the ray is in the jet; just outside, it is not."""
    model = jf.JetModel(**CFG)
    seg_in, seg_out, nint = model._jet_segments_numpy()
    lo_grid, hi_grid = model.z_im_1D[0], model.z_im_1D[-1]
    checked = 0
    for pix in np.flatnonzero(nint):
        for k in range(nint[pix]):
            for end, sign in ((seg_in[pix, k], -1.0), (seg_out[pix, k], 1.0)):
                if end in (lo_grid, hi_grid):
                    continue
                eps = 1e-9 * max(1.0, abs(end))
                p = np.array([pix, pix])
                assert model._inside_numpy(np.array([end - sign * eps]), p[:1])[0]
                assert not model._inside_numpy(np.array([end + sign * eps]), p[:1])[0]
                checked += 1
    assert checked > 10


@numba_only
def test_stored_path_lengths_add_up_to_the_segments():
    """The cells of a ray, cut at the segment ends, cover exactly the segments."""
    model = jf.JetModel(**CFG)
    model.precompute_state()
    iv = model._get_intervals()
    st = model._state
    for q, pix in enumerate(iv["order"]):
        k = iv["nint"][pix]
        total = np.sum(iv["seg_out"][pix, :k] - iv["seg_in"][pix, :k]) * model.rg
        stored = st["ds"][st["offsets"][q]:st["offsets"][q + 1]].sum()
        assert stored == pytest.approx(total, rel=1e-12, abs=1e-300)


def test_path_length_inside_the_jet_has_no_staircase():
    """
    Along a row of pixels across the jet edge, the path length inside the jet follows a
    brute-force estimate closely with the segments, but jumps by up to a cell when whole
    cells are counted by their centers -- the old staircase.
    """
    model = jf.JetModel(**dict(CFG, Nx=40, Ny=2, xmin=-40.0, xmax=40.0, ymin=-25.0, ymax=25.0))
    seg_in, seg_out, nint = model._jet_segments_numpy()
    seg_len = np.array([np.sum(seg_out[p, :nint[p]] - seg_in[p, :nint[p]])
                        for p in range(model.x_im_f.shape[0])])
    centers = 0.5 * (model.z_im_1D[1:] + model.z_im_1D[:-1])
    widths = np.diff(model.z_im_1D)
    fine = np.linspace(model.z_im_1D[0], model.z_im_1D[-1], 400001)
    step = fine[1] - fine[0]
    worst_seg, worst_cells = 0.0, 0.0
    for pix in range(model.x_im_f.shape[0]):
        brute = model._inside_numpy(fine, np.full(fine.shape, pix)).sum() * step
        whole = np.sum(widths[model._inside_numpy(centers, np.full(centers.shape, pix))])
        worst_seg = max(worst_seg, abs(seg_len[pix] - brute))
        worst_cells = max(worst_cells, abs(whole - brute))
    cell = widths.max()
    assert worst_seg < 5.0 * step          # the brute force's own resolution
    assert worst_cells > 0.2 * cell        # the staircase: a sizable fraction of a cell
