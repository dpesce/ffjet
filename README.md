# ffjet

[![Build status](https://github.com/dpesce/ffjet/actions/workflows/ci.yml/badge.svg)](https://github.com/dpesce/ffjet/actions)
[![Python versions](https://img.shields.io/badge/python-3.9|3.10|3.11|3.12|3.13-blue.svg)](https://github.com/dpesce/ffjet)
[![Code coverage](https://codecov.io/gh/dpesce/ffjet/branch/main/graph/badge.svg)](https://codecov.io/gh/dpesce/ffjet)
[![PyPI version](https://img.shields.io/pypi/v/ffjet.svg)](https://pypi.org/project/ffjet/)
[![conda-forge version](https://img.shields.io/conda/vn/conda-forge/ffjet.svg)](https://anaconda.org/conda-forge/ffjet)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22709525.svg)](https://doi.org/10.5281/zenodo.22709525)

A tool for simulating images and spectral energy distributions (SEDs) of astrophysical jets, particularly the relativistic jets produced by spinning black holes.  The underlying model assumes force-free electrodynamics and a nonthermal population of synchrotron-emitting electrons, and it can produce either total-intensity or full-Stokes (polarized) images.  Details are provided in [Pesce et al. (TBD)]().

## Installation

The code is only tested on Python 3.9 and higher, and it may break for earlier versions.  The latest release can be installed from PyPI:

```
    pip install ffjet
```

or from conda-forge:

```
    conda install -c conda-forge ffjet
```

There is an optional progress bar functionality that uses [`tqdm`](https://tqdm.github.io/); it can be installed together with ffjet using `pip install "ffjet[progress]"`, or using `conda install -c conda-forge ffjet tqdm`.

To install the development version from source instead, it is recommended that you use a virtual environment, e.g.:

```
    $ git clone https://github.com/dpesce/ffjet
    $ python -m venv .venv
    $ source .venv/bin/activate
    (.venv) $ cd ffjet
    (.venv) $ pip install .
```

You can also install the development version directly from GitHub:

```
    pip install "git+https://github.com/dpesce/ffjet.git"
```

## Checking that it works

There are a number of example scripts contained in the [examples](./examples/) folder. You can check to make sure your installation is working by running one of these scripts, e.g.:

```
    cd ./examples
    python ./simulate_jet_image.py
```

## Performance notes

The radiative transfer has two interchangeable back ends, selected with the `backend` argument of `JetModel` (or per call in `make_image`):

- `backend="numba"` (the default whenever [numba](https://numba.pydata.org/) is installed): a compiled kernel that integrates each line of sight through the cells that lie inside the jet, running rays in parallel across all cores.  The very first call on a given machine compiles the kernel (~5-10 s); the compiled code is cached on disk, so later sessions start in under a second.  The number of threads can be set with `jetfuncs.set_num_threads(n)`.
- `backend="numpy"`: the original vectorized loop over depth slices.  It is an order of magnitude slower and is kept as the reference implementation; the two back ends agree to floating-point roundoff.

Each line of sight is cut at the jet boundary: the segments of the ray that lie inside the jet are located once per model (by bisection, to roundoff), and a depth-grid cell that the boundary cuts counts only its part inside the jet.  Earlier versions counted whole cells by their centers, so the path length inside the jet jumped by up to a cell from one pixel to the next; in the outer jet of an M87-like model at 43 GHz this left Stokes U off by 13% rms per pixel at `Nz = 1600` and by 1.6% at `Nz = 12800`, against 0.03% now at `Nz = 1600`.

When computing many frequencies on the same model (e.g. an SED), call `model.precompute_state()` once after constructing the model.  This stores the frequency-independent physics of every jet cell (48 bytes per cell; the call returns the memory used), after which each `make_image` call only evaluates the synchrotron coefficients and the transfer integral -- roughly 3x faster again than the compiled kernel alone.  Use `model.clear_state()` to release the memory.  `model.precompute_state(polarized=True)` stores four further quantities per cell (80 bytes per cell) so that `make_image_polarized` can use the cached path as well; the extra state is a superset, and `make_image` reads the same arrays.

For single images, `model.build_field_table()` tabulates the axisymmetric part of the physics (fields, velocities, cooling, electron normalization) once on a 2-D grid and interpolates it per cell, which makes the compiled kernel about 1.8x faster at the price of a small, controlled interpolation error (total flux ~1e-5, per-pixel 99th percentile ~1e-4 with the default grid; see the docstring for measured values at other resolutions).  It also speeds up `precompute_state()`.  This mode is opt-in; `model.clear_field_table()` returns to the exact kernel.

The stagnation surface used for the field-parallel velocity is tabulated on `n_stagnation` field lines (default 1000; the build is vectorized and costs ~10 ms).

For parameter surveys, `jetfuncs.survey(configs, func)` evaluates `func(model)` for a list of `JetModel` configurations in parallel worker processes, dividing the machine's threads between them; see [examples/simulate_jet_survey.py](./examples/simulate_jet_survey.py).  Because the compiled kernels are already multi-threaded, the gain is modest (about 1.3-1.6x on a 10-core machine) and comes from overlapping the per-model setup and Python overhead.

Synchrotron emissivity and absorption integrals are computed at model construction; pass `stokes="IQV"` to `JetModel` to build the Stokes Q and V tables up front (`make_image_polarized` builds them on demand otherwise).

## Polarization

`model.make_image_polarized(frequency)` returns `(x, y, IQUV)` with `IQUV` of shape `(4, Ny, Nx)`, in the same units as `make_image`.  It solves the full 4x4 polarized transfer equation, including polarized emission and absorption in Q and V, Faraday rotation and Faraday conversion, for the same anisotropic double power-law electrons the unpolarized model uses.  Each cell is advanced with the exact constant-coefficient solution of the transfer equation, so no restriction on the Faraday depth per cell is needed.

```python
    x, y, IQUV = model.make_image_polarized(230.0)
    m_lin, m_circ, chi = jf.polarization_fractions(IQUV, floor=IQUV[0].max() * 1e-3)
```

Conventions: Q and U are in the astronomical convention, referred to (North, East), so the EVPA is `0.5*atan2(U, Q)` measured **east of north** and is perpendicular to the projected fluid-frame magnetic field for optically thin emission; V > 0 where the fluid-frame magnetic field points towards the observer.  Use `jf.evpa(Q, U)` and `jf.polarization_fractions(...)` rather than rederiving these.

Before plotting, pass the image through `jf.sky_view(x, y, IQUV)`.  jetfuncs places the observer on the −z side of the image plane, so a raw `pcolormesh(x, y, I)` is a mirror of the sky; `sky_view` re-labels the grid so that north is up and east is to the left.  In such a panel an EVPA of 0 is a vertical tick and the tick direction is `(-sin(EVPA), cos(EVPA))`.  `export_fits` accepts a `(4, Ny, Nx)` cube and writes a FITS STOKES axis.  Stokes I from the polarized transfer is not identical to `make_image`: polarized absorption couples Q, U and V back into I, which is a fraction of a percent where the jet is optically thin and up to several percent where it is thick.  See [examples/simulate_jet_image_polarized.py](./examples/simulate_jet_image_polarized.py).

The polarized emission and absorption coefficients follow Dexter (2016) Appendix A, with the anisotropy corrections of Tsunetoe et al. (2025); the cell solution is the closed form of Landi Degl'Innocenti & Landi Degl'Innocenti (1985) given in Dexter's Appendix D.

The Faraday rotativities are selected with the `rotativities` argument of `JetModel`:

- `rotativities="HS11"` (the default): Huang & Shcherbakov (2011), whose monoenergetic fitting formulae are integrated over the electron distribution.
- `rotativities="JO77"`: the power-law expressions of Jones & Odell (1977) / Sazonov (1969) as given in Dexter's Appendix B1, generalized here so that each segment of the broken power law is integrated over its own range of Lorentz factors.
- `rotativities="none"`: emission and absorption only.

A polarized image costs about 1.7x an unpolarized one with the default `"HS11"` rotativities, 1.8x with `"JO77"` and 1.4x with `"none"` (measured on a 120x120x400 grid, 0.77M jet cells).

## Kerr geodesics

By default the rays are straight lines of sight (`geometry="flat"`).  With `geometry="kerr"`, each ray is instead traced backwards from a distant observer along a null geodesic of the Kerr metric with the model's spin:

```python
    model = jf.JetModel(..., geometry="kerr")
    x, y, IQUV = model.make_image_polarized(86.0)
```

The jet itself is unchanged: its fields, flow, electrons, synchrotron coefficients and transfer were already computed in the Kerr metric.  What the ray geometry adds is:

- light bending: rays that pass near the hole are deflected toward it, so the bright base of the jet appears magnified, and rays inside the shadow end on the horizon;
- the photon's direction and energy in the frame of the local zero-angular-momentum observers, the frame the flow and field are given in, so that the redshift includes frame dragging;
- the polarization basis, carried along each ray by parallel transport (through its Walker-Penrose constant).

The cells of a ray are laid out on the same depth grid as in flat mode, in affine parameter about the ray's closest approach to the jet axis, and cut at the jet boundary.  Both back ends, `precompute_state` and `build_field_table` work in Kerr mode.  It requires numba even with `backend="numpy"`, which takes the rays from the compiled tracer and does the physics and the transfer itself.  The line of sight must be away from the jet axis (`|sin(inc)| >= 1e-6`).  `get_quantity` evaluates its direction-dependent quantities for the straight line of sight.  See [examples/simulate_jet_image_kerr.py](./examples/simulate_jet_image_kerr.py).

The tracer and the Kerr path are checked in `tests/test_geodesics.py` and `tests/test_kerr.py`: against the post-Newtonian deflection of light, Bardeen's shadow and a direct integration of parallel transport; against the flat model, which the Kerr path reproduces to ~1e-6 when its rays are made straight; and against the reflection symmetry of the model and convergence in the integration step.

## Citing ffjet

If you use ffjet in your research, please cite the paper describing the model (see above), together with the archived version of the code that you used.  Every release is archived on Zenodo: [10.5281/zenodo.22709525](https://doi.org/10.5281/zenodo.22709525) always points to the latest version, and the Zenodo page lists a separate DOI for each individual release.
