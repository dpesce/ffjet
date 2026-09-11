# ffjet

[![Build status](https://github.com/dpesce/ffjet/actions/workflows/ci.yml/badge.svg)](https://github.com/dpesce/ffjet/actions)
[![Python versions](https://img.shields.io/badge/python-3.9|3.10|3.11|3.12|3.13-blue.svg)](https://github.com/dpesce/ffjet)
[![Code coverage](https://codecov.io/gh/dpesce/ffjet/branch/main/graph/badge.svg)](https://codecov.io/gh/dpesce/ffjet)
[![PyPI version](https://img.shields.io/pypi/v/ffjet.svg)](https://pypi.org/project/ffjet/)
[![conda-forge version](https://img.shields.io/conda/vn/conda-forge/ffjet.svg)](https://anaconda.org/conda-forge/ffjet)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22709525.svg)](https://doi.org/10.5281/zenodo.22709525)

A tool for simulating images and spectral energy distributions (SEDs) of astrophysical jets, particularly the relativistic jets produced by spinning black holes.  The underlying model assumes force-free electrodynamics and a nonthermal population of synchrotron-emitting electrons.  Details are provided in [Pesce et al. (TBD)]().

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

When computing many frequencies on the same model (e.g. an SED), call `model.precompute_state()` once after constructing the model.  This stores the frequency-independent physics of every jet cell (44 bytes per cell; the call returns the memory used), after which each `make_image` call only evaluates the synchrotron coefficients and the transfer integral -- roughly 3x faster again than the compiled kernel alone.  Use `model.clear_state()` to release the memory.

For single images, `model.build_field_table()` tabulates the axisymmetric part of the physics (fields, velocities, cooling, electron normalization) once on a 2-D grid and interpolates it per cell, which makes the compiled kernel about 1.8x faster at the price of a small, controlled interpolation error (total flux ~1e-5, per-pixel 99th percentile ~1e-4 with the default grid; see the docstring for measured values at other resolutions).  It also speeds up `precompute_state()`.  This mode is opt-in; `model.clear_field_table()` returns to the exact kernel.

The stagnation surface used for the field-parallel velocity is tabulated on `n_stagnation` field lines (default 1000; the build is vectorized and costs ~10 ms).

For parameter surveys, `jetfuncs.survey(configs, func)` evaluates `func(model)` for a list of `JetModel` configurations in parallel worker processes, dividing the machine's threads between them; see [examples/simulate_jet_survey.py](./examples/simulate_jet_survey.py).  Because the compiled kernels are already multi-threaded, the gain is modest (about 1.3-1.6x on a 10-core machine) and comes from overlapping the per-model setup and Python overhead.

Synchrotron emissivity and absorption integrals are computed at model construction; pass `stokes="IQV"` to `JetModel` to also build the Stokes Q and V tables (not yet used by the radiative transfer).

## Citing ffjet

If you use ffjet in your research, please cite the paper describing the model (see above), together with the archived version of the code that you used.  Every release is archived on Zenodo: [10.5281/zenodo.22709525](https://doi.org/10.5281/zenodo.22709525) always points to the latest version, and the Zenodo page lists a separate DOI for each individual release.  The "Cite this repository" button on the GitHub page provides a ready-made citation.
