# ffjet

[![Build status](https://github.com/dpesce/ffjet/actions/workflows/ci.yml/badge.svg)](https://github.com/dpesce/ffjet/actions)
[![Python versions](https://img.shields.io/badge/python-3.9|3.10|3.11|3.12|3.13-blue.svg)](https://github.com/dpesce/ffjet)

A tool for simulating images and spectral energy distributions (SEDs) of astrophysical jets, particularly the relativistic jets produced by spinning black holes.  The underlying model assumes force-free electrodynamics and a nonthermal population of synchrotron-emitting electrons.  Details are provided in [Pesce et al. (TBD)]().

## Installation

The code is only tested on Python 3.9 and higher, and it may break for earlier versions.  It is recommended that you install ffjet using a virtual environment, e.g.:

```
    $ git clone https://github.com/dpesce/jjfet
    $ python -m venv .venv
    $ source .venv/bin/activate
    (.venv) $ pip install .
```

You can also install directly from GitHub:

```
    (.venv) $ pip install "git+https://github.com/dpesce/ffjet.git"
```

There is an optional progress bar functionality that uses `tqdm`; it can be installed using:

```
    (.venv) $ pip install "ffjet[progress] @ git+https://github.com/dpesce/ffjet.git"
```
