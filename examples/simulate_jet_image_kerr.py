###################################################
#                                                 #
# This example script makes full-Stokes images of #
# the core of the jet twice: once along straight  #
# lines of sight (the default), and once along    #
# null geodesics of the Kerr metric, which adds   #
# light bending, the photon's local direction and #
# energy (including frame dragging), and the      #
# transport of the polarization basis along each  #
# ray.  It plots total intensity and fractional   #
# circular polarization for both, and their       #
# ratio.                                          #
#                                                 #
###################################################

###################################################
# imports

import numpy as np
import jetfuncs as jf
import matplotlib.pyplot as plt

###################################################
# inputs that control the jet model

# BH mass, in solar masses
m = 6.2e9

# dimensionless spin
a = 0.9

# mass accretion rate
mdot = 5.45e-5

# inclination angle, in degrees
inc = 163.0

# jet collimation parameter
s = 0.6

# power-law index for injected electrons
p = 2.2

# jet power heating fraction
h = 0.0025

# anisotropy parameter; eta=1 is isotropic
eta = 0.01

###################################################
# inputs that control the image

# observing frequency, in GHz
frequency = 86.0

# a linear pixel grid around the black hole, in gravitational radii
Nx = Ny = 200
xmax = ymax = 50.0

# depth (z) direction; relevant for radiative transfer
zmin = 0.1
zmax = 1.0e8
Nz = 1600

###################################################
# the two images

###################################
# geometry="kerr" traces each ray #
# backwards from the observer     #
# along a null geodesic of the    #
# Kerr metric with the model's    #
# spin; everything else about the #
# model is the same.  It needs    #
# numba, which traces the rays.   #
###################################

images = {}
for geometry in ("flat", "kerr"):
    model = jf.JetModel(
        m=m,
        a=a,
        inc=inc,
        mdot=mdot,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        xmax=xmax,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        s=s,
        p=p,
        h=h,
        eta=eta,
        use_log_zgrid=True,
        geometry=geometry,
    )
    x, y, IQUV = model.make_image_polarized(frequency)
    images[geometry] = jf.sky_view(x, y, IQUV)
    print("%-4s: total I %.4e, V/I %+.3f %% (cgs intensity summed over pixels)"
          % (geometry, IQUV[0].sum(), 100.0 * IQUV[3].sum() / IQUV[0].sum()))

###################################################
# plot

fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.6))
x, y, flat = images["flat"]
_, _, kerr = images["kerr"]
vmax = np.log10(max(flat[0].max(), kerr[0].max()))
circular = {}
for row, (name, IQUV) in enumerate((("flat", flat), ("Kerr", kerr))):
    floor = IQUV[0].max() * 1.0e-3
    m_lin, m_circ, chi = jf.polarization_fractions(IQUV, floor=floor)
    circular[name] = m_circ

    ax = axes[row, 0]
    pc = ax.pcolormesh(x, y, np.log10(IQUV[0]), cmap="afmhot", vmin=vmax - 3.0, vmax=vmax)
    plt.colorbar(pc, ax=ax, label=r"$\log(I_\nu)$")
    ax.set_title("%s: total intensity" % name)

    ax = axes[row, 1]
    pc = ax.pcolormesh(x, y, 100.0 * m_circ, cmap="RdBu_r", vmin=-2.0, vmax=2.0)
    plt.colorbar(pc, ax=ax, label="circular polarization (%)")
    ax.set_title("%s: fractional circular polarization" % name)

# what Kerr changes
ax = axes[0, 2]
ratio = np.where(flat[0] > 0.0, kerr[0] / np.where(flat[0] > 0.0, flat[0], 1.0), np.nan)
pc = ax.pcolormesh(x, y, ratio, cmap="RdBu_r", vmin=0.0, vmax=2.0)
plt.colorbar(pc, ax=ax, label="Kerr / flat")
ax.set_title("total intensity, Kerr / flat")

ax = axes[1, 2]
pc = ax.pcolormesh(x, y, 100.0 * (circular["Kerr"] - circular["flat"]), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
plt.colorbar(pc, ax=ax, label="percentage points")
ax.set_title("circular polarization, Kerr - flat")

for ax in axes.flat:
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ ($r_g$)")
    ax.set_ylabel(r"$y$ ($r_g$)")
    ax.set_facecolor("black")

plt.tight_layout()
plt.savefig("jet_image_kerr.png", dpi=200, bbox_inches="tight")
plt.close()

###################################################
