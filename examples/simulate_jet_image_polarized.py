###################################################
#                                                 #
# This example script showcases the full-Stokes   #
# (polarized) jet model: it makes an I, Q, U, V   #
# image, converts it to flux units, and plots the #
# total intensity with polarization ticks and the #
# fractional linear and circular polarization.    #
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
frequency = 230.0

# image dimensions, in gravitational radii
xmin = ymin = -100.0
xmax = ymax = 100.0

# number of pixels
Nx = Ny = 200

# depth (z) direction; relevant for radiative transfer
zmin = 0.0
zmax = 400.0
Nz = 800

# distance to the source, in Mpc
D = 16.8

###################################################
# initialize the jet model and generate the image

model = jf.JetModel(
    m=m,
    a=a,
    inc=inc,
    mdot=mdot,
    Nx=Nx,
    Ny=Ny,
    Nz=Nz,
    xmin=xmin,
    xmax=xmax,
    ymin=ymin,
    ymax=ymax,
    zmin=zmin,
    zmax=zmax,
    s=s,
    p=p,
    h=h,
    eta=eta,
)

###################################
# make_image_polarized returns the #
# four Stokes parameters stacked   #
# along a leading axis, in the     #
# same units as make_image.        #
###################################

x, y, IQUV = model.make_image_polarized(frequency, show_progress=True)

###################################################
# re-orient into the observer's view of the sky

###################################
# jetfuncs places the observer on #
# the -z side of the image plane, #
# so plotting (x, y) directly     #
# would give a MIRROR of the sky. #
# sky_view() re-orients the image #
# so that north is up, east is to #
# the left, and the approaching   #
# jet points to the right.  It    #
# re-labels the grid rather than  #
# reflecting the data, so the     #
# Stokes values pass through it   #
# unchanged.  Everything below is #
# in this sky orientation.        #
###################################

x, y, IQUV = jf.sky_view(x, y, IQUV)

###################################################
# convert image units

# flux density, in Jy per pixel (the conversion is linear, so it applies to each
# Stokes parameter separately)
Snu = np.stack([jf.convert_units(model, S, output_units="flux", D=D) * 1.0e23 for S in IQUV])

print("integrated flux density (Jy):")
print("  I = %.4f" % Snu[0].sum())
print("  Q = %+.5f    U = %+.5f    V = %+.5f" % tuple(Snu[k].sum() for k in (1, 2, 3)))

###################################
# Conventions: the EVPA is        #
# measured east of north and is   #
# perpendicular to the projected  #
# magnetic field for optically    #
# thin emission; V > 0 where the  #
# field points at the observer.   #
###################################

m_lin_tot, m_circ_tot, evpa_tot = jf.polarization_fractions(*[S.sum() for S in Snu])
print("image-integrated fractions:")
print("  linear   = %.3f %%" % (100.0 * m_lin_tot))
print("  circular = %+.4f %%" % (100.0 * m_circ_tot))
print("  EVPA     = %+.2f deg" % np.degrees(evpa_tot))

# per-pixel maps; pixels below the floor are left as NaN so they do not colour the plot
floor = IQUV[0].max() * 1.0e-3
m_lin, m_circ, chi = jf.polarization_fractions(IQUV, floor=floor)

###################################################
# plot

Tb = jf.convert_units(model, IQUV[0], output_units="Tb", frequency=frequency)

fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))

# --- total intensity with polarization ticks
ax = axes[0]
ax.set_facecolor("black")
ax.set_aspect("equal")
vmax = np.log10(Tb.max())
pc = ax.pcolormesh(x, y, np.log10(Tb), cmap="afmhot", vmin=vmax - 3.0, vmax=vmax)
plt.colorbar(pc, ax=ax, label=r"$\log(T_b)$")

# Ticks along the EVPA, with length proportional to the linear polarization.  The EVPA
# is measured east of north, so zero is a vertical tick, 45 degrees runs from lower
# right to upper left, and the direction in display coordinates is (-sin, cos).
step = max(1, Nx // 25)
X, Y = np.meshgrid(x[::step], y[::step])
ml = m_lin[::step, ::step]
ch = chi[::step, ::step]
scale = 0.9 * step * (x[1] - x[0]) / max(np.nanmax(ml), 1e-12)
dx = -scale * ml * np.sin(ch)
dy = scale * ml * np.cos(ch)
ax.quiver(
    X, Y, dx, dy, color="cyan", pivot="middle", headwidth=0, headlength=0, headaxislength=0,
    scale_units="xy", scale=1.0, width=0.004,
)
ax.set_title("total intensity + EVPA")
ax.set_xlabel(r"$x$ ($r_g$)")
ax.set_ylabel(r"$y$ ($r_g$)")

# --- fractional linear polarization
ax = axes[1]
ax.set_facecolor("black")
ax.set_aspect("equal")
pc = ax.pcolormesh(x, y, 100.0 * m_lin, cmap="viridis", vmin=0.0)
plt.colorbar(pc, ax=ax, label="linear polarization (%)")
ax.set_title("fractional linear polarization")
ax.set_xlabel(r"$x$ ($r_g$)")
ax.set_ylabel(r"$y$ ($r_g$)")

# --- fractional circular polarization
ax = axes[2]
ax.set_facecolor("black")
ax.set_aspect("equal")
# lim = np.nanmax(np.abs(100.0 * m_circ))
lim = 2.0
pc = ax.pcolormesh(x, y, 100.0 * m_circ, cmap="RdBu_r", vmin=-lim, vmax=lim)
plt.colorbar(pc, ax=ax, label="circular polarization (%)")
ax.set_title("fractional circular polarization")
ax.set_xlabel(r"$x$ ($r_g$)")
ax.set_ylabel(r"$y$ ($r_g$)")

plt.tight_layout()
plt.savefig("jet_image_polarized.png", dpi=200, bbox_inches="tight")
plt.close()

###################################################
# export the Stokes cube as a FITS file
#
# export_fits accepts a (4, ny, nx) cube and writes it with a FITS STOKES axis
# (I = 1, Q = 2, U = 3, V = 4).  The cube is written in the sky orientation, so that
# it can be compared with an observed image directly.

x_reg, y_reg, _ = jf.interp_to_regular_grid(Snu[0], x, y, nx=256, ny=256)
cube = np.stack([jf.interp_to_regular_grid(P, x, y, nx=256, ny=256)[2] for P in Snu])

# angular offsets, in radians
rad_per_rg = model.rg / (D * 3.086e24)
jf.export_fits(
    "jet_image_polarized.fits",
    cube,
    x_reg * rad_per_rg,
    y_reg * rad_per_rg,
    observing_frequency_hz=frequency * 1.0e9,
    bunit="Jy/pix",
)

###################################################
