###################################################
#                                                 #
# This example script shows how to simulate a jet #
# image that spans several orders of magnitude in #
# spatial scale, using a logarithmically-spaced   #
# pixel grid.  It also showcases some regridding  #
# functionality that is useful to convert such    #
# log-spaced images into more typical fits files  #
# and similar.                                    #
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

###################################################
# inputs that control the image

# observing frequency, in GHz
frequency = 230.0

# we'll use a logarithmic grid in all dimensions
use_log_xgrid = use_log_ygrid = use_log_zgrid = True

# set the minimum and maximum values for the image dimensions, in gravitational radii
xmin = ymin = 0.1
xmax = ymax = 1.0e6

# set the number of (logarithmic) pixels for the image
Nx = Ny = 400

# set the dimensions and resolution along the depth (z) direction; relevant for radiative transfer
zmin = 0.1
zmax = 1.0e8
Nz = 1600

# power-law index for injected electrons
p = 2.2

# jet power heating fraction
h = 0.0025

# anisotropy parameter; eta=1 is isotropic
eta = 0.01

###################################################
# initialize the jet model and generate an image

# initialize
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
    use_log_xgrid=use_log_xgrid,
    use_log_ygrid=use_log_ygrid,
    use_log_zgrid=use_log_zgrid,
)

# generate image
x, y, I_nu = model.make_image(frequency, show_progress=True)

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
# intensities pass through it     #
# unchanged.  Everything below is #
# in this sky orientation.        #
###################################

x, y, I_nu = jf.sky_view(x, y, I_nu)

###################################################
# convert image units

###################################
# The image is natively produced  #
# in intensity units.  To convert #
# to other units, we can use the  #
# convert_units() function.       #
###################################

# distance to the source, in Mpc
D = 16.8

# luminosity density, in cgs units
Lnu = jf.convert_units(model, I_nu, output_units="luminosity")

# flux density, in cgs units
Snu = jf.convert_units(model, I_nu, output_units="flux", D=D)

# convert flux density to Jy
Snu_Jy = Snu * (1.0e23)

# brightness temperature, in K
Tb = jf.convert_units(model, I_nu, output_units="Tb", frequency=frequency)

###################################################
# plot image of innermost 200 rg

# specify new grid
x_new = np.linspace(-100.0, 100.0, 200)
y_new = np.linspace(-100.0, 100.0, 200)

# interpolate image to new grid -- reminder: interpolate intensity and not flux!
_, _, I_new = jf.interp_to_regular_grid(I_nu, x, y, x_new=x_new, y_new=y_new, method="linear")

# convert interpolated image to brightness temperature
Tb_new = jf.convert_units(model, I_new, x_new, y_new, output_units="Tb", frequency=frequency)

# plot
fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
cax = fig.add_axes([0.91, 0.1, 0.02, 0.8])
ax.set_facecolor("black")
ax.set_aspect("equal")
vmax = 10.5
vmin = vmax - 3.0
pc = ax.pcolormesh(x_new, y_new, np.log10(Tb_new), cmap="afmhot", vmax=vmax, vmin=vmin)
ax.set_xlabel(r"$x$ ($r_g$)")
ax.set_ylabel(r"$y$ ($r_g$)")
plt.colorbar(pc, cax=cax, label=r"$\log(T_b)$")
plt.savefig("jet_image_log.png", dpi=300, bbox_inches="tight")
plt.close()

###################################################
# export the image as a FITS file
#
# the image is written in the sky orientation, so that it can be compared with an
# observed image directly

# first, convert interpolated image to flux density
Snu_new = jf.convert_units(model, I_new, x_new, y_new, output_units="flux", D=D)
Snu_Jy_new = Snu_new * (1.0e23)

jf.export_fits(
    "jet_image_log.fits", Snu_Jy_new, x_new, y_new, observing_frequency_hz=230.0e9, bunit="Jy/pix"
)

###################################################
