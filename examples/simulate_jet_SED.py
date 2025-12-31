###################################################
#                                                 #
# This example script shows how to loop through   #
# frequency with the same underlying base jet     #
# model to produce an SED.                        #
#                                                 #
###################################################

###################################################
# imports

import numpy as np
import jetfuncs as jf
from tqdm import tqdm
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

# set the minimum and maximum values for the image dimensions, in gravitational radii
xmin = ymin = -100.0
xmax = ymax = 100.0

# set the number of pixels for the image
Nx = Ny = 100

# set the dimensions and resolution along the depth (z) direction; relevant for radiative transfer
zmin = 0.0
zmax = 400.0
Nz = 800

# power-law index for injected electrons
p = 2.2

# jet power heating fraction
h = 0.0025

# anisotropy parameter; eta=1 is isotropic
eta = 0.01

###################################################
# initialize the jet model

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
)

###################################################
# loop through frequency to generate an SED

# array of observing frequencies, in GHz
frequency_arr = 10.0 ** np.linspace(0.0, 6.0, 200)

# initialize an array to hold integrated luminosity density
luminosity = np.zeros_like(frequency_arr)

# loop through frequency
for ifreq in tqdm(range(len(frequency_arr))):
    # generate image at this frequency
    x, y, I_nu = model.make_image(frequency_arr[ifreq], show_progress=False)

    # convert intensity to luminosity density, in cgs units
    Lnu = jf.convert_units(model, I_nu, output_units="luminosity")

    # store
    luminosity[ifreq] = np.sum(Lnu)

###################################################
# plot the SED

fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(frequency_arr, frequency_arr * (1.0e9) * luminosity, "k-")
ax.loglog()
ax.set_xlim(np.min(frequency_arr), np.max(frequency_arr))
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel(r"$\nu L_{\nu}$ (erg/s)")
ax.grid(linewidth=0.5, linestyle="--", alpha=0.1)
plt.savefig("jet_SED.png", dpi=300, bbox_inches="tight")
plt.close()

###################################################
