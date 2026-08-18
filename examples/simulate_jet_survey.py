###################################################
#                                                 #
# This example script shows how to run a small    #
# parameter survey -- here SEDs for several       #
# heating fractions -- using jetfuncs.survey(),   #
# which evaluates a function on many jet models   #
# in parallel worker processes.                   #
#                                                 #
###################################################

###################################################
# imports

import numpy as np
import jetfuncs as jf
import matplotlib.pyplot as plt

###################################################
# baseline model (M87-like) and the parameter to survey

base = dict(
    m=6.5e9,
    a=0.9,
    inc=163.0,
    mdot=5.45e-5,
    s=0.6,
    p=2.14,
    eta=0.03,
    gamma_inf=4.0,
    Nx=100,
    Ny=100,
    Nz=400,
    xmin=0.1,
    xmax=1.0e5,
    ymin=0.1,
    ymax=1.0e5,
    zmin=0.1,
    zmax=1.0e7,
    use_log_xgrid=True,
    use_log_ygrid=True,
    use_log_zgrid=True,
)

# jet heating fractions to survey
h_values = 10.0 ** np.linspace(-4.0, -2.0, 5)
configs = [dict(base, h=h) for h in h_values]

# observing frequencies, in GHz
frequency_arr = 10.0 ** np.linspace(0.0, 6.0, 60)


###################################################
# the function evaluated on every model
#
# It must be defined at module level (the worker processes import it), and it
# receives a fully constructed JetModel.  Here it returns the SED in erg/s.


def sed(model, frequencies):
    model.precompute_state()  # frequency-independent physics, once per model
    nuLnu = np.zeros_like(frequencies)
    for i, freq in enumerate(frequencies):
        _, _, I_nu = model.make_image(freq)
        Lnu = jf.convert_units(model, I_nu, output_units="luminosity")
        nuLnu[i] = freq * 1.0e9 * np.sum(Lnu)
    return nuLnu


###################################################
# run the survey
#
# The `if __name__ == "__main__":` guard is required: worker processes are started
# with the "spawn" method and import this script to find `sed`.

if __name__ == "__main__":
    seds = jf.survey(configs, sed, func_kwargs=dict(frequencies=frequency_arr))

    ###############################################
    # plot

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    for h, nuLnu in zip(h_values, seds):
        ax.plot(frequency_arr, nuLnu, label=rf"$h = 10^{{{np.log10(h):.1f}}}$")
    ax.loglog()
    ax.set_xlim(np.min(frequency_arr), np.max(frequency_arr))
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(r"$\nu L_{\nu}$ (erg/s)")
    ax.legend(fontsize=7)
    ax.grid(linewidth=0.5, linestyle="--", alpha=0.1)
    plt.savefig("jet_survey.png", dpi=300, bbox_inches="tight")
    plt.close()
