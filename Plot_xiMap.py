import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from Libs.NiceColours import *

from HEGData.DMCData import *
from HEGData.LDAFits import *

mHa = 1000
ShowEot = True

HEGReference = HEG_Spink2013()
zeta_Spink = HEGReference.zeta

CX, CC = 0., 1.

X = np.load("Data/RPA-Xi.npz", allow_pickle=True)
RPAData = X['RPAData'][()]

fig, axs = plt.subplots(1, 3, figsize=(6,2), sharey=True)

for Krs, raw_rs in enumerate([1., 2., 5.]):
    rs = np.round(raw_rs*10)/10
    
    epsx_u, epsc_u = LDA_rPW92(rs, 0.)
    epsx_p, epsc_p = LDA_rPW92(rs, 1.)

    def Toepsc(Xic):
        return epsc_u + (Xic-1.)/(Xic[-1]-1.)*(epsc_p-epsc_u)

    epsx = RPAData[raw_rs]['Xix'] * epsx_u
    epsc = Toepsc(RPAData[raw_rs]['Xic'])

    epsx_ot = RPAData[raw_rs]['Xix_ot'] * epsx_u
    epsc_ot = Toepsc(RPAData[raw_rs]['Xic_ot'])

    ax = axs[Krs]

    ax.plot(epsx_ot/epsx_u, epsc_ot/epsc_u, "--",
            color=NiceColour("Orange"),
            label="cofe",
            )
    ax.plot(epsx/epsx_u, epsc/epsc_u, "-",
            color=NiceColour("Navy"),
            label="pol.",
            )

    
    Xix = epsx/epsx_u
    Xix_ot = epsx_ot/epsx_u

    Xix_f = np.linspace(1., 2**(1/3), 101)
    deps = np.interp(Xix_f, Xix, epsc) \
        - np.interp(Xix_f, Xix_ot, epsc_ot)

    if rs in HEGReference.rs:
        epsc_Spink = HEGReference.Getepsc(rs, zeta_Spink)
        epsx_Spink = epsx_u * ((1+zeta_Spink)**(4/3)
                               + (1-zeta_Spink)**(4/3))/2
        ax.scatter(epsx_Spink/epsx_u, epsc_Spink/epsc_u,
                   color='k',
                   )

        if ShowEot:
            Xix_Eot = epsx_Spink/epsx_u
            f_Eot = 2/(Xix_Eot**3)
            #Gx_Eot = -f_Eot/(3*f_Eot-2)
            epsx_Eot, epsc_Eot = LDA_Eot(rs, f_Eot)
            depsMax = np.abs(epsc_Eot-epsc_Spink).max()
        else:
            depsMax = np.abs(deps).max()
    else:
        depsMax = np.abs(deps).max()

    if False and ShowEot:
        ax.scatter(epsx_Eot/epsx_u, epsc_Eot/epsc_u,
                   marker="*",
                   color=NiceColour("Orange"),
                   )
        ax.text(0.5, 0.7, "Max Err:\n%.1f mHa"%(mHa*depsMax),
                fontsize=12,
                horizontalalignment="left",
                #verticalalignment="center",
                transform=ax.transAxes,
                )
    else:
        print("Max error = %.1f mHa"%(mHa*depsMax))
        ax.legend(loc="upper right",
                  frameon=False, borderpad=0.)
              
    ax.text(0.05, 0.05, "$r_s=%d$"%(rs),
            fontsize=14,
            transform=ax.transAxes,
            )

    ax.set_xlabel("$\\xi_{\\rm x}$ [unitless]", fontsize=14)

axs[0].set_ylabel("$\\bar{\\xi}_{\\rm c}$ [unitless]", fontsize=14)

fig.tight_layout(pad=0.3)
fig.savefig("FigxiMapping.pdf")

plt.show()

