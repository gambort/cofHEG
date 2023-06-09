from HEGData.LDAFits import *
from HEGData.DMCData import *
from Libs.FitLDAs import *


# Low-density limit
# Cinf, Cinfp = 0.9, 1.35 # Older data
Cinf, Cinfp = 0.8959, 1.33 # From 10.1063/1.5078565
Cx = 0.458165 # Exchange factor


def NiceArray(X):
    return "["+", ".join(["%6.3f"%(x) for x in np.atleast_1d(X)])+"]"



def DoPlot(HEGModel, zeta, epscData,
           rsMin=0.3, rsMax=200, spinFit=False,
           Inset=True):
    Units, UnitsLbl = 1000., "mHa"

    fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True, figsize=(6,6))

    rsp = np.logspace(-1, 2.5, 201)

    for Kz, z in enumerate(zeta):
        fb = fmap(z)
        Gx = 2*fb/(3*fb+1)

        zColp = zetaColour(z)
        
        print("zeta = %.3f, Gx = %.3f, fb = %.3f"\
              %(z, Gx, fb))
        ShowLabel = z<0.01

        
        rs_x = HEGModel.rs
        epsc_x_0 = HEGModel.Getepsc(rs_x, 0)
        epsx_x_0 = -Cx/rs_x
        
        epsx_x = epsx_x_0 * (2/fb)**(1/3)
        epsc_x = epscData[Kz,:]
        
        epsxc_x = epsx_x + epsc_x
        epsxc_x_0 = epsx_x_0 + epsc_x_0
        
        ax1.scatter(rs, epsc_x*Units,
                    color=zColp,
                    )
        ax2.scatter(rs, (epsxc_x - epsxc_x_0)*Units,
                    color=zColp,
        )
        ax3.scatter(rs, (epsxc_x/epsxc_x_0)*100.,
                    color=zColp,
        )

        epsx, epsc = LDA_cofe(rsp, fb=fb)
        epsx_0 = -Cx/rsp
        epsxc0 = LDA_cofe(rsp, fb=2, Sum=True)

        ax1.semilogx(rsp, epsc*Units,
                     color=zColp,
                     label="cofe" if ShowLabel else "",
        )
        ax2.semilogx(rsp, (epsx+epsc-epsxc0)*Units,
                     color=zColp,
        )
        ax3.semilogx(rsp, ((epsx+epsc)/epsxc0)*100.,
                     color=zColp,
        )


        epsx_P, epsc_P = LDA_PW92(rsp, z)
        epsxc0_P = LDA_PW92(rsp, 0.0, Sum=True)
        
        ax1.semilogx(rsp, epsc_P*Units, ":",
                     color=zColp,
                     label="PW92" if ShowLabel else "",
        )
        ax2.semilogx(rsp, (epsx_P+epsc_P-epsxc0_P)*Units, ":",
                     color=zColp,
        )
        ax3.semilogx(rsp, ((epsx_P+epsc_P)/epsxc0_P)*100., ":",
                     color=zColp,
        )

        epsx_r, epsc_r = LDA_rPW92(rsp, z)
        epsxc0_r = LDA_rPW92(rsp, 0.0, Sum=True)
        
        ax1.semilogx(rsp, epsc_r*Units, "-.",
                     color=zColp,
                     label="rPW92" if ShowLabel else "",
        )
        ax2.semilogx(rsp, (epsx_r+epsc_r-epsxc0_r)*Units, "-.",
                     color=zColp,
        )
        ax3.semilogx(rsp, ((epsx_r+epsc_r)/epsxc0_r)*100., "-.",
                     color=zColp,
        )


    for ax in (ax1,ax2,ax3):
        1
        #for Zone in (2.5,100,):
        #    ax.semilogx([Zone,Zone],[-1e3,1e3],":k")

    ax3.set_position([0.12,0.10,0.87,0.30])
    ax2.set_position([0.12,0.41,0.87,0.28])
    ax1.set_position([0.12,0.70,0.87,0.28])
    
    ax1.set_ylabel("$\\epsilon_{\\rm c}^{\\rm SD}$ [mHa]",
                   labelpad=0., fontsize=14)
    ax1.legend(loc="lower right", ncol=2)

    ax2.set_ylabel("$\\Delta\\epsilon_{\\rm xc}^{\\rm SD}$ [mHa]",
                   labelpad=0., fontsize=14)

    ax3.set_ylabel("xc enhance. [%]", fontsize=14)
    
    ax1.axis([rsMin, rsMax,-75,2])
    ax2.axis([rsMin, rsMax,-230,10])
    ax3.axis([rsMin, rsMax, 97, 123])
    ax2.set_yticks([-200,-150,-100,-50,0])
    ax2.set_yticklabels([-200,-150,-100,-50,0],rotation=45)

    ax3.set_xlabel("Wigner-Seitz radius, $r_s$ [Bohr]", fontsize=14)
    
    # Add the inset
    if Inset:
        axI = ax2.inset_axes([0.54,0.34,0.44,0.5])
        Range = (1000,-1000)

        for rsT in (2,): # The rs to show
            kI = np.argmin(np.abs(rsT-rs))
            rsI = rs[kI]

            # Inset - add the points
            for Q in (0.1, 0.3, 0.5, 0.7, 0.9):
                cc = zetaColour(Q)
                zetaQ = np.linspace(Q-0.1, Q+0.1, 7)
                fQ = fmap(zetaQ)

                axI.plot(fQ, LDA_cofe(rsI, fb=fQ, Sum=True)*Units,
                         color=cc)

            f = fmap(zeta)
            epsxI = -Cx/rsI * (2/f)**(1/3)
            epscI = epscData[:, kI]
            epsxcI = epsxI + epscI

            axI.scatter(f, epsxcI*Units,
                        zorder=1000,
                        c=[zetaColour(z) for z in zeta])
            
            axI.text(0.1+0.2*rsI,0.85, "$r_s=%.0f$"%(rsI),
                     transform=axI.transAxes,
                     )
    
            Range = (min(Range[0],LDA_cofe(rsI, fb=1., Sum=True)),
                     max(Range[1],LDA_cofe(rsI, fb=2., Sum=True)))

        if (Range[1]-Range[0])*Units>35: S=20
        elif (Range[1]-Range[0])*Units<20: S=5
        else: S=10

        A = np.floor(Range[0]*Units/S)*S
        B = np.ceil(Range[1]*Units/S)*S

        TT = np.arange(A,(B+S),S, dtype=int)
        axI.set_yticks(TT)
        axI.set_yticklabels(TT, rotation=45)
        axI.set_ylabel("$\\epsilon_{\\rm xc}$ [mHa]", fontsize=12)

        axI.set_xlabel("$\\bar{f}$ [unitless]", fontsize=12)
        axI.set_xticks([1,1.5,2],
                       ["1","3/2","2"])
        axI.axis([1,2,A,B])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    spinFit = False

    print("="*72)
    print("Doing the fits")
    HEGModel = HEG_Spink2013()
    Pz = {}
    for FitMode in ("rPW92", "cofe"):
        print("Fit mode = %s"%(FitMode))
        Pz[FitMode], zeta, rs, epscData \
            = DoFit(HEGModel, FitMode=FitMode)

    print("="*72)
    print("Doing the plots")
    DoPlot(HEGModel, zeta, epscData, spinFit=spinFit)
    plt.savefig("FigepsxcFit.pdf")
    plt.show()
    
    print("="*72)
    for FitMode in ("rPW92", "cofe"):
        print("# - python code for %s"%(FitMode))
        for z in sorted(list(Pz[FitMode])):
            print("ec%-3d"%(np.round(z*100))
                  + " = F(rs, (%.6f, %.4f, %.4f, %.4f, %.4f, %.4f))"\
                  %(Pz[FitMode][z]))

    for FitMode in ("rPW92", "cofe"):
        print("%% - latex table for %s"%(FitMode))
        for z in sorted(list(Pz[FitMode])):
            print("%.2f & "%(z)
                  + "%.6f & %.4f & %.4f & %.4f & %.4f & %.4f \\\\"\
                  %(Pz[FitMode][z]))

