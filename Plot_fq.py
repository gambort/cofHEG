import numpy as np
import matplotlib.pyplot as plt
from Libs.NiceColours import *

ShowOccs = True
ShowMultiOccs = True
ShowEnhance = False

ConstrainGx = False

Range = 1.4
n0 = 3.

def zeta_to_f(zeta):
    if not(ConstrainGx):
        f = 2 - 4/3*zeta**2 + 0.16979*zeta**3 + 0.16355*zeta**4
        Gx = -f/(3*f-2)
    else:
        Gx = -(1+zeta**2)/2
        f = 2*Gx/(3*Gx+1)

    return f, Gx

def Get_kf(n=3, f=2):
    return (6.*np.pi**2*n/f)**(1/3)

def IWP91(zeta):
    y1 = np.maximum((1+zeta)**(1/3),1e-8)
    y2 = np.maximum((1-zeta)**(1/3),1e-8)
    return 0.5*(1. + (y1*y2*(y1+y2) + y1**3*np.log(y1) + y2**3*np.log(y2)
                      - 2.*np.log(y1+y2) ) / (2.*(1-np.log(2.))) )

def Occ(n=3, f=None, zeta=None, kFonly=False):
    kf = Get_kf(n)
    
    if not(f is None):
        kff = Get_kf(n, f)
        if kFonly: return kff
        xx = np.array([0, kff, kff, kf*Range])
        yy = np.array([f, f, 0, 0])
    elif not(zeta is None):
        nu, nd = n * (1+np.abs(zeta))/2, n * (1-np.abs(zeta))/2

        if nd>0.:
            kfu = Get_kf(nu, f=1.)
            kfd = Get_kf(nd, f=1.)
            if kFonly: return kfu,kfd
            xx = np.array([0, kfd, kfd, kfu, kfu, kf*1.7])
            yy = np.array([2, 2, 1, 1, 0, 0])
        else:
            if kFonly: return Get_kf(n, f=1.),0.
            return Occ(n, f=1.)
    else:
        if kFonly: return kf
        xx = np.array([0, kf, kf, kf*1.7])
        yy = np.array([2, 2, 0, 0])

    nTest = 0.
    for k in range(1,len(xx)):
        VV = (xx[k]**3 - xx[k-1]**3)/(6.*np.pi**2)
        nTest += VV*yy[k]
    print(nTest)
    return xx, yy

if ShowOccs:
    fig, ax = plt.subplots(1, 1, figsize=(6,2))

    zeta = 0.5
    f, Gx = zeta_to_f(zeta)

    print("zeta = %.3f, Gx = %.3f, f = %.3f"%(zeta, Gx, f))


    kfu, kfd = Occ(zeta=zeta, kFonly=True)
    xx,yy = Occ(zeta=zeta)
    ax.plot(xx, yy, color=NiceColour("Navy"), lw=3)

    kf = Occ(kFonly=True)
    xx,yy = Occ()
    ax.plot(xx, yy, color=NiceColour("Teal"), dashes=(3,1,1,1), lw=3)

    kff = Occ(f=f, kFonly=True)
    xx,yy = Occ(f=f)
    ax.plot(xx, yy, color=NiceColour("Orange"), dashes=(2,1), lw=3)

    for K, (Lbl,Col) in enumerate([("unpolarized", "Teal"),
                                   ("polarized", "Navy"),
                                   ("cofe", "Orange"),
                                   ]):
        AddBorder(
            ax.text(0.12+K*0.25, 0.1, Lbl, color=NiceColour(Col),
                    horizontalalignment="center",
                    fontsize=14, transform=ax.transAxes)
        )


    ax.axis([0, 1.3*Get_kf(), 0., 2.1])

    ax.set_xticks([kfu, kfd, kf, kff],
                  ['$k_F^{\\uparrow}$', '$k_F^{\\downarrow}$',
                   '$k_F$', '$\\bar{k}_F^{\\rm cofe}$'])
    
    ax.set_yticks([0,0.5,1,1.5,2.])
    ax.set_yticklabels(['0', '', '$\\uparrow$',
                        '', '$\\uparrow\\!\\!\\downarrow$'])
    ax.set_xlabel("Wavenumber, $q$", fontsize=14)
    ax.set_ylabel("Occ., $f_q$", fontsize=14)
    plt.tight_layout(pad=0.3)

    plt.savefig("FigOccs.pdf")

if ShowMultiOccs:
    fig, axs = plt.subplots(4, 1, figsize=(6,3), sharex=True)

    #Gx_x = np.array([-0.5, -0.6, -0.8, -1.0])
    #zeta_x = np.sqrt(-1-2*Gx_x)

    zeta_x = [0, 0.34, 0.66, 1.]
        
    
    for zeta, ax in zip(zeta_x,axs):
        f, Gx = zeta_to_f(zeta)
        #Gx = -(1+zeta**2)/2
        #f = 2*Gx/(3*Gx+1)

        print("zeta = %.3f, Gx = %.3f, f = %.3f"%(zeta, Gx, f))


        xx,yy = Occ(zeta=zeta)
        ax.plot(xx, yy, color=NiceColour("Navy"), lw=3)

        xx,yy = Occ(f=f)
        ax.plot(xx, yy, color=NiceColour("Orange"), dashes=(2,1), lw=3)


        XLbl, YLbl, InterpLbl = 0.77, 0.97, "$G_{\\rm x}=%.2f$"%(Gx)
        XLbl, YLbl, InterpLbl = 0.02, 0.47, "$\zeta=%.2f$, $\\bar{f}=%.2f$"\
            %(zeta, f)

        AddBorder(
            ax.text(XLbl, YLbl, InterpLbl,
                    verticalalignment="top",
                    fontsize=14,
                    transform=ax.transAxes,
                    )
        )
        if ax==axs[3]:
            for K, (Lbl,Col) in enumerate([
                    ("polarized gas", "Navy"),
                    ("cofe gas", "Orange"),
            ]):
                AddBorder(
                    ax.text(0.55+K*0.25, 0.95, Lbl, color=NiceColour(Col),
                            horizontalalignment="center",
                            verticalalignment="top",
                            fontsize=14, transform=ax.transAxes)
                )


        ax.axis([0, Range*Get_kf(), -0.1, 2.1])
    
        ax.set_xticks([])
        ax.set_yticks([0,0.5,1,1.5,2.])
        ax.set_yticklabels(['0','','1','','2'])
        #ax.set_yticklabels(['0', '', '$\\uparrow$',
        #                    '', '$\\uparrow\\!\\!\\!\\!\\downarrow$'])

    ax.set_xlabel("Wavenumber, $q$", fontsize=14)

    fig.supylabel("Occupation factor, $f_q$", fontsize=14)

    plt.subplots_adjust(hspace=0.)
    plt.tight_layout(pad=0.3, h_pad=0.0, w_pad=0.0)
    plt.savefig("FigMultiOccs.pdf")

if ShowEnhance:
    fig, ax = plt.subplots(1, 1, figsize=(6,2))

    zeta = np.linspace(0, 1, 81)
    Gx = -(1+zeta**2)/2
    f = 2*Gx/(3*Gx+1)

    Fpol = ((1+zeta)**(4/3) + (1-zeta)**(4/3))/2.
    Fcofe = (3. + 1./Gx)**(1/3)

    Fldpol  = 1.96 - Fpol
    Fldcofe = 1.96 - Fcofe

    Fhdpol  = IWP91(zeta)# From WP91
    Fhdcofe = f/2.
    
    def Xi(F,x=None):
        if x is None:
            return (F-F[0])/(F[-1]-F[0])
        else:
            return np.interp(x, zeta, Xi(F))

    ax.plot(zeta, Xi(Fpol),
            color=NiceColour("Navy"),
            label="x/c(ld)",
            )
    ax.plot(zeta, Xi(Fhdpol), dashes=(2,1),
            color=NiceColour("Navy"),
            label="c(hd)",
            )
    ax.text(0.6, Xi(Fhdpol, 0.6)-0.1, "pol",
            color=NiceColour("Navy"), fontsize=14,
            )
    
    ax.plot(zeta, Xi(Fcofe),
            color=NiceColour("Orange"),
            )
    ax.plot(zeta, Xi(Fhdcofe), dashes=(2,1),
            color=NiceColour("Orange"),
            )
    ax.text(0.4, Xi(Fhdcofe, 0.4)+0.1, "cofe",
            color=NiceColour("Orange"), fontsize=14,
            )

    ax.legend(loc="upper left")
    ax.axis([0, 1, -0.01, 1.01])

    ax.set_xticks([0,0.5,1])
    ax.set_xticklabels(['$0$','$\\frac{1}{2}$', '$1$'])
    ax.set_xlabel("Effective polarization, $\\zeta$", fontsize=14)
    ax.set_ylabel("Enhancement", fontsize=14)
    plt.tight_layout(pad=0.3)

plt.show()
