import numpy as np
from HEGData.LDAFits import fmap
from Libs.RPAHelper import *

import matplotlib.pyplot as plt
from Libs.NiceColours import *

AllText = []

PreCompute = False
RunTests = False


##### Fine grid ####
def GetGrid(N1=51, N=11):
    # Special grid
    x0, wx0 = GaussLegWeights(N1, a=0., b=1.)
    xp, wxp = GaussLegWeights(N , a=0., b=1.)
    for S in range(50):
        if S==0:
            x, wx = x0*1., wx0*1.
        else:
            x = np.hstack((x, xp+S))
            wx = np.hstack((wx, wxp))

    return x, wx



##################
if RunTests:
    ####
    x, wx = GetGrid()
    gamma, wgamma = x, wx
    # Q grid
    q, wq = x, wx

    Q, Gamma = np.meshgrid(q, gamma)

    def IQG_Q2(X, Pre=1.):
        Y = Pre * X/Q**2
        return wgamma.dot(-Y+np.log(1+Y)).dot(wq*q**3)

    IP = {}
    for NP, (N1,N) in enumerate(((51, 11), (101,11))):
        x, wx = GetGrid(N1, N)
        gamma, wgamma = x, wx
        q, wq = x, wx

        Q, Gamma = np.meshgrid(q, gamma)

        PP = np.logspace(-3, 7, 101)
        IP[NP] = 0.*PP
        for K, P in enumerate(PP):
            if (K%10)==0: print(K)
            IP[NP][K] = -IQG_Q2(CLindImag(Q, Gamma), Pre=P)

        plt.loglog(PP, IP[NP])

    print(IP[0]/IP[1])
    plt.show()

    quit()
##################



fig, ax = plt.subplots(1, figsize=(6,3))

# High density limit is larger values of Pre

if PreCompute:
    rsAll = np.array([0.5, 1, 2, 3, 5, 10, 20])
    nAll = 1./(4.*np.pi/3 * rsAll**3)
else:
    # For plotting
    nAll = np.logspace(-6,6,3)

try:
    AD = np.load("Data/RPA-Xi.npz", allow_pickle=True)
    RPAData = AD['RPAData'][()]
except:
    RPAData = {}

def IQG_Q2(X, Pre=1.):
    Y = Pre * X/Q**2
    return 1./Pre * wgamma.dot(-Y+np.log(1+Y)).dot(wq*q**3)

for Kn, n in enumerate(nAll):
    rs = 0.62035/n**(1/3)
    rs = np.round(rs*1000)/1000
    kF = (3.*np.pi**2*n)**(1/3)
    print("rs = %.3f, kF = %.3f"%(rs, kF))
    Pre = 1./(2.*np.pi*kF)

    x, wx = GetGrid()
    q, wq = x, wx
    gamma, wgamma = x, wx
                
    # Mesh
    Q, Gamma = np.meshgrid(q, gamma)

    zzeta = np.sqrt(np.linspace(0.00001, 0.99999, 31))
    
    Xix = 0.*zzeta
    Xic = 0.*zzeta
    Xix_ot = 0.*zzeta
    Xic_ot = 0.*zzeta

    IRaw = IQG_Q2(CLindImag(Q, Gamma), Pre=Pre)
    for Kz, zeta in enumerate(zzeta):
        ft = fmap(zeta)
        Gx = -ft/(3*ft-2)

        hp, hm = (1+zeta)**(1/3), (1-zeta)**(1/3)
        g = (3.+1/Gx)**(1/3)


        Ih   = IQG_Q2(hm/2 * CLindImag(Q/hm, Gamma/hm)
                      + hp/2 * CLindImag(Q/hp, Gamma/hp), Pre=Pre)
        Ig   = IQG_Q2(1/g**2 * CLindImag(Q/g , Gamma/g ), Pre=Pre)

        if False:
            Ihp  = IQG_Q2(hp/2 * CLindImag(Q/hp, Gamma/hp), Pre=Pre)
            Ihm  = IQG_Q2(hm/2 * CLindImag(Q/hm, Gamma/hm), Pre=Pre)
            print("hp/2 = %.3f hm/2 = %.3f 1/g^2 = %.3f"\
                  %(hp/2, hm/2, g**(-2)))
            print("Ihp/IRaw = %.3f, Ihm/IRaw = %.3f, Ig/IRaw = %.3f"\
                  %(Ihp/IRaw, Ihm/IRaw, Ig/IRaw))
        
        Xic[Kz]    = Ih / IRaw
        Xic_ot[Kz] = Ig / IRaw

        Xix[Kz] = (hp**4 + hm**4)/2.
        Xix_ot[Kz] = g

    RPAData[rs] = {
        'Gx': -0.5*(1+zzeta**2), 
        'Xix':Xix, 'Xic':Xic,
        'Xix_ot':Xix_ot, 'Xic_ot':Xic_ot,
    }
    np.savez("Data/RPA-Xi.npz", RPAData=RPAData)

    zeta = zzeta
    f = fmap(zeta)
    Gx = -f/(3*f-2)

    Deltac_RPA = (Xic_ot - Xic)/Xic

    Lbl = "$n=10^{%d}$"%(np.log10(n))

    ax.plot(f, Xic, "-.", color=NiceColour(Kn),
            label="pol. c factor" if Kn==0 else "")
    ax.plot(f, Xic_ot, "-", color=NiceColour(Kn),
            label="cof c factor" if Kn==0 else "")
    
    X = (2/f)**(1/3)
    Xic_ot_ld = 1. + (Xic_ot[-1]-Xic_ot[0])/(2**(1/3)-1)*(X-1)
    ax.plot(f, Xic_ot_ld, ":", color=NiceColour(Kn),
            label="cof x factor" if Kn==0 else "")

    wf = 1+0*f
    _,P=FitPolyfb(f, wf, Xic_ot, 1., Xic_ot[-1])
    print(P)
        

    i0 = np.argmin(np.abs(f-1.1))
    txt = ax.text(f[i0], Xic_ot[i0], Lbl,
                  color=NiceColour(Kn),
                  fontsize=14,)
    AllText += [txt] 
    AddBorder(txt)
        
ax.set_ylabel("$\\xi_{\\rm c}$ enhancement", fontsize=14)
ax.legend(loc="lower right", fontsize=14)
    
ax.set_xlim([1,2])   
ax.set_xlabel("(Average) occupation factor, $f$", fontsize=14)

#adjust_text(AllText)
    
plt.tight_layout(pad=0.3)

plt.savefig("FigRPA.pdf")
plt.show()
