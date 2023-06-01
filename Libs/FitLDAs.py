import numpy as np
import scipy.optimize as opt

from HEGData.LDAFits import *


# Low-density limit
# Cinf, Cinfp = 0.9, 1.35 # Older data
Cinf, Cinfp = 0.8959, 1.33 # From 10.1063/1.5078565
Cx = 0.458165 # Exchange factor



def c_PW92(zeta=None, fb=None,
           Scale1=1.):
    c0_0 = 0.031091  # c0 unpolarized
    c0_1 = c0_0/2.   # c0 polarized
    if not(zeta is None):
        # Note, Scale1=1.08 accommodates the new data from Luo and Alavi 2022
        fzeta = ((1+zeta)**(4/3) + (1-zeta)**(4/3)-2)/(2**(4/3)-2)
        c0 = c0_0 - 0.00988 * fzeta*(1.-zeta**4) \
            - (c0_0-c0_1) * fzeta*zeta**4
        c1 = Scale1*(0.04664  - 0.02074 * fzeta*(1.-zeta**4) \
                     - 0.02105 * fzeta*zeta**4)
        return c0, c1
    else:
        c0_0 = 0.031091  # c0 unpolarized
        c0_1 = 0.0155455 # c0 polarized
        c1_0 = Scale1*0.04664  # c1 unpolarized
        c1_1 = Scale1*(0.04664 - 0.02105)  # c1 polarized

        # Linear interpolation
        c0 = c0_0 * (fb-1) + c0_1 * (2-fb)
        c1 = c1_0 * (fb-1) + c1_1 * (2-fb)
        
        return c0, c1
    
def Lim_to_Param(alpha, c0, c1, Cinf, Cinfp, fx, spinFit=False):
    A = c0
    beta1 = np.exp(-c1/(2.*c0))/(2.*c0)
    beta2 = 2.*A*beta1**2
    #beta2 = 0.

    beta4 = alpha/(Cinf - Cx*fx)
    beta3 = beta4**2*Cinfp/max(alpha, 1e-4)

    return (A, alpha, beta1, beta2, beta3, beta4)

def Fit_alpha(rs, epsc, zeta=None, fb=None,
              alpha_low = 0.05, alpha_high = 1.0,
              UseMinimize=True):
    
    c0, c1 = c_PW92(zeta=zeta, fb=fb)
    if zeta is None:
        fx = (2/fb)**(1/3)
    else:
        fx = ((1+zeta)**(4/3)+(1-zeta)**(4/3))/2

    def GetErr(alpha):
        Pu = Lim_to_Param(alpha, c0, c1, Cinf, Cinfp, fx)
        return np.mean(np.abs(F(rs, Pu)-epsc))

    if UseMinimize:
        res = opt.minimize_scalar(GetErr, bounds=[alpha_low, alpha_high])
        alpha_u = res.x
        Err_u = GetErr(alpha_u)
    else:
        # First do a scan
        N = 80
        a = np.linspace(alpha_low, alpha_high, N)
        e = 0.*a
        for k in range(N): e[k] = GetErr(a[k])
        k0 = np.argmin(e)
        k0 = min(max(k0, 1), N-1)
        alpha_low  = a[k0-1]
        alpha_high = a[k0+1]

        # Use bisection method to obtain alpha
        alpha0, alpha1 = alpha_low, alpha_high
        Err0 = GetErr(alpha0)
        Err1 = GetErr(alpha1)
        for bi in range(30):
            #print(alpha0, Err0, alpha1, Err1) # For debugging
            alphan = (alpha0+alpha1)/2.
            Errn = GetErr(alphan)
            if Err0<Err1:
                alpha1, Err1 = alphan, Errn
            else:
                alpha0, Err0 = alphan, Errn


        alpha_u, Err_u = alpha0, Err0

    Pu = Lim_to_Param(alpha_u, c0, c1, Cinf, Cinfp, fx)

    if zeta is None:
        print("alpha(%.3f) = %.5f : Err = %.4f"%(fb  , alpha_u, Err_u))
    else:
        print("alpha(%.3f) = %.5f : Err = %.4f"%(zeta, alpha_u, Err_u))
    return Pu


def DoFit(HEGModel, FitMode="SPI"):
    Pz = {}

    rs = HEGModel.rs
    epsx = -Cx/rs

    zeta = HEGModel.zeta

    FitID = FitMode.upper()[:3]
    
    epscData = np.zeros((len(zeta), len(HEGModel.rs)))

    Pz = {}
    for Kz, z in enumerate(zeta):
        epsc = HEGModel.Getepsc(rs, z)
        epscData[Kz,:] = epsc*1.

        if FitID in ("NEW", "MAP", "COF", "EOT"):
            fb = fmap(z)
            Pz[z] = Fit_alpha(rs, epsc, fb=fb)
        else:
            Pz[z] = Fit_alpha(rs, epsc, zeta=z)

    return Pz, zeta, rs, epscData
