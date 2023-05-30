import numpy as np
import scipy.linalg as la

def GaussLegWeights(n,a=-1,b=1):
    """Generates the abscissa and weights for a Gauss-Legendre quadrature.
    between a and b"""

    x,w=np.polynomial.legendre.leggauss(n)
    S=(b-a)/2.
    return (x+1)*S+a,w*S

def CLindImag(Q=0,gam=0):
    """
    Unitless
    """

    if (not hasattr(Q, "__shape__")):
        1
    elif (not hasattr(gam, "__shape__")):
        1
    elif (Q.shape==gam.shape):
        1
    else:
        print("Error: q and omega are ill-shaped")
        return 0
    
    
    Qp=Q+1
    Qm=Q-1

    gam2=gam**2
    Q2=Q**2

    C=1+(gam2-Qp*Qm)/(4*Q)*np.log((Qp**2+gam2)/(Qm**2+gam2)) \
        + gam*(np.arctan(Qm/gam)-np.arctan(Qp/gam))

    return C

def FitPolyfb(fb, wfb, F, F0=0., F1=0., N=5):    
    a0 = F1
    a1 = (F0-F1)

    Flin = a0 + a1*(fb-1)
    
    dF = F-Flin
    #print("%.4f %.4f %.4f %.4f"%(F[0], Flin[0], F[-1], Flin[-1]))

    fp = np.ones((len(fb),N))
    fp[:,1] = (fb-1)
    fp[:,2] = (fb-1)*(2-fb)
    fp[:,3] = (fb-1)**2*(2-fb)
    if N>4:
        fp[:,4] = (fb-1)**2*(2-fb)**2
    if N>5:
        for k in range(5,N):
            p1 = int(np.ceil(k/2))
            p2 = k - p1
            fp[:,k] = (fb-1)**p1*(2-fb)**p2

    ff = fp[:,2:]

    A = np.einsum('kp,kq,k->pq', ff, ff, wfb)
    b = np.einsum('kp,k,k->p', ff, dF, wfb)

    cf = la.solve(A, b)
    cp = np.hstack(([a0,a1],cf))

    if False:
        print("""Fitting functions are:
    1, (fb-1), (fb-1)*(2-fb), (fb-1)**2*(2-fb1),
    (fb-1)**2*(2-fb)**2, (fb-1)**3*(2-fb)**2, etc
in the order of coefficients.
""")

    FFit = np.dot(fp, cp)

    Err = np.dot(wfb, np.abs(F-FFit))
    print("Err = %.5f"%(Err))
    
    return FFit, cp
