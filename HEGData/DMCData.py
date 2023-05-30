import numpy as np

# Mappings using constant eps_x
def zeta_to_f_epsx(zetap):
    f = np.linspace(2., 1., 601)
    Fac_f = (2/f)**(1/3)
    
    zeta = np.linspace(0., 1., 601)
    Fac_z = ((1+zeta)**(4/3) + (1-zeta)**(4/3))/2

    zeta_f = np.interp(Fac_f, Fac_z, zeta)
    return np.interp(zetap, zeta_f, f)

# Mappings using constant eps_x
def f_to_zeta_epsx(fp):
    f = np.linspace(2., 1., 601)
    Fac_f = (2/f)**(1/3)
    
    zeta = np.linspace(0., 1., 601)
    Fac_z = ((1+zeta)**(4/3) + (1-zeta)**(4/3))/2

    zeta_f = np.interp(Fac_f, Fac_z, zeta)
    return np.interp(-fp, -f, zeta_f)

def KSEn(rs, zeta):
    # From PW92
    CK = 3/10*(9.*np.pi/4.)**(2/3)
    Cx = 3/4/np.pi*(9.*np.pi/4.)**(1/3)
    
    ts = CK/rs**2 * ((1+zeta)**(5/3) + (1-zeta)**(5/3))/2
    epsx = -Cx/rs * ((1+zeta)**(4/3) + (1-zeta)**(4/3))/2
    return ts + epsx

def xEn(rs, zeta):
    # From PW92
    Cx = 3/4/np.pi*(9.*np.pi/4.)**(1/3)
    
    epsx = -Cx/rs * ((1+zeta)**(4/3) + (1-zeta)**(4/3))/2
    return epsx

def tsEn(rs, zeta):
    # From PW92
    CK = 3/10*(9.*np.pi/4.)**(2/3)
    
    ts = CK/rs**2 * ((1+zeta)**(5/3) + (1-zeta)**(5/3))/2
    return ts

class HEG_GridData:
    def __init__(self):
        self.Data = {}
        self.rs = []
        self.zeta = []

    def GetKS(self, rs, zeta):
        return KSEn(rs, zeta)

    def Getts(self, rs, zeta):
        return tsEn(rs, zeta)
        
    def Gete(self, rs=100., zeta=0., allow_interp=True):
        # Return e data
        #
        # returns None for values outside range (allow_interp=True)
        # or not in data (allow_intero=False)

        if hasattr(rs, "__len__"):
            N = len(rs)
            e = np.zeros((N,))
            for k in range(N):
                e[k] = self.Gete(rs[k],zeta, allow_interp)
            return e
        
        if rs in self.Data:
            if allow_interp:
                if (np.min(zeta)<0.) or (np.max(zeta)>1.): return None
                return np.interp(zeta, self.Data[rs]['zeta'],
                                 self.Data[rs]['e'])
            else:
                if hasattr(zeta, "__len__"):
                    print("Can only do non-interpolate mode for scalar rs, zeta")
                    return None
                ii = np.argwhere((self.Data[rs]['zeta']-zeta)
                                 <1e-3).reshape((-1))
                if len(ii)==1:
                    return self.Data[rs]['e'][ii]
                else: return None
        else: return None
        
    def Getepsc(self, rs=100., zeta=0., allow_interp=True):
        # Return epsc data
        #
        # returns None for values outside range (allow_interp=True)
        # or not in data (allow_intero=False)

        e = self.Gete(rs, zeta, allow_interp)
        return e - self.GetKS(rs, zeta)

    def Getepsxc(self, rs=100., zeta=0., allow_interp=True):
        # Return epsxc data
        #
        # returns None for values outside range (allow_interp=True)
        # or not in data (allow_intero=False)

        e = self.Gete(rs, zeta, allow_interp)
        return e - self.Getts(rs, zeta)

class HEG_Zong2002(HEG_GridData):
    def __init__(self):
        array = np.array
        self.Data = \
{40.0: {'e': array([-0.01761874, -0.01761648, -0.0176027 , -0.01756742]),
        'zeta': array([0.   , 0.333, 0.667, 1.   ])},
 50.0: {'e': array([-0.0144495 , -0.0144495 , -0.01444981, -0.01444725, -0.01444418,
       -0.01443771, -0.01442492]),
        'zeta': array([0.   , 0.185, 0.333, 0.519, 0.667, 0.852, 1.   ])},
 60.0: {'e': array([-0.01226009, -0.01225933, -0.01226016, -0.01225982, -0.01225874,
       -0.01225594, -0.01225084]),
        'zeta': array([0.   , 0.185, 0.333, 0.519, 0.667, 0.852, 1.   ])},
 70.0: {'e': array([-0.01065715, -0.01065691, -0.01065811, -0.01065858, -0.01065797,
       -0.01065666, -0.01065334]),
        'zeta': array([0.   , 0.185, 0.333, 0.519, 0.667, 0.852, 1.   ])},
 75.0: {'e': array([-0.01000569, -0.01000596, -0.01000688, -0.01000717, -0.01000439]),
        'zeta': array([0.   , 0.185, 0.333, 0.667, 1.   ])},
 85.0: {'e': array([-0.00892009, -0.00892076, -0.0089215 , -0.00892055]),
        'zeta': array([0.   , 0.333, 0.667, 1.   ])},
 100.0: {'e': array([-0.00767679, -0.0076767 , -0.0076782 , -0.00767881]),
         'zeta': array([0.   , 0.333, 0.667, 1.   ])}}
        
        self.rs = np.array(list(self.Data))
        self.zeta = self.Data[70.0]['zeta']

class HEG_Spink2013(HEG_GridData):
    def __init__(self):
        array = np.array
        self.Data = \
{0.5: {'e': array([3.430114, 3.692876, 4.441646, 5.824982]),
       'zeta': array([0.  , 0.34, 0.66, 1.  ])},
 1.0: {'e': array([0.587801, 0.649192, 0.823944, 1.146342]),
       'zeta': array([0.  , 0.34, 0.66, 1.  ])},
 2.0: {'e': array([0.0023805, 0.0160276, 0.054752 , 0.126293 ]),
       'zeta': array([0.  , 0.34, 0.66, 1.  ])},
 3.0: {'e': array([-0.0670754, -0.0616045, -0.046082 , -0.0172784]),
       'zeta': array([0.  , 0.34, 0.66, 1.  ])},
 5.0: {'e': array([-0.0758811, -0.0742084, -0.0695484, -0.0607175]),
       'zeta': array([0.  , 0.34, 0.66, 1.  ])},
 10.0: {'e': array([-0.05351165, -0.0532142 , -0.0523752 , -0.05073375]),
        'zeta': array([0.  , 0.34, 0.66, 1.  ])},
 20.0: {'e': array([-0.03176865, -0.03171567, -0.03159407, -0.03131604]),
        'zeta': array([0.  , 0.34, 0.66, 1.  ])}}
        
        self.rs = np.array(list(self.Data))
        self.zeta = self.Data[5.0]['zeta']

        
class HEG_Azadi2022:
    def __init__(self):
        self.Fit = {
            0.0 : [-0.131,  1.01 , 0.323],
            0.5 : [-0.151,  1.31 , 0.362],
            1.0 : [-0.0626, 0.978, 0.191],
        }

        self.rs = np.array([30,40,50,60,70,80])
        self.zeta = np.linspace(0.,1.,5)
        
    def Getepsc(self, rs, zeta):
        F_0 = self.Fit[0.0]
        F_1 = self.Fit[1.0]
        F_m = self.Fit[0.5]
        epsc_0 = F_0[0]/(1. + F_0[1]*np.sqrt(rs) + F_0[2]*rs)
        epsc_1 = F_1[0]/(1. + F_1[1]*np.sqrt(rs) + F_1[2]*rs)
        epsc_m = F_m[0]/(1. + F_m[1]*np.sqrt(rs) + F_m[2]*rs)

        # Quadratic fit (on zeta^2) to the three values
        epsc = epsc_0 \
            - zeta**2/3. * (epsc_1 - 16*epsc_m + 15*epsc_0) \
            + zeta**4/3. * (4*epsc_1 - 16*epsc_m + 12*epsc_0)
        
        return epsc

    def Getepsxc(self, rs, zeta):
        return self.Getepsc(rs, zeta) + xEn(rs, zeta)

    def Gete(self, rs, zeta):
        return self.GetKS(rs, zeta) + self.Getepsc(rs, zeta)
    
class HEG_SCE:
    def __init__(self):
        self.Cinf = -9/10
        self.Cpinf = 1.30 # Fit to rs=80 from Azari

        self.rs = np.linspace(80., 200., 11)
        self.zeta = np.array([0.,])

    def Getepsxc(self, rs, zeta=0.):
        return self.Cinf/rs/(1. - self.Cpinf/self.Cinf/np.sqrt(rs))

    def Getepsc(self, rs, zeta):
        return self.Getepsxc(rs, zeta) - xEn(rs, zeta)

    def Gete(self, rs, zeta):
        return tsEn(rs, zeta) + self.epsxc(rs, zeta)
    

def zetaColour(zeta):
    N = len(np.array([zeta]).reshape((-1,)))
    if N>1:
        cc = []
        for z in zeta:
            cc += [zetaColour(z)]
        return cc
    
    c0  = np.array([0.        , 0.50196078, 0.50196078])
    c12 = np.array([0.        , 0.50980392, 0.78431373])
    c1  = np.array([0.90196078, 0.09803922, 0.29411765])

    x = zeta**2
    if x<0.5:
        return tuple((1.-2.*x) * c0 + 2.*x * c12)
    else:
        return tuple((2.-2.*x) * c12 + (2.*x-1.) * c1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Units, UnitsLbl = 1000., "mHa"
    
    HEGData_1 = HEG_Zong2002()
    HEGData_2 = HEG_Spink2013()
    HEGData_3 = HEG_Azadi2022()
    HEGData_4 = HEG_SCE()

    fig, (ax,axIn) = plt.subplots(2,1, figsize=(6,4))
    ax.semilogx()

    for HEGD, m in zip((HEGData_1, HEGData_2, HEGData_3),
                       ("o", "s", "v")):
        rs = HEGD.rs
        zeta = HEGD.zeta
        for z in zeta:
            cc = zetaColour(z)

            ax.scatter(rs, Units*HEGD.Getepsc(rs, z),
                       color=cc, marker=m,
            )

            rs0 = np.array([10.,])
            axIn.scatter(z, Units*HEGD.Getepsc(rs0, z),
                         color=cc, marker=m,
                        )

    rs = HEGData_4.rs
    for z in HEGData_1.zeta:
        cc = zetaColour(z)
        ax.plot(rs, -Units*HEGData_4.Getepsc(rs, z),
                color=cc)
    
    ax.axis([4.,150.,-30.,0.])
    ax.set_xlabel("Wigner-Seitz radius, $r_s$ [Bohr]", fontsize=14)
    ax.set_ylabel("$\\epsilon_{\\rm c}$ [mHa]", fontsize=14)

    axIn.axis([-0.03,1.03,-20,-9])
    axIn.set_xlabel("Spin-polarization, $\\zeta$", fontsize=14)
    axIn.set_ylabel("$\\epsilon_{\\rm c}$ [mHa]", fontsize=14)

    plt.tight_layout(pad=0.3)
    plt.show()

    
