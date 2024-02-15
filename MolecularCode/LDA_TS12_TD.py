#!/home/timgould/psi4conda/bin/python3
import numpy as np
from pyscf import gto, scf, dft, tddft


#Basis = 'def2-tzvp'
Basis = 'aug-cc-pvqz'

#xcStr = 'LDA, PW91'
xcStr = 'svwn'

kcal = 627.5

Sym = True

#####################################################################
# Benchmark data

GeomStr = """
NF 1.31698 1.3079
NH 1.034 1.0362
NO- 1.258 1.262
O2 1.20752 1.2156 11
PF 1.5897 1.5849 12
PH 1.4223 1.4302 13,14
S2 1.8892 1.8983 15
SO 1.481087 1.49197 16,17
"""

Geom = {}
for L in GeomStr.split('\n'):
    X = L.split()
    if len(X)<3: continue
    Geom[X[0]] = [ float(X[1]), float(X[2]) ]

EnStr = """
C 29.14
NF 34.32
NH 35.93
NO- 17.30
O2 22.64
O 45.37
PF 20.27
PH 21.90
S2 13.44
S 26.41
Si 18.01
SO 18.16
"""

En = {}
for L in EnStr.split('\n'):
    X = L.split()
    if len(X)<2: continue
    En[X[0]] = float(X[1])

#####################################################################

try:
    X = np.load("./Cache/ST-LDA.npz", allow_pickle=True)
    AllData = X['AllData'][()]
except:
    AllData = {}


if not(Basis in AllData): AllData[Basis] = {}
print(list(AllData))
print(list(AllData[Basis]))

for Mol in En:
    print("="*72)

    if Mol=="NO-": Q = -1
    else: Q = 0
    
    if Mol in Geom:
        MolStr1 = "%s\n%s 1 %.4f"%(Mol[0], Mol[1], Geom[Mol][0])
        MolStr2 = "%s\n%s 1 %.4f"%(Mol[0], Mol[1], Geom[Mol][1])
    else:
        MolStr1 = "%s"%(Mol)
        MolStr2 = MolStr1

    mol = gto.Mole()
    mol.build(
        atom = MolStr1,  # in Angstrom
        charge = Q,
        basis = Basis,
        symmetry = Sym,
        spin = 2,
        verbose = 0,
    )

    mf = dft.UKS(mol)
    mf.xc = xcStr
    E1 = mf.kernel()

    mol.build(
        atom = MolStr2,  # in Angstrom
        basis = Basis,
        charge = Q,
        symmetry = Sym,
        spin = 2,
        verbose = 0,
    )

    mf = dft.UKS(mol)
    mf.xc = xcStr
    E2 = mf.kernel()


    mytd = tddft.TDDFT(mf)
    mytd.nstates = 10
    E3, _ = mytd.kernel()

    print(E1, E2)

    En_TDLDA = kcal*(E3[0] + E2 - E1)

    New_TDLDA = ( En_TDLDA, En_TDLDA - En[Mol] )
    
    AllData[Basis][Mol+'_TD_NoS'] = ( En_TDLDA, En_TDLDA - En[Mol] )
    print("TDLDA Singlet-triplet gap = %6.2f %6.2f [kcal/mol]"\
          %AllData[Basis][Mol+"_TD"])
    print("TDLDA Singlet-triplet gap = %6.2f %6.2f [kcal/mol]"\
          %AllData[Basis][Mol+'_TD_NoS'] )
    np.savez("./Cache/ST-LDA.npz", AllData=AllData)
