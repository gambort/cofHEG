#!/home/timgould/psi4conda/bin/python3

import psi4
from psi4Engine.Engine import psi4Engine, TextDFA
from Broadway.OOELDA import *
from Broadway.OOEDFT import *
from Broadway.Helpers import *
        
from psi4.driver.procrouting.response.scf_response import tdscf_excitations


psi4.set_output_file("__ST-LDA.out")

#Basis = 'def2-tzvp'
Basis = 'aug-cc-pvqz'

MaxIter = 50

kcal = 627.5

NoSym = False

TD_States = [3,0,0,0,3,0,0,0]

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
    
    
psi4.set_options({
    'basis': Basis,
    'reference': 'uhf',
    'save_jk': True, # Does not work with symmetry
    'fail_on_maxiter': False,
})


for Mol in En:
    print("="*72)

    DoEDFT = not(Mol in AllData[Basis])
    DoTDDFT = not(Mol+"_TD" in AllData[Basis])

    if not(DoEDFT) and not(DoTDDFT):
        print("Skipping %s"%(Mol))
        continue

    if Mol in Geom:
        if Mol=="NO-": Q = -1
        else: Q = 0

        MolStr1 = "%d 3\n%s\n%s 1 %.4f"%(Q, Mol[0], Mol[1], Geom[Mol][0])
        MolStr2 = "%d 3\n%s\n%s 1 %.4f"%(Q, Mol[0], Mol[1], Geom[Mol][1])

        if NoSym:
            MolStr1 += "\nsymmetry c1\n"
            MolStr2 += "\nsymmetry c1\n"

        print(MolStr1)
        print(MolStr2)
        
        psi4.geometry(MolStr1)
        E0, wfn = psi4.energy('svwn', return_wfn=True)
        E_TS_TD = E0
                
        if DoEDFT:        
            Engine = psi4Engine(wfn, Report=-1)
            XHelp = OOELDAExcitationHelper(Engine, Report=3)
            XHelp.SeteLDAType('NO')
        
            E_TS = XHelp.SolveTS(MaxIter=MaxIter)
            XHelp.ShowQuadrature()

        psi4.geometry(MolStr2)
        E0, wfn = psi4.energy('svwn', return_wfn=True)
        

        if DoTDDFT:
            res = tdscf_excitations(wfn, states=TD_States[:wfn.nirrep()], triplets='none')
            E_TDLDA = [ 0. ]
            for r in res:
                E_TDLDA += [ E0 + r['EXCITATION ENERGY'] ]

            E_SX_TD = E_TDLDA[1]

        if DoEDFT:        
            Engine = psi4Engine(wfn, Report=-1)
            XHelp = OOELDAExcitationHelper(Engine, Report=3)
            XHelp.SeteLDAType('NO')
            
            E_SX = XHelp.SolveSX(MaxIter=MaxIter)
            XHelp.ShowQuadrature()
    else:
        MolStr = "0 3\n%s"%(Mol)

        if NoSym:
            MolStr += "\nsymmetry c1\n"

        psi4.geometry(MolStr)
        E0, wfn = psi4.energy('svwn', return_wfn=True)
        E_TS_TD = E0

        if DoTDDFT:
            res = tdscf_excitations(wfn, states=TD_States[:wfn.nirrep()], triplets='none')
            E_TDLDA = [ E0 ]
            for r in res:
                E_TDLDA += [ E0 + r['EXCITATION ENERGY'] ]

            E_SX_TD = E_TDLDA[1]

        if DoEDFT:        
            Engine = psi4Engine(wfn, Report=-1)
            XHelp = OOELDAExcitationHelper(Engine, Report=3)
            XHelp.SeteLDAType('NO')
        
        
            E_TS = XHelp.SolveTS(MaxIter=MaxIter)
            XHelp.ShowQuadrature()
        
            E_SX = XHelp.SolveSX(MaxIter=MaxIter)
            XHelp.ShowQuadrature()

    if DoEDFT:
        En_ELDA = kcal*(E_SX - E_TS)
        AllData[Basis][Mol] = ( En_ELDA, En_ELDA - En[Mol] )
        print("eLDA  Singlet-triplet gap = %6.2f %6.2f [kcal/mol]"\
              %AllData[Basis][Mol])

    if DoTDDFT:
        En_TDLDA = kcal*(E_SX_TD - E_TS_TD)
        AllData[Basis][Mol+'_TD'] = ( En_TDLDA, En_TDLDA - En[Mol] )
        print("TDLDA Singlet-triplet gap = %6.2f %6.2f [kcal/mol]"\
              %AllData[Basis][Mol+"_TD"])

    np.savez("./Cache/ST-LDA.npz", AllData=AllData)

def Report(Suff=''):
    print("="*72)
    print("Suffix = %s"%(Suff))
    print("%8s %6s %6s"%('Sys', 'EST', 'Error'))
    Data = []
    for M in En:
        Mol = M + Suff
        print("%-8s "%(M) \
              + "%6.2f %6.2f [kcal/mol]"%AllData[Basis][Mol])
        Data += [AllData[Basis][Mol][1]]

    Data = np.array(Data)
    Avgs = {
        'MSE': np.mean(Data),
        'MAE': np.mean(np.abs(Data)),
        'RMSD': np.sqrt(np.mean(Data**2)),
    }

    for Key in Avgs:
        print("%-8s "%(Key) \
              + "%6s %6.2f [kcal/mol]"%('-', Avgs[Key]))


Report('_TD')
Report('_TD_NoS')
Report()
