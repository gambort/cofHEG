#!/home/timgould/psi4conda/bin/python3

import psi4
from psi4Engine.Engine import psi4Engine, TextDFA
from Broadway.OOELDA import *
from Broadway.OOEDFT import *
from Broadway.Helpers import *
        

psi4.set_output_file("__IPLDA.out")


Basis = 'aug-cc-pvqz'
#Basis = 'def2-qzvppd'

psi4.set_options({
    'basis': Basis,
    'reference': 'uhf',
    'fail_on_maxiter': False,
})


Atoms = [
    '', 'H', 'He',
    'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar',
]

try:
    X = np.load("./Cache/IP_LDA.npz", allow_pickle=True)
    AtomData = X['AtomData'][()]
except:
    AtomData = { }

if not(Basis in AtomData):
    AtomData[Basis] = {}
else:
    print(list(AtomData[Basis]))

def NtoQ(N):
    for R in [(0,4), (4,10), (10,12), (12,18), (18,20), (20,30), (30,36)]:
        if N>R[0] and N<=R[1]:
            if (R[1]-R[0])==6:
                return 1 + min((N-R[0]), 6-(N-R[0]))
            
    return 1 + N%2

def Ntof(N, Q):
    f = np.zeros((20,))
    Nu = (N + Q - 1)//2
    Nd = (N - Q + 1)//2

    f[:Nu] += 1.
    f[:Nd] += 1.
    kTo = Nu-1

    f = f[:Nu]
    
    return f, kTo

for Z, Atom in enumerate(Atoms):
    if Z<2: continue

    if Atom in AtomData[Basis]: continue
    #if Z<13: continue

    print('='*72)
    print("%2s Z = %d, Q = %d"%(Atom, Z, NtoQ(Z)))

    Q  = NtoQ(Z  )
    Qp = NtoQ(Z-1)

    psi4.geometry("""%d %d\n%s\nsymmetry c1"""%(0, Q , Atom))
    E0, wfn = psi4.energy('svwn', return_wfn=True)
    Nf = wfn.nalpha()

    psi4.geometry("""%d %d\n%s\nsymmetry c1"""%(1, Qp, Atom))
    Ep, wfn = psi4.energy('svwn', return_wfn=True)

    IP_svwn = Ep - E0
    print("IP(SVWN)   = %7.2f eV"%(IP_svwn*eV))

    Engine = psi4Engine(wfn, Report=-1)

    f , kTo  = Ntof(Z  , Q )
    fp, kTop = Ntof(Z-1, Qp)

    f  = f[:Nf]
    fp = fp[:Nf]

    Plan  = { '1RDM': f , 'kTo': (kTo ,), 'Extra': None,
              'Hx':[(1.,1.,f ),],  'xcDFA':[(1.,1.,f ),] }
    Planp = { '1RDM': fp, 'kTo': (kTop,), 'Extra': None,
              'Hx':[(1.,1.,fp),],  'xcDFA':[(1.,1.,fp),] }

    XHelp = OOELDAExcitationHelper(Engine, Report=-1)
    XHelp.Setxi(0.)

    XHelp.SeteLDAType('NO', a=1/3)
    Ep = XHelp.Solver(Planp)
    E0 = XHelp.Solver(Plan )

    IP_dwocc = Ep - E0
    print("IP(dwocc)  = %7.2f eV"%(IP_dwocc*eV))

    
    XHelp.SeteLDAType('NO', a=1  )
    Ep = XHelp.Solver(Planp)
    E0 = XHelp.Solver(Plan )

    IP_wocc = Ep - E0
    print("IP(wocc)   = %7.2f eV"%(IP_wocc*eV))

    AtomData[Basis][Atom] = [ IP_svwn*eV, IP_dwocc*eV, IP_wocc*eV ]
    
    np.savez("./Cache/IP_LDA.npz", AtomData=AtomData)

print("="*72)
print("# N El %7s %7s %7s"%('SVWN', 'dwocc', 'wocc'))
for Z, Atom in enumerate(Atoms):
    if Atom in AtomData[Basis]:
        print("%3d %2s "%(Z, Atom)
              + "%7.2f %7.2f %7.2f"%tuple(AtomData[Basis][Atom])) 
      
