#!/home/timgould/psi4conda/bin/python3

import psi4
from psi4Engine.Engine import psi4Engine, TextDFA
from Broadway.OOELDA import *
from Broadway.OOEDFT import *
from Broadway.Helpers import *
        

psi4.set_output_file("__IPLDA.out")

import sys

#Basis = 'aug-cc-pvtz'
Basis = 'aug-cc-pvqz'
#Basis = 'def2-qzvppd'

if len(sys.argv)>2:
    Basis = sys.argv[1]

psi4.set_options({
    'basis': Basis,
    'reference': 'uhf',
    'fail_on_maxiter': False,
})

DFA = {
    "name": "pw92",
    "x_functionals": {"LDA_X":{},},
    "c_functionals": {"LDA_C_PW":{}},
}

AllData = {}
E0 , EE0 = None, None
for D in (5,3,1):
    psi4.geometry("""%d %d\n%s\nsymmetry c1"""%(0, D, "Fe"))

    E, wfn = psi4.energy('scf', dft_functional=DFA, return_wfn=True)
    if E0 is None: E0 = E

    print("E(LDA ) = %10.4f Ha, Gap = %5.2f eV"%(E, eV*(E - E0)))

    Engine = psi4Engine(wfn, Report=-1)

    f = np.zeros(16)
    f[:wfn.nalpha()] += 1.
    f[:wfn.nbeta() ] += 1.
    kTo = wfn.nalpha()

    Plan  = { '1RDM': f , 'kTo': (kTo ,), 'Extra': None,
              'Hx':[(1.,1.,f ),],  'xcDFA':[(1.,1.,f ),] }

    XHelp = OOELDAExcitationHelper(Engine, Report=3)
    XHelp.Setxi(0.)

    XHelp.SeteLDAType('NO', a=1/3)
    EE = XHelp.Solver(Plan )
    if EE0 is None: EE0 = EE

    print("E(ELDA) = %10.4f Ha, Gap = %5.2f eV"%(EE, eV*(EE - EE0)))

    AllData[D] = (E, EE)

    psi4.core.clean()

print("="*72)

Experiment = {
    5: 0.00, # +- 0.06 [eV]
    3: 1.47, # +- 0.06 [eV]
    1: 3.05, # From 5 average
}

for D in (5,3,1):
    print("Degen = %d"%(D))
    E, EE = AllData[D]
    print("Exper.  Gap = %5.2f eV"%( Experiment[D] ) )
    print("E(LDA ) Gap = %5.2f eV [%10.4f Ha]"%(eV*(E  - E0 ), E ) )
    print("E(ELDA) Gap = %5.2f eV [%10.4f Ha]"%(eV*(EE - EE0), EE) )
