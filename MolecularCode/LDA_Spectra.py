#!/home/timgould/psi4conda/bin/python3

import psi4
from psi4Engine.Engine import psi4Engine, TextDFA
from Broadway.OOELDA import *
from Broadway.OOEDFT import *
from Broadway.Helpers import *
        

psi4.set_output_file("__LDA.out")

import sys

if len(sys.argv)<2:
    print("Must specify an input file")
    quit()

try:
    FullMolID = sys.argv[1]
    F = open(FullMolID)
    MolStr = "".join(list(F))
    F.close()
    psi4.geometry(MolStr)
except:
    print("%s is not a valid input file"%(sys.argv[1]))
    quit()


print("Molecule:")
print(MolStr)

Options = {
    'BASIS': 'cc-pvdz',
    'REPORT': 1,
    'MEM': '8gb',
    'PRE': 10.0,
    'MAXITER': 50,
    'A': 0.33333333333,
}

# Process input variables
for k in range(2, len(sys.argv)):
    q = sys.argv[k].split('=')
    var = q[0].upper()[:8]
    if len(q)>1:
        val = "".join(q[1:])
    else:
        val = None
    Options[var] = val


print("Options = ", Options)

if ('MEM' in Options):
    psi4.set_memory(Options['MEM'])
if ('MEMORY' in Options):
    psi4.set_memory(Options['MEMORY'])


if ('QUICK' in Options):
    Options['BASIS'] = 'def2-msvp'
    psi4.set_options({'dft_radial_points':50, 'dft_spherical_points':110})
    
psi4.set_options({
    'basis': Options['BASIS'],
    'reference': 'rhf',
    'save_jk': True, # Does not work with symmetry
})


print("Basis: %s"%(Options['BASIS']))


try:
    E0, wfn = psi4.energy('svwn', return_wfn=True)
except:
    E0, wfn = psi4.energy('scf', return_wfn=True)
    psi4.set_options({'maxiter':1, 'fail_on_maxiter':False, 'reference':'rhf',})
    E0, wfn = psi4.energy('svwn', return_wfn=True)
    


# Do a TD-DFT calculations
print("Running TDDFT")
from psi4.driver.procrouting.response.scf_response import tdscf_excitations
res = tdscf_excitations(wfn, states=12, triplets='none')
Omega_TDLDA = [ 0. ]
for r in res:
    Omega_TDLDA += [ r['EXCITATION ENERGY'] ]
print("Done TDDFT")

Engine = psi4Engine(wfn)

print("="*72)
if ('CLDA' in Options):
    XHelp = OOExcitationHelper(Engine,
                               Report=int(Options['REPORT']),
    )
    Quad = False
    print("Running combination LDA (SVWN)")
else:
    XHelp = OOELDAExcitationHelper(Engine,
                                   Report=int(Options['REPORT']),
    )

    IgnorefDeriv = ('IGNORE' in Options)

    XHelp.SeteLDAType('RE', IgnorefDeriv=IgnorefDeriv)
    if 'UNP' in Options: XHelp.SeteLDAType('Unpolarized')
    if 'SQ' in Options: XHelp.SeteLDAType('SQ', IgnorefDeriv=IgnorefDeriv)
    if 'NO' in Options: XHelp.SeteLDAType('NO', IgnorefDeriv=IgnorefDeriv, a=float(Options['A']))
    if ('OT' in Options) or ('ON' in Options): XHelp.SeteLDAType('OT', IgnorefDeriv=IgnorefDeriv)

    if XHelp.eLDAType in ('ON', 'OT'):
        print("Running OT-ELDA")
    elif XHelp.eLDAType in ('SQ', 'SQ'):
        print("Running square case")
    elif XHelp.eLDAType in ('OP', 'NO'):
        print("Running optimized and normed case")
    elif XHelp.eLDAType in ('U', 'UN'):
        print("Running unpolarized ELDA")
    else:
        print("Running fbar ELDA")
    Quad = True

XHelp.Setxi(0.)
    

print("="*72)

if 'TO' in Options:
    XHelp.SetTo(Options['TO'])
if 'FROM' in Options:
    XHelp.SetFrom(Options['FROM'])

def ShowEn(ID, E, Ep):
    print("%-4s : %11.4f [Ha], Gap = %6.2f [eV]"\
          %(ID, E, eV*(E-Ep)), flush=True)

print("EDFA gaps:", flush=True)
E_GS = XHelp.SolveGS(MaxIter=int(Options['MAXITER']))
if Quad: XHelp.ShowQuadrature()
eps_From = XHelp.epsilonE[:XHelp.kl]*1.
C0 = 1.*XHelp.CE


E_TS = XHelp.SolveTS(MaxIter=int(Options['MAXITER']))
if Quad: XHelp.ShowQuadrature()


E_SX = XHelp.SolveSX(MaxIter=int(Options['MAXITER']))
if Quad: XHelp.ShowQuadrature()
eps_To   = XHelp.epsilonE[XHelp.kl:]*1.

E_DX = XHelp.SolveDX(MaxIter=int(Options['MAXITER']))
if Quad: XHelp.ShowQuadrature()


print("="*72)
print("Excitation GS %12.4f %8.2f"%(E_GS, 0.))
print("Excitation TS %12.4f %8.2f"%(E_TS, (E_TS-E_GS)*eV))
print("Excitation SX %12.4f %8.2f"%(E_SX, (E_SX-E_GS)*eV))
if not(E_DX is None):
    print("Excitation DX %12.4f %8.2f"%(E_DX, (E_DX-E_GS)*eV))

###################################################################################
# Guess the excited states that lie lower than
# 1.5 max(Omega_DX, Omega_TDLDA[1])
Excite = {}

# Get all pairs allowed by symmetries
SymUnique = set(list(XHelp.Engine.Sym_k)) # Unique symmetries
FromToPairs = []
for S1 in SymUnique:
    k = np.argwhere(XHelp.Engine.Sym_k==S1).reshape((-1,))
    try:
        kFrom = k[k<XHelp.kl].max() # From the max in each sym
    except:
        continue
    for S2 in SymUnique:
        k = np.argwhere(XHelp.Engine.Sym_k==S2).reshape((-1,))
        kTo   = k[k>XHelp.kh].min() # To the min in each sym
        FromToPairs += [(kFrom, kTo)]

print('='*72)
print("Occupied orbitals")
print(NiceArr(eps_From*eV))
print("Unoccupied orbitals")
print(NiceArr(eps_To*eV))
print('='*72)


for kFrom, kTo in FromToPairs:
    eT = eps_To[kTo-XHelp.kl]
    eF = eps_From[kFrom]
    
    deps = eT - eF

    if deps*eV < float(Options['PRE']):
        print("(%3d, %3d) %6.2f | "%(kFrom, kTo, deps*eV), end='')
        
        Excite[deps] = (kFrom, kTo)

print()
        
###################################################################################

# Sort them and get the From and To
Excite = [(deps, Excite[deps][0], Excite[deps][1]) for deps in sorted(Excite)]

SymFrom = XHelp.Engine.Sym_k[XHelp.kh]
SymTo   = XHelp.Engine.Sym_k[XHelp.kh]

Omega_ID = [ ('X', 0,0, SymFrom, SymTo, True) ]
Omega_ELDA = [ 0. ]
for deps, kFrom, kTo in Excite:
    print('='*72)
    print("Doing %3d -> %3d"%(kFrom, kTo))
    
    XHelp.SetFrom(kFrom)
    XHelp.SetTo(kTo)

    XHelp.epsilonE = XHelp.epsilon0
    XHelp.CE = XHelp.C0

    SymFrom = XHelp.Engine.Sym_k[kFrom]
    SymTo   = XHelp.Engine.Sym_k[kTo  ]

    #XHelp.CE = 1.*C0

    E_SX_I = XHelp.SolveSX(MaxIter=int(Options['MAXITER']))
    if Quad and not(E_SX_I is None): XHelp.ShowQuadrature()

    #GetPropsLDA(XHelp.LastPlan, Report=True) # For debugging purposes
    
    E_DX_I = XHelp.SolveDX(MaxIter=int(Options['MAXITER']))
    if Quad and not(E_DX_I is None): XHelp.ShowQuadrature()
    
    #GetPropsLDA(XHelp.LastPlan, Report=True) # For debugging purposes

    # Test the double excitation for rotation to a lower state
    OK = (XHelp.epsilonE[XHelp.kTo] - XHelp.epsilonE[XHelp.kFrom])>0.

    if not(E_SX_I is None):
        Omega_ID += [ ('S', kFrom, kTo, SymFrom, SymTo, True), ]
        Omega_ELDA += [ E_SX_I - E_GS, ]
        DE = E_SX_I - E_GS
        print("Excitation energy = %7.2f vs %7.2f (delta = %7.2f) from orbitals"\
              %( DE*eV, deps*eV, (DE - deps)*eV ))
    if not(E_DX_I is None) and OK:
        Omega_ID += [ ('D', kFrom, kTo, SymFrom, SymTo, True), ]
        Omega_ELDA += [ E_DX_I - E_GS, ]
        print("Excitation energy = %7.2f"\
              %((E_DX_I-E_GS)*eV))



# Check for better ground state
Indx = np.argsort(Omega_ELDA)
Omega_ELDA = np.sort(Omega_ELDA) - np.min(Omega_ELDA)
Omega_ID = [ Omega_ID[I] for I in Indx ]



print("== TDLDA ==")
for k in range(len(Omega_TDLDA)):
    print("Energy %2d = %7.2f eV"%( k, Omega_TDLDA[k]*eV))

print("== ELDA ==")
for k in range(len(Omega_ELDA)):
    print("Energy %2d = %7.2f eV"%( k, Omega_ELDA[k]*eV) \
          + "  %s %03d->%03d %d->%d %s"%Omega_ID[k])


