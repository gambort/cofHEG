import numpy as np
import psi4

np.set_printoptions(precision=4, suppress=True)

class ForceHelper:
    def __init__(self, basisset):
        self.basisset = basisset
        self.molecule = basisset.molecule()
        self.mints = psi4.core.MintsHelper(wfn.basisset())
        
        self.nbf = basisset.nbf()
        self.atomindx = {}

        # Distances
        self.R12 = self.molecule.distance_matrix().to_array(dense=True)
        self.xyz = self.molecule.full_geometry().to_array(dense=True)

        # Get the atoms
        self.natom = 0
        for p in range(self.nbf):
            a = basisset.function_to_center(p)
            self.natom = max(self.natom, a+1)
            
            if a in self.atomindx:
                self.atomindx[a] += [p]
            else:
                self.atomindx[a]  = [p]

        for a in self.atomindx:
            self.atomindx[a] = np.array(self.atomindx[a])

        self.ERI = None

    def GetIndex(self, atom=0):
        return self.atomindx[atom % self.natom]

    def GetBasis(self, atom=0):
        a = atom % self.natom
        # Not implemented

    def GetDFBasis(self, atom=0):
        a = atom % self.natom
        # Not implemented

    # Get the nuclear force
    def ForceNuclear(self):
        return self.molecule.nuclear_repulsion_energy_deriv1().to_array(dense=True)

    # Get the one electron integral force
    def ForceOEI(self, D_ao, epsD_ao):
        F_OEI = np.zeros((self.natom, 3))
        
        for a in range(AtomFH.natom):
            dT = self.mints.ao_oei_deriv1("KINETIC", a)
            dV = self.mints.ao_oei_deriv1("POTENTIAL", a)
            dS = self.mints.ao_oei_deriv1("OVERLAP", a)

            for alpha in range(3):
                F_OEI[a,alpha]  = np.vdot(dT[alpha].to_array(dense=True), D_ao)
                F_OEI[a,alpha] += np.vdot(dV[alpha].to_array(dense=True), D_ao)
                F_OEI[a,alpha] -= np.vdot(dS[alpha].to_array(dense=True), epsD_ao)

        return F_OEI

    # Get a crude approximation to the ERI force
    def CrudeForceERI(self, D_ao):
        F = np.zeros((self.natom,3))

        S_ao = self.mints.ao_overlap().to_array(dense=True)
        SD = S_ao * D_ao

        for a1 in range(self.natom):
            indx1 = self.atomindx[a1]
            N1 = np.sum(SD[indx1,:][:,indx1])

            for a2 in range(self.natom):
                if a1==a2: continue # Exclude self-self
                
                indx2 = self.atomindx[a2]
                N2 = np.sum(SD[indx2,:][:,indx2])

                F12 = (self.xyz[a2,:] - self.xyz[a1,:])/self.R12[a1,a2]**3

                F[a1,:] += F12*N1*N2

        return F

    # Get the ERI force
    def ForceERI(self, D_ao,
                 DFName = None,
                 Moved=False):
        # Create the density fit stuff if required

        psi4.core.clean()
        
        if (self.ERI is None) or Moved:
            self.ERI = {}

            # Create the density fit basis functions        
            if DFName is None: DFName = self.basisset.name()
            aux_basis = psi4.core.BasisSet.build\
                (wfn.molecule(), "DF_BASIS_SCF", "",
                 'JKFIT', DFName)
            zero_basis = psi4.core.BasisSet.zero_ao_basis_set()
            
            # Create the integral factory for DF
            self.ERI['IntFac_DF'] = psi4.core.IntegralFactory\
                (aux_basis, zero_basis, aux_basis, zero_basis)

            SAB = np.squeeze(mints.ao_eri(aux_basis, zero_basis,
                                          self.basisset, self.basisset))
            metric = mints.ao_eri(aux_basis, zero_basis, aux_basis, zero_basis)
            self.ERI['ERI_DF'] = np.squeeze(metric)
            
            metric.power(-1.0, 1e-14)
            metric = np.squeeze(metric)

            self.ERI['To_DF'] = np.tensordot(metric, SAB, axes=((1,),(0,)))

        # Calculate the force
        for a in range(self.natom):
            print("Computing dERI")
            print(self.ERI['IntFac_DF'])
            #dERI = np.squeeze(
            self.mints.ao_tei_deriv1(a, 0.0, self.ERI['IntFac_DF']) # ao_tei_deriv1
            #    )
            #print(dERI.shape)

            

if __name__ == "__main__":
    psi4.set_memory('8gb')
    
    MolStr1 = """
O     -1.088248    0.844052   -0.021009
H     -2.035650    1.008848   -0.555136
H     -1.108334    1.591056    0.786194
symmetry c1
"""

    MolStr2 = """
B
H  1 4.0
symmetry c1
"""
    
    psi4.set_options({
        "basis": "cc-pvtz",
        "reference": "rhf",
    })
    
    psi4.geometry(MolStr2) # Change number here
    G = psi4.gradient("pbe").to_array(dense=True)
    E, wfn = psi4.energy("pbe", return_wfn=True)
    AtomFH = ForceHelper(wfn.basisset())

    S = wfn.S().to_array(dense=True)
    D = wfn.Da().to_array(dense=True)*2.

    NOcc = wfn.nalpha()
    epsilon = wfn.epsilon_a().to_array(dense=True)
    C = wfn.Ca().to_array(dense=True)
    epsD = 2.*np.einsum('k,pk,qk->pq', epsilon[:NOcc], C[:,:NOcc], C[:,:NOcc])

    # Get the 1RDM forces
    F_OEI = np.zeros((AtomFH.natom, 3))
    F_TEI = np.zeros((AtomFH.natom, 3))
    
    mints = psi4.core.MintsHelper(wfn.basisset())
    for a in range(AtomFH.natom):
        dT = mints.ao_oei_deriv1("KINETIC", a)
        dV = mints.ao_oei_deriv1("POTENTIAL", a)
        dS = mints.ao_oei_deriv1("OVERLAP", a)

        dERI = mints.ao_tei_deriv1(a)

        for alpha in range(3):
            F_OEI[a,alpha]  = np.vdot(dT[alpha].to_array(dense=True), D)
            F_OEI[a,alpha] += np.vdot(dV[alpha].to_array(dense=True), D)
            F_OEI[a,alpha] -= np.vdot(dS[alpha].to_array(dense=True), epsD)

            X = np.tensordot(dERI[alpha].to_array(dense=True),
                             D, axes=((0,1),(0,1)))
            F_TEI[a,alpha] = 0.5*np.vdot(D, X)

    print("Nuclear force")
    print(AtomFH.ForceNuclear())
    print("1RDM force")
    print(F_OEI)
    print(AtomFH.ForceOEI(D, epsD))
        
    F = - (F_OEI + AtomFH.ForceNuclear())

    print("2RDM force")
    print(F_TEI)

    print("Crude 2RDM force")
    print(AtomFH.CrudeForceERI(D))

    AtomFH.ForceERI(D)

    F -= F_TEI

    print("Our force")
    print(F)
    print("True force")
    print(G)
    print("Error")
    print(F-G)
    
    
