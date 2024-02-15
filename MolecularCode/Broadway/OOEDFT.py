from Broadway.EDFT import CoreExcitationHelper
from Broadway.pEDFT import pExcitationHelper
from Broadway.LDAFits import *

import numpy as np
import scipy.linalg as la
import numpy.random as ra

eV = 27.211

zero_round = 1e-5

# Generate U or 
def CondenseFList(FMap, FList, Mode='U', f=None):
    N = 0
    for k in FMap:
        N = max(np.max(FMap[k])+1, N)
    U = np.zeros((N,N))
    for k1 in FList:
        for k2 in FList:
            if Mode=='U' and k2>=k1: continue
            elif Mode=='F' and k2>k1: continue

            for a in FMap[k1]:
                if Mode=='U':
                    Delta = f[k2] * FList[k2][a,FMap[k2]] - f[k1] * FList[k1][a,FMap[k2]]
                    U[a,FMap[k2]] = -Delta
                    U[FMap[k2],a] =  Delta
                else:
                    Delta = 0.5*(FList[k1][a,FMap[k2]] + FList[k2][a,FMap[k2]])

                    U[a,FMap[k2]] = Delta
                    U[FMap[k2],a] = Delta
    return U

# Estimate of the Hessian
def EstimateIH(f, epsilon, eta=0.1):
    H0 = -2.*(f[:,None] - f[None,:])*(epsilon[:,None] - epsilon[None,:])
    #return np.sign(H0)/np.maximum(np.abs(H0), eta)
    return np.abs(H0)/(H0**2 + eta**2)


class OOExcitationHelper(pExcitationHelper):
    # Set default xi
    def Setxi(self, xi):
        # This is the scaling factor for sr density-driven correlations
        if xi is None:
            self.xi = 0.35
        else:
            self.xi = xi
            
    # Set the internal properties
    def SetProps(self, pEDFT = False,
                 MaxIter = 200, ShowIter = 20,
                 DECut = 1e-7, DepsCut = 1e-6,
                 Fail = True,
                 **kwargs):
        self.Props['MaxIter'] = MaxIter
        self.Props['ShowIter'] = ShowIter
        self.Props['DECut']   = DECut
        self.Props['DepsCut'] = DepsCut
        self.Props['Fail']    = Fail
        self.Props['pEDFT']   = pEDFT
        
    # Generic solver routine - currently can only call SolverpEDFT
    def Solver(self, Plan, kFrom=None, kTo=None,
               Dipole = False, # Return the dipole as well
               **kwargs):
        return self.SolverOOEDFT(Plan, **kwargs)
        
    def SolverOOEDFT(self, Plan, **kwargs):
        return self.SolverOOEDFT_LS(Plan, **kwargs)
    
    # Solve the pEDFT equations iteratively using a
    # line search after NLineStep iterations
    def SolverOOEDFT_LS(self, Plan,
                        Reset = True, # Reset to ground state
                        
                        Dipole = False, # Return the dipole as well
                        
                        delta = 1.0, # Default delta guess
                        NLineStep = 50, # Start quadratic refinement at this step
                        Prefactor = True, # Use an approximate 2nd deriv prefactor

                        IH_eta = 0.1, # Factor for approximate 2nd derivative
                        
                        MixUOld = 0.3, # Mix this much of the old U (negative is random mixing)
                        DE_Break = 1e-6, # Break when energy varies less than this
                        UM_Break = 1e-4, # Break when ||U|| is less than this
                        MaxIter = 500, # Maximum number of steps to take
                        **kwargs):
        if self.Props['pEDFT']:
            self.SolverpEDFT(Plan, Dipole, **kwargs)
        elif Reset:
            self.CE = 1.*self.C0
            self.epsilonE = 1.*self.epsilon0

        E, FList, FMap = self.GetEnergyFocks(Plan)
        
        E_Trial, F_Trial = self.GetEnergy(Plan)

        # Convert F_Trial into orbital basis
        F_Trial = np.einsum('pq,pj,qk->jk', F_Trial, self.CE, self.CE)

        # Get the occupation factors in full size
        NOrb = F_Trial.shape[0]
        f = np.zeros((NOrb,))
        f_ = Plan['1RDM']
        f[:len(f_)] = f_

        if Prefactor:
            # Guess second deriative
            eps = np.diag(F_Trial)*1.
            IH0 = EstimateIH(f, eps, eta = IH_eta)
        else:
            IH0 = 1.

        # Initialize last steps
        P = np.eye(NOrb)
        EOld = E_Trial
        UOld = None
        POld = None
        UOver = 1.
        delta_Old = delta

        if self.Report>0:
            print("%4s %12s %7s %6s %6s %6s [ %23s ]"\
                  %('step', 'En [Ha]', 'DE [eV]', 'delta', '||U||', 'U_ang',
                    'Line search E [eV]'),
                  flush=True)


        # Mask out terms that shouldn't interact because of
        # symmetry
        SymMask = 1.

        SUnique = set(list(self.Engine.Sym_k)) # Unique symmetries
        if len(SUnique)>1:
            Mask = 0. * F_Trial

            for S in SUnique:
                k = np.argwhere(self.Engine.Sym_k==S).reshape((-1,))
                Mask[(k[:,None], k)] = 1.


        self.Converged = False
        for step in range(MaxIter):
            E, FList, FMap = self.GetEnergyFocks(Plan)

            U = IH0 * CondenseFList(FMap, FList, f=f, Mode='U')

            U *= SymMask
            

            if self.Report>=10: # Full debug mode
                k1 = max(self.kh-2,0)
                k2 = min(self.kl+3,NOrb)

                print("Error in U = %.10f"%(np.mean(np.abs(U + U.T))))
                print("k = [ %s ]"%( " ".join(["%3d"%(x) for x in range(k1,k2)])))
                if step==0:
                    print(FMap)
                    for k in FList:
                        print(k)
                        print(f[k]*FList[k][k1:k2,k1:k2])
                
                print(U[k1:k2,k1:k2])

            if Prefactor:
                # Updated guessed second deriative
                for k in FList:
                    eps[FMap[k]] = np.diag(FList[k])[FMap[k]]
                IH0 = EstimateIH(f, eps, eta = IH_eta)

            # Mix int he last step for faster convergence
            if not(UOld is None):
                if MixUOld>=0.:
                    U = (1-MixUOld)*U + MixUOld*UOld
                else:
                    X = np.abs(UOld)
                    x = (1-X)/2 + X*ra.rand()
                    U = x*U + (1-x)*UOld

                UM = np.sqrt(np.sum(U**2))

                if (UM*UMOld)>1e-10:
                    UOver = np.vdot(U, UOld)/UM/UMOld
                else:
                    UOver = 1.0
            else:
                UOver = 1.0
                UM = np.sqrt(np.sum(U**2))


            if (step>=NLineStep):
                # Make a qudaratic fit to determine best change

                deltas = [0, 1.5*delta, 3.0*delta]
                
                C1 = self.CE.dot(la.expm( deltas[1]*U))
                C2 = self.CE.dot(la.expm( deltas[2]*U))


                E1, _ = self.GetEnergy(Plan, C=C1)
                E2, _ = self.GetEnergy(Plan, C=C2)

                pE = np.polyfit(deltas, [E, E1, E2], 2)

                if pE[0]>0.:
                    delta_Opt = pE[1]/(-2.*pE[0])
                    delta_Opt = np.sign(delta_Opt) * min(deltas[2], np.abs(delta_Opt))
                else:
                    delta_Opt = deltas[2]*np.sign(E-E2)
            else:
                E1 = E_Trial
                E2 = E_Trial
                delta_Opt = delta_Old

            delta_Old = (delta_Opt + delta_Old)/2

            if self.Report>0:
                print("%4d %12.5f %7.3f %6.3f %6.4f %6.3f [ %7.2f %7.2f %7.2f ]"\
                      %(step, E, eV*(E-E_Trial), delta_Opt, UM, UOver,
                        (E-E_Trial)*eV, (E1-E_Trial)*eV, (E2-E_Trial)*eV ),
                      flush=True)

            EU = la.expm( delta_Opt*U)
            P = P.dot(EU)
            self.CE = self.CE.dot(EU)

            if (step>0) and ((np.abs(EOld - E)<DE_Break)
                             or (UM < UM_Break*(1-np.abs(MixUOld)))):
                self.Converged = True
                break

            EOld = E
            UOld = U
            UMOld = UM


        if self.Report>3: # Full debug mode
            k1 = max(self.kh-2,0)
            k2 = min(self.kl+3,NOrb)

            print("Final U - with k = [ %s ]"%( " ".join(["%3d"%(x) for x in range(k1,k2)])))
            P = la.expm(U)
            print(P[k1:k2,k1:k2])

        SC = self.Engine.S_ao.dot(self.CE)
        for k in FList:
            self.epsilonE[FMap[k]] = np.diag(FList[k])[FMap[k]]
            self.LastF = (SC).dot(FList[k]).dot(SC.T)

        self.LastPlan = Plan
        self.Lastf = f
        
        f = np.array(Plan['1RDM'])
        self.LastD = np.einsum('pk,qk,k', self.CE[:,:len(f)], self.CE[:,:len(f)], f)
        
        if self.Converged:
            1
        elif self.Props['Fail']:
            print("Error! Failed to converge - returning None")
            return None
        else:
            print("Warning! Failed to converge - returning last calc")

        return E

    # Solve the pEDFT equations iteratively using
    # L-BFGS
    def SolverOOEDFT_L_BFGS(self, Plan,
                            Dipole = False, # Return the dipole as well
                            
                            delta = 0.1, # Default delta guess
                            MixUOld = 0.7, # Mix this much of the old U (negative is random mixing)
                            DE_Break = 1e-6, # Break when energy varies less than this
                            MaxIter = 200, # Maximum number of steps to take
                            NBack = 10, # Number of back steps to include
                            **kwargs):
        if self.Props['pEDFT']:
            self.SolverpEDFT(Plan, Dipole, **kwargs)
        else:
            self.CE = 1.*self.C0
            self.epsilonE = 1.*self.epsilon0

        E, FList, FMap = self.GetEnergyFocks(Plan)
        E_Trial, F_Trial = self.GetEnergy(Plan)

        # Convert F_Trial into orbital basis
        F_Trial = np.einsum('pq,pj,qk->jk', F_Trial, self.CE, self.CE)

        # Get the occupation factors in full size
        NOrb = F_Trial.shape[0]
        f = np.zeros((NOrb,))
        f_ = Plan['1RDM']
        f[:len(f_)] = f_

        # Initialize last steps
        x = np.zeros((NBack, NOrb, NOrb)) # Positions
        g = np.zeros((NBack, NOrb, NOrb)) # Gradients

        cache_step = 0

        # Guess second deriative
        eps = np.diag(F_Trial)
        IH0 = EstimateIH(f, eps)
        
        P = np.eye(NOrb) # Unitary matrix
        for step in range(MaxIter):
            E, FList, FMap = self.GetEnergyFocks(Plan)

            # Get U
            U = -CondenseFList(FMap, FList, f=f, Mode='U') # Actual gradient

            cache_step = step%NBack

            x[cache_step,:,:] = la.logm(P)
            g[cache_step,:,:] = U
            

            if step < NBack:
                Indx = np.arange(step+1)
            else:
                Indx = np.hstack((np.arange(cache_step+1, NBack), np.arange(cache_step+1)))
            #print("%10.5f %7.2f"%(E, eV*(E-E_Trial)))

            # Update
            EU = la.expm(-delta*IH0*U)
            P = P.dot(EU)
            self.CE = self.CE.dot(EU)


        self.Lastf = f
        self.LastPlan = Plan
        
        return E

class ExcitationHelper(OOExcitationHelper):
    1


if __name__ == "__main__":
    1
