"""
Module that contains the functions used for
computing RHF integrals and matrices.

Diego Ontiveros
"""

import numpy as np
import sympy as sm
from scipy.special import erf

##################### EXTERNAL CONSTANTS, SYMBOLS AND FUNCTIONS #################

# Constants and Symbols
pi = np.pi
W = sm.symbols("W")

# Fit parameters for the STO-NG (d,alpha)
coef = {
    "1g": [[1],[0.270950]],
    "2g": [[0.678914,0.430129],[0.151623,0.851819]],
    "3g": [[0.444635,0.535328,0.154329],[0.109818,0.405771,2.22766]]
}

def print_title(text,before=15,after=15,separator="-",head=2,tail=1):
    """Prints text in a title-style."""
    separator = str(separator)
    print("\n"*head,separator*before,text,separator*after,"\n"*tail)

def NG_parser(NG):
    """ 
    Parser for the number of gaussians.
    Takes a general input (int, 1, or str, '1g') and returns the NG, d and a vectors.
    """
    if type(NG) == str:
        d,a = coef[NG.lower()]
        NG = int(NG[0])
    elif type(NG) == int:
        d,a = coef[f"{NG}g"]
    else: 
        raise SyntaxError("Insert a valid number of gaussians. Eg, ""2g"" or 2")
    return NG,d,a

def Fo(t): 
    """Auxiliary function"""
    if t<1e-6: return 1-t/3
    else: return 0.5*np.sqrt(pi/t)*erf(np.sqrt(t))


#############################  MAIN CLASES  #################################

class Molecule():
    def __init__(self,
        geometry:np.ndarray,
        charges:np.ndarray,
        effective_charges:np.ndarray,
        N_electrons: int):
        """
        Creates a Molecule object of the molecule that will be used for the RHF calculations.

        Parameters
        ----------
        `geometry` : Array with the positions of the atoms in the molecule.
        `charges` : Array with the nuclear charged of each atom.
        `effective_charges` : Array with the effective charges of each atom.
        `N_electrons` : Number of electrons of the molecule.
        """

        self.geometry = geometry
        self.charges = charges
        self.effective_charges = effective_charges
        self.Natoms = len(charges)
        self.N_electrons = N_electrons

    def get_parameters(self):
        pass

       
class RHF():

    def __init__(self,
        molecule: Molecule = None,
        NG: int|str = None):
        """
        Class that contains the RHF calculations for two-electron diatomic molecules using the STO-NG basis set.

        Parameters
        ----------
        (Optional, can also be loaded with the load() function)
        `molecule` : Molecule ocject the RHF will be based on.
        `NG` : Number of primitive gaussians used for the STO-NG basis. Accepts int (3) or str ("3g").
        """
        self.molecule = molecule
        self.NG = NG
        self.m = molecule.Natoms
        self.N = molecule.N_electrons


        pass



    def SCF(self,eps:float=1e-4,max_iter:int=100):

        config = self.molecule.geometry
        m = self.m
        NG = self.NG
        Za,Zb = self.molecule.charges
        R = abs(config[1]-config[0])

        ############# compute initial integrals in other function !

        # Compute molecular integrals (S,T,V --> H)
        S = molecular_matrix(m,NG,config,_S)
        T = molecular_matrix(m,NG,config,_T)
        V = molecular_matrix(m,NG,config,_V,Za,Zb,config)

        H = T+V

        # Print initial integrals           ################ +FUNCTION
        print("\nOverlap Matrix (S):\n",S)
        print("\nKinetic Matrix (T):\n",T)
        print("\nPotential Matrix (V):\n",V)
        print("\nHamiltonian Matrix (H):\n",H)


        # Compute bielectronic integrals
        bielectronic = bielectronic_matrix()
        bielectronic = [0.7746,0.5697,0.4441,0.2970]
        ###################Print bielectronic function


        # Compute transformation matrix X from S so that XSX=1
        Seval,U = np.linalg.eigh(S)
        S12 = np.diag(Seval**-0.5)
        X = U@S12     #U@S12@U.T.conj() also works  

        print("\nSquareroot inverse of S\n",S12)
        print("\nUnitary Matrix U\n",U)
        print("\nTransformation Matrix X\n",X)
        print("\nMatrix product XSX = 1\n",X.T@S@X)


        # Guess density matrix (at zero)
        P0 = np.zeros(shape=(m,m))
        print("\nGuess P0\n",P0)


        # SCF LOOP
        print_title("Entering SCF loop")
        n_iterations = 0
        while True:
            n_iterations += 1
            print_title(f"SCF Iternation: {n_iterations}",head=1,tail=0,before=10,after=0)

            # SCF step
            G = G_matrix(P0.copy(),bielectronic)    # Bielectronic Matrix
            F = H + G                               # Fock Matrix
            Ft =  X.T.conj()@F@X                    # Tranformed Fock Matrix
            e,Ct = np.linalg.eigh(Ft)               # Orbital Energies and transformed coefs
            C = X@Ct                                # Orbital Coeffitients
            Pt = P_matrix(m,N,C)                    # Denisty Matrix
            Eelec = 0.5*np.sum(Pt*(H+F))            # Electronic Energy E0

            # Printing current iteration results
            print("\nBielectronic Matrix G:\n",G)
            print("\nFock Matrix F:\n",F)
            print("\nTransformed Fock Matrix Ft\n",Ft)
            print("\nOrbital Energies e:\n",e)
            print("\nOrbital Coeffitients C:\n",C)
            print("\nDensity Matrix P\n",Pt)
            print("\nElectronic Energy: \n", Eelec)

            # Convergence
            if converged(P0,Pt,eps):
                print_title("Ending SCF loop. CONVERGED!")

                # Function to print info!

                # Energy contributions
                Vnn = Za*Zb/R           # Nuclear Repulsion
                Eelec = Eelec           # Electronic Energy
                E = Eelec + Vnn         # Total Energy
                mulliken = Pt@S         # Mulliken Population Matrix

                # Printing converged results
                print("\nNumber of SCF iterations:",n_iterations)
                print("\nElectronic Energy: Eelec =",Eelec)
                print("Nuclear Repulsion:   Vnn = ",Vnn)
                print("Total Energy:          E = ",Eelec+Vnn)

                print("\nOrbital Energies e:\n",e)
                print("\nOrbital Coeffitients C:\n",C)
                print("\nMulliken Population Matrix:\n",mulliken)

                break

            if n_iterations >= max_iter:
                print_title("Ending SCF loop. NOT CONVERGED! :(")
                break

            P0 = Pt.copy()  # Updating Density Matrix
        
        pass



# Matrix elements
def molecular_matrix(N,NG,R,f,*params):
    """
    Constructs the full molecular matrix for a given contribution function (f).
    
    Parameters 
    ----------
    `N` : Number of basis used
    `NG`: Number of primitive gaussians in each basis
    `R` : Configuration array. [Ra,Rb] for diatomic molecules.
    `f` : Individual gaussian integral function to construct matrix of (overlap, _S, kinetic, _T, potential, _V)
    `*params` : Extra parameters the functions may need. (Za, Zb for potential _V)
    
    Returns
    ----------
    `M` : N x N matrix for the given integral.
    """
    
    NG,d,a = NG_parser(NG)          # Get NG and coefitients
    a = np.array(a) * 1.24**2       # Correcting alphas by the exponent         ##################### OJO

    # Loop for all basis functions (N) and primitives (NG)
    M = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):

            # Current configurations 
            Ra = R[i]
            Rb = R[j]
                            
            for p in range(NG):
                for q in range(NG):

                    # Updating each matrix coeffitient with STO-NG formula
                    M[i,j] += d[p]*d[q]*f(a[p],a[q],Ra,Rb, *params)
    return M

def bielectronic_matrix():
    pass


################  Individual gaussian molecular integrals  #####################

def Norm(a,b):
    """The normalization constant for a given integral <A|B>"""
    return ((2*a/pi)*(2*b/pi))**(3/4)

def _S(a,b,Ra,Rb): 
    """Overlap Integral"""
    return Norm(a,b)*(pi/(a+b))**(3/2) * np.exp(-a*b/(a+b)*(Ra-Rb)**2)

def _T(a,b,Ra,Rb): 
    """Kinetic Integral"""
    fracc = a*b/(a+b)
    return fracc*(3-2*fracc*(Ra-Rb)**2)*_S(a,b,Ra,Rb)

def _Vp(a,b,Ra,Rb,Z,Rc):
    """Individual Electron-Nucleus atraction Integral"""
    Rp = (a*Ra + b*Rb)/(a+b)
    t = (a+b)*(Rp-Rc)**2

    return -2*pi/(a+b)*Z * np.exp(-a*b/(a+b)*(Ra-Rb)**2)*Fo(t) * Norm(a,b)

def _V(a,b,Ra,Rb,Za,Zb,R):
    """Electron-Nucleus atraction Integral. Accounts for both nucleus."""
    return _Vp(a,b,Ra,Rb,Za,R[0]) + _Vp(a,b,Ra,Rb,Zb,R[1])


def _ijkl(a,b,c,d,Ra,Rb,Rc,Rd):     ####################### OJO
    """Bi-electronic Integral"""
    Rp = (a*Ra + b*Rb)/(a+b)
    Rq = (c*Rc + d*Rd)/(c+d)
    norms = Norm(a,b)*Norm(c,d)
    coefs = ((a+b)*(c+d)*np.sqrt(a+b+c+d))
    exponents = -a*b/(a+b)*(Ra-Rb)**2 - c*d/(c+d)*(Rc-Rd)**2
    t = (a+b)*(c+d)/(a+b+c+d)*(Rp-Rq)**2

    return 2*pi**(5/2)*norms/coefs*np.exp(exponents)*Fo(t)


######################### SCF Functions #############################


def G_matrix(P,bielectronic):
    """
    Computes bielectornic matrix.\n
    Parameters: 
    ------------
    `P` : m x m density matrix.
    `bielectornic`: bielectronic integrals tensor.

    Returns: 
    ------------
    `G` : m x m bielectronic matrix.
    """
    m = len(P)
    G = np.zeros(shape=(m,m))
    for mu in range(m):
        for nu in range(m):

            for l in range(m):
                for s in range(m):
 
                    # Getting ijkl and index of the integral it corresponds
                    munusl = f"{mu+1}{nu+1}{s+1}{l+1}"
                    mulsnu = f"{mu+1}{l+1}{s+1}{nu+1}"
                    i1 = bie_index(munusl)    
                    i2 = bie_index(mulsnu)


                    G[mu,nu] += P[l,s]*(bielectronic[i1] - 0.5*bielectronic[i2])
                   

                    # G[mu,nu] += P[l,s]*(bielectronic[mu,nu,s,l] - 0.5*bielectronic[mu,l,s,nu])

    return G

def P_matrix(m,N,C):
    """
    Computes Density matrix for the given coeffitients matrix.\n
    Parameters: `C` : m x m coefitients matrix.\n
    Returns: `P` : m x m density matrix.
    """
    P = np.zeros(shape=(m,m))
    for l in range(m):
        for s in range(m):
            for j in range(int(N/2)):
                P[l,s] += 2*C[l,j]*C[s,j]

    return P


def converged(P0,Pt,eps):
    """
    Determines if a SCF step is converged with a certain tolerance (eps).
    `P0` : Old Density matrix. (m x m array)
    `Pt` : New Density matrix. (m x m array)
    `eps`: Tolerance value.
    """
    m = len(Pt)
    diff = np.sqrt(np.sum((Pt-P0)**2)/m**2)
    
    if diff<=eps: return True
    else: return False



def bie_index(ijkl:str):
    """Takes string of the "ijkl" indices of the bielectronic integral and returns
    the index of the bielectronic parametrized array of integrals it corresponds."""
    i,j,k,l =[int(t) for t in ijkl]
    sij = i+j if i!=j else i
    skl = k+l if k!=l else k
    sijkl = [sij,skl]
    if len(set(ijkl)) <= 1: return 0            # (ii|ii) cases
    elif sorted(sijkl) == [1,2]: return 1       # (ii|jj) cases
    elif sorted(sijkl) == [1,3]: return 2       # (ii|ij) cases
    elif sorted(sijkl) == [2,3]: return 2       # (jj|ij) cases
    elif sorted(sijkl) == [3,3]: return 3       # (ij|ij) cases
    else: raise ValueError("Bielectronic indices nor valid.")


# Input parameters
NG = 3          # Number of primitive gaussian functions per basis
R = 1.4      # Distance between atoms
N = 2           # Number of Basis functions
Ra = 0          # Position of H(A)
Rb = Ra+R       # Position of H(B)
Za,Zb = 1,1     # Atomic charges of each nucleus
za,zb = 1.24,1.24
config = np.array([Ra,Rb])  # Positions Array
print("\nInput parameters:")
print(f"{R = }  {NG = }")



H2 = Molecule(
    geometry=[Ra,Rb],
    charges=[Za,Zb],
    effective_charges=[za,zb],
    N_electrons = N
    )

scf = RHF(H2,NG=3)
scf.SCF()



