"""
Module for the RHF program.
Contains the Molecule class, which allows to create a molecule with its geometry and information (charges, electrons,..)
And the RHF class, which reads a Molecule object and performs RHF calculations using SCF with the STO-NG basis set.

Diego Ontiveros
"""

import sys, os
import numpy as np
from scipy.special import erf

##################### EXTERNAL CONSTANTS, SYMBOLS AND FUNCTIONS #################

# Constants and Symbols
pi = np.pi

# Fit parameters for the STO-NG (d,alpha)
coef = {
    "1g": [[1],[0.270950]],
    "2g": [[0.678914,0.430129],[0.151623,0.851819]],
    "3g": [[0.444635,0.535328,0.154329],[0.109818,0.405771,2.22766]]
}

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
    return NG,np.array(d),np.array(a)

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
        N_electrons: int,
        molecule_label:str=None):
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
        self.molecule_label = molecule_label
        #self.labels +atom label implementation with dict charges?
        

       
class RHF():

    def __init__(self,
        molecule: Molecule,
        NG: int|str):
        """
        Class that contains the RHF calculations for two-electron diatomic molecules using the STO-NG basis set.

        Parameters
        ----------
        `molecule` : Molecule ocject the RHF will be based on.
        `NG` : Number of primitive gaussians used for the STO-NG basis. Accepts int (3) or str ("3g").
        """
        self.molecule = molecule
        self.NG = NG
        self.m = molecule.Natoms
        self.N = molecule.N_electrons


    def SCF(self,
        P0:np.ndarray=None,
        eps:float=1e-4,
        max_iter:int=100,
        file_name:str = "out.log",
        print_options:list=["all"]):
        """
        Employs SCF method to obtain the orbital energies and coefitients.
        Writes output in the specified file. 

        Parameters
        ----------
        `P0` : Optional. m x m Initial guess density matrix. By default zero.
        `eps` : Optional. Tolerance for the SCF convergence. By default 1e-4.
        `max_iter` : Optional. Maximum number of iterations for the SCF loop. By default 100.
        `file_name`: Output file name.
        `print_options` : Optional. List with the results to print. Defaults to "all"
                "molecular" --> Prints molecular integrals.\n
                "bielectronic" --> Prints bielectronic integrals.\n
                "transformation" --> Prints transformation matrix.\n
                "guess density" --> Prints the initial guess density.\n
                "iterations" --> Prints information of each SCF iteration.\n
                "all" --> Prints everything. \n
                None or "nothing" --> Prints only final results and steps.

        Returns
        ---------
        `Vnn` : Nuclear repulsion energy.
        `Eelec` : Electronic energy.
        `E` : Total energy. Eelec + Vnn.
        Only returns if the SCF converges.
        """
        m = self.m
        NG = self.NG
        N = self.molecule.N_electrons
        config = self.molecule.geometry
        Za,Zb = self.molecule.charges
        R = abs(config[1]-config[0])

        not_print = None in print_options or "nothing" in print_options

        # Compute molecular integrals (S,T,V --> H)
        S,T,V = self.molecular_integrals()
        H = T+V

        # Compute bielectronic integrals        
        bielectronic = self.bielectronic_integrals()


        # Compute transformation matrix X from S so that XSX=1
        Seval,U = np.linalg.eigh(S)
        S12 = np.diag(Seval**-0.5)
        X = U@S12     #U@S12@U.T also works  

        # Guess density matrix (at zero)
        if P0 is None: P0 = np.zeros(shape=(m,m))


        # Printing Initial information Section
        if not not_print:
            print_title("Initial Molecule Information")
            if "all" in print_options or "molecular" in print_options:
                print_molecular_matrices(S,T,V,H)
                                    
            if "all" in print_options or "bielectronic" in print_options:
                print_bielectronic_tensor(bielectronic)

            if "all" in print_options or "transformation" in print_options:
                print_transformation_matrix(S12,U,X,S)

            if "all" in print_options or "guess density" in print_options:
                print("\nGuess P0\n",P0)    

            # Could use an implementation of the match/case from Python 3.10, but brings less compatibility.

        # Writing to output file
        original_stdout = sys.stdout
        outFile = open(file_name,"a")
        sys.stdout = outFile
        print_title("Initial Molecule Information")
        print_molecular_matrices(S,T,V,H)
        print_bielectronic_tensor(bielectronic)
        print_transformation_matrix(S12,U,X,S)
        print("\nGuess P0\n",P0)  
        print_title("Entering SCF loop")
        sys.stdout = original_stdout

        # SCF LOOP
        if not not_print: print_title("Entering SCF loop")
        n_iterations = 0
        while True:

            n_iterations += 1

            # SCF step
            G = G_matrix(P0,bielectronic)    # Bielectronic Matrix
            F = H + G                        # Fock Matrix
            Ft =  X.T.conj()@F@X             # Tranformed Fock Matrix
            e,Ct = np.linalg.eigh(Ft)        # Orbital Energies and transformed coefs
            C = X@Ct                         # Orbital Coeffitients
            Pt = P_matrix(N,C)               # Denisty Matrix
            Eelec = 0.5*np.sum(Pt*(H+F))     # Electronic Energy E0


            # Printing and writing current iteration results
            if not not_print:
                if "all" in print_options or "iterations" in print_options:
                    print_title(f"SCF Iternation: {n_iterations}",head=1,tail=0,before=10,after=0)
                    print_current_iteration(G,F,Ft,e,C,Pt,Eelec)
            
            sys.stdout = outFile
            print_title(f"SCF Iternation: {n_iterations}",head=1,tail=0,before=10,after=0)
            print_current_iteration(G,F,Ft,e,C,Pt,Eelec)
            sys.stdout = original_stdout


            # Convergence
            if converged(P0,Pt,eps):

                # Energy contributions
                Vnn = Za*Zb/R           # Nuclear Repulsion
                Eelec = Eelec           # Electronic Energy
                E = Eelec + Vnn         # Total Energy
                mulliken = Pt@S         # Mulliken Population Matrix

                # Printing and writing converged results
                if not not_print:
                    print_title("Ending SCF loop. CONVERGED!")
                    print_converged(n_iterations,Eelec,Vnn,E,e,C,mulliken)
                
                sys.stdout = outFile
                print_title("Ending SCF loop. CONVERGED!")
                print_converged(n_iterations,Eelec,Vnn,E,e,C,mulliken)
                sys.stdout = original_stdout
                outFile.close()

                return Vnn,Eelec,E

            if n_iterations >= max_iter:
                print_title("Ending SCF loop. NOT CONVERGED! :(")

                sys.stdout = outFile
                print_title("Ending SCF loop. NOT CONVERGED! :(")
                sys.stdout = original_stdout
                outFile.close()
                break

            P0 = Pt.copy()  # Updating Density Matrix
        sys.stdout = original_stdout


    def PES(self,R_array:np.ndarray=None,file_name="PES.log"):
        """
        Returns the PES of the fundamental state calculated at the R_array positions.
        Calculates also the equilibrium configuration.
        Writes results in the specified file.

        Parameters
        ----------
        `R_array` : Optional. Internuclear distances array. By default makes an array
            from 1 unit before the current distance to 5 after.
        `file_name`: Output file name.

        Returns
        -------
        `energies` : PES for the fundamental state.
        `R_array` : Array of distances used.
        `minE` : Minimum Energy found.
        `minR` : Equilibrium distance found.
        """
        try: os.remove(file_name)
        except FileNotFoundError: pass

        if R_array is None:
            R = abs(self.molecule.geometry[1] - self.molecule.geometry[0])
            R_array = np.linspace(R-1,R+5,100)
        if self.molecule.molecule_label is None: molecule_label = "1"
        else: molecule_label = self.molecule.molecule_label


        print_title(f"Starting PES Calculation for {molecule_label}")
        # Calculating PES for the R_array
        energies = np.zeros(len(R_array))
        for i,Ri in enumerate(R_array):
            with open(file_name,"a") as outFile:
                outFile.write(f"\n\n---------------- R = {Ri}\n\n")
            self.molecule.geometry = np.array([0,Ri])
            Vnn,Eelec,E = self.SCF(file_name=file_name, print_options=["nothing"])
            energies[i] = E
        

        # Searching minimum.
        minE = np.min(energies)
        minEi = np.argmin(energies)
        minR = R_array[minEi]

        print_title("PES Calculation ended",head=1)
    
                                
        return energies,R_array,minE,minR
        

    def molecular_integrals(self):
        """
        Computes matrices made from the molecular integrals.

        Returns 
        ----------
        `S` : m x m array. Overlap integral matrix.
        `T` : m x m array. Kinetic integral matrix.
        `V` : m x m array. Electron-nucleus potential integral matrix.

        """
        m = self.m                          # Number of basis functions
        NG = self.NG                        # Number of primitive gaussians
        config = self.molecule.geometry     # Geometry array
        z = self.molecule.effective_charges # Effective charges array
        Za,Zb = self.molecule.charges       # Nuclear charges

        S = molecular_matrix(m,NG,config,z,_S)
        T = molecular_matrix(m,NG,config,z,_T)
        V = molecular_matrix(m,NG,config,z,_V,Za,Zb,config)

        return S,T,V

    def bielectronic_integrals(self):
        """
        Computes tensor made from the bielectronic (ij|kl) integrals.

        Returns
        -------
        `bielectronic` : m x m x m x m array. Bielectronic integrals tensor.
        """

        m = self.m                          # Number of basis functions
        NG = self.NG                        # Number of primitive gaussians
        config = self.molecule.geometry     # Geometry array
        z = self.molecule.effective_charges # Effective charges array

        bielectronic = bielectronic_matrix(m,NG,config,z)

        return bielectronic


#################### Initial Matrices (molecular, bielectronic) ##############

def molecular_matrix(N,NG,R,z,f,*params):
    """
    Constructs the full molecular matrix for a given contribution function (f).
    
    Parameters 
    ----------
    `N` : Number of basis functions used.
    `NG`: Number of primitive gaussians in each basis.
    `R` : Configuration array. [Ra,Rb] for diatomic molecules.
    `z` : Effective charges array.
    `f` : Individual gaussian integral function to construct matrix of (overlap, _S, kinetic, _T, potential, _V)
    `*params` : Extra parameters the functions may need. (Za, Zb for potential _V)
    
    Returns
    ----------
    `M` : N x N matrix for the given integral.
    """
    za,zb = z[0],z[1]               # Get effective charges
    NG,d,a = NG_parser(NG)          # Get NG and coefitients
    a = np.array([a*za**2,a*zb**2]) # Correct exponents by effective charge

    # Loop for all basis functions (N) and primitives (NG)
    M = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):

            # Current gaussian center 
            Ra,Rb = R[i],R[j]
                            
            for p in range(NG):
                for q in range(NG):

                    # Current GTO exponent
                    alpha,beta = a[i,p], a[j,q]

                    # Updating each matrix coeffitient with STO-NG formula
                    M[i,j] += d[p]*d[q]*f(alpha,beta,Ra,Rb, *params)

    return M

def bielectronic_matrix(N,NG,R,z):
    """
    Constructs full bielectronic tensor for each ijkl integral.

    Parameters
    ----------
    `N` : Number of basis functions used.
    `NG`: Number of primitive gaussians in each basis.
    `R` : Configuration array. [Ra,Rb] for diatomic molecules.
    `z` : Effective charges array.

    Returns
    ----------
    `M` : N x N x N x N bielectronic tensor.
    """
    za,zb = z[0],z[1]                   # Get effective charges
    NG,d,a = NG_parser(NG)              # Get NG and coefitients
    a = np.array([a*za**2,a*zb**2])     # Correct exponents by effective charge

    # For each gaussian center (atom)
    M = np.zeros(shape=(N,N,N,N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):

                    # Current gaussian center
                    Ra,Rb = R[i],R[j]
                    Rc,Rd = R[k],R[l]

                    # For each primitive gaussian
                    for p in range(NG):
                        for q in range(NG):
                            for r in range(NG):
                                for s in range(NG):

                                    # Current GTO parameters (d and alpha exponents)
                                    d_coefs = d[p]*d[q]*d[r]*d[s]
                                    alpha,beta = a[i,p], a[j,q]
                                    gamma,delta = a[k,r], a[l,s]

                                    # Updating each matrix coeffitient with STO-NG formula
                                    M[i,j,k,l] += d_coefs*_ijkl(alpha,beta,gamma,delta, Ra,Rb,Rc,Rd)
    
    return M


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


def _ijkl(a,b,c,d,Ra,Rb,Rc,Rd):
    """Bi-electronic Integral """
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
    Computes bielectornic matrix.

    Parameters
    ------------
    `P` : m x m density matrix.
    `bielectornic`: bielectronic integrals tensor.

    Returns
    ------------
    `G` : m x m bielectronic matrix.
    """
    m = len(P)
    G = np.zeros(shape=(m,m))
    for mu in range(m):
        for nu in range(m):

            for l in range(m):
                for s in range(m):
 
                    G[mu,nu] += P[l,s]*(bielectronic[mu,nu,s,l] - 0.5*bielectronic[mu,l,s,nu])

    return G

def P_matrix(N,C):
    """
    Computes Density matrix for the given coeffitients matrix.

    Parameters
    -----------
     `C` : m x m coefitients matrix.

    Returns
    -----------
    `P` : m x m density matrix.
    """
    m = len(C)
    P = np.zeros(shape=(m,m))
    for l in range(m):
        for s in range(m):
            for j in range(int(N/2)):
                P[l,s] += 2*C[l,j]*C[s,j]

    return P


def converged(P0,Pt,eps):
    """
    Determines if a SCF step is converged with a certain tolerance (eps).

    Parameters
    -----------
    `P0` : Old Density matrix. (m x m array)
    `Pt` : New Density matrix. (m x m array)
    `eps`: Tolerance value.

    Returns
    ----------
    `bool` : True or False whether it is converged or not.
    """
    m = len(Pt)
    diff = np.sqrt(np.sum((Pt-P0)**2)/m**2)
    
    if diff<=eps: return True
    else: return False


##################### PRINTING FUNCTIONS ######################

def print_title(text,before=15,after=15,separator="-",head=2,tail=1):
    """Prints text in a title-style."""
    separator = str(separator)
    print("\n"*head,separator*before,text,separator*after,"\n"*tail)

def print_molecular_matrices(S,T,V,H):
    """Prints molecular matrices S,T,V and H."""
    print("\nOverlap Matrix (S):\n",S)
    print("\nKinetic Matrix (T):\n",T)
    print("\nPotential Matrix (V):\n",V)
    print("\nHamiltonian Matrix (H):\n",H)

def print_bielectronic_tensor(tensor):
    """Prints bielectronic tensor in a legible way."""
    m = len(tensor)
    u = 2
    print("\nBielectronic Tensor (ij|kl):")
    print(" I  J  K  L     (ij|kl)")
    print_title("-",before=10,after=15,head=0,tail=0)
    for i in range(m):
        for j in range(m):
            for k in range(m):
                for l in range(m):

                    print(f" {i+1}  {j+1}  {k+1}  {l+1}   {tensor[i,j,k,l]}")

def print_transformation_matrix(S12,U,X,S):
    """Prints transformation matrix X information."""
    print("\nSquareroot inverse of S:\n",S12)
    print("\nUnitary Matrix U:\n",U)
    print("\nTransformation Matrix X:\n",X)
    print("\nMatrix product X.TSX = 1:\n",X.T@S@X)

def print_current_iteration(G,F,Ft,e,C,Pt,Eelec):
    """Prints results of the current iteration."""
    print("\nBielectronic Matrix G:\n",G)
    print("\nFock Matrix F:\n",F)
    print("\nTransformed Fock Matrix Ft:\n",Ft)
    print("\nOrbital Energies e:\n",e)
    print("\nOrbital Coeffitients C:\n",C)
    print("\nDensity Matrix P:\n",Pt)
    print("\nElectronic Energy: \n", Eelec)

def print_converged(n_iterations,Eelec,Vnn,E,e,C,mulliken):
    """Prins information of the converged results."""
    print("\nNumber of SCF iterations:",n_iterations)
    print("\nElectronic Energy: Eelec =",Eelec)
    print("Nuclear Repulsion:   Vnn = ",Vnn)
    print("Total Energy:          E = ",E)

    print("\nOrbital Energies e:\n",e)
    print("\nOrbital Coeffitients C:\n",C)
    print("\nMulliken Population Matrix:\n",mulliken)   
    print() 

