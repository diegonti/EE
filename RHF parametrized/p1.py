"""
Practice Session 3 -  Problem 1
Parametrized RHF for H2 molecule.
Diego Ontiveros
"""

import numpy as np
from time import time
to = time()

def print_title(text,before=15,after=15,separator="-",head=2,tail=1):
    """Prints text in a title-style."""
    separator = str(separator)
    print("\n"*head,separator*before,text,separator*after,"\n"*tail)

def bie_index(ijkl:str):
    """Takes string of the "ijkl" indices of the bielectronic integral and returns
    the index of the bielectronic parametrized array of integrals it corresponds."""
    i,j,k,l =[int(t) for t in ijkl]
    sij = i+j if i!=j else i
    skl = k+l if k!=l else k
    sijkl = [sij,skl]
    if len(set(ijkl)) <= 1: return 0              # (ii|ii) cases
    elif sorted(sijkl) == [1,2]: return 1       # (ii|jj) cases
    elif sorted(sijkl) == [1,3]: return 2       # (ii|ij) cases
    elif sorted(sijkl) == [2,3]: return 2       # (jj|ij) cases
    elif sorted(sijkl) == [3,3]: return 3       # (ij|ij) cases
    else: raise ValueError("Bielectronic indices nor valid.")

def G_matrix(P):
    """
    Computes bielectornic matrix.\n
    Parameters: `P` : m x m density matrix.\n
    Returns: `G` : m x m bielectronic matrix.
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

    return G

def P_matrix(m,C):
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


######################### MAIN PROGRAM ####################

print_title("Initial Molecular Integrals")
# Inputs
N = 2           # Number of electrons
m = 2           # Number of basis functions
R = 1.4         # H-H distance
Za,Zb = 1,1     # Nuclear atomic charges
z = 1.24        # exponent
eps = 1e-4      # SCF tolerance
max_iter = 100  # Maximum number of iterations

# Parametrized Hamiltonian and Overlapping matrices
S = [1,0.6593]
t = [0.7600,0.2365]
v = [-1.2266,-0.5974,-0.6538]
v = [v[0]+v[2], 2*v[1]]

S = np.array([S,S[::-1]])       # Overlap Matrix
T = np.array([t,t[::-1]])       # Kinetix Matrix
V = np.array([v,v[::-1]])       # VNe Matrix
H = T+V                         # Hamiltonian

print("\nOverlap Matrix S:\n",S)
print("\nKinetic Matrix T:\n",T)
print("\ne-N Potential Matrix V:\n",V)
print("\nHamiltonian Matrix H:\n",H)


# Bielectronic integrals array
bielectronic = [0.7746,0.5697,0.4441,0.2970]


# Transformation Matrix X from S so that XSX=1
Seval,U = np.linalg.eigh(S)
S12 = np.diag(Seval**-0.5)
print("\nSquareroot inverse of S\n",S12)
print("\nUnitary Matrix U\n",U)

X = U@S12      #U@S12@U.T.conj() also works   

print("\nTransformation MatrixX\n",X)
print("\nMatrix product XSX = 1\n",X.T@S@X)


# Guess density matrix (at zero)
P0 = np.zeros(shape=(m,m))
print("\nGuess P0\n",P0)


# print("\n\n","-"*15,"Entering SCF loop","-"*15,"\n")
print_title("Entering SCF loop")

n_iterations = 0
while True:
    n_iterations +=1
    print_title(f"SCF Iternation: {n_iterations}",head=1,tail=0,before=10,after=0)

    # SCF step
    G = G_matrix(P0.copy())     # Bielectronic Matrix
    F = H + G                   # Fock Matrix
    Ft =  X.T.conj()@F@X        # Tranformed Fock Matrix
    e,Ct = np.linalg.eigh(Ft)   # Orbital Energies and transformed coefs
    C = X@Ct                    # Orbital Coeffitients
    Pt = P_matrix(m,C)          # Denisty Matrix
    Eelec = 0.5*np.sum(Pt*(H+F))
    
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
        print("Total energy:          E = ",Eelec+Vnn)

        print("\nOrbital Energies e:\n",e)
        print("\nOrbital Coeffitients C:\n",C)
        print("\nMulliken Population Matrix:\n",mulliken)

        break

    if n_iterations >= max_iter:
        print_title("Ending SCF loop. NOT CONVERGED! :(")
        break

    P0 = Pt.copy()  # Updating Density Matrix

tf = time()
print(f"\nProcess finished in {tf-to:.5f}s.\n")








