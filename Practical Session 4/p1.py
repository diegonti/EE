"""
FULL. BIN.
"""
import numpy as np
import sympy as sm
from scipy.special import erf
import matplotlib.pyplot as plt
from time import time
to = time()


# Constants and Symbols
pi = np.pi
W = sm.symbols("W")

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
    return NG,d,a

def Norm(a,b):
    """The normalization constant for a given integral <A|B>"""
    return ((2*a/pi)*(2*b/pi))**(3/4)


def Fo(t): 
    """Auxiliary function"""
    if t<1e-6: return 1-t/3
    else: return 0.5*np.sqrt(pi/t)*erf(np.sqrt(t))


# Individual gaussian integrals
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

# Matrix elements
def matrix(N,NG,R,f,*params):
    """
    Constructs the full matrix for a given contribution functions.
    
    Parameters 
    ----------
    `N` : Number of basis used
    `NG`: Number of primitive gaussians in each basis
    `R` : Configuration array. [Ra,Rb] for diatomic molecules.
    `f` : Individual gaussian integral function to construct matrix of (overlap, _S, kinetic, _T, potential, _V)
    `*params` : Extra parameters the functions may need. (Za, Zb for potential _V)
    
    Returns
    -----------
    `M` :: NxN matrix for the given integral.
    """
    
    NG,d,a = NG_parser(NG)          # Get NG and coefitients
    a = np.array(a) * 1.24**2       # Correcting alphas by the exponent

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


def _ijkl(a,b,c,d,Ra,Rb,Rc,Rd):
    Rp = (a*Ra + b*Rb)/(a+b)
    Rq = (c*Rc + d*Rd)/(c+d)
    norms = Norm(a,b)*Norm(c,d)
    coefs = ((a+b)*(c+d)*np.sqrt(a+b+c+d))
    exponents = -a*b/(a+b)*(Ra-Rb)**2 - c*d/(c+d)*(Rc-Rd)**2
    t = (a+b)*(c+d)/(a+b+c+d)*(Rp-Rq)**2

    return 2*pi**(5/2)*norms/coefs*np.exp(exponents)*Fo(t)


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
    sorted_sijkl = sorted(sijkl)
    if sorted_sijkl == [1,1]: return 0         # (11|11) case
    elif sorted_sijkl == [2,2]: return 1       # (22|22) case
    elif sorted_sijkl == [1,2]: return 2       # (ii|jj) cases
    elif sorted_sijkl == [1,3]: return 3       # (11|ij) cases
    elif sorted_sijkl == [2,3]: return 4       # (22|ij) cases
    elif sorted_sijkl == [3,3]: return 5       # (ij|ij) cases
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
N = 2               # Number of electrons
m = 2               # Number of basis functions
R = 1.4632          # He-H distance
Za,Zb = 2,1         # Nuclear atomic charges
za,zb = 2.0925,1.24 # Exponents
eps = 1e-4          # SCF tolerance
max_iter = 10       # Maximum number of iterations

# Parametrized Hamiltonian and Overlapping matrices
S = [1,0.4508]
t = [2.1643,0.7600,0.1670]
v = np.array(
    [[-4.1398,-1.1029,-1.2652],
    [-0.6772,-0.4113,-1.2266]])
vaa = v[0,0] + v[1,0]
vbb = v[0,2] + v[1,2]
vab = v[0,1] + v[1,1]

S = np.array([S,S[::-1]])
T = np.array([[2.1643,0.1617],[0.1617,0.7600]])
V = np.array([[vaa,vab],[vab,vbb]])
H = T+V

print("\nOverlap Matrix S:\n",S)
print("\nKinetic Matrix T:\n",T)
print("\ne-N Potential Matrix V:\n",V)
print("\nHamiltonian Matrix H:\n",H)


# Bielectronic integrals
bielectronic = [1.3072,0.7746,0.6057,0.4373,0.3118,0.1773]


# Transformation Matrix X from S so that XSX=1
Seval,U = np.linalg.eigh(S)
S12 = np.diag(Seval**-0.5)
print("\nSquareroot inverse of S\n",S12)
print("\nUnitary Matrix U\n",U)

X = U@S12     #U@S12@U.T.conj() also works   

print("\nTransformation Matrix X\n",X)
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
    G = G_matrix(P0.copy())         # Bielectronic Matrix
    F = H + G                       # Fock Matrix
    Ft =  X.T.conj()@F@X            # Tranformed Fock Matrix
    e,Ct = np.linalg.eigh(Ft)       # Orbital Energies and transformed coefs
    C = X@Ct                        # Orbital Coeffitients
    Pt = P_matrix(m,C)              # Denisty Matrix
    Eelec = 0.5*np.sum(Pt*(H+F))    # Electronic Energy E0
    
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

tf = time()
print(f"\nProcess finished in {tf-to:.5f}s.\n")




