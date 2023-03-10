"""
Practice Session 2 -  Problem 5
H2+ energies made from 1s STO-NG.
Computes E1 and E2 for a given R and NG.
Plots the PES of the ground and first excited state for the given NG.
Diego Ontiveros
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


###################### MAIN PROGRAM #################


# Input parameters
NG = 3          # Number of primitive gaussian functions per basis
R = 2.0035      # Distance between atoms
N = 2           # Number of Basis functions
Ra = 0          # Position of H(A)
Rb = Ra+R       # Position of H(B)
Za,Zb = 1,1     # Atomic charges of each nucleus
config = np.array([Ra,Rb])  # Positions Array

print("\nInput parameters:")
print(f"{R = }  {NG = }")


# Integral Matrices (Overlap and Hamiltonian)
S = matrix(N,NG,config,_S)
T = matrix(N,NG,config,_T)
V = matrix(N,NG,config,_V,Za,Zb,config)
H = T+V

print("\nOverlap Matrix (S):\n",S)
print("\nKinetic Matrix (T):\n",T)
print("\nPotential Matrix (V):\n",V)
print("\nHamiltonian Matrix (H):\n",H)


# Solving for W1, W2 with SymPy by solving Characterystic polynomial# 
M = sm.Matrix(H-W*S)
W1,W2 = sm.solve(M.det())
print("\nElectronic Energies (W):\n",W1,W2)

Vnn = 1/R
print("\nNuclear Repulsion (Vnn):\n",Vnn)

E1,E2 = W1+1/R, W2+1/R
print("\nTotal Energies (E):\n",E1,E2)



# Plots PES of ground and first excited state
n = 100                         # Number of study points
ao = 0.529177249                # 1ao (au) = 0.529177249 Armsotrongs
R = np.linspace(0.2,5.,n)/ao    # Values for R(A-B) distance

energies = np.zeros((2,n))
for i,Ri in enumerate(R):
    config = np.array([0,Ri])
    S = matrix(N,NG,config,_S)
    T = matrix(N,NG,config,_T)
    V = matrix(N,NG,config,_V,Za,Zb,config)
    H = T+V

    M = sm.Matrix(H-W*S)
    W1,W2 = sm.solve(M.det())   
    Vnn = 1/Ri
    E1,E2 = W1+1/Ri, W2+1/Ri
    energies[0][i] = E1
    energies[1][i] = E2

plt.plot(R,energies[0],c="r")
plt.plot(R,energies[1],c="b")


plt.title("PES for the two lowest states of H$_2^+$")
plt.xlabel("R (a.u)");plt.ylabel("E ($E_h$)")
plt.xticks(np.arange(0,round(R[-1]+1),1))
plt.xlim(R[0],R[-1]); plt.ylim(ymax=1)
plt.legend(["$E_1$","$E_2$"])


plt.savefig(f"p5{NG}G.jpg",dpi=600)

tf = time()
print(f"\nProcess finished in {tf-to:.2f}s.")
plt.show()