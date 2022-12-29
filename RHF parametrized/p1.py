"""
Practice Session 3 -  Problem 1
Parametrized RHF for H2 molecule.
Diego Ontiveros
"""

import numpy as np

def bie_index(ijkl):
    """Takes string od the indices of the bielectronic integral and returns
    the index of the bielectronic parametrized array of integrals it corresponds."""
    i,j,k,l =[int(t) for t in ijkl]
    sij = i+j if i!=j else i
    skl = k+l if k!=l else k
    sijkl = [sij,skl]
    if len(set(ijkl))<=1: return 0
    elif sorted(sijkl) == [1,2]: return 1
    elif sorted(sijkl) == [1,3]: return 2
    elif sorted(sijkl) == [2,3]: return 2
    elif sorted(sijkl) == [3,3]: return 3
    else: raise ValueError("Bielectronic indices nor valid.")

def G_matrix(m,P):
    """Computed bielectornic matrix."""
    G = np.zeros(shape=(m,m))

    for mu in range(m):
        for nu in range(m):

            for l in range(m):
                for s in range(m):

                    munusl = f"{mu+1}{nu+1}{s+1}{l+1}"
                    mulsnu = f"{mu+1}{l+1}{s+1}{nu+1}"
                    i1 = bie_index(munusl)    
                    i2 = bie_index(mulsnu)
                    # print(munusl,mulsnu)
                    # print(i1,i2)

                    G[mu,nu] += P[l,s]*(bielectronic[i1] - 0.5*bielectronic[i2])

    return G

def P_matrix(m,C):
    P = np.zeros(shape=(m,m))
    for l in range(m):
        for s in range(m):
            for j in range(int(N/2)):
                P[l,s] += 2*C[l,j]*C[s,j]

    return P


def converged(P0,Pt,eps):
    """Determines if a step is converged with a certain tolerance (eps)."""
    m = len(Pt)
    diff = 0
    for mu in range(m):
        for nu in range(m):
            diff += (Pt[nu,mu]-P0[nu,mu])**2
    # Probar  de hacerlo con arrays!!!

    diff = np.sqrt(diff/m**2)

    if diff<=eps: return True
    else: return False



N = 2           # Number of electrons
m = 2           # Number of basis functions
R = 1.4         # H-H distance
z = 1.24        # exponent
eps = 1e-4      # SCF tolerance
np.random.seed(3333)

# Parametrized Hamiltonian and Overlapping matrices
S = [1,0.6593]
t = [0.7600,0.2365]
v = [-1.2266,-0.5974,-0.6538]

S = np.array([S,S[::-1]])
T = np.array([t,t[::-1]])
V = np.array([[-1.2266,-0.5974],[-0.5974,-0.6538]])
H = T+V

# print("Overlap Matrix:\n",S)
# print("Kinetic Matrix:\n",T)
# print("e-N Potential Matrix:\n",V)
# print("Hamiltonian Matrix:\n",H)

# Bielectronic integrals
bielectronic = [0.7746,0.5697,0.4441,0.2970]


# Unitary matrix to build X
Uij = 1/np.sqrt(2)
U = np.array([[Uij,Uij],[Uij,-Uij]])

Seval,U = np.linalg.eigh(S)
X = U@np.linalg.inv(np.diag(np.sqrt(Seval)))
print("U\n",X.T@S@X)


# Randomly initialized MO
C = np.random.uniform(0,1,size=(m,m))
P0 = np.zeros(shape=(m,m))
print("P0\n",P0)


while True:
    print()
    G = G_matrix(m,P0.copy())
    print("G\n",G)
    F = H + G

    Ft =  X.T.conj()@F@X
    print("Ft\n",Ft)

    e,Ct = np.linalg.eigh(Ft)
    print("e\n",e)
    C = X@Ct
    print("C\n",C)

    Pt = P_matrix(m,C)
    # Pt = 2*C@C.T
    print("P\n",Pt)
    print("P\n",P0)


    if converged(P0,Pt,eps):
        print("\nConverged!")
        print(C)
        print(e)
        break



    P0 = Pt.copy()










