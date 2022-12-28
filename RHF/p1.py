"""
Practice Session 3 -  Problem 1
Parametrized HF for H2 molecule.
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

                    G[mu,nu] += P[mu,nu]*(bielectronic[i1] - 0.5*bielectronic[i2])

    return G

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
eps = 1e-4

# Parametrized Hamiltonian and Overlapping matrices
S = [1,0.6593]
t = [0.7600,0.2365]
v = [-1.2266,-0.5974,-0.6538]

S = np.array([S,S[::-1]])
T = np.array([t,t[::-1]])
V = np.array([[-1.2266,-0.5974],[-0.5974,-0.6538]])
H = T+V

print("Overlap Matrix:\n",S)
print("Kinetic Matrix:\n",T)
print("e-N Potential Matrix:\n",V)
print("Hamiltonian Matrix:\n",H)

# Bielectronic integrals
bielectronic = [0.7746,0.5697,0.4441,0.2970]


# Unitary matrix to build X
Uij = 1/np.sqrt(2)
U = np.array([[Uij,Uij],[Uij,-Uij]])

# Randomly initialized MO
C = np.random.uniform(-1,1,size=N)

while True:


    if converged(P0,Pt,eps):
        print("Converged!")




Seval,Sevec = np.linalg.eig(S)
X = U@np.linalg.inv(np.diag(np.sqrt(Seval)))
print(X)

P = 2*np.outer(C,C)

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

                G[mu,nu] += P[mu,nu]*(bielectronic[i1] - 0.5*bielectronic[i2])

print(G)

F = H + G

Ft = X.T.conj()@F@X
print("Transformed Fock Matrix:\n",Ft)

e,Ct = np.linalg.eig(F)
C = X@Ct
