"""
Practice Session 3 -  Problem 2
Parametrized RHF for HHe+ molecule.
Diego Ontiveros
"""
import numpy as np
import matplotlib.pyplot as plt

S = [0.4508]
t = [0.21643,2.1643,0.7600]
v = [[-4.1398,-1.1029,-1.2652],
    [-0.6772,-0.4113,-1.226]]
jk = [1.3072,0.7746,0.6057,0.4373,0.3118,0.1773]



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
                    ######### Mirar bien las integrales! Ojo, usted!
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
    diff = np.sqrt(np.sum((Pt-P0)**2)/m**2)
    
    if diff<=eps: return True
    else: return False



N = 2               # Number of electrons
m = 2               # Number of basis functions
R = 1.4632          # H-H distance

Za,Zb = 2,1         # Nuclear atomic charges
za,zb = 2.0925,1.24 # Exponents
eps = 1e-4          # SCF tolerance
max_iter = 10       # Maximum number of iterations
np.random.seed(3333)

# Parametrized Hamiltonian and Overlapping matrices
S = [1,0.6593]
t = [0.7600,0.2365]
v = [-1.2266,-0.5974,-0.6538]

S = np.array([S,S[::-1]])
T = np.array([t,t[::-1]])
V = np.array([[-1.2266,-0.5974],[-0.5974,-0.6538]])
H = T+V

print("\nOverlap Matrix:\n",S)
print("\nKinetic Matrix:\n",T)
print("\ne-N Potential Matrix:\n",V)
print("\nHamiltonian Matrix:\n",H)


# Bielectronic integrals
bielectronic = [0.7746,0.5697,0.4441,0.2970]


# Unitary matrix to build X
Uij = 1/np.sqrt(2)
U = np.array([[Uij,Uij],[Uij,-Uij]])

Seval,U = np.linalg.eigh(S)
S12 = np.diag(Seval**-0.5)
print("\nS12\n",S12)
print("\nU\n",U)

# S12 = np.linalg.inv(np.diag(np.sqrt(Seval)))
# print("\nS12\n",S12)

X = U@S12@U.T       #U@S12@U.T.conj()         ###### Probar U_dada y U_evec

# Posible 
# S12 = S[0,1]
# X[0,0] = 1.0/np.sqrt(2.0*(1.0+S12))
# X[1,0] = X[0,0]
# X[0,1] = 1.0/np.sqrt(2.0*(1.0-S12))
# X[1,1] = -X[0,1]

print("\nX\n",X)
print("\nXSX\n",X.T@S@X)





# Randomly initialized MO
C = np.random.uniform(0,1,size=(m,m))
P0 = np.zeros(shape=(m,m))
print("\nP0\n",P0)

print("\n\nEntering SCF loop...")
n_iterations = 0
while True:
    n_iterations +=1
    print(f"\nIternation {n_iterations}")

    # Bielectronic Matrix
    G = G_matrix(m,P0.copy())
    print("G\n",G)

    # Fock Matrix
    F = H + G
    print("F\n",F)

    

    # Tranformed Fock Matrix
    Ft =  X.T.conj()@F@X
    print("Ft\n",Ft)


    # Orbital energies and coefs
    e,Ct = np.linalg.eigh(Ft)
    print("e\n",e)
    C = X@Ct
    print("C\n",C)

    # Denisty Matrix
    Pt = P_matrix(m,C)
    # Pt = 2*C@C.T
    print("P\n",Pt)
    print("P0\n",P0)

    Eelec = 0.5*np.sum(Pt*(H+F))

    print("\nElcetronic Energy: \n", Eelec)

    # Convergence
    if converged(P0,Pt,eps):
        print("\nConverged!")
        print(C)
        print(e)

        # Functon to print info!
        Vnn = 1/R
        Eelec = Eelec
        E = Eelec + Za*Zb/R
        Mulliken = Pt@S
        print("\nElectronic Energy: Eelec =",Eelec)
        print("Nuclear Repulsion:   Vnn = ",Vnn)
        print("Total energy:          E = ",Eelec+Vnn)

        break

    if n_iterations >= max_iter:
        print("\n Not converged!")
        break

    
    P0 = Pt.copy()







