"""
Practice Session 3 -  Problem 1
Parametrized HF for H2 molecule.
Diego Ontiveros
"""

import numpy as np

def bie_index(ijkl):
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


Uij = 1/np.sqrt(2)

N = 2
m = 2
R = 1.4
z = 1.24

S = [1,0.6593]
t = [0.7600,0.2365]
v = [-1.2266,-0.5974,-0.6538]
bielectronic = [0.7746,0.5697,0.4441,0.2970]

S = np.array([S,S[::-1]])
T = np.array([t,t[::-1]])
V = np.array([[-1.2266,-0.5974],[-0.5974,-0.6538]])
H = T+V

U = np.array([[Uij,Uij],[Uij,-Uij]])

C = np.random.uniform(-1,1,size=N)



print("Overlap Matrix:\n",S)
print("Kinetic Matrix:\n",T)
print("e-N Potential Matrix:\n",V)
print("Hamiltonian Matrix:\n",H)

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
