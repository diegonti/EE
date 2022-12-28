"""
Practice Session 2 -  Problem 5
H2+ PES made from 1s STO-NG. Energies.
Diego Ontiveros
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import sympy as sm
from time import time
to = time()
pi = np.pi

W = sm.symbols("W")

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
def _S(a,b,Ra,Rb): return Norm(a,b)*(pi/(a+b))**(3/2) * np.exp(-a*b/(a+b)*(Ra-Rb)**2)

def _T(a,b,Ra,Rb): 
    fracc = a*b/(a+b)
    return fracc*(3-2*fracc*(Ra-Rb)**2)*_S(a,b,Ra,Rb)

def _Vp(a,b,Ra,Rb,Z,Rc):
    Rp = (a*Ra + b*Rb)/(a+b)
    t = (a+b)*(Rp-Rc)**2

    return -2*pi/(a+b)*Z * np.exp(-a*b/(a+b)*(Ra-Rb)**2)*Fo(t) * Norm(a,b)

def _V(a,b,Ra,Rb,Za,Zb,R):
    return _Vp(a,b,Ra,Rb,Za,R[0]) + _Vp(a,b,Ra,Rb,Zb,R[1])

# Matrix elements
def matrix(N,NG,R,f,*params):
    
    NG,d,a = NG_parser(NG)
    a = np.array(a) * 1.24**2

    M = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):
            # if i==j: Rij = 0
            # else: Rij = R
            Ra = R[i]
            Rb = R[j]
                            
            for p in range(NG):
                for q in range(NG):

                    M[i,j] += d[p]*d[q]*f(a[p],a[q],Ra,Rb, *params)
    return M

def W1b(H,S): return (H[0,0] + H[0,1])/(1+S[0,1])
def W2b(H,S): return (H[0,0] - H[0,1])/(1-S[0,1])


# ao = 0.529177249
# n = 100
N = 2
NG = 3
R = 2.0035
Ra = 0
Rb = Ra+R
config = np.array([Ra,Rb])

Za,Zb = 1,1

S = matrix(N,NG,config,_S)
T = matrix(N,NG,config,_T)
V = matrix(N,NG,config,_V,Za,Zb,config)
H = T+V

print("Overlap Matrix:\n",S)
print("Kinetic Matrix:\n",T)
print("Potential Matrix:\n",V)

M = sm.Matrix(H-W*S)
W1,W2 = sm.solve(M.det())
print("Electronic Energies:\n",W1,W2)

Vnn = 1/R
print("Nuclear Repulsion:\n",Vnn)

E1,E2 = W1+1/R, W2+1/R
print("Total Energies:\n",E1,E2)




# n = 5000                        # Number of study points
# ao = 0.529177249                # 1ao (au) = 0.529177249 Armsotrongs
# k = np.linspace(0.4,2.,n)       # Values for k parameter
# R = np.linspace(0.8,2.,n)/ao    # Values for R(A-B) distance

# kk,RR = np.meshgrid(k,R)        # All possible k-R combinations
# E1 = W1(kk,RR)+1/RR              # Feed to the energy integral  
# minE = E1.min()                  # Minimum total enegry


# print()
# print("Optimized k:           ",mink)
# print("Optimized R (a.u):     ",minR)
# print()
# print("Minimum Energy (a.u):   ",minE)  
# print("Electronic Energy (a.u):",minE-1/minR)
# print("Nuclear Repulsion (a.u): ",1/minR)





tf = time()
print(f"\nProcess finished in {tf-to:.2f}s.")
plt.show()