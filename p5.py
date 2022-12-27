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
    return ((2*a/pi)*(2*b/pi))**(3/4)

def Fo(t): 
    """Auxiliary function"""
    return 0.5*np.sqrt(pi/t)*erf(np.sqrt(t))

def g(r,a):
    "Normalized gaussian function in spherical coordinates"
    return (2*a/pi)**(3/4) * np.exp(-a*r**2)


# Individual gaussian integrals
def _S(a,b,R): return Norm(a,b)*(pi/(a+b))**(3/2) * np.exp(-a*b/(a+b)*R**2)

def _T(a,b,R): 
    fracc = a*b/(a+b)
    return fracc*(3-2*fracc*R**2)*_S(a,b,R)

def _Vp(a,b,R,Z,Rc):
    Rp = (a*0 + b*R)/(a+b)
    t = (a+b)*(Rp-Rc)**2

    if t<1e-6: F = 1-t/3
    else: F = Fo(t)

    return -2*pi/(a+b)*Z * np.exp(-a*b/(a+b)*R**2)*F * Norm(a,b)

def _V(a,b,R,Za,Zb):
    return _Vp(a,b,R,Za,0) + _Vp(a,b,R,Zb,R)

# Matrix elements
def matrix(N,NG,R,f,*params):
    
    NG,d,a = NG_parser(NG)
    a = np.array(a) * 1.24**2

    M = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):

                            
            for p in range(NG):
                for q in range(NG):
                    if i==j: Rij = 0
                    else: Rij = R

                    
                    M[i,j] += d[p]*d[q]*f(a[p],a[q],Rij, *params)
    return M

def W1b(H,S): return (H[0,0] + H[0,1])/(1+S[0,1])
def W2b(H,S): return (H[0,0] - H[0,1])/(1-S[0,1])

# ao = 0.529177249
# n = 100
N = 2
NG = 1
R = 2
Ra = 0
Rb = Ra+R

# R = np.linspace(0.2,5.,n)/ao 
Za,Zb = 1,1

S = matrix(N,NG,R,_S)
T = matrix(N,NG,R,_T)
V = matrix(N,NG,R,_V,Za,Zb)
H = T+V

print(S)
print(T)
print(V)

M = sm.Matrix(H-W*S)
W1,W2 = sm.solve(M.det())
print(W1)
print(W2)



print(W1b(H,S))
print(W2b(H,S))




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