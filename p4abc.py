"""
Practice Session 2 -  Problem 4abc
H2+ PES made from 1s STOs. Optimized parameters.
Diego Ontiveros
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
to = time()
pi = np.pi

# Energy Integrals
def Sab(k,R): return np.exp(-k*R) * (1+k*R+(k**2*R**2)/3)
def Haa(k,R): return 0.5*k**2 - k - 1/R + np.exp(-2*k*R)*(k+1/R)
def Hab(k,R): return -0.5*k*k*Sab(k,R) + k*(k-2)*(1+k*R)*np.exp(-k*R)

def W1(k,R): return (Haa(k,R) + Hab(k,R))/(1+Sab(k,R))
def W2(k,R):return (Haa(k,R) - Hab(k,R))/(1-Sab(k,R))

################# MAIN PROGRAM ##############

n = 5000                        # Number of study points
ao = 0.529177249                # 1ao (au) = 0.529177249 Angstroms
k = np.linspace(0.4,2.,n)       # Values for k parameter
R = np.linspace(0.8,2.,n)/ao    # Values for R(A-B) distance

kk,RR = np.meshgrid(k,R)        # All possible k-R combinations
E1 = W1(kk,RR)+1/RR             # Feed to the energy integral  
minE = E1.min()                 # Minimum total enegry

# From the E1 matrix whe can find the indices of the minimum energy,
# which can be translated to the minimum R (rows) and mininum k (colums)           
minR_index,mink_index = np.unravel_index(np.argmin(E1),E1.shape)    # Idices of the minimum R and k
mink = k[mink_index]        # Optimized k
minR = R[minR_index]        # Optimized R

# Printing results
print()
print("Optimized k:           ",mink)
print("Optimized R (a.u):     ",minR)
print()
print("Minimum Energy (a.u):   ",minE)  
print("Electronic Energy (a.u):",minE-1/minR)
print("Nuclear Repulsion (a.u): ",1/minR)


# Plot Settings
fig = plt.figure(figsize=(8,3.5))
ax1 = fig.add_subplot(121,projection="3d")
ax2 = fig.add_subplot(122)

R = np.linspace(0.2,5.,n)/ao                    # Wider range of R to visualize PES
ax1.plot_surface(kk,RR,E1,cmap="gnuplot")       # E1(k,R) surface (every k,R pair)
ax2.plot(R,W1(mink,R)+1/R, c="r")               # E1(kopt,R) plot (PES1)
ax2.plot(R,W2(mink,R)+1/R, c="b")               # E2(kopt,R) plot (PES2)
ax2.annotate(
# Label and coordinate
f'R$_e$ = {minR:.3f}', xy=(minR, minE),xytext=(minR+3, minE-0.1) ,
horizontalalignment="center",
# Custom arrow
arrowprops=dict(arrowstyle='->',lw=1)
)

ax1.view_init(11,-43,0)
ax1.set_title("$E_1$ as a function of k and R")
ax2.set_title("PES for the two lowest states of H$_2^+$")
ax1.set_xlabel("k");ax1.set_ylabel("R (a.u)");ax1.set_zlabel("E ($E_h$)")
ax2.set_xlabel("R (a.u)");ax2.set_ylabel("E ($E_h$)")
ax2.set_xticks(np.arange(0,round(R[-1]+1),1))
ax2.set_xlim(R[0],R[-1]); ax2.set_ylim(ymax=1)
ax2.legend(["$E_1$","$E_2$"])

fig.tight_layout(w_pad=3)
fig.savefig("p4abc.jpg",dpi=600)

tf = time()
print(f"\nProcess finished in {tf-to:.2f}s.")
plt.show()