"""
Practice Session 1 -  Problem 1
Finding optimized paramaters and energy for GTO.
Diego Ontiveros
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
to = time()

pi = np.pi

# GTO integrals as a function of the exponent
def Sij(a1,a2): return (pi/(a1+a2))**(3/2)
def Tij(a1,a2): return 3*(a1*a2*pi**(3/2))/((a1+a2)**(5/2))
def Vij(a1,a2): return -2*pi/(a1+a2)
def Wij(a1,a2): return (Tij(a1,a2)+Vij(a1,a2))/Sij(a1,a2)

# Range of alphas exponents
a = np.linspace(0.01,5,10000)

# Computed W array for all alphas
W = Wij(a,a)

# Optimized parameters
minW = np.min(W)
minWi = np.argmin(W)
mina = a[minWi]

# Printing Results
real_energy = -0.5
err = 100*abs((minW-real_energy)/real_energy)
print(f"Optimal exponent for GTO 1s orbital:  {mina:.8f}")
print(f"Energy with optimal exponent:        {minW:.8f} a.u")
print(f"Relative error: {err:.2f} %")

# Ploting W(alpha) vas alpha
plt.plot(a,W,c="r")
plt.annotate(
# Label and coordinate
f'$\\alpha$ = {mina:.4f}', xy=(mina, minW),xytext=(mina+0.5, minW+2) ,
horizontalalignment="center",
# Custom arrow
arrowprops=dict(arrowstyle='->',lw=1)
)
plt.ylabel("$W (E_h)$");plt.xlabel("$\\alpha$")
plt.xlim(a[0],a[-1])

plt.savefig("p1.jpg",dpi=600)

tf = time()
print(f"\nProcess finished in {tf-to:.2f}s.")
plt.show()