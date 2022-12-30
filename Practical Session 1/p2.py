"""
Practice Session 1 -  Problem 2
Finding optimized paramaters and energy for STO-1G.
Diego Ontiveros
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from time import time
to = time()

pi = np.pi

def S(r,a,f):
    """Overlap integral"""
    dr = r[1]-r[0]
    return pi**(-0.5) * (2*a/pi)**(3/4) * simpson(f,dx=dr)

def f(r,a):
    """Function to integrate 4pi(r**2*STO-STO1G)"""
    return 4*pi*r**2*np.exp(-(r+a*r**2))    # Add jacobian 4pir**2!!!

# GTO variational integrals as a function of the exponent
def Sij(a1,a2): return (pi/(a1+a2))**(3/2)
def Tij(a1,a2): return 3*(a1*a2*pi**(3/2))/((a1+a2)**(5/2))
def Vij(a1,a2): return -2*pi/(a1+a2)
def Wij(a1,a2): return (Tij(a1,a2)+Vij(a1,a2))/Sij(a1,a2)

# Integral r parameters
ri,rf = 0,100   # Integral limits. We consider a large number as inf. 100 has proven to be enough
n = 10000       # Number of r points
h = (rf-ri)/n   # dr. Space between points

r = np.arange(ri,rf+h,h)
a_array = np.linspace(0.1,0.5,50000)

# Calculating integral for each value of a 
S_array = []
for a in a_array:
    integrand = f(r,a)
    s = S(r,a,integrand)
    S_array.append(s)

# Finding a that maximizes S integral
S_array = np.array(S_array)
Smax = np.max(S_array)
amax = a_array[np.argmax(S_array)]
energy = Wij(amax,amax)
real_energy = -0.5
err = 100*abs((energy-real_energy)/real_energy)

# Printing results
print()
print(f"Optimal exponent for STO-1G :  {amax:.8f}")
print(f"Energy with optimal exponent: {energy:.8f} a.u")
print(f"Relative error: {err:.2f} %")

# Plotting S vs alpha
plt.plot(a_array,S_array,c="red")
plt.annotate(
# Label and coordinate
f'$\\alpha$ = {amax:.5f}', xy=(amax, Smax),xytext=(amax+0, Smax-0.05*Smax) ,
horizontalalignment="center",
# Custom arrow
arrowprops=dict(arrowstyle='->',lw=1)
)

plt.xlabel("$\\alpha$");plt.ylabel("S")
plt.xlim(a_array[0],a_array[-1])
plt.savefig("p2.jpg",dpi=600)

tf = time()
print(f"\nProcess finished in {tf-to:.2f}s.")
plt.show()