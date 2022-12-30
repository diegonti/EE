"""
Practice Session 1 -  Problem 3
Ploting X1s (H and He) functions expressed with 
STO, STO-1G, STO-2G and STO-3G 
Diego Ontiveros
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from time import time
to = time()
pi = np.pi

# STO-NG coefitients d_i and alpha_i
coef = {
    "1g": [[1],[0.270950]],
    "2g": [[0.678914,0.430129],[0.151623,0.851819]],
    "3g": [[0.444635,0.535328,0.154329],[0.109818,0.405771,2.22766]]
}

# Orbital functions
def GTO(a,z,r):
    """GTO wave function for a given a (alpha)"""
    a = a*z**2      # Rescaling alpha for a given z 
    return (2*a/pi)**(3/4) * np.exp(-a*r**2)


def STO(z,r):
    """STO wave function for a given z"""
    return np.sqrt(z**3/pi) * np.exp(-z*r)


def STONG(r,z,ng):
    """STO-NG function for a given n"""
    if type(ng) == str:
        d,a = coef[ng.lower()]
    elif type(ng) == int:
        d,a = coef[f"{ng}g"]
    else: print("Insert a valid number of gaussians. Eg, ""2g"" ")

    sto = 0
    for ai,di in zip(a,d):
        sto += di*GTO(ai,z,r)

    return sto


def RD(r,f):
    """Radial Distribution of a funciton f."""
    return 4*pi*r**2*abs(f)**2      # 4pir**2 == jacobian


def meanR(r,f):
    """Mean radius for a wavefunction f. <r>=<f|rf>."""
    return simpson(r*f,r)


def orbitalR(r,rd,init):
    """Finds R such that P(r)=0.99 (orbital radius).
    Starts from init distance (mean r value)."""
    dr = r[1]-r[0]
    r_mean_index = np.where(np.isclose(r,init,atol=dr)==True)[0][0]
    for ri in range(r_mean_index,len(r)):
        slice = rd[0:ri]
        P = simpson(slice,dx=dr)
        if P>=0.99:
            return r[ri]

##########################  MAIN PROGRAM ########################

# Initial parameters
z = 2                       # Atomic number (z(H)=1 z(He)=2) 
n = 50000                   # number of r points (bigger=more accuracy but more time)
r = np.linspace(0,10,n)     # Positions array
print("\nNuclear Charge: Z = ",z,"\n")

## Plots and Plot Settings
fig,(ax1,ax2) = plt.subplots(1,2, figsize=(8,4))
colors = ["royalblue","limegreen","orange","red"]

# Plotting STO function
print("STO:")
X1s = STO(z,r)                  # STO function
rd = RD(r,X1s)                  # Radial distribution (RD)
r_maxP = r[np.argmax(rd)]       # R that maximizes RD
r_mean = meanR(r,rd)            # Average e-N distance <r>
r_orbit = orbitalR(r,rd,r_mean) # Radius of orbital (r at P=0.99)

print(f"Radius with maximum probability: {r_maxP}")      
print(f"Mean radius value <r>:           {r_mean}") 
print(f"Orbital radius (r at P=0.99):    {r_orbit}")

ax1.plot(r,X1s, c=colors[-1],label=f"STO $(\chi_{{1s}})$")                
ax1.plot(r,X1s**2,"--", c=colors[-1],label=f"STO $(\chi_{{1s}}^{2})$")
ax2.plot(r,rd, c=colors[-1],label=f"STO $(\chi_{{1s}})$")


# Plotting STO-NG functions
for i in range(1,3+1):
    print(f"\nSTO-{i}G:")

    X1s = STONG(r,z,ng=i)               # STO-NG function
    rd = RD(r,X1s)                      # Radial Distribution (RD)
    r_maxP = r[np.argmax(rd)]           # Raidus at maximum Probability denisty
    r_mean = meanR(r,rd)                # Average e-N distance <r> 
    r_orbit = orbitalR(r,rd,r_mean)     # Orbital Radius (such that P=0.99)

    print(f"Radius with maximum probability: {r_maxP}")      
    print(f"Mean radius value <r>:           {r_mean}")   
    print(f"Orbital radius (r at P=0.99):    {r_orbit}")


    ax1.plot(r,X1s, c=colors[i-1],alpha=0.75,label=f"STO-{i}G $(\chi_{{1s}})$")             # Wave function
    ax1.plot(r,X1s**2, "--",c=colors[i-1],alpha=0.75,label=f"STO-{i}G $(\chi_{{1s}}^{2})$") # Square Wave function
    ax2.plot(r,rd, c=colors[i-1],alpha=0.75,label=f"STO-{i}G $(\chi_{{1s}})$")       # Radial distribution


# Plot parameters
ax1.set_xlabel("r ($a_0$)");ax2.set_xlabel("r ($a_0$)")
ax1.set_ylabel("$\chi$");ax2.set_ylabel("$4πr^2|\chi|^2$")
ax1.set_xlim(0,6/z);ax2.set_xlim(0,6/z)
ax1.set_ylim(0);ax2.set_ylim(0)

ax1.legend(fontsize="small",frameon=False)
ax2.legend(fontsize="small",frameon=False)
    

tf = time()
print(f"\nProcess finished in {tf-to:.2f}s.")
fig.tight_layout()
fig.savefig(f"p3z{z}.jpg",dpi=600)
plt.show()

# The higher the contraction GTOs, the better fit to the STO.
# He+ has 2 protons+ at the nucleus which imply a higher attraction force
# on the electron, thus the lower mean r values in He+ with respect to H·.