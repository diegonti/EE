"""
Practice Session 2 -  Problem 4d
H2+ PES made from 1s STOs. WF visualization.
Diego Ontiveros
"""
from scipy.integrate import simpson
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

# Wave-functions (atomic and for each state)
def s1a(k,ra): return k**(3/2)*pi**(-1/2)*np.exp(-k*ra)
def s1b(k,rb): return k**(3/2)*pi**(-1/2)*np.exp(-k*rb)

def f1(ra,rb,k,R): return (s1a(k,ra)+s1b(k,rb))/(np.sqrt(2*(1+Sab(k,R))))
def f2(ra,rb,k,R): return (s1a(k,ra)-s1b(k,rb))/(np.sqrt(2*(1-Sab(k,R))))


################# MAIN PROGRAM ##############

# Calculating optimized parameters (same as in problem 4.abc)
n = 5000                        # Number of study points
ao = 0.529177249                # 1ao (au) = 0.529177249 Armsotrongs
k = np.linspace(0.4,2.,n)       # Values for k parameter
R = np.linspace(0.8,2.,n)/ao    # Values for R(A-B) distance

kk,RR = np.meshgrid(k,R)        # All possible k-R combinations
E1 = W1(kk,RR)+1/RR              # Feed to the energy integral  
minE = E1.min()                  # Minimum total enegry

# From the w1 matrix whe can find the indices of the minimum energy,
# which can be translated to the minimum R (rows) and mininum k (colums)           
minR_index,mink_index = np.unravel_index(np.argmin(E1),E1.shape)    # Idices of the minimum R and k
mink = k[mink_index]        # Optimized k
minR = R[minR_index]        # Optimized R

print()
print("Optimized k:           ",mink)
print("Optimized R (a.u):     ",minR)
print()
print("Minimum Energy (a.u):   ",minE)  
print("Electronic Energy (a.u):",minE-1/minR)
print("Nuclear Repulsion (a.u): ",1/minR)

# Using optimized parameters to represent WaveFunctions

z = np.linspace(-minR,2*minR,1000)
ra = abs(z)
rb = abs(z-minR)
phi1 = f1(ra,rb,mink,minR)
phi2 = f2(ra,rb,mink,minR)
stoA = s1a(mink,ra)
stoB = s1a(mink,rb)


# Plot Settings
width,height = 7,7
fig,ax = plt.subplots(2,2,figsize=(width,height))

plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["mathtext.fontset"] = "cm"

# Plotting WF and |WF|^2
wavefunctions = [phi1,phi2]
for i,wf in enumerate(wavefunctions):

    ax[i,0].plot(z,stoA,"r",alpha=0.5,label= "$1s_a$")
    ax[i,0].plot(z,(-1)**i*stoB,"b",alpha=0.5,label= f"${(-1)**i}s_b$")

    ax[i,1].plot(z,stoA**2,"r",alpha=0.5, label= "$|1s_a|^2$")
    ax[i,1].plot(z,stoB**2,"b",alpha=0.5,label= f"$|{(-1)**i}s_b|^2$")

    ax[i,0].plot(z,wf,c="k", label=f"$\\varphi_{i+1}$")
    ax[i,1].plot(z,wf**2,c="k",label=f"$|\\varphi_{i+1}|^2$")

    ax[i,0].set_ylabel("$\\varphi$")
    ax[i,1].set_ylabel("$|\\varphi|^2$")

# Drawing atoms on the plots (more visual)
for axi in ax.flatten():
    axi.axhline(0,c="k",lw=0.5)
    axi.set_xlim(z[0],z[-1])
    axi.set_xlabel("z (a.u)")
    axi.legend(fontsize="small")
    axi.scatter(0,0,100,color="r")
    axi.scatter(minR,0,100,color="b")

# Figure titles
y0 = ax[0,0].get_window_extent().y0
inv = fig.transFigure.inverted()
h0 = inv.transform((y0,1))[0]

fig.text(0.5,h0-0.06,"Wavefunctions for anti-bonding state",
va="center",ha="center",size = 14)

fig.suptitle("Wavefunctions for bonding state",size=14)
fig.tight_layout(h_pad=4)
fig.savefig("p4d.jpg",dpi=600)

tf = time()
print(f"\nProcess finished in {tf-to:.2f}s.")
plt.show()