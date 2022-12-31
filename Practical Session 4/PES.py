"""
Practice Session 4 - Problem 1
General RHF SCF program for two-electron diatomic molecules.
Generates a plot of the PES.
Here is used for the examples of H2 and HeH+.

Diego Ontiveros
"""

# Imported from the module RHF created, 
# which contains all the functions and classes needed.
from RHF import Molecule, RHF
import matplotlib.pyplot as plt
import numpy as np

# INPUT PARAMETERS
N = 2
RHHe = 1.4632
RHH = 1.4
ZH,ZHe = 1,2
zH,zHe = 1.24,2.0926
R_array = np.linspace(1,6,100)

# GENERATING MOLECULES 
H2 = Molecule(
    geometry=[0,RHH],
    charges=[ZH,ZH],
    effective_charges=[zH,zH],
    N_electrons = N
    )

HeH = Molecule(
    geometry=[0, RHHe],
    charges=[ZHe,ZH],
    effective_charges=[zHe,zH],
    N_electrons = N
)

# SCF PES PROCEIDURE for H2
scf = RHF(H2,NG=3)
PES_H2, r = scf.PES()

# SCF PES PROCEIDURE for HeH
scf = RHF(HeH,NG=3)
PES_HeH, r = scf.PES()

# Plotting PES
fig,(ax1,ax2) = plt.subplots(1,2, figsize=(9,5))

ax1.plot(R_array,PES_H2,c="r")
ax2.plot(R_array,PES_HeH,c="r")

ax1.set_xlim(R_array[0],R_array[-1]);ax2.set_xlim(R_array[0],R_array[-1])
ax1.set_xlabel("R (a.u.)"); ax2.set_xlabel("R (a.u.)")
ax1.set_ylabel("E (a.u.)"); ax2.set_ylabel("E (a.u.)")

fig.tight_layout()

fig.savefig("PES.jpg",dpi=600)
plt.show()
