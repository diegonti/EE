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
from time import time
import matplotlib.pyplot as plt
import numpy as np
to = time()

# INPUT PARAMETERS
N = 2                   # Number of electrons
RHHe = 1.4632           # H-He distance in HeH+
RHH = 1.4               # H-H distance in H2
ZH,ZHe = 1,2            # H and He nuclear charges
zH,zHe = 1.24,2.0926    # H and He effective charges
R_array = np.linspace(0.5,8,1000)

# GENERATING MOLECULES 
H2 = Molecule(
    geometry=[0,RHH],
    charges=[ZH,ZH],
    effective_charges=[zH,zH],
    N_electrons = N,
    molecule_label="H2"
    )

HeH = Molecule(
    geometry=[0, RHHe],
    charges=[ZHe,ZH],
    effective_charges=[zHe,zH],
    N_electrons = N,
    molecule_label="HeH+"
)

# SCF PES PROCEIDURE for H2
scf = RHF(H2,NG=3)
PES_H2, r, Emin_H2, Re_H2 = scf.PES(R_array,file_name="PES_H2.log")


# SCF PES PROCEIDURE for HeH+
scf = RHF(HeH,NG=3)
PES_HeH, r, Emin_HeH,Re_HeH = scf.PES(R_array,file_name="PES_HeH.log")


print("H2 minimum Energy: ", Emin_H2)
print("H2 optimized distance: ", Re_H2)
print("HeH+ minimum Energy: ", Emin_HeH)
print("HeH+ optimized distance: ", Re_HeH)

# Plotting PES
fig,(ax1,ax2) = plt.subplots(1,2, figsize=(9,5))

ax1.plot(r,PES_H2,c="r")
ax2.plot(r,PES_HeH,c="r")

ax1.set_xlim(R_array[0],R_array[-1]);ax2.set_xlim(R_array[0],R_array[-1])
ax1.set_xlabel("R (a.u.)"); ax2.set_xlabel("R (a.u.)")
ax1.set_ylabel("E (a.u.)"); ax2.set_ylabel("E (a.u.)")

fig.tight_layout()
fig.savefig("PES.jpg",dpi=600)

tf = time()
print(f"\nProcess finished in {tf-to:.4f}s\n")
plt.show()
