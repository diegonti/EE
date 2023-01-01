"""
Practice Session 4 - Problem 1
General RHF SCF program for two-electron diatomic molecules.
Performes a single point calculation for a fixed geometry.
Here is used for the examples of H2 and HeH+.

Diego Ontiveros
"""


# Imported from the module RHF created, 
# which contains all the functions and classes needed.
from RHF import Molecule, RHF
from time import time
import os
to = time()


# INPUT PARAMETERS
N = 2                   # Number of electrons
RHHe = 1.4632           # H-He distance in HeH+
RHH = 1.4               # H-H distance in H2
ZH,ZHe = 1,2            # H and He nuclear charges
zH,zHe = 1.24,2.0926    # H and He effective charges
file_name = "SP.log"    # Output file

# To overwrite file
try: os.remove("SP.log")
except FileNotFoundError: pass

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

# SCF PROCEIDURE. Change HeH to H2 to calculate the H2 molecule.
scf = RHF(HeH,NG=3)
scf.SCF(print_options=["all"],file_name=file_name)

tf = time()
print(f"\nProcess finished in {tf-to:.4f}s\n")