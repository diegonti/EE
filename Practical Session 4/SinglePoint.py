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


# INPUT PARAMETERS
N = 2
RHHe = 1.4632
RHH = 1.4
ZH,ZHe = 1,2
zH,zHe = 1.24,2.0926

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
scf.SCF(print_options=["all"])