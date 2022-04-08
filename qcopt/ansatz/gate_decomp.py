"""
03/29/2022 - Nicholas Allen

Auxiliary file for extending the StandardEquivalenceLibrary to aid when decompose_toffoli = 2.
"""
import warnings
from qiskit.qasm import pi
from qiskit.circuit import EquivalenceLibrary, Parameter, QuantumCircuit, QuantumRegister

from qiskit.circuit.library.standard_gates import (
    UGate,
    U3Gate,
)
from qiskit.circuit.equivalence_library import StandardEquivalenceLibrary

_cel = CustomEquivalenceLibrary = StandardEquivalenceLibrary

# U gate

q = QuantumRegister(1, 'q')
theta = Parameter('theta')
phi = Parameter('phi')
lam = Parameter('lam')
u_to_u3 = QuantumCircuit(q)
u_to_u3.append(U3Gate(theta, phi, lam), [0])
_cel.add_equivalence(UGate(theta, phi, lam), u_to_u3)
