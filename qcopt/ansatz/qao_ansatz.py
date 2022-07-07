"""
07/16/2021 - Teague Tomesh

The functions in the file are used to generate the Quantum Alternatiing
Operator Ansatz (QAO-Ansatz) for solving constrained combinatorial optimization
problems.

Note that the QAO-Ansatz differs from the typical Quantum Approximate Optimization
Algorithm in the structure of the quantum circuits it uses. In this file, all
mentions of "qaoa" actually refer to the QAO-Ansatz.
"""
from typing import List, Optional, Union

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate
from qiskit.circuit import EquivalenceLibrary
from qiskit.circuit.library.standard_gates import RXGate
from qiskit.transpiler.passes import Unroller
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler import PassManager
from qiskit.converters import circuit_to_dag, dag_to_circuit

from qcopt.ansatz.gate_decomp import CustomEquivalenceLibrary

import qcopt


def apply_mixer(
    circ: QuantumCircuit,
    G: nx.Graph,
    beta: Union[float, List],
    barriers: bool,
    decompose_toffoli: int,
    mixer_order: List[int],
    verbose: int = 0,
):

    # apply partial mixers U_M^i according to the order given in mixer_order
    for i, qubit in enumerate(mixer_order):
        if isinstance(beta, List):
            cur_angle = beta[i]
        else:
            cur_angle = beta

        neighbors = list(G.neighbors(qubit))

        if verbose > 0:
            print("qubit:", qubit, "neighbors:", neighbors)

        # Construct a multi-controlled Toffoli gate, with open-controls on q's neighbors
        # Qiskit has bugs when attempting to simulate custom controlled gates.
        # Instead, wrap a regular toffoli with X-gates
        ctrl_qubits = [circ.qubits[j] for j in neighbors]
        if decompose_toffoli > 0:
            # Implement the multi-controlled Rx rotation using the decomposition given by
            # Lemma 5.1 in Barenco et. al. (https://arxiv.org/pdf/quant-ph/9503016.pdf)
            for ctrl in ctrl_qubits:
                circ.x(ctrl)
            circ.rz(np.pi / 2, circ.qubits[qubit])
            circ.ry(cur_angle, circ.qubits[qubit])
            circ.mcx(ctrl_qubits, circ.qubits[qubit])
            circ.ry(-1 * cur_angle, circ.qubits[qubit])
            circ.mcx(ctrl_qubits, circ.qubits[qubit])
            circ.rz(-1 * np.pi / 2, circ.qubits[qubit])
            for ctrl in ctrl_qubits:
                circ.x(ctrl)
        else:
            mcrx = ControlledGate(
                "mcrx",
                len(neighbors) + 1,
                [2 * cur_angle],
                num_ctrl_qubits=len(neighbors),
                ctrl_state="0" * len(neighbors),
                base_gate=RXGate(2 * cur_angle),
            )
            circ.append(mcrx, ctrl_qubits + [circ.qubits[qubit]])

        if barriers > 1:
            circ.barrier()


def apply_phase_separator(circ: QuantumCircuit, gamma: float, G: nx.Graph):
    for qb in G.nodes:
        circ.rz(2 * gamma, qb)


def split_and_cyclic_shift(circuit, n, k, decompose=0):
    # (i)
    circuit.cx(n-2, n-1)
    circuit.cry(2 * np.arccos(np.sqrt(1/n)), n-1, n-2)
    circuit.cx(n-2, n-1)

    # (ii)_l
    for l_index in range(2, k+1):
        circuit.cx(n-l_index-1, n-1)
        if decompose > 0:
            circuit.ry(np.arccos(np.sqrt(l_index / n)), n-l_index-1)
            circuit.ccx(n-1, n-l_index, n-l_index-1)
            circuit.ry(-1*np.arccos(np.sqrt(l_index / n)), n-l_index-1)
            circuit.ccx(n-1, n-l_index, n-l_index-1)
        else:
            ccry = qiskit.circuit.ControlledGate(
                    "ccry",
                    3,
                    [2 * np.arccos(np.sqrt(l_index / n))],
                    num_ctrl_qubits=2,
                    ctrl_state="11",
                    base_gate=qiskit.circuit.library.standard_gates.RYGate(2 * np.arccos(np.sqrt(l_index / n))),
                )
            circuit.append(ccry, [n-1, n-l_index, n-l_index-1])
        circuit.cx(n-l_index-1, n-1)


def dicke_state(circuit, l, decompose=0):
    """Efficient state preparation of a Dicke State.

    Based on the efficient circuit implementation given in the
    paper https://arxiv.org/pdf/1904.07358.pdf.

    Arguments
    ---------
    circuit: QuantumCircuit
    l: int - Hamming weight of the Dicke State
    decompose: int - Optional. Level of decomposition for the multi-controlled
                     Ry gates.
    """
    n = circuit.num_qubits
    for i in range(1, l+1):
        circuit.x(n-i)
    m = n
    k = l
    while m >= k + 2:
        split_and_cyclic_shift(circuit, m, k, decompose=decompose)
        m = m - 1
    while k >= 1:
        split_and_cyclic_shift(circuit, m, k, decompose=decompose)
        m = m - 1
        k = k - 1


def gen_qaoa(
    G: nx.Graph,
    P: int,
    mixer_order: Optional[List[int]] = None,
    params: List = [],
    init_state: str = None,
    individual_partial_mixers: bool = False,
    barriers: int = 1,
    decompose_toffoli: int = 1,
    verbose: int = 0,
):

    nq = len(G.nodes)
    qaoa_circ = QuantumCircuit(nq, name="q")

    if not mixer_order:
        mixer_order = list(sorted(list(G.nodes)))

    # Step 1: Jump Start
    if init_state is None:
        # for now, select the all zero state
        init_state = "0" * nq
    elif init_state == "W":
        # Prepare the |W> initial state (https://arxiv.org/pdf/1904.07358.pdf)
        # dicke_state() adds the gates in-place
        dicke_state(qaoa_circ, 1, decompose=decompose_toffoli)
        qaoa_circ.barrier()
    else:
        for qb, bit in enumerate(reversed(init_state)):
            if bit == "1":
                qaoa_circ.x(qb)

    if barriers > 0:
        qaoa_circ.barrier()

    # Step 2: Alternate applications of the mixer and driver unitaries
    if individual_partial_mixers:
        assert len(params) == P * (len(G.nodes) + 1), "Incorrect number of parameters!"
        betas, gammas = [], []
        for i in range(P):
            chunk = params[i * (nq + 1) : (i + 1) * (nq + 1)]
            betas.append(list(chunk[:-1]))
            gammas.append(chunk[-1])
    else:
        assert len(params) == 2 * P, "Incorrect number of parameters!"
        betas = [a for i, a in enumerate(params) if i % 2 == 0]
        gammas = [a for i, a in enumerate(params) if i % 2 == 1]

    if verbose > 0:
        print("betas:", betas)
        print("gammas:", gammas)

    for beta, gamma in zip(betas, gammas):
        apply_mixer(qaoa_circ, G, beta, barriers, decompose_toffoli, mixer_order, verbose=verbose)
        if barriers > 0:
            qaoa_circ.barrier()

        apply_phase_separator(qaoa_circ, gamma, G)
        if barriers > 0:
            qaoa_circ.barrier()

    if decompose_toffoli > 1:
        basis_gates_u = ["u1", "u2", "u3", "cx", "u"]
        basis_gates = ["u1", "u2", "u3", "cx"]
        pass_ = Unroller(basis_gates_u)
        pm = PassManager(pass_)
        qaoa_circ = pm.run(qaoa_circ)

        bt_pass = BasisTranslator(CustomEquivalenceLibrary, basis_gates)
        dag_out = bt_pass.run(circuit_to_dag(qaoa_circ))
        qaoa_circ = dag_to_circuit(dag_out)

    return qaoa_circ
