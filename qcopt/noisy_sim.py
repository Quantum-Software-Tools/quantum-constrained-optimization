import numpy as np
import qiskit
import qcopt

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer import AerSimulator


def get_backend(p_Xerr=0.001, p_Zerr=0.001, p_Yerr=0.003, max_parallel_threads=0):
    # QuantumError objects
    error_single_qubit = pauli_error([('X',p_Xerr), ('Z',p_Zerr), ('Y',p_Yerr), ('I', 1 - (p_Xerr + p_Zerr + p_Yerr))])
    error_two_qubit = error_single_qubit.tensor(error_single_qubit) # A chance of single-qubit error on each participating qubit

    # Add errors to noise model
    noise_pauli = NoiseModel()
    noise_pauli.add_all_qubit_quantum_error(error_single_qubit, ["u1", "u2", "u3", "rz", "sx"])
    noise_pauli.add_all_qubit_quantum_error(error_two_qubit, ["cx"])

    return AerSimulator(noise_model=noise_pauli, method='density_matrix', max_parallel_threads=max_parallel_threads)

def execute_and_prune(circuit, graph, backend, shots=8192):
    # Execute
    transpiled_circuit = qiskit.transpile(circuit, backend)
    transpiled_circuit.save_state()
    result = backend.run(transpiled_circuit).result()
    noisy_probs = result.data()["density_matrix"].probabilities_dict(decimals=7)

    # Prune
    pruned_probs = {}
    for noisy_bitstring, probability in noisy_probs.items():
        while not qcopt.graph_funcs.is_indset(noisy_bitstring, graph):
            reversed_bitstr = noisy_bitstring[::-1]
            # Find one invalid edge, randomly remove one of the nodes from the IS
            hot_nodes = [idx for idx, bit in enumerate(reversed_bitstr) if bit == "1"]
            invalid_edges = []
            for hot_node in hot_nodes:
                for neighbor in list(graph[hot_node]):
                    if neighbor in hot_nodes:
                        invalid_edges.append((hot_node, neighbor))

            # Randomly select an invalid edge and remove the first index
            # Since each invalid edge appears twice in the list, with both
            # orderings, this is equivalent to tossing a 50/50 coin.
            selected_edge = invalid_edges[np.random.randint(0, len(invalid_edges))]
            bitstr_list = list(reversed_bitstr)
            bitstr_list[selected_edge[0]] = "0"
            noisy_bitstring = "".join(bitstr_list[::-1])
        # At this point noisy_bitstring should be a valid MIS,
        # add its probability to the dictionary
        try:
            pruned_probs[noisy_bitstring] += probability
        except KeyError:
            pruned_probs[noisy_bitstring] = probability

    return pruned_probs
