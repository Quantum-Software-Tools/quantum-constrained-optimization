import networkx as nx
import numpy as np

import qcopt

######
# total entangling gate counts
def one_ancilla_barenco(gate_dict, largest_native_control):
    native_counts = 0 # total number of native entangling gates
    for gate, num_counts in gate_dict.items():
        num_controls = int(gate.split('_')[1])
        if num_controls <= largest_native_control:
            native_counts += 2 * num_counts
        elif largest_native_control == 1:
            # Special cases for S_2 gate set
            if num_controls == 2:
                native_counts += 6 * num_counts
            elif num_controls == 3:
                native_counts += 18 * num_counts
            elif num_controls == 4:
                native_counts += 42 * num_counts
            else:
                # Asymptotic case for S_2 gate set
                native_counts += (16 * num_controls - 8) * num_counts
        elif largest_native_control == 2:
            # Special cases for S_3 gate set
            if num_controls == 3:
                native_counts += 4 * num_counts
            elif num_controls == 4:
                native_counts += 10 * num_counts
            else:
                # Asymptotic case for S_3 gate set
                native_counts += (8 * num_controls - 24) * num_counts
        else:
            raise Exception("Only CX and Toffolis supported")

    return native_counts

def n_ancilla_barenco(gate_dict, largest_native_control):
    native_counts = 0
    for gate, num_counts in gate_dict.items():
        num_controls = int(gate.split('_')[1])
        if num_controls <= largest_native_control:
            native_counts += 2 * num_counts
        elif largest_native_control == 1:
            # Special cases for S_2 gate set
            if num_controls == 2:
                native_counts += 6 * num_counts
            else:
                # Asymptotic case for S_2 gate set
                native_counts += (6 * num_controls) * num_counts
        elif largest_native_control == 2:
            # Asymptotic case for S_3 gate set
            native_counts += (2 * num_controls - 2) * num_counts
        else:
            raise Exception("Only CX and Toffolis supported")

    return native_counts

def no_ancilla_qutrit(gate_dict):
    native_counts = 0
    for gate, num_counts in gate_dict.items():
        num_controls = int(gate.split('_')[1])
        if num_controls % 2 == 0 :
            native_counts += (6 * (num_controls - 1) + 4) * num_counts
        else:
            native_counts += (6 * (num_controls - 1) + 2) * num_counts

    return native_counts

######
# basis gate counts
def one_ancilla_barenco_basis_counts(gate_dict, largest_native_control):
    basis_gates = [0 for _ in range(largest_native_control)]
    for gate, num_counts in gate_dict.items():
        num_controls = int(gate.split('_')[1])
        if num_controls <= largest_native_control:
            basis_gates[num_controls-1] += 2 * num_counts
        elif largest_native_control == 1:
            # Special cases for S_2 gate set
            if num_controls == 2:
                basis_gates[0] += 6 * num_counts
            elif num_controls == 3:
                basis_gates[0] += 18 * num_counts
            elif num_controls == 4:
                basis_gates[0] += 42 * num_counts
            else:
                # Asymptotic case for S_2 gate set
                basis_gates[0] += (16 * num_controls - 8) * num_counts
        elif largest_native_control == 2:
            # Special cases for S_3 gate set
            if num_controls == 3:
                basis_gates[1] += 4 * num_counts
            elif num_controls == 4:
                basis_gates[1] += 10 * num_counts
            else:
                # Asymptotic case for S_3 gate set
                basis_gates[1] += (8 * num_controls - 24) * num_counts
        else:
            raise Exception("Only CX and Toffolis supported")

    return basis_gates

def n_ancilla_barenco_basis_counts(gate_dict, largest_native_control):
    basis_gates = [0 for _ in range(largest_native_control)]
    for gate, num_counts in gate_dict.items():
        num_controls = int(gate.split('_')[1])
        if num_controls <= largest_native_control:
            basis_gates[num_controls-1] += 2 * num_counts
        elif largest_native_control == 1:
            # Special cases for S_2 gate set
            if num_controls == 2:
                basis_gates[0] += 6 * num_counts
            else:
                # Asymptotic case for S_2 gate set
                basis_gates[0] += (6 * num_controls) * num_counts
        elif largest_native_control == 2:
            # Asymptotic case for S_3 gate set
            basis_gates[1] += (2 * num_controls - 2) * num_counts
        else:
            raise Exception("Only CX and Toffolis supported")

    return basis_gates

def no_ancilla_qutrit_basis_counts(gate_dict):
    native_counts = 0
    for gate, num_counts in gate_dict.items():
        num_controls = int(gate.split('_')[1])
        if num_controls % 2 == 0 :
            native_counts += (6 * num_controls - 8) * num_counts
        else:
            native_counts += (6 * num_controls - 4) * num_counts

    return [native_counts]
######



def gen_3_regular_graph(size):
    """Generate a random 3-regular graph."""
    while True:
        G = nx.random_regular_graph(3, size)
        if nx.is_connected(G):
            return G

def gen_erdos_renyi_graph(size, avg_degree):
    """Generate a random Erdos-Renyi graph."""
    edge_probability = avg_degree / (size - 1)
    while True:
        G = nx.generators.random_graphs.erdos_renyi_graph(size, edge_probability)
        if nx.is_connected(G):
            return G

def dqva_gate_count(G, nu):
    """Take in a NetworkX graph G and compute the number
    of basis gates required in the quantum circuit.

    nu=10 is a good starting point - this will have to scale with graph size.
    DQVA randomly picks vertices in G to hit with a partial mixer.

    Returns
    -------
    gate_counts Dict: {c_1_Rx: a, c_2_Rx: b, ...}
    """
    gate_counts = {}
    for node in np.random.choice(list(G.nodes), size=nu):
        num_controls = len(list(G.neighbors(node)))
        key = f"c_{num_controls}_Rx"
        try:
            gate_counts[key] += 1
        except KeyError:
            gate_counts[key] = 1
    return gate_counts

def sa_qaoa_gate_count(G, p):
    """Take in a NetworkX graph G and compute the number
    of basis gates required in the quantum circuit.

    The parameter p is the depth of the SA-QAOA. One layer of SA-QAOA hits every vertex
    with its partial mixer, and that layer is repeated p times.

    Returns
    -------
    gate_counts Dict: {c_1_Rx: a, c_2_Rx: b, ...}
    """
    gate_counts = {}
    for node in G:
        num_controls = len(list(G.neighbors(node)))
        key = f"c_{num_controls}_Rx"
        try:
            gate_counts[key] += 1
        except KeyError:
            gate_counts[key] = 1
    gate_counts = {key: p * value for key, value in gate_counts.items()}
    return gate_counts

def ma_qaoa_gate_count(G):
    """Take in a NetworkX graph G and compute the number
    of basis gates required in the quantum circuit.

    Usually we just have p=1 for MA-QAOA. In that single layer, every vertex
    is hit with its partial mixer.

    Returns
    -------
    gate_counts Dict: {c_1_Rx: a, c_2_Rx: b, ...}
    """
    gate_counts = {}
    for node in G:
        num_controls = len(list(G.neighbors(node)))
        key = f"c_{num_controls}_Rx"
        try:
            gate_counts[key] += 1
        except KeyError:
            gate_counts[key] = 1
    return gate_counts

def main():
    """This function should generate a bunch of random graphs
    with increasing size. And for each graph, calculate the gate
    counts incurred by DQVA, SA-QAOA, and MA-QAOA.

    Do we want to just focus on 3-regular graphs for now because
    then we will only ever have C^3 Rx gates. Should make the
    analysis easier.
    We can potentially look at Erdos-Renyi graphs later if we want but lets
    get 3-regular working first.
    """

    # This first part is what I imagine will be needed to generate Figure 5
    graph_sizes = [10, 20, 60, 80, 100, 200, 500, 1000]
    repetitions = 100

    all_gate_counts = {}
    for size in graph_sizes:
        # Should average over many repetitions (maybe only necessary for Erdos-Renyi)
        gate_counts = {'dqva': [], 'sa-qaoa': [], 'ma-qaoa': []}
        for rep in range(repetitions):
            G = gen_3_regular_graph(size)
            for key in gate_counts.keys():
                if key == 'dqva':
                    logical_gate_count = dqva_gate_count(G, 10)
                elif key == 'sa-qaoa':
                    logical_gate_count = sa_qaoa_gate_count(G, 10)
                elif key == 'ma-qaoa':
                    logical_gate_count = ma_qaoa_gate_count(G)

                #gate_counts[key].append(get_decomposed_counts(logical_gate_count, basis_gates, decomp_method))

        all_gate_counts[size] = gate_counts



    # This second part is what I imagine will be needed to generate the table
    # comparing our work with the Qiskit and tket compiler.
    graph_sizes = [5, 10, 15, 20]
    repetitions = 20

    all_gate_counts = {}
    for size in graph_sizes:
        gate_counts = {}
        for rep in range(repetitions):
            G = gen_3_regular_graph(size)
            mixer_order = list(np.random.permutation(list(G.nodes)))
            sa_qaoa_circuit = qcopt.ansatz.qao_ansatz.gen_qaoa(
                    G=G,
                    P=10,
                    mixer_order=mixer_order,
                    params=np.random.uniform(low=0.0, high=2 * np.pi, size=20),
                    individual_partial_mixers=0,
                    barriers=0,
                    decompose_toffoli=0
            ) # you may want to play around with the decompose_toffoli parameter to get the right level

            ma_qaoa_circuit = qcopt.ansatz.qao_ansatz.gen_qaoa(
                    G=G,
                    P=1,
                    mixer_order=mixer_order,
                    params=np.random.uniform(low=0.0, high=2 * np.pi, size=size+1),
                    individual_partial_mixers=1,
                    barriers=0,
                    decompose_toffoli=0
            )

            # Please ignore the horrible naming right now.
            # qlsa is what we used to call the DQVA when we
            # restricted the number of partial mixers
            num_partial_mixers = 10
            dqva_circuit = qcopt.ansatz.qlsa.gen_qlsa(
                    G=G,
                    P=1,
                    params=np.random.uniform(low=0.0, high=2 * np.pi, size=num_partial_mixers),
                    barriers=0,
                    decompose_toffoli=0,
                    mixer_order=mixer_order,
                    param_lim=num_partial_mixers,
            )


            # Once all the circuits are generated, unroll the multi-controlled gates using
            # different compilers:

            # Qiskit (optimization_level=3)

            # tket

            # Our work


if __name__ == "__main__":
    main()
