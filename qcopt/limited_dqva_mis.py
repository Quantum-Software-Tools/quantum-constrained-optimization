import time, random, queue, copy, itertools
import numpy as np
import networkx as nx

from networkx.algorithms.community.kernighan_lin import kernighan_lin_bisection
from scipy.optimize import minimize

import qiskit
from qiskit import Aer
from qiskit.quantum_info import Statevector

from qcopt.ansatz import qlsa
from qcopt.utils import graph_funcs, helper_funcs


def solve_mis(
    init_state,
    G,
    P=1,
    m=1,
    mixer_order=None,
    threshold=1e-5,
    cutoff=1,
    sim="aer",
    shots=8192,
    verbose=0,
    param_lim=None,
    threads=0,
):
    """
    Find the MIS of G using Quantum Local Search (QLS), this
    ansatz is composed of two types of unitaries: the cost unitary U_C and the
    mixer unitary U_M. The mixer U_M is made up of individual partial mixers
    which are independently parametrized.

    QLS's key feature is the parameter limit which truncates the number of
    partial mixers that are applied at any one time, and its dynamic reuse of
    quantum resources (i.e. the partial mixers for qubits which are in the MIS
    are turned off and applied to other qubits not currently in the set)
    """

    # Initialization
    if sim == "statevector" or sim == "qasm":
        backend = Aer.get_backend(sim + "_simulator", max_parallel_threads=threads)
    elif sim == "aer":
        backend = Aer.get_backend(
            name="aer_simulator", method="statevector", max_parallel_threads=threads
        )
    elif sim == "cloud":
        raise Exception("NOT YET IMPLEMENTED")
    else:
        raise Exception("Unknown simulator:", sim)

    # Select an ordering for the partial mixers
    if mixer_order == None:
        cur_permutation = list(np.random.permutation(list(G.nodes)))
    else:
        cur_permutation = mixer_order

    history = []

    # This function will be what scipy.minimize optimizes
    def f(params):
        # Generate a circuit
        circ = qlsa.gen_qlsa(
            G,
            P=P,
            params=params,
            init_state=cur_init_state,
            barriers=0,
            decompose_toffoli=1,
            mixer_order=cur_permutation,
            verbose=0,
            param_lim=param_lim,
        )

        if sim == "qasm" or sim == "aer":
            circ.measure_all()

        # Compute the cost function
        result = qiskit.execute(circ, backend=backend, shots=shots).result()
        if sim == "statevector":
            statevector = Statevector(result.get_statevector(circ))
            probs = helper_funcs.strip_ancillas(statevector.probabilities_dict(decimals=5), circ)
        elif sim == "qasm" or sim == "aer":
            counts = result.get_counts(circ)
            probs = helper_funcs.strip_ancillas(
                {key: val / shots for key, val in counts.items()}, circ
            )

        avg_cost = 0
        for sample in probs.keys():
            x = [int(bit) for bit in list(sample)]
            # Cost function is Hamming weight
            avg_cost += probs[sample] * sum(x)

        # Return the negative of the cost for minimization
        # print('Expectation value:', avg_cost)
        return -avg_cost

    # Begin outer optimization loop
    best_indset = init_state
    best_init_state = init_state
    cur_init_state = init_state
    best_params = None
    best_perm = copy.copy(cur_permutation)

    # Randomly permute the order of mixer unitaries m times
    for mixer_round in range(1, m + 1):
        mixer_history = []
        inner_round = 1
        new_hamming_weight = helper_funcs.hamming_weight(cur_init_state)

        # Attempt to improve the Hamming weight until no further improvements can be made
        while True:
            print(
                "Start round {}.{}, Initial state = {}".format(
                    mixer_round, inner_round, cur_init_state
                )
            )

            # Begin Inner variational loop
            num_nonzero = len(G.nodes()) - helper_funcs.hamming_weight(cur_init_state)
            if param_lim is None:
                num_params = min(P * (len(G.nodes()) + 1), (P + 1) * (num_nonzero + 1))
            else:
                num_params = param_lim
            print("\tNum params =", num_params)
            # Important to start from random initial points
            # init_params = np.zeros(num_params)
            init_params = np.random.uniform(low=0.0, high=2 * np.pi, size=num_params)
            print("\tCurrent Mixer Order:", cur_permutation)

            out = minimize(f, x0=init_params, method="COBYLA")

            opt_params = out["x"]
            opt_cost = out["fun"]
            # print('\tOptimal Parameters:', opt_params)
            print("\tOptimal cost:", opt_cost)

            # Get the results of the optimized circuit
            opt_circ = qlsa.gen_qlsa(
                G,
                P=P,
                params=opt_params,
                init_state=cur_init_state,
                barriers=0,
                decompose_toffoli=1,
                mixer_order=cur_permutation,
                verbose=0,
                param_lim=param_lim,
            )

            if sim == "qasm" or sim == "aer":
                opt_circ.measure_all()

            result = qiskit.execute(opt_circ, backend=backend, shots=shots).result()
            if sim == "statevector":
                statevector = Statevector(result.get_statevector(opt_circ))
                probs = helper_funcs.strip_ancillas(
                    statevector.probabilities_dict(decimals=5), opt_circ
                )
            elif sim == "qasm" or sim == "aer":
                counts = result.get_counts(opt_circ)
                probs = helper_funcs.strip_ancillas(
                    {key: val / shots for key, val in counts.items()}, opt_circ
                )

            # Select the top [cutoff] counts
            top_counts = sorted(
                [(key, val) for key, val in probs.items() if val > threshold],
                key=lambda tup: tup[1],
                reverse=True,
            )[:cutoff]

            # Check if we have improved the Hamming weight
            best_hamming_weight = helper_funcs.hamming_weight(best_indset)
            better_strs = []
            for bitstr, prob in top_counts:
                this_hamming = helper_funcs.hamming_weight(bitstr)
                if graph_funcs.is_indset(bitstr, G) and this_hamming > best_hamming_weight:
                    better_strs.append((bitstr, this_hamming))
            better_strs = sorted(better_strs, key=lambda t: t[1], reverse=True)

            # Save current results to history
            inner_history = {
                "mixer_round": mixer_round,
                "inner_round": inner_round,
                "cost": opt_cost,
                "init_state": cur_init_state,
                "mixer_order": copy.copy(cur_permutation),
                "num_params": num_params,
            }
            mixer_history.append(inner_history)

            # If no improvement was made, break and go to next mixer round
            if len(better_strs) == 0:
                print(
                    "\tNone of the measured bitstrings had higher Hamming weight than:", best_indset
                )
                break

            # Otherwise, save the new bitstring and repeat
            best_indset, new_hamming_weight = better_strs[0]
            best_init_state = cur_init_state
            best_params = opt_params
            best_perm = copy.copy(cur_permutation)
            cur_init_state = best_indset
            print(
                "\tFound new independent set: {}, Hamming weight = {}".format(
                    best_indset, new_hamming_weight
                )
            )

            # Go through another execution of this While loop, with the same
            # mixer order
            inner_round += 1

        # Save the history of the current mixer round
        history.append(mixer_history)

        # Choose a new permutation of the mixer unitaries that have NOT been set to identity
        identity_mixers = [
            i for i in range(len(cur_init_state)) if list(reversed(cur_init_state))[i] == "1"
        ]
        non_identity_mixers = [
            i for i in range(len(cur_init_state)) if list(reversed(cur_init_state))[i] == "0"
        ]
        permutation = np.random.permutation(non_identity_mixers)
        perm_queue = queue.Queue()
        for p in permutation:
            perm_queue.put(p)
        for i, mixer in enumerate(cur_permutation):
            if mixer in identity_mixers:
                continue
            else:
                cur_permutation[i] = perm_queue.get()

    print("\tRETURNING, best hamming weight:", new_hamming_weight)
    return best_indset, best_params, best_init_state, best_perm, history
