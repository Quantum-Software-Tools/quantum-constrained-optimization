from typing import List, Optional
import time, random, queue, copy, itertools
import numpy as np
import networkx as nx

from networkx.algorithms.community.kernighan_lin import kernighan_lin_bisection
from scipy.optimize import minimize

import qiskit
from qiskit import Aer
from qiskit.quantum_info import Statevector

import qcopt


def solve_mis(
    init_state: str,
    G: nx.Graph,
    P: int = 1,
    individual_partial_mixers: bool = False,
    mixer_order: Optional[List[int]] = None,
    threshold: float = 1e-5,
    cutoff: int = 1,
    sim: str = "statevector",
    shots: int = 8192,
    verbose: int = 0,
    threads: int = 0,
    noisy: bool = False,
):
    """
    Find the MIS of G using the Quantum Alternating Operator Ansatz (QAOA).

    The structure of the mixer unitaries keeps the initial state within the feasible
    subspace of possible MIS solutions. Furthermore, the mixer can be parameterized
    by a single angle which is shared amongst all of the partial mixers:

        U_C_P(gamma_P) * U_M_P(beta_P) * ... * U_C_1(gamma_1) * U_M_1(beta_1)|0>

    Or the mixer may be parameterized by a vector of angles, one for each partial mixer:

        U_C_P(gamma_P) * U_M_P([b_1,..., b_n]_P) * ... * U_C_1(gamma_1) * U_M_1(b_1,..., b_n]_P)|0>
    """

    # Initialization
    backend = Aer.get_backend("aer_simulator_statevector", max_parallel_threads=threads)
    if sim == "cloud":
        raise Exception("NOT YET IMPLEMENTED!")
    elif sim not in ["statevector", "qasm"]:
        raise Exception("Unknown simulator:", sim)
    noisy_backend = qcopt.noisy_sim.get_backend(max_parallel_threads=threads)

    # Select an ordering for the partial mixers
    if mixer_order == None:
        cur_permutation = list(np.random.permutation(list(G.nodes)))
    else:
        cur_permutation = mixer_order
    if verbose > 0:
        print("Mixer order:", cur_permutation)

    # This function will be what scipy.minimize optimizes
    def f(params: List):
        # Generate a QAOA circuit
        circ = qcopt.ansatz.qao_ansatz.gen_qaoa(
            G,
            P,
            cur_permutation,
            params=params,
            init_state=init_state,
            individual_partial_mixers=individual_partial_mixers,
            barriers=0,
            decompose_toffoli=1,
            verbose=0,
        )

        # Compute the cost function
        if not noisy:
            if sim == "qasm":
                circ.measure_all()
            elif sim == "statevector":
                circ.save_statevector()

            result = qiskit.execute(circ, backend=backend, shots=shots).result()
            if sim == "statevector":
                probs = Statevector(result.get_statevector(circ)).probabilities_dict(decimals=7)
            elif sim == "qasm":
                probs = {key: val / shots for key, val in result.get_counts(circ).items()}
        else:
            probs = qcopt.noisy_sim.execute_and_prune(circ, G, noisy_backend)

        avg_cost = 0
        for sample in probs.keys():
            x = [int(bit) for bit in list(sample)]
            # Cost function is Hamming weight
            avg_cost += probs[sample] * sum(x)

        # Return the negative of the cost for minimization
        # print('Expectation value:', avg_cost)
        return -avg_cost

    # Begin variational optimization loop
    if individual_partial_mixers:
        num_params = P * (len(G.nodes) + 1)
    else:
        num_params = 2 * P

    init_params = np.random.uniform(low=0.0, high=2 * np.pi, size=num_params)

    if verbose:
        print("\tNum params =", num_params)
        print("\tCurrent Mixer Order:", cur_permutation)

    out = minimize(f, x0=init_params, method="COBYLA")

    opt_params = out["x"]
    opt_cost = out["fun"]
    if verbose:
        print("\tOptimal cost:", opt_cost)

    # Construct the fully optimized circuit
    opt_circ = qcopt.ansatz.qao_ansatz.gen_qaoa(
        G,
        P,
        cur_permutation,
        params=opt_params,
        init_state=init_state,
        individual_partial_mixers=individual_partial_mixers,
        barriers=0,
        decompose_toffoli=1,
        verbose=verbose,
    )

    if not noisy:
        if sim == "qasm":
            opt_circ.measure_all()
        elif sim == "statevector":
            opt_circ.save_statevector()

        result = qiskit.execute(opt_circ, backend=backend, shots=shots).result()
        if sim == "statevector":
            probs = Statevector(result.get_statevector(opt_circ)).probabilities_dict(decimals=7)
        elif sim == "qasm":
            probs = {key: val / shots for key, val in result.get_counts(opt_circ).items()}
    else:
        probs = qcopt.noisy_sim.execute_and_prune(opt_circ, G, noisy_backend)

    # Select the most probable bitstring as the output of the optimization
    top_counts = sorted(
        [(key, val) for key, val in probs.items() if val > threshold],
        key=lambda tup: tup[1],
        reverse=True,
    )[:cutoff]

    best_hamming_weight = 0
    best_indset = None
    for bitstr, prob in top_counts:
        this_hamming = qcopt.helper_funcs.hamming_weight(bitstr)
        if (
            qcopt.graph_funcs.is_indset(bitstr, G, little_endian=True)
            and this_hamming > best_hamming_weight
        ):
            best_hamming_weight = this_hamming
            best_indset = bitstr

    if verbose:
        print(f"\tFound new independent set: {best_indset}, Hamming weight = {best_hamming_weight}")

    # Save the output of the variational optimization
    data_dict = {
        "best_indset": best_indset,
        "cost": opt_cost,
        "function_evals": out["nfev"],
        "init_state": init_state,
        "mixer_order": cur_permutation,
        "num_params": num_params,
        "opt_params": opt_params,
        "P": P,
        "individual_partial_mixers": individual_partial_mixers,
        "noisy": noisy,
    }

    return data_dict
