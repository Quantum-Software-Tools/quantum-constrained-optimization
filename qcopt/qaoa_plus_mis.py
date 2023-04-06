from scipy.optimize import minimize
import numpy as np
import qiskit
from qiskit import Aer

import qcopt


def solve_mis(P, G, Lambda, shots=1024, threads=0, noisy=False):

    backend = Aer.get_backend("aer_simulator_statevector", max_parallel_threads=threads)
    noisy_backend = qcopt.noisy_sim.get_backend(max_parallel_threads=threads)

    def f(params):
        circ = qcopt.qaoa_plus.construct_qaoa_plus(P, G, params, measure=False)
        if not noisy:
            circ.save_statevector()

            result = qiskit.execute(circ, backend=backend, shots=shots).result()
            probs = qiskit.quantum_info.Statevector(result.get_statevector(circ)).probabilities_dict(
                decimals=7
            )
        else:
            probs = qcopt.noisy_sim.execute_and_prune(circ, G, noisy_backend)

        return -1 * expectation_value(probs, G, Lambda)

    init_params = np.random.uniform(low=0.0, high=2 * np.pi, size=2 * P)
    out = minimize(f, x0=init_params, method="COBYLA")

    return out


def expectation_value(probs, G, Lambda):
    energy = 0
    for bitstr, probability in probs.items():
        temp_energy = qcopt.helper_funcs.hamming_weight(bitstr)
        for edge in G.edges():
            q_i, q_j = edge
            rev_bitstr = list(reversed(bitstr))
            if rev_bitstr[q_i] == "1" and rev_bitstr[q_j] == "1":
                temp_energy += -1 * Lambda

        energy += probability * temp_energy

    return energy


def get_ranked_probs(P, G, params, threads=0):
    backend = Aer.get_backend("aer_simulator_statevector", max_parallel_threads=threads)
    circ = qcopt.ansatz.qaoa_plus.construct_qaoa_plus(P, G, params=params, measure=False)
    circ.save_statevector()
    result = qiskit.execute(circ, backend=backend).result()
    probs = qiskit.quantum_info.Statevector(result.get_statevector(circ)).probabilities_dict(
        decimals=5
    )

    sorted_probs = [
        (bitstr, p, qcopt.utils.graph_funcs.is_indset(bitstr, G)) for bitstr, p in probs.items()
    ]
    sorted_probs = sorted(sorted_probs, key=lambda p: p[1], reverse=True)

    return sorted_probs


def get_approximation_ratio(out, P, G, brute_force_output=None, shots=8192, threads=0):
    if brute_force_output:
        with open(brute_force_output, "r") as outfile:
            for i, line in enumerate(outfile):
                if i == 1:
                    opt_mis = int(line.split()[-1])
    else:
        print("No brute force file exists, finding now...")
        opt_mis = qcopt.utils.helper_funcs.brute_force_search(G)[1]

    circ = qcopt.ansatz.qaoa_plus.construct_qaoa_plus(P, G, params=out["x"], measure=True)
    result = qiskit.execute(
        circ, backend=Aer.get_backend("aer_simulator", max_parallel_threads=threads), shots=shots
    ).result()
    counts = result.get_counts(circ)

    # Equation 8
    # Approximation ratio is computed using ONLY valid independent sets
    # E(gamma, beta) = SUM_bitstrs { (bitstr_counts / total_shots) * hamming_weight(bitstr) } / opt_mis
    numerator = 0
    for bitstr, count in counts.items():
        if qcopt.utils.graph_funcs.is_indset(bitstr, G):
            numerator += count * qcopt.utils.helper_funcs.hamming_weight(bitstr) / shots
    ratio_eqn8 = numerator / opt_mis

    sorted_counts = sorted(
        [(bitstr, counts[bitstr] / shots) for bitstr in counts.keys()],
        key=lambda p: p[1],
        reverse=True,
    )
    most_likely_ratio = 0
    for bitstr, prob in sorted_counts:
        if qcopt.utils.graph_funcs.is_indset(bitstr, G):
            most_likely_ratio = qcopt.utils.helper_funcs.hamming_weight(bitstr) / opt_mis
            break

    return ratio_eqn8, most_likely_ratio


def top_strs(counts, G, top=5):
    total_shots = sum(counts.values())
    probs = [(bitstr, counts[bitstr] / total_shots) for bitstr in counts.keys()]
    probs = sorted(probs, key=lambda p: p[1], reverse=True)
    opt_mis = qcopt.utils.helper_funcs.brute_force_search(G)[1]

    for i in range(top):
        ratio = qcopt.utils.helper_funcs.hamming_weight(probs[i][0]) * probs[i][1] / opt_mis
        print(
            "{} ({}) -> {:.4f}%, Ratio = {:.4f}, Is MIS? {}".format(
                probs[i][0],
                qcopt.utils.helper_funcs.hamming_weight(probs[i][0]),
                probs[i][1] * 100,
                ratio,
                qcopt.utils.graph_funcs.is_indset(probs[i][0], G),
            )
        )
