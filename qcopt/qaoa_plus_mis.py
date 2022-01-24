from scipy.optimize import minimize
import numpy as np
import qiskit
from qiskit import Aer

import qcopt


def solve_mis(P, G, Lambda, threads=0):

    backend = Aer.get_backend("aer_simulator", max_parallel_threads=threads)

    def f(params):
        circ = qcopt.ansatz.qaoa_plus.construct_qaoa_plus(P, G, params, measure=True)

        result = qiskit.execute(circ, backend=backend, shots=8192).result()
        counts = result.get_counts(circ)

        return -1 * expectation_value(counts, G, Lambda)

    init_params = np.random.uniform(low=0.0, high=2 * np.pi, size=2 * P)
    out = minimize(f, x0=init_params, method="COBYLA")

    return out


def expectation_value(counts, G, Lambda):
    total_shots = sum(counts.values())
    energy = 0
    for bitstr, count in counts.items():
        temp_energy = qcopt.helper_funcs.hamming_weight(bitstr)
        for edge in G.edges():
            q_i, q_j = edge
            rev_bitstr = list(reversed(bitstr))
            if rev_bitstr[q_i] == "1" and rev_bitstr[q_j] == "1":
                temp_energy += -1 * Lambda

        energy += count * temp_energy / total_shots

    return energy

def get_ranked_probs(P, G, params, shots=8192, threads=0):
    circ = qcopt.ansatz.qaoa_plus.construct_qaoa_plus(P, G, params=params, measure=True)
    result = qiskit.execute(circ, backend=Aer.get_backend("aer_simulator", max_parallel_threads=threads), shots=shots).result()
    counts = result.get_counts(circ)

    probs = [(bitstr, counts[bitstr] / shots, qcopt.utils.graph_funcs.is_indset(bitstr, G)) for bitstr in counts.keys()]
    probs = sorted(probs, key=lambda p: p[1], reverse=True)

    return probs


def get_approximation_ratio(out, P, G, shots=8192, threads=0):
    opt_mis = qcopt.utils.helper_funcs.brute_force_search(G)[1]

    circ = qcopt.ansatz.qaoa_plus.construct_qaoa_plus(P, G, params=out["x"], measure=True)
    result = qiskit.execute(circ, backend=Aer.get_backend("aer_simulator", max_parallel_threads=threads), shots=shots).result()
    counts = result.get_counts(circ)

    # Approximation ratio is computed using ONLY valid independent sets
    # E(gamma, beta) = SUM_bitstrs { (bitstr_counts / total_shots) * hamming_weight(bitstr) } / opt_mis
    numerator = 0
    for bitstr, count in counts.items():
        if qcopt.utils.graph_funcs.is_indset(bitstr, G):
            numerator += count * qcopt.utils.helper_funcs.hamming_weight(bitstr) / shots
    ratio = numerator / opt_mis

    # ratio = sum([count * hamming_weight(bitstr) / shots for bitstr, count in counts.items() \
    #             if is_indset(bitstr, G)]) / opt_mis

    return ratio


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
