import qiskit


def construct_qaoa_plus(P, G, params, barriers=False, measure=False):
    assert len(params) == 2 * P, "Number of parameters should be 2P"

    nq = len(G.nodes())
    circ = qiskit.QuantumCircuit(nq, name="q")

    # Initial state
    circ.h(range(nq))

    gammas = [param for i, param in enumerate(params) if i % 2 == 0]
    betas = [param for i, param in enumerate(params) if i % 2 == 1]
    for i in range(P):
        # Phase Separator Unitary
        for edge in G.edges():
            q_i, q_j = edge
            circ.rz(gammas[i] / 2, [q_i, q_j])
            circ.cx(q_i, q_j)
            circ.rz(-1 * gammas[i] / 2, q_j)
            circ.cx(q_i, q_j)
            if barriers:
                circ.barrier()

        # Mixing Unitary
        for q_i in range(nq):
            circ.rx(-2 * betas[i], q_i)

    if measure:
        circ.measure_all()

    return circ
