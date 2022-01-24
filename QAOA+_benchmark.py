#!/usr/bin/env python
"""
Use the benchmark graphs to test the performance of QAOA+
"""
import os, sys, argparse, glob
import numpy as np
import pickle, random
from pathlib import Path

import qcopt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=None, help="path to dqva project")
    parser.add_argument(
        "--graph", type=str, default=None, help="glob path to the benchmark graph(s)"
    )
    parser.add_argument("-P", type=int, default=1, help="P-value for algorithm")
    parser.add_argument("--reps", type=int, default=4, help="Number of repetitions to run")
    parser.add_argument("-v", type=int, default=1, help="verbose")
    parser.add_argument(
        "--threads", type=int, default=0, help="Number of threads to pass to Aer simulator"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    DQVAROOT = args.path
    if DQVAROOT[-1] != "/":
        DQVAROOT += "/"
    sys.path.append(DQVAROOT)

    all_graphs = glob.glob(DQVAROOT + args.graph)
    graph_type = all_graphs[0].split("/")[-2]

    savepath = DQVAROOT + f"benchmark_results/QAOA+_P{args.P}_qasm/{graph_type}/"
    Path(savepath).mkdir(parents=True, exist_ok=True)

    for graphfn in all_graphs:
        graphname = graphfn.split("/")[-1].strip(".txt")
        cur_savepath = savepath + f"{graphname}/"
        Path(cur_savepath).mkdir(parents=True, exist_ok=True)

        G = qcopt.utils.graph_funcs.graph_from_file(graphfn)
        print(graphname, G.edges())

        data_list = []
        for rep in range(1, args.reps + 1):
            for Lambda in np.arange(0.1, 10, 0.7):
                data_dict = {"lambda": Lambda, "graph": graphfn}
                out = qcopt.qaoa_plus_mis.solve_mis(args.P, G, Lambda, threads=args.threads)
                data_dict["fevals"] = out["nfev"]
                data_dict["opt_params"] = out["x"]

                # Compute the approximation ratio by summing over only valid ISs and by taking the most likely IS
                ratios = qcopt.qaoa_plus_mis.get_approximation_ratio(
                    out,
                    args.P,
                    G,
                    brute_force_output=None,  # f"{DQVAROOT}benchmark_graphs/brute_force_outputs/{graph_type}/{graphname}_brute_force.out",
                    threads=args.threads,
                )
                data_dict["ratios"] = ratios

                # Sort the output bitstrings by their probabilities
                ranked_probs = qcopt.qaoa_plus_mis.get_ranked_probs(
                    args.P, G, out["x"], threads=args.threads
                )
                data_dict["ranked_probs"] = ranked_probs

                print(f"lambda: {Lambda:.3f}, ratios = {ratios[0]:.3f}, {ratios[1]:.3f}")

                data_list.append(data_dict)

            # Save the results
            savename = f"{graphname}_QAOA+_P{args.P}_rep{rep}.pickle"
            with open(cur_savepath + savename, "ab") as pf:
                pickle.dump(data_list, pf)


if __name__ == "__main__":
    main()
