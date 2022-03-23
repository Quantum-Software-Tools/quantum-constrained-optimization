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
    parser.add_argument("--lowerlim", type=float, default=0.1, help="Lower limit in the lambda range")
    parser.add_argument("--upperlim", type=float, default=4, help="Uppler limit in the lambda range")
    parser.add_argument("--step", type=float, default=0.25, help="Step size in the lambda range")
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
        cur_savepath = savepath + f"{graphname}/extra_lambda/"
        Path(cur_savepath).mkdir(parents=True, exist_ok=True)

        G = qcopt.utils.graph_funcs.graph_from_file(graphfn)
        print(graphname, G.edges())

        for rep in range(1, args.reps + 1):
            data_list = []
            for Lambda in np.arange(args.lowerlim, args.upperlim, args.step)
                out = qcopt.qaoa_plus_mis.solve_mis(args.P, G, Lambda, threads=args.threads)

                # Compute the approximation ratio by summing over only valid ISs and by taking the most likely IS
                ratios = qcopt.qaoa_plus_mis.get_approximation_ratio(
                    out,
                    args.P,
                    G,
                    brute_force_output=f"{DQVAROOT}benchmark_graphs/brute_force_outputs/{graph_type}/{graphname}_brute_force.out",
                    threads=args.threads,
                )

                data_dict = {
                    "lambda": Lambda,
                    "graph": graphfn,
                    "P": args.P,
                    "function_evals": out["nfev"],
                    "opt_params": out["x"],
                    "ratios": ratios,
                }

                print(f"lambda: {Lambda:.3f}, ratios = {ratios[0]:.3f}, {ratios[1]:.3f}")

                data_list.append(data_dict)

            # Save the results
            savename = f"{graphname}_QAOA+_P{args.P}_rep{rep}.pickle"
            with open(cur_savepath + savename, "ab") as pf:
                pickle.dump(data_list, pf)


if __name__ == "__main__":
    main()
