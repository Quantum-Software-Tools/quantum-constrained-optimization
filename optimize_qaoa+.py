#!/usr/bin/env python
"""
Optimize the QAOA+ on a benchmark graph for a given value of p and lambda
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
    parser.add_argument("--name", type=str, default='test', help="Give a unique name to distinguish the save file")
    parser.add_argument("-v", type=int, default=1, help="verbose")
    parser.add_argument(
        "--threads", type=int, default=0, help="Number of threads to pass to Aer simulator"
    )
    parser.add_argument("--lamda", type=float, default=0.1, help="Value of the penalty factor lambda")
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

    savepath = DQVAROOT + f"benchmark_results/QAOA+_P{args.P}_data/{graph_type}/"
    Path(savepath).mkdir(parents=True, exist_ok=True)

    for graphfn in all_graphs:
        graphname = graphfn.split("/")[-1].strip(".txt")
        cur_savepath = savepath + f"{graphname}/lambda_{args.lamda}_runs/"
        Path(cur_savepath).mkdir(parents=True, exist_ok=True)

        G = qcopt.utils.graph_funcs.graph_from_file(graphfn)
        if args.v:
            print(f'Evaluating rep{args.name} p = {args.P} QAOA+ on {graph_type}/{graphname} with lambda = {args.lamda:.4f}')

        out = qcopt.qaoa_plus_mis.solve_mis(args.P, G, args.lamda, threads=args.threads)

        data_dict = {
            "lambda": args.lamda,
            "graph": graphfn,
            "P": args.P,
            "function_evals": out["nfev"],
            "opt_params": out["x"],
            "opt_objfun": -1 * out["fun"],
        }

        # Save the results
        savename = f"{graphname}_QAOA+_P{args.P}_lambda{args.lamda}_rep{args.name}.pickle"
        with open(cur_savepath + savename, "ab") as pf:
            pickle.dump(data_dict, pf)


if __name__ == "__main__":
    main()

