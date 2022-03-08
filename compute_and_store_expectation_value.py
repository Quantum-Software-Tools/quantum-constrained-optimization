#!/usr/bin/env python
import sys
import glob
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import qiskit
import qcopt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=None, help="path to project")
    parser.add_argument("-P", type=int, default=1, help="P-value for QAOA")
    parser.add_argument("--graphtype", type=str, default=None, help='Graph type to load (d3, p20, p50, p80)')
    parser.add_argument("--graphname", type=str, default=None, help="Graph name (1, 2, 1 to 10, 11 to 30, ....)")
    parser.add_argument("--repname", type=str, default=None, help="Rep name (1, 1 to 5, extra 1, extra 1 to 5, ...)")
    args = parser.parse_args()
    return args

def get_pickle(ROOT, args, N , graph_name, rep_glob, verbose=0):
    retval = []

    base_path = f'{ROOT}/benchmark_results/QAOA+_P{args.P}_qasm/N{N}_{args.graphtype}_graphs/{graph_name}/{rep_glob}'
    pickle_files = glob.glob(base_path)

    for pklfile in pickle_files:
        if verbose:
            print('Loading pickle file:', pklfile)
        with open(pklfile, 'rb') as pf:
            res = pickle.load(pf)
        return res


def get_output_dist_dict(probs, G):
    output_dict = {hammingweight: {"valid": 0, "invalid": 0} for hammingweight in range(len(G.nodes)+1)}
    for bitstring, p in probs.items():
        if qcopt.graph_funcs.is_indset(bitstring, G):
            output_dict[int(qcopt.helper_funcs.hamming_weight(bitstring))]["valid"] += p
        else:
            output_dict[int(qcopt.helper_funcs.hamming_weight(bitstring))]["invalid"] += p
    return output_dict


def main():
    args = parse_args()

    ROOT = args.path
    if ROOT[-1] != "/":
        ROOT += "/"
    sys.path.append(ROOT)

    # Parse input params
    csv_savepath = ROOT + f"benchmark_results/QAOA+_expectation_values/"
    Path(csv_savepath).mkdir(parents=True, exist_ok=True)
    csv_savename = f"qaoa+_P{args.P}_{args.graphtype}_graphs_{'_'.join(args.graphname.split())}_rep{'_'.join(args.repname.split())}.csv"

    # Everything is hardcoded to 20 nodes
    num_nodes = 20
    # Only allow a single p and graph_type, but multiple graph_names and reps
    if 'to' in args.graphname:
        lowerlim = int(args.graphname.split()[0])
        upperlim = int(args.graphname.split()[-1])
        graph_names = [f"G{i}" for i in np.arange(lowerlim, upperlim + 1)]
    else:
        graph_names = [f"G{args.graphname}"]

    if 'extra' in args.repname:
        if 'to' in args.repname:
            lowerlim = int(args.repname.split()[1])
            upperlim = int(args.repname.split()[-1])
            rep_globs = [f"extra_lambda/*rep{i}.pickle" for i in np.arange(lowerlim, upperlim + 1)]
        else:
            rep_globs = [f"extra_lambda/*rep{args.repname.split()[-1]}.pickle"]
    else:
        if 'to' in args.repname:
            lowerlim = int(args.repname.split()[0])
            upperlim = int(args.repname.split()[-1])
            rep_globs = [f"*rep{i}.pickle" for i in np.arange(lowerlim, upperlim + 1)]
        else:
            rep_globs = [f"*rep{args.repname}.pickle"]

    backend = qiskit.Aer.get_backend("aer_simulator_statevector", max_parallel_threads=4)

    columns = ["lambda", "expectation_value", "p", "N", "graph_type", "graph_name", "rep_name"]
    df = pd.DataFrame(columns=columns)

    for graph_name in graph_names:
        for rep_glob in rep_globs:
            qaoa_plus_data = get_pickle(ROOT, args, num_nodes, graph_name, rep_glob, verbose=0)
            if "extra" in rep_glob:
                cur_rep_name = f"extra_{rep_glob.split('*')[-1].strip('.pickle')}"
            else:
                cur_rep_name = rep_glob.split('*')[-1].strip('.pickle')

            print(f'Processing QAOA+ p = {args.P}, {args.graphtype} graph {graph_name} {cur_rep_name}')
            dist_savepath = ROOT + f"benchmark_results/QAOA+_output_distributions/P{args.P}_{args.graphtype}/{graph_name}_{cur_rep_name}/"
            Path(dist_savepath).mkdir(parents=True, exist_ok=True)
            for data_dict in qaoa_plus_data:
                rounded_lambda = round(data_dict['lambda'], 3)
                graph_fn = '/'.join(data_dict['graph'].split('/')[-3:])
                G = qcopt.graph_funcs.graph_from_file(ROOT + graph_fn)

                # Evaluate the optimized circuit
                print(f'\tSimulating circuit, lambda = {rounded_lambda}')
                circ = qcopt.ansatz.qaoa_plus.construct_qaoa_plus(args.P, G, data_dict['opt_params'], measure=False)
                circ.save_statevector()
                result = qiskit.execute(circ, backend=backend).result()
                probs = qiskit.quantum_info.Statevector(result.get_statevector(circ)).probabilities_dict(decimals=7)

                # Compute the expectation of the objective function
                print(f'\t\tComputing expectation value...')
                expected_energy = qcopt.qaoa_plus_mis.expectation_value(probs, G, rounded_lambda)

                # Save to the DataFrame
                save_values = [rounded_lambda, expected_energy, args.P, num_nodes,
                               args.graphtype, graph_name, cur_rep_name]
                df = pd.concat([df, pd.DataFrame([save_values], columns=columns)])

                # Analyze the valid/invalid output states in the final distribution
                print(f'\t\tAnalyzing output distribution...')
                output_dist_dict = get_output_dist_dict(probs, G)

                # Save to pickle
                with open(dist_savepath + f"hamming_histogram_lambda_{rounded_lambda}.pickle", "wb") as pf:
                    pickle.dump(output_dist_dict, pf)

                # Save the top 100 most probable bitstrings to pickle
                top_strings = sorted([(key, val) for key, val in probs.items()], key=lambda p: p[1], reverse=True)[:100]
                with open(dist_savepath + f"top_100strings_lambda_{rounded_lambda}.pickle", "wb") as pf:
                    pickle.dump(top_strings, pf)

    # Save the DataFrame as csv
    df.to_csv(csv_savepath + csv_savename)



if __name__ == '__main__':
    main()
