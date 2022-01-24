#!/usr/bin/env python
import argparse, glob, sys, time
from pathlib import Path

import qcopt


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-p', type=str, default=None,
    #                    help='path to DQVA directory')
    parser.add_argument('-g', type=str, default=None,
                        help='Graph file name')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    graphfiles = glob.glob(args.g)
    for gfile in graphfiles:
        graphtype = gfile.split('/')[-2]
        graphname = gfile.split('/')[-1].strip('.txt')

        G = qcopt.utils.graph_funcs.graph_from_file(gfile)
        print(f'graphtype: {graphtype}, graphname: {graphname}')
        print(f'Loaded graph with {len(G.nodes)} nodes')

        start = time.time()
        opt_strs, opt_mis = qcopt.utils.helper_funcs.brute_force_search(G)
        end = time.time()
        print('Finished brute force search in {:.3f} min'.format((end - start) / 60))

        outdir = f'brute_force_outputs/{graphtype}/'
        Path(outdir).mkdir(parents=True, exist_ok=True)
        outfile = outdir + graphname + '_brute_force.out'
        with open(outfile, 'w') as fn:
            fn.write(f'{graphtype}, {graphname}\n')
            fn.write(f'Optimal MIS is {opt_mis}\n')
            fn.write(f'Optimal MIS:\n')
            for bitstr in opt_strs:
                fn.write(f'\t{bitstr}, valid: {qcopt.utils.graph_funcs.is_indset(bitstr, G)}\n')

if __name__ == '__main__':
    main()
