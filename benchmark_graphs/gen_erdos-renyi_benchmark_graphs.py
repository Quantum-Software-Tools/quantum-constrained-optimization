#!/usr/bin/env python
import os
import glob
import networkx as nx

import qcopt

def is_unique(folder, G):
    all_graphs = glob.glob(folder + "/*")

    for graph in all_graphs:
        cur_G = qcopt.graph_funcs.graph_from_file(graph)
        if nx.is_isomorphic(G, cur_G):
            return False

    return True

N = 20

for pval in [50, 80]:
    folder = f'N{N}_p{pval}_graphs/'
    print(folder)
    if not os.path.isdir(folder):
      os.mkdir(folder)

    n = int(folder.split("_")[0][1:])
    p = int(folder.split("_")[1][1:]) / 100
    print("Nodes: {}, probability: {}".format(n, p))

    count = 0
    while count < 30:
        G = nx.generators.random_graphs.erdos_renyi_graph(n, p)
        if nx.is_connected(G) and is_unique(folder, G):
            count += 1
            edges = list(G.edges())

            with open(folder + "/G{}.txt".format(count), "w") as fn:
                edgestr = "".join(["{}, ".format(e) for e in edges])
                edgestr = edgestr.strip(", ")
                fn.write(edgestr)
