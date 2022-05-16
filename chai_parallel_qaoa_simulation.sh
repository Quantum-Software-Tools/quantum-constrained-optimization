#!/bin/bash

date +%m_%d_%y-%H.%M.%S

for graph in {6..60}
do
  for pval in {1..7}
  do
    for var in 1 11 21 31 41
    do
      max=$(expr $var + 9)
      echo $var $max
      time parallel ./optimize_qaoa+.py -p /home/ttomesh/quantum-constrained-optimization/ --graph benchmark_graphs/N10_d3_graphs/G$graph.txt -P $pval --name {1} -v 1 --threads 2 --lamda 0.1 ::: $(eval echo "{$var..$max}")
    done
  done
done
