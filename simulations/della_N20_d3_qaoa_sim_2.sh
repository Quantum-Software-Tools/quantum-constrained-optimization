#!/bin/bash

date +%m_%d_%y-%H.%M.%S

for pval in {1..7}
do
  for var in 26
  do
    max=$(expr $var + 24)
    echo $var $max
    time parallel python /projects/MARTONOSI/teague/quantum-constrained-optimization/optimize_qaoa+.py -p /projects/MARTONOSI/teague/quantum-constrained-optimization/ --graph benchmark_graphs/N20_d3_graphs/G$1.txt -P $pval --name {1} -v 1 --threads 2 --lamda $2 ::: $(eval echo "{$var..$max}")
  done
done
