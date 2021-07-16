# quantum-constrained-optimization

## Overview
The code in this repository implements the algorithms and simulations in [Approaches to Constrained Quantum Approximate Optimization](https://arxiv.org/abs/2010.06660). The `ansatz\` directory contains multiple files which implement different variational ansatzes for solving the Maximum Independent Set (MIS) problem. The functions for executing the variational optimizations are in `mis.py`, each different ansatz has its own function and take a target graph as input.

## Citation
If you use the code in this repository, please cite our paper:

    Zain H. Saleem, Teague Tomesh, Bilal Tariq, and Martin Suchara, Approaches to
    Constrained Quantum Approximate Optimization, arXiv preprint, arXiv:2010.06660 (2021).
