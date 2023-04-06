# quantum-constrained-optimization

## Overview
The code in this repository implements the algorithms and simulations in [Approaches to Constrained Quantum Approximate Optimization](https://arxiv.org/abs/2010.06660). The `ansatz\` directory contains multiple files which implement different variational ansatzes for solving the Maximum Independent Set (MIS) problem. The functions for executing the variational optimizations are in `mis.py`, each different ansatz has its own function and take a target graph as input.

## Getting Started
The code in this repo can be installed as a Python module by following the instructions below. Either Python 3.6 or a newer version should be used. We highly recommend the use of Python virtual environments to simplify the installation experience and reduce the chance of package conflicts.

```bash
git clone https://github.com/teaguetomesh/quantum-constrained-optimization.git
python3 -m venv your_new_venv       # create a new Python virtual environment
source your_new_venv/bin/activate   # activate that virtual environment
cd quantum-constrained-optimization
pip install -r requirements.txt     # install the package dependecies
pip install -e .                    # install the qcopt package
```

After the `qcopt` package has been installed it can be imported into your Python scripts via:

```python
import qcopt
```

## Citation
If you use the code in this repository, please cite our paper:

    Zain H. Saleem, Teague Tomesh, Bilal Tariq, and Martin Suchara, Approaches to
    Constrained Quantum Approximate Optimization, arXiv preprint, arXiv:2010.06660 (2021).
