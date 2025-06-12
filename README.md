MeanFieldIncrements
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/nmayhall-vt/meanfieldincrements/workflows/CI/badge.svg)](https://github.com/nmayhall-vt/meanfieldincrements/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/nmayhall-vt/MeanFieldIncrements/branch/main/graph/badge.svg)](https://codecov.io/gh/nmayhall-vt/MeanFieldIncrements/branch/main)

MFI - A Python package for mean-field increment calculations and quantum many-body operator manipulation.

## Features

- **Site and LocalOperator classes**: Core infrastructure for quantum many-body calculations
- **Many-Body Expansion (MBE)**: Efficient representation and manipulation of density matrices
- **Operator Spaces**: Flexible system for creating and manipulating quantum operators
  - Pauli operators for qubit systems
  - Spin operators for arbitrary spin systems  
  - Fermionic creation/annihilation operators
  - Easy tensor product construction for multi-site operators
- **Partial trace operations**: Efficient reduced density matrix calculations
- **Cumulant expansions**: Tools for computing n-body correlation corrections

## Quick Start

```python
import meanfieldincrements as mfi
import numpy as np

# Create sites
site0 = mfi.Site(0, 2)  # Qubit at site 0
site1 = mfi.Site(1, 2)  # Qubit at site 1

# Create operators using built-in operator spaces
pauli_ops = mfi.PauliHilbertSpace(2).create_operators()
spin_ops = mfi.SpinHilbertSpace(2).create_operators()

# Build two-site operators
two_site_ops = pauli_ops.kron(spin_ops)
xx_interaction = mfi.LocalOperator(two_site_ops['XSx'], [site0, site1])

# Use with many-body expansion
sites = [site0, site1]
rho = mfi.MBEState(sites).initialize_mixed()
```

## Operator Spaces

The package provides several built-in operator spaces:

- **PauliHilbertSpace**: Pauli operators (I, X, Y, Z) for qubit systems
- **SpinHilbertSpace**: Spin operators (Sx, Sy, Sz, S±) for arbitrary spin
- **FermionHilbertSpace**: Creation/annihilation operators for fermionic systems

Create custom operator spaces by subclassing `HilbertSpace`.

### Copyright

Copyright (c) 2025, Nick Mayhall

#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.