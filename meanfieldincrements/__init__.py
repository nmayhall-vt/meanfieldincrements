"""MFI"""

import numpy as np

# Import HilbertSpace classes first (no dependencies)
from .HilbertSpace import HilbertSpace, PauliHilbertSpace, SpinHilbertSpace, FermionHilbertSpace

# Import SiteOperators (depends on HilbertSpace)
from .SiteOperators import SiteOperators

# Import Site class and its factory functions (depends on HilbertSpace, uses lazy import for SiteOperators)
from .Site import (
    Site, 
    # Factory functions for common site types
    qubit_site, spin_site, fermion_site, multi_qubit_site,
    # Chain creation functions  
    create_sites, create_qubit_chain, create_spin_chain, create_fermion_chain
)

# Import other core classes (minimal dependencies)
from .LocalTensor import LocalTensor
from .PauliString import PauliString
from .MBEState import MBEState

# Import GeneralHamiltonian and related utilities (depends on Site, LocalOperator, SiteOperators)
from .GeneralHamiltonian import (
    GeneralHamiltonian,
    # Pre-built Hamiltonian constructors
    build_heisenberg_hamiltonian,
    build_ising_hamiltonian,
    from_pauli_strings
)

# from .Marginals import Marginal, Cumulant, Marginals, Cumulants 

from ._version import __version__