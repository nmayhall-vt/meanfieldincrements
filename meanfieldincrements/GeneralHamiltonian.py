import numpy as np
import copy as cp
from typing import Dict, List, Union
from .Site import Site
from .SiteOperators import SiteOperators


class GeneralHamiltonian:
    """
    Simplified General Hamiltonian class for linear combinations of tensor products of site operators.
    
    Represents H = ∑_i h_i ⊗_j O_ij where:
    - h_i are coefficients  
    - O_ij are operators on site j for term i
    
    The Hamiltonian is defined on a fixed set of sites and stores terms as a dictionary
    mapping operator string tuples to coefficients.
    
    Example:
        # Two-site Heisenberg model: H = J(X⊗X + Y⊗Y + Z⊗Z)
        sites = [Site(0, PauliHilbertSpace(2)), Site(1, PauliHilbertSpace(2))]
        terms = {
            ('X', 'X'): 1.0,
            ('Y', 'Y'): 1.0, 
            ('Z', 'Z'): 1.0
        }
        hamiltonian = GeneralHamiltonian(sites, terms)
    """
    
    def __init__(self, sites: List[Site], terms: Dict = None):
        """
        Initialize a GeneralHamiltonian.
        
        Args:
            sites (List[Site]): List of sites the Hamiltonian acts on
            terms (dict, optional): Dictionary mapping operator string tuples to coefficients.
                Keys are tuples of operator labels, values are complex coefficients.
                Example: {('X', 'Y'): 0.5, ('Z', 'Z'): -0.3}
        """
        self.sites = sites
        self.terms = terms if terms is not None else {}
        self._validate_terms()
    
    def _validate_terms(self):
        """Validate that all terms have correct structure."""
        for op_string, coeff in self.terms.items():
            if not isinstance(op_string, tuple):
                raise ValueError(f"Term keys must be tuples, got {type(op_string)}")
            if len(op_string) != len(self.sites):
                raise ValueError(f"Operator string length {len(op_string)} doesn't match number of sites {len(self.sites)}")
    
    def __add__(self, other: 'GeneralHamiltonian') -> 'GeneralHamiltonian':
        """Add two Hamiltonians together."""
        if not isinstance(other, GeneralHamiltonian):
            raise TypeError("Can only add GeneralHamiltonian to GeneralHamiltonian")
        
        # Check if sites are compatible
        if len(self.sites) != len(other.sites):
            raise ValueError("Cannot add Hamiltonians with different numbers of sites")
        
        # Check site dimensions match
        for i, (s1, s2) in enumerate(zip(self.sites, other.sites)):
            if s1.dimension != s2.dimension:
                raise ValueError(f"Site {i} dimension mismatch: {s1.dimension} vs {s2.dimension}")
        
        # Combine terms
        new_terms = cp.deepcopy(self.terms)
        for op_string, coeff in other.terms.items():
            if op_string in new_terms:
                new_terms[op_string] += coeff
            else:
                new_terms[op_string] = coeff
        
        return GeneralHamiltonian(self.sites, new_terms)
    
    def __sub__(self, other: 'GeneralHamiltonian') -> 'GeneralHamiltonian':
        """Subtract two Hamiltonians."""
        return self + (-1.0 * other)
    
    def __mul__(self, scalar: Union[float, complex]) -> 'GeneralHamiltonian':
        """Multiply Hamiltonian by a scalar."""
        new_terms = {op_string: scalar * coeff for op_string, coeff in self.terms.items()}
        return GeneralHamiltonian(self.sites, new_terms)
    
    def __rmul__(self, scalar: Union[float, complex]) -> 'GeneralHamiltonian':
        """Right multiplication by scalar."""
        return self * scalar
    
    def matrix(self, site_ops: Dict[type, SiteOperators]) -> np.ndarray:
        """
        Returns the matrix representation of the Hamiltonian.
        
        Args:
            site_ops: Dictionary mapping HilbertSpace types to SiteOperators instances
                     e.g., {PauliHilbertSpace: pauli_ops, SpinHilbertSpace: spin_ops}
                     
        Returns:
            np.ndarray: Matrix representation of the full Hamiltonian
        """
        if not self.terms:
            # Return zero matrix
            total_dim = np.prod([site.dimension for site in self.sites])
            return np.zeros((total_dim, total_dim), dtype=complex)
        
        # Build operator libraries for each site
        libraries = []
        for site in self.sites:
            hilbert_type = type(site.hilbert_space)
            if hilbert_type in site_ops:
                libraries.append(site_ops[hilbert_type])
            else:
                # Fallback: create operators from the site's Hilbert space
                libraries.append(site.create_operators())
        
        # Start with zero matrix
        total_dim = np.prod([site.dimension for site in self.sites])
        result = np.zeros((total_dim, total_dim), dtype=complex)
        
        # Add each term
        for op_string, coeff in self.terms.items():
            term_matrix = self._get_term_matrix(op_string, libraries)
            result += coeff * term_matrix
        
        return result
    
    def _get_term_matrix(self, op_string: tuple, libraries: List[SiteOperators]) -> np.ndarray:
        """Get matrix for a single term."""
        if len(self.sites) == 1:
            # Single site
            op_label = op_string[0]
            if op_label not in libraries[0]:
                raise KeyError(f"Operator '{op_label}' not found in operator library")
            return libraries[0][op_label]
        else:
            # Multi-site: use kron products
            combined_ops = libraries[0]
            for lib in libraries[1:]:
                combined_ops = combined_ops.kron(lib)
            
            op_name = ''.join(op_string)
            if op_name not in combined_ops:
                raise KeyError(f"Combined operator '{op_name}' not found")
            
            return combined_ops[op_name]
    
    def __getitem__(self, key: tuple) -> Union[float, complex]:
        """Get coefficient for an operator string."""
        return self.terms[key]
    
    def __setitem__(self, key: tuple, value: Union[float, complex]):
        """Set coefficient for an operator string."""
        if not isinstance(key, tuple):
            raise TypeError("Key must be tuple of operator strings")
        if len(key) != len(self.sites):
            raise ValueError(f"Operator string length {len(key)} doesn't match number of sites {len(self.sites)}")
        self.terms[key] = value
    
    def __iter__(self):
        """Iterate over coefficient values."""
        return iter(self.terms.values())
    
    def __len__(self):
        """Number of terms in the Hamiltonian."""
        return len(self.terms)
    
    def __contains__(self, key: tuple):
        """Check if operator string is in the Hamiltonian."""
        return key in self.terms
    
    def __str__(self):
        """Pretty printing of the Hamiltonian."""
        if not self.terms:
            return f"GeneralHamiltonian({len(self.sites)} sites): (empty)"
        
        lines = [f"GeneralHamiltonian ({len(self.sites)} sites, {len(self.terms)} terms):"]
        
        # Sort terms by magnitude for nice display
        sorted_terms = sorted(self.terms.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for op_string, coeff in sorted_terms[:10]:  # Show up to 10 terms
            op_str = '⊗'.join(op_string)
            if isinstance(coeff, complex):
                if np.isreal(coeff):
                    coeff_str = f"{coeff.real:+.6f}"
                else:
                    coeff_str = f"{coeff:+.6f}"
            else:
                coeff_str = f"{coeff:+.6f}"
            lines.append(f"  {coeff_str} * {op_str}")
        
        if len(self.terms) > 10:
            lines.append(f"  ... and {len(self.terms) - 10} more terms")
        
        return '\n'.join(lines)
    
    def __repr__(self):
        """String representation for debugging."""
        return f"GeneralHamiltonian(sites={len(self.sites)}, terms={len(self.terms)})"
    def items(self):
        """Iterate over (operator string, coefficient) pairs."""
        return self.terms.items()

# Convenience functions for building common Hamiltonians


def build_ising_hamiltonian(sites: List[Site],
                           J: Union[float, complex] = 1.0,
                           h: Union[float, complex] = 0.0,
                           periodic: bool = False) -> GeneralHamiltonian:
    """
    Build an Ising model Hamiltonian H = -J∑⟨i,j⟩ ZᵢZⱼ - h∑ᵢ Xᵢ.
    
    Args:
        sites (List[Site]): List of sites
        J (float or complex): Coupling strength
        h (float or complex): Transverse field strength
        periodic (bool): Whether to include periodic boundary conditions
        
    Returns:
        GeneralHamiltonian: Ising model Hamiltonian
    """
    n_sites = len(sites)
    terms = {}
    
    # ZZ coupling terms
    for i in range(n_sites - 1):
        op_list = ['I'] * n_sites
        op_list[i] = 'Z'
        op_list[i + 1] = 'Z'
        terms[tuple(op_list)] = -J
    
    if periodic and n_sites > 2:
        op_list = ['I'] * n_sites
        op_list[0] = 'Z'
        op_list[n_sites - 1] = 'Z'
        terms[tuple(op_list)] = -J
    
    # Transverse field terms
    if h != 0:
        for i in range(n_sites):
            op_list = ['I'] * n_sites
            op_list[i] = 'X'
            terms[tuple(op_list)] = -h
    
    return GeneralHamiltonian(sites, terms)


def build_heisenberg_hamiltonian(sites: List[Site], 
                                coupling: Union[float, complex] = 1.0,
                                periodic: bool = False) -> GeneralHamiltonian:
    """
    Build a Heisenberg model Hamiltonian H = J∑⟨i,j⟩ (S^x_i S^x_j + S^y_i S^y_j + S^z_i S^z_j).
    
    Uses appropriate spin operators based on each site's HilbertSpace type:
    - SpinHilbertSpace: Uses 'Sx', 'Sy', 'Sz' (proper spin operators)
    - PauliHilbertSpace: Uses 'X', 'Y', 'Z' (Pauli matrices)
    - FermionHilbertSpace: Not supported (raises error)
    - Generic HilbertSpace: Uses 'X', 'Y', 'Z' (assumes Pauli-like operators)
    
    Args:
        sites (List[Site]): List of sites  
        coupling (float or complex): Coupling strength J
        periodic (bool): Whether to include periodic boundary conditions
        
    Returns:
        GeneralHamiltonian: Heisenberg model Hamiltonian
        
    Raises:
        ValueError: If any site uses FermionHilbertSpace (incompatible with Heisenberg model)
    """
    from .HilbertSpace import SpinHilbertSpace, PauliHilbertSpace, FermionHilbertSpace
    
    n_sites = len(sites)
    terms = {}
    
    # Determine the appropriate operator names for each site
    def get_spin_operators(site):
        """Get the appropriate spin operator names for a site."""
        if isinstance(site.hilbert_space, SpinHilbertSpace):
            return ['Sx', 'Sy', 'Sz']
        elif isinstance(site.hilbert_space, PauliHilbertSpace):
            return ['X', 'Y', 'Z']
        elif isinstance(site.hilbert_space, FermionHilbertSpace):
            raise ValueError(f"Site {site.label} uses FermionHilbertSpace, which is incompatible with Heisenberg model")
        else:
            # Generic HilbertSpace - assume Pauli-like operators
            return ['X', 'Y', 'Z']
    
    # Get operator names for each site
    site_operators = [get_spin_operators(site) for site in sites]
    
    # Determine pairs for nearest-neighbor interactions
    pairs = []
    for i in range(n_sites - 1):
        pairs.append((i, i + 1))
    
    if periodic and n_sites > 2:
        pairs.append((n_sites - 1, 0))
    
    # Add terms for each pair and each spin component (x, y, z)
    for i, j in pairs:
        for component in range(3):  # 0=x, 1=y, 2=z components
            # Create operator string with identity on all sites except i and j
            op_list = ['I'] * n_sites
            op_list[i] = site_operators[i][component]  # Sx/X, Sy/Y, or Sz/Z for site i
            op_list[j] = site_operators[j][component]  # Sx/X, Sy/Y, or Sz/Z for site j
            terms[tuple(op_list)] = coupling
    
    return GeneralHamiltonian(sites, terms)