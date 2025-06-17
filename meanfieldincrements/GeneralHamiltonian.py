import numpy as np
import copy as cp
from typing import Dict, List, Tuple, Union, Optional
from itertools import combinations

from .Site import Site
from .LocalTensor import LocalTensor
from .SiteOperators import SiteOperators


class GeneralHamiltonian:
    """
    General Hamiltonian class for linear combinations of tensor products of site operators.
    
    Represents H = ∑_i h_i ⊗_j O_ij where:
    - h_i are coefficients
    - O_ij are operators on site j for term i
    
    The Hamiltonian is stored as a dictionary mapping operator string tuples to coefficients.
    Each operator string tuple contains labels for operators that must be provided by
    SiteOperators libraries.
    
    Example:
        # Two-site Heisenberg model: H = J(X⊗X + Y⊗Y + Z⊗Z)
        terms = {
            ('X', 'X'): 1.0,
            ('Y', 'Y'): 1.0, 
            ('Z', 'Z'): 1.0
        }
        hamiltonian = GeneralHamiltonian(terms)
    """
    
    def __init__(self, terms: Optional[Dict[Tuple[str, ...], Union[float, complex]]] = None):
        """
        Initialize a GeneralHamiltonian.
        
        Args:
            terms (dict, optional): Dictionary mapping operator string tuples to coefficients.
                Keys are tuples of operator labels, values are complex coefficients.
                Example: {('X', 'Y', 'I'): 0.5, ('Z', 'Z', 'Z'): -0.3j}
        """
        self.terms = terms if terms is not None else {}
        self._validate_terms()
    
    def _validate_terms(self):
        """Validate that all terms have consistent structure."""
        if not self.terms:
            return
            
        # Check that all operator strings have the same length
        lengths = [len(op_string) for op_string in self.terms.keys()]
        if len(set(lengths)) > 1:
            raise ValueError(f"Inconsistent operator string lengths: {set(lengths)}")
    
    @property
    def n_sites(self) -> int:
        """Get the number of sites this Hamiltonian acts on."""
        if not self.terms:
            return 0
        return len(next(iter(self.terms.keys())))
    
    @property
    def n_terms(self) -> int:
        """Get the number of terms in the Hamiltonian."""
        return len(self.terms)
    
    def add_term(self, operator_strings: Tuple[str, ...], coefficient: Union[float, complex]):
        """
        Add a term to the Hamiltonian.
        
        Args:
            operator_strings (tuple): Tuple of operator labels, one per site
            coefficient (float or complex): Coefficient for this term
            
        Example:
            >>> ham.add_term(('X', 'Y', 'Z'), 0.5)
        """
        if not isinstance(operator_strings, tuple):
            operator_strings = tuple(operator_strings)
            
        # Check consistency with existing terms
        if self.terms and len(operator_strings) != self.n_sites:
            raise ValueError(f"Operator string length {len(operator_strings)} "
                           f"doesn't match existing terms ({self.n_sites} sites)")
        
        if operator_strings in self.terms:
            self.terms[operator_strings] += coefficient
        else:
            self.terms[operator_strings] = coefficient
    
    def remove_term(self, operator_strings: Tuple[str, ...]):
        """Remove a term from the Hamiltonian."""
        if not isinstance(operator_strings, tuple):
            operator_strings = tuple(operator_strings)
        
        if operator_strings in self.terms:
            del self.terms[operator_strings]
        else:
            raise KeyError(f"Term {operator_strings} not found in Hamiltonian")
    
    def get_coefficient(self, operator_strings: Tuple[str, ...]) -> Union[float, complex]:
        """Get the coefficient for a specific operator string."""
        if not isinstance(operator_strings, tuple):
            operator_strings = tuple(operator_strings)
        return self.terms.get(operator_strings, 0.0)
    
    def get_local_tensor(self, 
                          operator_strings: Tuple[str, ...], 
                          sites: List[Site],
                          operator_libraries: Union[SiteOperators, List[SiteOperators]]) -> LocalTensor:
        """
        Convert an operator string to a LocalTensor using provided operator libraries.
        
        Args:
            operator_strings (tuple): Tuple of operator labels
            sites (list): List of Site objects 
            operator_libraries (SiteOperators or list): Either a single SiteOperators 
                instance (same for all sites) or a list of SiteOperators (one per site)
                
        Returns:
            LocalTensor: The resulting tensor product operator
            
        Example:
            >>> pauli_ops = PauliHilbertSpace(2).create_operators()
            >>> sites = [Site(0, 2), Site(1, 2)]
            >>> local_op = ham.get_local_tensor(('X', 'Y'), sites, pauli_ops)
        """
        if not isinstance(operator_strings, tuple):
            operator_strings = tuple(operator_strings)
            
        if len(operator_strings) != len(sites):
            raise ValueError(f"Operator string length {len(operator_strings)} "
                           f"doesn't match number of sites {len(sites)}")
        
        # Handle single operator library for all sites vs per-site libraries
        if isinstance(operator_libraries, SiteOperators):
            libraries = [operator_libraries] * len(sites)
        else:
            libraries = operator_libraries
            if len(libraries) != len(sites):
                raise ValueError(f"Number of operator libraries {len(libraries)} "
                               f"doesn't match number of sites {len(sites)}")
        
        # Build the tensor product
        if len(sites) == 1:
            # Single site case
            op_label = operator_strings[0]
            if op_label not in libraries[0]:
                raise KeyError(f"Operator '{op_label}' not found in operator library")
            matrix = libraries[0][op_label]
            return LocalTensor(matrix, sites)
        else:
            # Multi-site case - use kron products
            combined_ops = libraries[0]
            for lib in libraries[1:]:
                combined_ops = combined_ops.kron(lib)
            
            # Build the operator name
            op_name = ''.join(operator_strings)
            if op_name not in combined_ops:
                raise KeyError(f"Combined operator '{op_name}' not found. "
                             f"Individual operators: {operator_strings}")
            
            matrix = combined_ops[op_name]
            return LocalTensor(matrix, sites)
    
    def get_term_matrix(self,
                       operator_strings: Tuple[str, ...],
                       sites: List[Site],
                       operator_libraries: Union[SiteOperators, List[SiteOperators]]) -> np.ndarray:
        """
        Get the matrix representation of a single term (without coefficient).
        
        Args:
            operator_strings (tuple): Tuple of operator labels
            sites (list): List of Site objects
            operator_libraries: Operator libraries
            
        Returns:
            np.ndarray: Matrix representation of the operator tensor product
        """
        local_op = self.get_local_tensor(operator_strings, sites, operator_libraries)
        return local_op.unfold().tensor
    
    def to_matrix(self, 
                  sites: List[Site],
                  operator_libraries: Union[SiteOperators, List[SiteOperators]]) -> np.ndarray:
        """
        Convert the full Hamiltonian to a matrix representation.
        
        Args:
            sites (list): List of Site objects
            operator_libraries: Operator libraries
            
        Returns:
            np.ndarray: Matrix representation of the full Hamiltonian
            
        Example:
            >>> sites = [Site(0, 2), Site(1, 2)]
            >>> pauli_ops = PauliHilbertSpace(2).create_operators()
            >>> H_matrix = ham.to_matrix(sites, pauli_ops)
        """
        if not self.terms:
            # Return zero matrix of appropriate size
            total_dim = np.prod([site.dimension for site in sites])
            return np.zeros((total_dim, total_dim), dtype=complex)
        
        # Start with zero matrix
        first_term = next(iter(self.terms.keys()))
        first_matrix = self.get_term_matrix(first_term, sites, operator_libraries)
        result = np.zeros_like(first_matrix, dtype=complex)
        
        # Add all terms
        for op_strings, coeff in self.terms.items():
            term_matrix = self.get_term_matrix(op_strings, sites, operator_libraries)
            result += coeff * term_matrix
        
        return result
    
    def to_local_tensors(self,
                          sites: List[Site],
                          operator_libraries: Union[SiteOperators, List[SiteOperators]]) -> List[LocalTensor]:
        """
        Convert all terms to LocalTensor instances.
        
        Args:
            sites (list): List of Site objects
            operator_libraries: Operator libraries
            
        Returns:
            list: List of (coefficient, LocalTensor) tuples
        """
        result = []
        for op_strings, coeff in self.terms.items():
            local_op = self.get_local_tensor(op_strings, sites, operator_libraries)
            # Scale by coefficient
            scaled_op = local_op.scale(coeff)
            result.append(scaled_op)
        return result
    
    def __add__(self, other: 'GeneralHamiltonian') -> 'GeneralHamiltonian':
        """Add two Hamiltonians together."""
        if not isinstance(other, GeneralHamiltonian):
            raise TypeError("Can only add GeneralHamiltonian to GeneralHamiltonian")
        
        if self.n_sites > 0 and other.n_sites > 0 and self.n_sites != other.n_sites:
            raise ValueError(f"Cannot add Hamiltonians with different numbers of sites: "
                           f"{self.n_sites} vs {other.n_sites}")
        
        new_terms = cp.deepcopy(self.terms)
        for op_strings, coeff in other.terms.items():
            if op_strings in new_terms:
                new_terms[op_strings] += coeff
            else:
                new_terms[op_strings] = coeff
        
        return GeneralHamiltonian(new_terms)
    
    def __sub__(self, other: 'GeneralHamiltonian') -> 'GeneralHamiltonian':
        """Subtract two Hamiltonians."""
        return self + (-1.0 * other)
    
    def __mul__(self, scalar: Union[float, complex]) -> 'GeneralHamiltonian':
        """Multiply Hamiltonian by a scalar."""
        new_terms = {op_strings: scalar * coeff 
                    for op_strings, coeff in self.terms.items()}
        return GeneralHamiltonian(new_terms)
    
    def __rmul__(self, scalar: Union[float, complex]) -> 'GeneralHamiltonian':
        """Right multiplication by scalar."""
        return self * scalar
    
    def __truediv__(self, scalar: Union[float, complex]) -> 'GeneralHamiltonian':
        """Divide Hamiltonian by a scalar."""
        return self * (1.0 / scalar)
    
    def conjugate(self) -> 'GeneralHamiltonian':
        """Return complex conjugate of the Hamiltonian."""
        new_terms = {op_strings: np.conj(coeff) 
                    for op_strings, coeff in self.terms.items()}
        return GeneralHamiltonian(new_terms)
    
    def is_hermitian(self,
                    sites: List[Site],
                    operator_libraries: Union[SiteOperators, List[SiteOperators]],
                    rtol: float = 1e-10) -> bool:
        """
        Check if the Hamiltonian is Hermitian.
        
        Args:
            sites (list): List of Site objects
            operator_libraries: Operator libraries
            rtol (float): Relative tolerance for comparison
            
        Returns:
            bool: True if Hamiltonian is Hermitian
        """
        matrix = self.to_matrix(sites, operator_libraries)
        return np.allclose(matrix, matrix.conj().T, rtol=rtol)
    
    def get_expectation_value(self,
                             state: np.ndarray,
                             sites: List[Site],
                             operator_libraries: Union[SiteOperators, List[SiteOperators]]) -> Union[float, complex]:
        """
        Compute expectation value ⟨ψ|H|ψ⟩ for a given state.
        
        Args:
            state (np.ndarray): State vector (1D) or density matrix (2D)
            sites (list): List of Site objects
            operator_libraries: Operator libraries
            
        Returns:
            float or complex: The expectation value
        """
        H_matrix = self.to_matrix(sites, operator_libraries)
        
        if state.ndim == 1:  # State vector
            return np.real_if_close(np.conj(state) @ H_matrix @ state).item()
        elif state.ndim == 2:  # Density matrix
            return np.real_if_close(np.trace(H_matrix @ state))
        else:
            raise ValueError("State must be 1D vector or 2D density matrix")
    
    def filter_terms(self, filter_func) -> 'GeneralHamiltonian':
        """
        Filter terms based on a function.
        
        Args:
            filter_func: Function that takes (operator_strings, coefficient) and returns bool
            
        Returns:
            GeneralHamiltonian: New Hamiltonian with filtered terms
            
        Example:
            >>> # Keep only terms with non-zero coefficients
            >>> filtered = ham.filter_terms(lambda op_str, coeff: abs(coeff) > 1e-10)
        """
        new_terms = {op_str: coeff for op_str, coeff in self.terms.items() 
                    if filter_func(op_str, coeff)}
        return GeneralHamiltonian(new_terms)
    
    def get_terms_by_weight(self, n_body: int) -> 'GeneralHamiltonian':
        """
        Get all terms with a specific number of non-identity operators.
        
        Args:
            n_body (int): Number of non-identity operators
            
        Returns:
            GeneralHamiltonian: New Hamiltonian with only n-body terms
        """
        def is_n_body(op_strings, coeff):
            # Count non-identity operators (assuming 'I' is identity)
            non_identity_count = sum(1 for op in op_strings if op != 'I')
            return non_identity_count == n_body
        
        return self.filter_terms(is_n_body)
    
    def simplify(self, rtol: float = 1e-12) -> 'GeneralHamiltonian':
        """
        Remove terms with coefficients smaller than rtol.
        
        Args:
            rtol (float): Tolerance for removing small terms
            
        Returns:
            GeneralHamiltonian: Simplified Hamiltonian
        """
        return self.filter_terms(lambda op_str, coeff: abs(coeff) > rtol)
    
    def __repr__(self) -> str:
        return f"GeneralHamiltonian(n_sites={self.n_sites}, n_terms={self.n_terms})"
    
    def __str__(self) -> str:
        if not self.terms:
            return "GeneralHamiltonian: (empty)"
        
        lines = [f"GeneralHamiltonian ({self.n_sites} sites, {self.n_terms} terms):"]
        
        # Sort terms by magnitude of coefficient for nice display
        sorted_terms = sorted(self.terms.items(), 
                            key=lambda x: abs(x[1]), reverse=True)
        
        for op_strings, coeff in sorted_terms[:10]:  # Show up to 10 terms
            op_str = '⊗'.join(op_strings)
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


# Convenience functions for building common Hamiltonians

def build_heisenberg_hamiltonian(n_sites: int, 
                                coupling: Union[float, complex] = 1.0,
                                periodic: bool = False) -> GeneralHamiltonian:
    """
    Build a Heisenberg model Hamiltonian H = J∑⟨i,j⟩ (XᵢXⱼ + YᵢYⱼ + ZᵢZⱼ).
    
    Args:
        n_sites (int): Number of sites
        coupling (float or complex): Coupling strength J
        periodic (bool): Whether to include periodic boundary conditions
        
    Returns:
        GeneralHamiltonian: Heisenberg model Hamiltonian
    """
    ham = GeneralHamiltonian()
    
    # Determine pairs
    pairs = []
    for i in range(n_sites - 1):
        pairs.append((i, i + 1))
    
    if periodic and n_sites > 2:
        pairs.append((n_sites - 1, 0))
    
    # Add terms for each pair and each Pauli component
    for i, j in pairs:
        for pauli in ['X', 'Y', 'Z']:
            # Create operator string with identity on all sites except i and j
            op_list = ['I'] * n_sites
            op_list[i] = pauli
            op_list[j] = pauli
            ham.add_term(tuple(op_list), coupling)
    
    return ham


def build_ising_hamiltonian(n_sites: int,
                           J: Union[float, complex] = 1.0,
                           h: Union[float, complex] = 0.0,
                           periodic: bool = False) -> GeneralHamiltonian:
    """
    Build an Ising model Hamiltonian H = -J∑⟨i,j⟩ ZᵢZⱼ - h∑ᵢ Xᵢ.
    
    Args:
        n_sites (int): Number of sites
        J (float or complex): Coupling strength
        h (float or complex): Transverse field strength
        periodic (bool): Whether to include periodic boundary conditions
        
    Returns:
        GeneralHamiltonian: Ising model Hamiltonian
    """
    ham = GeneralHamiltonian()
    
    # ZZ coupling terms
    for i in range(n_sites - 1):
        op_list = ['I'] * n_sites
        op_list[i] = 'Z'
        op_list[i + 1] = 'Z'
        ham.add_term(tuple(op_list), -J)
    
    if periodic and n_sites > 2:
        op_list = ['I'] * n_sites
        op_list[0] = 'Z'
        op_list[n_sites - 1] = 'Z'
        ham.add_term(tuple(op_list), -J)
    
    # Transverse field terms
    if h != 0:
        for i in range(n_sites):
            op_list = ['I'] * n_sites
            op_list[i] = 'X'
            ham.add_term(tuple(op_list), -h)
    
    return ham


def from_pauli_strings(pauli_strings: List[str], 
                      coefficients: List[Union[float, complex]]) -> GeneralHamiltonian:
    """
    Create a GeneralHamiltonian from a list of Pauli strings and coefficients.
    
    Args:
        pauli_strings (list): List of Pauli strings (e.g., ['XYZ', 'IXI'])
        coefficients (list): List of coefficients
        
    Returns:
        GeneralHamiltonian: Hamiltonian constructed from Pauli strings
        
    Example:
        >>> ham = from_pauli_strings(['XYZ', 'IXI', 'ZZI'], [0.5, -0.3, 0.1])
    """
    if len(pauli_strings) != len(coefficients):
        raise ValueError("Number of Pauli strings must match number of coefficients")
    
    ham = GeneralHamiltonian()
    for pauli_str, coeff in zip(pauli_strings, coefficients):
        op_tuple = tuple(pauli_str)
        ham.add_term(op_tuple, coeff)
    
    return ham