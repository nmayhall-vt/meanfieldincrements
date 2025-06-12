import numpy as np
import copy as cp
from typing import List, Union, Dict, Tuple, Optional, Any
from .Site import Site
from .LocalOperator import LocalOperator


class GeneralHamiltonian:
    """
    A general quantum Hamiltonian class for arbitrary operators on sites of any dimension.
    
    The Hamiltonian is represented as:
    H = Σ_i c_i ⊗_j O_{i,j}
    
    where:
    - c_i are complex coefficients
    - O_{i,j} is an operator acting on site j for term i
    - Sites can have arbitrary local dimensions
    - Operators are user-defined through an operator library
    
    Attributes:
        sites (List[Site]): List of sites the Hamiltonian acts on
        operator_library (Dict): Maps operator labels to their matrix representations
        terms (Dict[Tuple[str, ...], complex]): Maps operator strings to coefficients
    
    Examples:
        >>> # Define sites with different dimensions
        >>> sites = [Site(0, 2), Site(1, 3), Site(2, 2)]  # qubit, qutrit, qubit
        >>> 
        >>> # Define operator library
        >>> operators = {
        >>>     "I2": np.eye(2),           # 2x2 identity
        >>>     "I3": np.eye(3),           # 3x3 identity  
        >>>     "sigma_x": pauli_x,        # 2x2 Pauli-X
        >>>     "sigma_z": pauli_z,        # 2x2 Pauli-Z
        >>>     "a": creation_op_3x3,      # 3x3 creation operator
        >>>     "ad": annihilation_op_3x3  # 3x3 annihilation operator
        >>> }
        >>> 
        >>> # Create Hamiltonian
        >>> H = GeneralHamiltonian(sites, operators)
        >>> H.add_term(("sigma_x", "I3", "I2"), 1.0)      # X ⊗ I ⊗ I
        >>> H.add_term(("I2", "a", "sigma_z"), 0.5)       # I ⊗ a ⊗ Z
        >>> 
        >>> # Get matrix representation
        >>> matrix = H.matrix()
    """
    
    def __init__(self, sites: List[Site], operator_library: Dict[str, np.ndarray]):
        """
        Initialize a general Hamiltonian.
        
        Args:
            sites (List[Site]): List of Site objects defining the system
            operator_library (Dict[str, np.ndarray]): Maps operator labels to matrices
                Each matrix must be compatible with at least one site dimension
        """
        self.sites = sites
        self.operator_library = operator_library.copy()
        self.terms = {}  # Maps tuple of operator labels to coefficient
        
        # Validate operator library
        self._validate_operator_library()
    
    def _validate_operator_library(self):
        """Validate that all operators in the library are proper matrices."""
        for label, op in self.operator_library.items():
            if not isinstance(op, np.ndarray):
                raise ValueError(f"Operator '{label}' must be a numpy array")
            if op.ndim != 2:
                raise ValueError(f"Operator '{label}' must be a 2D array (matrix)")
            if op.shape[0] != op.shape[1]:
                raise ValueError(f"Operator '{label}' must be square")
    
    def _validate_term(self, operators: Tuple[str, ...]):
        """
        Validate that a term is compatible with the system.
        
        Args:
            operators (Tuple[str, ...]): Tuple of operator labels
            
        Raises:
            ValueError: If term is invalid
        """
        if len(operators) != len(self.sites):
            raise ValueError(f"Operator string length ({len(operators)}) must match number of sites ({len(self.sites)})")
        
        for i, (op_label, site) in enumerate(zip(operators, self.sites)):
            if op_label not in self.operator_library:
                raise ValueError(f"Unknown operator '{op_label}' at position {i}")
            
            op_matrix = self.operator_library[op_label]
            if op_matrix.shape[0] != site.dimension:
                raise ValueError(f"Operator '{op_label}' has dimension {op_matrix.shape[0]} "
                               f"but site {i} has dimension {site.dimension}")
    
    def add_operator(self, label: str, matrix: np.ndarray):
        """
        Add an operator to the library.
        
        Args:
            label (str): Label for the operator
            matrix (np.ndarray): Matrix representation of the operator
        """
        if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
            raise ValueError("Operator must be a 2D numpy array")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Operator must be square")
        
        self.operator_library[label] = matrix.copy()
    
    def add_term(self, operators: Union[Tuple[str, ...], List[str]], coefficient: Union[float, complex] = 1.0):
        """
        Add a term to the Hamiltonian.
        
        Args:
            operators (Tuple[str, ...] or List[str]): Sequence of operator labels
            coefficient (float or complex): Coefficient for this term
        
        Examples:
            >>> H.add_term(("sigma_x", "I", "sigma_z"), 1.5)
            >>> H.add_term(["a", "ad", "I"], -0.5j)
        """
        # Convert to tuple for hashing
        if isinstance(operators, list):
            operators = tuple(operators)
        elif not isinstance(operators, tuple):
            raise ValueError("Operators must be a tuple or list of strings")
        
        # Validate the term
        self._validate_term(operators)
        
        # Add or update coefficient
        if operators in self.terms:
            self.terms[operators] += coefficient
            # Remove term if coefficient becomes effectively zero
            if abs(self.terms[operators]) < 1e-15:
                del self.terms[operators]
        else:
            if abs(coefficient) >= 1e-15:  # Only add non-zero terms
                self.terms[operators] = coefficient
    
    def remove_term(self, operators: Union[Tuple[str, ...], List[str]]):
        """
        Remove a term from the Hamiltonian.
        
        Args:
            operators (Tuple[str, ...] or List[str]): Operator sequence to remove
        """
        if isinstance(operators, list):
            operators = tuple(operators)
        
        if operators in self.terms:
            del self.terms[operators]
        else:
            raise KeyError(f"Term {operators} not found in Hamiltonian")
    
    def get_coefficient(self, operators: Union[Tuple[str, ...], List[str]]) -> Union[float, complex]:
        """
        Get the coefficient of a specific term.
        
        Args:
            operators (Tuple[str, ...] or List[str]): Operator sequence to look up
            
        Returns:
            float or complex: Coefficient (0 if term not present)
        """
        if isinstance(operators, list):
            operators = tuple(operators)
        return self.terms.get(operators, 0.0)
    
    def matrix(self) -> np.ndarray:
        """
        Convert the Hamiltonian to its full matrix representation.
        
        Returns:
            np.ndarray: Matrix representation of the Hamiltonian
        """
        if not self.terms:
            # Empty Hamiltonian
            total_dim = np.prod([site.dimension for site in self.sites])
            return np.zeros((total_dim, total_dim), dtype=complex)
        
        # Calculate total dimension
        total_dim = np.prod([site.dimension for site in self.sites])
        result = np.zeros((total_dim, total_dim), dtype=complex)
        
        # Add each term
        for operators, coeff in self.terms.items():
            # Build the tensor product for this term
            term_matrix = None
            for op_label in operators:
                op_matrix = self.operator_library[op_label]
                if term_matrix is None:
                    term_matrix = op_matrix.copy()
                else:
                    term_matrix = np.kron(term_matrix, op_matrix)
            
            result += coeff * term_matrix
        
        return result
    
    def to_local_operator(self) -> LocalOperator:
        """
        Convert the Hamiltonian to a LocalOperator.
        
        Returns:
            LocalOperator: LocalOperator representation of the Hamiltonian
        """
        matrix = self.matrix()
        return LocalOperator(matrix, self.sites)
    
    def __add__(self, other: 'GeneralHamiltonian') -> 'GeneralHamiltonian':
        """Add two Hamiltonians together."""
        if not isinstance(other, GeneralHamiltonian):
            return NotImplemented
        
        if self.sites != other.sites:
            raise ValueError("Cannot add Hamiltonians with different sites")
        
        # Merge operator libraries
        merged_library = self.operator_library.copy()
        for label, op in other.operator_library.items():
            if label in merged_library:
                if not np.allclose(merged_library[label], op):
                    raise ValueError(f"Operator '{label}' has different definitions in the two Hamiltonians")
            else:
                merged_library[label] = op
        
        # Create result
        result = GeneralHamiltonian(self.sites, merged_library)
        
        # Add terms from self
        for operators, coeff in self.terms.items():
            result.terms[operators] = coeff
        
        # Add terms from other
        for operators, coeff in other.terms.items():
            if operators in result.terms:
                result.terms[operators] += coeff
                # Remove if coefficient becomes zero
                if abs(result.terms[operators]) < 1e-15:
                    del result.terms[operators]
            else:
                if abs(coeff) >= 1e-15:
                    result.terms[operators] = coeff
        
        return result
    
    def __sub__(self, other: 'GeneralHamiltonian') -> 'GeneralHamiltonian':
        """Subtract two Hamiltonians."""
        return self + (-1.0 * other)
    
    def __mul__(self, scalar: Union[float, complex]) -> 'GeneralHamiltonian':
        """Scale the Hamiltonian by a scalar."""
        result = GeneralHamiltonian(self.sites, self.operator_library)
        
        for operators, coeff in self.terms.items():
            new_coeff = scalar * coeff
            if abs(new_coeff) >= 1e-15:
                result.terms[operators] = new_coeff
        
        return result
    
    def __rmul__(self, scalar: Union[float, complex]) -> 'GeneralHamiltonian':
        """Right multiplication (scalar * Hamiltonian)."""
        return self.__mul__(scalar)
    
    def __neg__(self) -> 'GeneralHamiltonian':
        """Negate the Hamiltonian."""
        return self * (-1.0)
    
    def __len__(self) -> int:
        """Return the number of terms in the Hamiltonian."""
        return len(self.terms)
    
    def __contains__(self, operators: Union[Tuple[str, ...], List[str]]) -> bool:
        """Check if a term is in the Hamiltonian."""
        if isinstance(operators, list):
            operators = tuple(operators)
        return operators in self.terms
    
    def __iter__(self):
        """Iterate over (operators, coefficient) pairs."""
        return iter(self.terms.items())
    
    def copy(self) -> 'GeneralHamiltonian':
        """Create a deep copy of the Hamiltonian."""
        result = GeneralHamiltonian(self.sites, self.operator_library)
        result.terms = cp.deepcopy(self.terms)
        return result
    
    def conjugate(self) -> 'GeneralHamiltonian':
        """
        Return the Hermitian conjugate of the Hamiltonian.
        
        Note: This only conjugates coefficients. For proper Hermitian conjugate,
        the operator library should contain the conjugate operators.
        """
        result = GeneralHamiltonian(self.sites, self.operator_library)
        
        for operators, coeff in self.terms.items():
            result.terms[operators] = np.conj(coeff)
        
        return result
    
    def is_hermitian(self, tolerance: float = 1e-12) -> bool:
        """
        Check if the Hamiltonian is Hermitian by comparing with its conjugate transpose.
        
        Args:
            tolerance (float): Numerical tolerance for comparison
            
        Returns:
            bool: True if Hamiltonian is Hermitian
        """
        matrix = self.matrix()
        return np.allclose(matrix, matrix.conj().T, atol=tolerance)
    
    def simplify(self, tolerance: float = 1e-15) -> 'GeneralHamiltonian':
        """
        Remove terms with coefficients below tolerance.
        
        Args:
            tolerance (float): Minimum absolute coefficient to keep
            
        Returns:
            GeneralHamiltonian: Simplified Hamiltonian
        """
        result = GeneralHamiltonian(self.sites, self.operator_library)
        
        for operators, coeff in self.terms.items():
            if abs(coeff) >= tolerance:
                result.terms[operators] = coeff
        
        return result
    
    def __repr__(self) -> str:
        """Return detailed string representation."""
        if not self.terms:
            return f"GeneralHamiltonian(sites={len(self.sites)}, terms=0)"
        
        terms_str = []
        for operators, coeff in list(self.terms.items())[:3]:  # Show first 3 terms
            if np.isreal(coeff):
                coeff_str = f"{coeff.real:.3g}"
            else:
                coeff_str = f"({coeff.real:.3g}{coeff.imag:+.3g}j)"
            terms_str.append(f"{coeff_str}*{operators}")
        
        if len(self.terms) > 3:
            terms_str.append("...")
        
        return f"GeneralHamiltonian({' + '.join(terms_str)})"
    
    def __str__(self) -> str:
        """Return user-friendly string representation."""
        if not self.terms:
            return "H = 0"
        
        terms_str = []
        for i, (operators, coeff) in enumerate(self.terms.items()):
            # Format coefficient
            if np.isreal(coeff):
                if i == 0:
                    if coeff == 1.0:
                        coeff_str = ""
                    elif coeff == -1.0:
                        coeff_str = "-"
                    else:
                        coeff_str = f"{coeff.real:.3g}"
                else:
                    if coeff == 1.0:
                        coeff_str = " + "
                    elif coeff == -1.0:
                        coeff_str = " - "
                    elif coeff > 0:
                        coeff_str = f" + {coeff.real:.3g}"
                    else:
                        coeff_str = f" - {abs(coeff.real):.3g}"
            else:
                if i == 0:
                    coeff_str = f"({coeff.real:.3g}{coeff.imag:+.3g}j)"
                else:
                    coeff_str = f" + ({coeff.real:.3g}{coeff.imag:+.3g}j)"
            
            # Add operator string
            op_str = "⊗".join(operators)
            if coeff_str.endswith(('+', '-')) or coeff_str == "":
                terms_str.append(f"{coeff_str}{op_str}")
            else:
                terms_str.append(f"{coeff_str}*{op_str}")
        
        return f"H = {''.join(terms_str)}"


class PauliHamiltonian(GeneralHamiltonian):
    """
    Specialized Hamiltonian class for qubits using Pauli operators.
    
    This is a convenience subclass that automatically provides the standard
    Pauli operator library for qubit systems.
    
    Examples:
        >>> # All sites must be qubits
        >>> sites = [Site(0, 2), Site(1, 2), Site(2, 2)]
        >>> H = PauliHamiltonian(sites)
        >>> H.add_term(("X", "I", "Z"), 1.0)   # X ⊗ I ⊗ Z
        >>> H.add_term(("Y", "Y", "I"), 0.5)   # Y ⊗ Y ⊗ I
    """
    
    # Standard Pauli operators
    PAULI_OPERATORS = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }
    
    def __init__(self, sites: List[Site]):
        """
        Initialize a Pauli Hamiltonian.
        
        Args:
            sites (List[Site]): List of Site objects (must all be qubits)
            
        Raises:
            ValueError: If any site is not a qubit
        """
        # Validate that all sites are qubits
        for i, site in enumerate(sites):
            if not site.is_qubit():
                raise ValueError(f"Site {i} (dimension {site.dimension}) is not a qubit. "
                               f"PauliHamiltonian only supports qubits.")
        
        super().__init__(sites, self.PAULI_OPERATORS)
    
    def add_pauli_string(self, pauli_string: str, coefficient: Union[float, complex] = 1.0):
        """
        Add a Pauli string term (convenience method).
        
        Args:
            pauli_string (str): String of Pauli operators (e.g., "XZI", "YYX")
            coefficient (float or complex): Coefficient for this term
        
        Examples:
            >>> H.add_pauli_string("XZI", 1.5)
            >>> H.add_pauli_string("YYX", -0.5j)
        """
        if len(pauli_string) != len(self.sites):
            raise ValueError(f"Pauli string length ({len(pauli_string)}) must match number of sites ({len(self.sites)})")
        
        # Convert string to tuple of operators
        operators = tuple(pauli_string)
        self.add_term(operators, coefficient)
    
    @classmethod
    def from_pauli_strings(cls, sites: List[Site], pauli_dict: Dict[str, Union[float, complex]]) -> 'PauliHamiltonian':
        """
        Create a PauliHamiltonian from a dictionary of Pauli strings.
        
        Args:
            sites (List[Site]): List of qubit sites
            pauli_dict (Dict[str, float/complex]): Maps Pauli strings to coefficients
            
        Returns:
            PauliHamiltonian: Constructed Hamiltonian
            
        Examples:
            >>> sites = [Site(0, 2), Site(1, 2)]
            >>> pauli_terms = {"XI": 1.0, "ZZ": -0.5, "YY": 0.3}
            >>> H = PauliHamiltonian.from_pauli_strings(sites, pauli_terms)
        """
        H = cls(sites)
        for pauli_string, coeff in pauli_dict.items():
            H.add_pauli_string(pauli_string, coeff)
        return H


def create_identity_operators(dimensions: List[int]) -> Dict[str, np.ndarray]:
    """
    Create identity operators for sites with given dimensions.
    
    Args:
        dimensions (List[int]): List of local dimensions
        
    Returns:
        Dict[str, np.ndarray]: Maps "I_d" to d×d identity matrix
    """
    operators = {}
    for d in set(dimensions):
        operators[f"I_{d}"] = np.eye(d, dtype=complex)
    return operators


def create_spin_operators(spin: Union[float, int]) -> Dict[str, np.ndarray]:
    """
    Create spin operators for given spin value.
    
    Args:
        spin (float or int): Spin value (e.g., 0.5 for spin-1/2, 1 for spin-1)
        
    Returns:
        Dict[str, np.ndarray]: Spin operators {S_x, S_y, S_z, S_+, S_-, I}
    """
    from scipy.sparse import csr_matrix
    import scipy.sparse as sp
    
    # Dimension is 2*spin + 1
    dim = int(2 * spin + 1)
    
    # Magnetic quantum numbers: m = -spin, -spin+1, ..., +spin
    m_values = np.arange(-spin, spin + 1)
    
    # Identity
    I = np.eye(dim, dtype=complex)
    
    # S_z operator (diagonal)
    S_z = np.diag(m_values.astype(complex))
    
    # S_+ and S_- operators
    S_plus = np.zeros((dim, dim), dtype=complex)
    S_minus = np.zeros((dim, dim), dtype=complex)
    
    for i, m in enumerate(m_values):
        if i < dim - 1:  # S_+ raises m by 1
            S_plus[i + 1, i] = np.sqrt(spin * (spin + 1) - m * (m + 1))
        if i > 0:  # S_- lowers m by 1  
            S_minus[i - 1, i] = np.sqrt(spin * (spin + 1) - m * (m - 1))
    
    # S_x and S_y from ladder operators
    S_x = 0.5 * (S_plus + S_minus)
    S_y = -0.5j * (S_plus - S_minus)
    
    return {
        'I': I,
        'S_x': S_x,
        'S_y': S_y, 
        'S_z': S_z,
        'S_plus': S_plus,
        'S_minus': S_minus
    }


def create_bosonic_operators(max_occupation: int) -> Dict[str, np.ndarray]:
    """
    Create bosonic creation and annihilation operators.
    
    Args:
        max_occupation (int): Maximum occupation number (Hilbert space truncation)
        
    Returns:
        Dict[str, np.ndarray]: Bosonic operators {a, a_dag, n, I}
    """
    dim = max_occupation + 1
    
    # Creation operator a†
    a_dag = np.zeros((dim, dim), dtype=complex)
    for n in range(dim - 1):
        a_dag[n + 1, n] = np.sqrt(n + 1)
    
    # Annihilation operator a
    a = a_dag.conj().T
    
    # Number operator n = a†a
    n = a_dag @ a
    
    # Identity
    I = np.eye(dim, dtype=complex)
    
    return {
        'I': I,
        'a': a,
        'a_dag': a_dag,
        'n': n
    }


# Convenience functions for common models
def transverse_field_ising_model_general(sites: List[Site], J: float = 1.0, h: float = 1.0) -> PauliHamiltonian:
    """
    Create a Transverse Field Ising Model using the general framework.
    
    H = -J Σ_{<i,j>} Z_i Z_j - h Σ_i X_i
    
    Args:
        sites (List[Site]): List of qubit sites
        J (float): Ising coupling strength
        h (float): Transverse field strength
        
    Returns:
        PauliHamiltonian: TFIM Hamiltonian
    """
    H = PauliHamiltonian(sites)
    n_sites = len(sites)
    
    # Add ZZ interactions (nearest neighbor)
    for i in range(n_sites - 1):
        operators = ['I'] * n_sites
        operators[i] = 'Z'
        operators[i + 1] = 'Z'
        H.add_term(tuple(operators), -J)
    
    # Add X terms
    for i in range(n_sites):
        operators = ['I'] * n_sites
        operators[i] = 'X'
        H.add_term(tuple(operators), -h)
    
    return H


def heisenberg_model_general(sites: List[Site], Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0) -> PauliHamiltonian:
    """
    Create a Heisenberg model using the general framework.
    
    H = Σ_{<i,j>} (Jx X_i X_j + Jy Y_i Y_j + Jz Z_i Z_j)
    
    Args:
        sites (List[Site]): List of qubit sites
        Jx, Jy, Jz (float): Coupling strengths
        
    Returns:
        PauliHamiltonian: Heisenberg Hamiltonian
    """
    H = PauliHamiltonian(sites)
    n_sites = len(sites)
    
    # Add nearest neighbor interactions
    for i in range(n_sites - 1):
        base_operators = ['I'] * n_sites
        
        # XX interaction
        if abs(Jx) > 1e-15:
            operators = base_operators.copy()
            operators[i] = 'X'
            operators[i + 1] = 'X'
            H.add_term(tuple(operators), Jx)
        
        # YY interaction  
        if abs(Jy) > 1e-15:
            operators = base_operators.copy()
            operators[i] = 'Y'
            operators[i + 1] = 'Y'
            H.add_term(tuple(operators), Jy)
        
        # ZZ interaction
        if abs(Jz) > 1e-15:
            operators = base_operators.copy()
            operators[i] = 'Z'
            operators[i + 1] = 'Z'
            H.add_term(tuple(operators), Jz)
    
    return H