import numpy as np
from typing import Dict, Optional
from itertools import product
from fractions import Fraction


class HilbertSpace:
    """
    Base class representing a local Hilbert space with a given dimension.
    
    This class serves as the foundation for specific types of Hilbert spaces
    like spin systems or Pauli operator spaces.
    
    Attributes:
        dimension (int): The dimension of the Hilbert space
        name (str): Optional name for the Hilbert space
    """
    
    def __init__(self, dimension: int, name: Optional[str] = None):
        """
        Initialize a Hilbert space.
        
        Args:
            dimension (int): The dimension of the Hilbert space (must be positive)
            name (str, optional): Name for this Hilbert space
        """
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        
        self.dimension = dimension
        self.name = name or f"dim{dimension}"
    
    def build_operators(self) -> Dict[str, np.ndarray]:
        """
        Build default operators for this Hilbert space.
        Base implementation only provides the identity operator.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping operator names to matrices
        """
        return {"I": np.eye(self.dimension)}
    
    def create_operators(self):
        """
        Convenience method to create SiteOperators from this Hilbert space.
        
        Returns:
            SiteOperators: A SiteOperators instance for this space
        """
        from .SiteOperators import SiteOperators
        return SiteOperators(self)
    
    def __repr__(self) -> str:
        return f"HilbertSpace(dimension={self.dimension}, name='{self.name}')"


class PauliHilbertSpace(HilbertSpace):
    """
    Hilbert space for Pauli operators on n qubits.
    
    This class represents a Hilbert space of dimension 2^n and builds
    all possible Pauli operator strings of length n.
    """
    
    # Pauli matrices (reusing from existing PauliString class concept)
    PAULI_MATRICES = {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }
    
    def __init__(self, dimension: int):
        """
        Initialize Pauli Hilbert space.
        
        Args:
            dimension (int): Must be a power of 2 (2^n for n qubits)
        """
        # Check if dimension is a power of 2
        if dimension <= 0 or (dimension & (dimension - 1)) != 0:
            raise ValueError(f"Pauli operators require dimension to be a power of 2, got {dimension}")
        
        self.n_qubits = int(np.log2(dimension))
        super().__init__(dimension, f"Pauli{self.n_qubits}Q")
    
    def build_operators(self) -> Dict[str, np.ndarray]:
        """
        Build all Pauli operators for n qubits.
        
        For n qubits, this generates 4^n operators corresponding to all
        possible Pauli strings (e.g., for 2 qubits: II, IX, IY, IZ, XI, XX, ...).
        
        Returns:
            Dict[str, np.ndarray]: All Pauli operators
        """
        if self.n_qubits == 1:
            return self.PAULI_MATRICES.copy()
        
        # Generate all Pauli strings of length n_qubits
        operators = {}
        pauli_labels = ['I', 'X', 'Y', 'Z']
        
        for pauli_string in product(pauli_labels, repeat=self.n_qubits):
            label = ''.join(pauli_string)
            
            # Compute tensor product of single-qubit Paulis
            op = self.PAULI_MATRICES[pauli_string[0]]
            for i in range(1, self.n_qubits):
                op = np.kron(op, self.PAULI_MATRICES[pauli_string[i]])
            
            operators[label] = op
        
        return operators


class SpinHilbertSpace(HilbertSpace):
    """
    Hilbert space for spin-j systems.
    
    The dimension determines the spin: j = (dimension - 1) / 2.
    Builds standard spin operators Sx, Sy, Sz, S+, S-.
    
    States are ordered as |j,j⟩, |j,j-1⟩, ..., |j,-j⟩.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize spin Hilbert space.
        
        Args:
            dimension (int): Dimension = 2j + 1 where j is the spin
        """
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        
        # Use Fraction for exact arithmetic with half-integer spins
        j_numerator = dimension - 1
        j_denominator = 2
        j_fraction = Fraction(j_numerator, j_denominator)
        
        if j_fraction < 0:
            raise ValueError(f"Invalid dimension {dimension} for spin system. "
                           f"Must be 2j+1 for positive j.")
        
        self.spin = j_fraction
        self.j = float(j_fraction)  # Keep float version for calculations
        super().__init__(dimension, f"Spin{j_fraction}")
    
    def build_operators(self) -> Dict[str, np.ndarray]:
        """
        Build spin operators using the standard ladder operator approach.
        
        First constructs S+ and S- operators, then derives Sx, Sy, Sz.
        
        States are ordered as |j,j⟩, |j,j-1⟩, ..., |j,-j⟩ where:
        - Index 0 corresponds to m = j
        - Index k corresponds to m = j - k
        - Index 2j corresponds to m = -j
        
        Returns:
            Dict[str, np.ndarray]: Spin operators
        """
        operators = {}
        j = self.j
        dim = self.dimension
        
        # Identity
        operators['I'] = np.eye(dim, dtype=complex)
        
        # Sz operator (diagonal)
        # Sz|j,m⟩ = m|j,m⟩ where m = j - k for index k
        sz_diagonal = [j - k for k in range(dim)]
        operators['Sz'] = np.diag(sz_diagonal).astype(complex)
        
        # S+ and S- (raising and lowering operators)
        sp = np.zeros((dim, dim), dtype=complex)
        sm = np.zeros((dim, dim), dtype=complex)
        
        for k in range(dim):
            m = j - k  # Current m quantum number for index k
            
            # S+|j,m⟩ = √[j(j+1) - m(m+1)]|j,m+1⟩
            # This connects index k to index k-1 (if k > 0)
            if k > 0:  # Can raise to m+1 if not at maximum
                coeff_plus = np.sqrt(j * (j + 1) - m * (m + 1))
                sp[k - 1, k] = coeff_plus
            
            # S-|j,m⟩ = √[j(j+1) - m(m-1)]|j,m-1⟩
            # This connects index k to index k+1 (if k < dim-1)
            if k < dim - 1:  # Can lower to m-1 if not at minimum
                coeff_minus = np.sqrt(j * (j + 1) - m * (m - 1))
                sm[k + 1, k] = coeff_minus
        
        operators['S+'] = sp
        operators['S-'] = sm
        
        # Derive Sx and Sy from ladder operators
        # Sx = (S+ + S-) / 2
        # Sy = (S+ - S-) / (2i)
        operators['Sx'] = (sp + sm) / 2
        operators['Sy'] = (sp - sm) / (2j)  # Note: 2i not 2j
        
        return operators


class FermionHilbertSpace(HilbertSpace):
    """
    Hilbert space for fermionic systems (dimension = 2).
    
    Builds creation/annihilation operators and number operator.
    """
    
    def __init__(self):
        """Initialize fermionic Hilbert space (always dimension 2)."""
        super().__init__(2, "Fermion")
    
    def build_operators(self) -> Dict[str, np.ndarray]:
        """
        Build fermionic operators.
        
        Returns:
            Dict[str, np.ndarray]: Fermionic operators (c, c†, n, I)
        """
        return {
            'I': np.array([[1, 0], [0, 1]], dtype=complex),  # Identity
            'c': np.array([[0, 1], [0, 0]], dtype=complex),  # Annihilation
            'cdag': np.array([[0, 0], [1, 0]], dtype=complex),  # Creation  
            'n': np.array([[0, 0], [0, 1]], dtype=complex),  # Number operator
        }