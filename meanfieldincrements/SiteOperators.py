import numpy as np
from typing import Dict, Optional, Union
from .HilbertSpace import HilbertSpace


class SiteOperators:
    """
    Container for matrix representations of local operators on a HilbertSpace.
    
    This class stores a dictionary mapping string representations to numerical
    (matrix) representations of operators. It supports tensor products with
    other SiteOperators to create operators on composite Hilbert spaces.
    
    Attributes:
        hilbert_space (HilbertSpace): The Hilbert space these operators act on
        operators (Dict[str, np.ndarray]): Dictionary of operators
    """
    
    def __init__(self, hilbert_space: HilbertSpace, operators: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize SiteOperators.
        
        Args:
            hilbert_space (HilbertSpace): The Hilbert space for these operators
            operators (Dict[str, np.ndarray], optional): Pre-defined operators.
                If None, uses hilbert_space.build_operators()
        """
        self.hilbert_space = hilbert_space
        self.operators = operators if operators is not None else hilbert_space.build_operators()
        
        # Validate all operators have correct dimensions
        for name, op in self.operators.items():
            self._validate_operator(name, op)
    
    def _validate_operator(self, name: str, operator: np.ndarray) -> None:
        """Validate that an operator has the correct dimensions."""
        expected_shape = (self.hilbert_space.dimension, self.hilbert_space.dimension)
        if operator.shape != expected_shape:
            raise ValueError(f"Operator '{name}' has shape {operator.shape}, "
                           f"expected {expected_shape} for dimension {self.hilbert_space.dimension}")
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Get an operator by name."""
        return self.operators[key]
    
    def __setitem__(self, key: str, value: np.ndarray) -> None:
        """Set an operator, with validation."""
        self._validate_operator(key, value)
        self.operators[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if an operator exists."""
        return key in self.operators
    
    def keys(self):
        """Get operator names."""
        return self.operators.keys()
    
    def items(self):
        """Get operator name-matrix pairs."""
        return self.operators.items()
    
    def values(self):
        """Get operator matrices."""
        return self.operators.values()
    
    def kron(self, other: 'SiteOperators') -> 'SiteOperators':
        """
        Tensor product with another SiteOperators.
        
        Creates a new SiteOperators on a composite Hilbert space where:
        - The dimension is the product of the two dimensions
        - Operator names are concatenated
        - Operator matrices are Kronecker products
        
        Args:
            other (SiteOperators): The other SiteOperators to tensor with
            
        Returns:
            SiteOperators: New SiteOperators on the composite space
            
        Example:
            >>> spin_ops = SiteOperators(SpinHilbertSpace(2))
            >>> pauli_ops = SiteOperators(PauliHilbertSpace(2))  
            >>> combined = spin_ops.kron(pauli_ops)
            >>> print(combined.keys())  # ['ISx', 'ISy', 'ISz', 'XSx', 'XSy', ...]
        """
        # Create new composite Hilbert space
        new_dim = self.hilbert_space.dimension * other.hilbert_space.dimension
        new_name = f"{self.hilbert_space.name}⊗{other.hilbert_space.name}"
        new_hilbert_space = HilbertSpace(new_dim, new_name)
        
        # Combine all pairs of operators
        new_operators = {}
        for name1, op1 in self.operators.items():
            for name2, op2 in other.operators.items():
                new_name = name1 + name2
                new_op = np.kron(op1, op2)
                new_operators[new_name] = new_op
        
        return SiteOperators(new_hilbert_space, new_operators)
    
    def add_operator(self, name: str, matrix: np.ndarray) -> None:
        """
        Add a new operator.
        
        Args:
            name (str): Name for the operator
            matrix (np.ndarray): The operator matrix
        """
        self[name] = matrix
    
    def get_commutator(self, op1_name: str, op2_name: str) -> np.ndarray:
        """
        Compute commutator [A, B] = AB - BA.
        
        Args:
            op1_name (str): Name of first operator
            op2_name (str): Name of second operator
            
        Returns:
            np.ndarray: The commutator matrix
        """
        A = self[op1_name]
        B = self[op2_name]
        return A @ B - B @ A
    
    def get_anticommutator(self, op1_name: str, op2_name: str) -> np.ndarray:
        """
        Compute anticommutator {A, B} = AB + BA.
        
        Args:
            op1_name (str): Name of first operator
            op2_name (str): Name of second operator
            
        Returns:
            np.ndarray: The anticommutator matrix
        """
        A = self[op1_name]
        B = self[op2_name]
        return A @ B + B @ A
    
    def get_expectation_value(self, operator_name: str, state: np.ndarray) -> Union[float, complex]:
        """
        Compute expectation value ⟨ψ|O|ψ⟩ for a given state.
        
        Args:
            operator_name (str): Name of the operator
            state (np.ndarray): State vector (1D) or density matrix (2D)
            
        Returns:
            float or complex: The expectation value
        """
        op = self[operator_name]
        
        if state.ndim == 1:  # State vector
            return np.real_if_close(np.conj(state) @ op @ state)
        elif state.ndim == 2:  # Density matrix
            return np.real_if_close(np.trace(op @ state))
        else:
            raise ValueError("State must be 1D vector or 2D density matrix")
    
    def get_variance(self, operator_name: str, state: np.ndarray) -> Union[float, complex]:
        """
        Compute variance ⟨O²⟩ - ⟨O⟩² for a given state.
        
        Args:
            operator_name (str): Name of the operator
            state (np.ndarray): State vector (1D) or density matrix (2D)
            
        Returns:
            float or complex: The variance
        """
        op = self[operator_name]
        op_squared = op @ op
        
        exp_val = self.get_expectation_value(operator_name, state)
        exp_val_squared = np.real_if_close(exp_val * np.conj(exp_val))
        
        if state.ndim == 1:  # State vector
            exp_op_squared = np.real_if_close(np.conj(state) @ op_squared @ state)
        else:  # Density matrix
            exp_op_squared = np.real_if_close(np.trace(op_squared @ state))
        
        return exp_op_squared - exp_val_squared
    
    def is_hermitian(self, operator_name: str, rtol: float = 1e-10) -> bool:
        """
        Check if an operator is Hermitian.
        
        Args:
            operator_name (str): Name of the operator
            rtol (float): Relative tolerance for comparison
            
        Returns:
            bool: True if operator is Hermitian
        """
        op = self[operator_name]
        return np.allclose(op, op.conj().T, rtol=rtol)
    
    def is_unitary(self, operator_name: str, rtol: float = 1e-10) -> bool:
        """
        Check if an operator is unitary.
        
        Args:
            operator_name (str): Name of the operator
            rtol (float): Relative tolerance for comparison
            
        Returns:
            bool: True if operator is unitary
        """
        op = self[operator_name]
        identity = np.eye(op.shape[0])
        return np.allclose(op @ op.conj().T, identity, rtol=rtol)
    
    def get_eigenvalues(self, operator_name: str) -> np.ndarray:
        """
        Get eigenvalues of an operator.
        
        Args:
            operator_name (str): Name of the operator
            
        Returns:
            np.ndarray: Array of eigenvalues
        """
        op = self[operator_name]
        if self.is_hermitian(operator_name):
            return np.linalg.eigvals(op).real
        else:
            return np.linalg.eigvals(op)
    
    def __repr__(self) -> str:
        op_names = list(self.operators.keys())
        return f"SiteOperators(dim={self.hilbert_space.dimension}, operators={op_names})"
    
    def __str__(self) -> str:
        lines = [f"SiteOperators on {self.hilbert_space}"]
        lines.append(f"  {len(self.operators)} operators:")
        
        # Group operators by type if possible
        op_names = sorted(self.operators.keys())
        if len(op_names) <= 10:
            lines.append(f"    {op_names}")
        else:
            lines.append(f"    {op_names[:5]} ... {op_names[-3:]}")
        
        return "\n".join(lines)