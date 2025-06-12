import numpy as np


class PauliString:
    '''
    This module defines the `PauliString` class, which represents a Pauli operator string
    and provides methods for its manipulation and conversion to matrix form.
    
    Classes:
        PauliString: Represents a Pauli operator string with an associated coefficient.
    
    Attributes:
        PAULI_I (np.ndarray): The identity matrix used in the matrix representation.
        PAULI_X (np.ndarray): The Pauli-X matrix used in the matrix representation.
        PAULI_Y (np.ndarray): The Pauli-Y matrix used in the matrix representation.
        PAULI_Z (np.ndarray): The Pauli-Z matrix used in the matrix representation.
    
    Example:
        # Create a PauliString object
        ps = PauliString("XZI", coeff=0.5)
        # Get the matrix representation
        matrix = ps.matrix()
        # Extract a local Pauli string
        local_ps = ps.get_local_string([0, 2])
    '''
    PAULI_X = np.array([[0, 1], [1, 0]])
    PAULI_Y = np.array([[0, -1j], [1j, 0]])
    PAULI_Z = np.array([[1, 0], [0, -1]])
    PAULI_I = np.array([[1, 0], [0, 1]])

    def __init__(self, string: str, coeff=1.0):
        self.string = string
        self.coeff = coeff
    
    def __repr__(self):
        return f"PauliString(string='{self.string}', coeff={self.coeff})"
    
    def __str__(self):
        if self.coeff == 1.0:
            return self.string
        elif self.coeff == -1.0:
            return f"-{self.string}"
        else:
            return f"{self.coeff}*{self.string}"

    def matrix(self):
        """
        Convert the Pauli operator string to a matrix representation.
        
        Returns:
            np.ndarray: The matrix representation of the Pauli operator.
        """
        # Initialize with identity (but we'll replace this with the first operator)
        result = None
        
        for char in self.string:
            if char == 'X':
                pauli_op = self.PAULI_X
            elif char == 'Y':
                pauli_op = self.PAULI_Y
            elif char == 'Z':
                pauli_op = self.PAULI_Z
            elif char == 'I':
                pauli_op = self.PAULI_I
            else:
                raise ValueError(f"Unknown Pauli character: {char}")
            
            if result is None:
                result = pauli_op.copy()
            else:
                result = np.kron(result, pauli_op)
        
        # Handle empty string case
        if result is None:
            result = np.array([[1.0]])
        
        return self.coeff * result
    
    def get_local_string(self, sites: list[int]):
        """
        Get the local Pauli string for the specified sites.
        
        Args:
            sites (list[int]): List of site indices where the Pauli operators act.
        
        Returns:
            PauliString: The local Pauli string for the specified sites.
        """
        if not sites:
            return PauliString("", self.coeff)
        
        # Check bounds
        for site in sites:
            if site < 0 or site >= len(self.string):
                raise IndexError(f"Site index {site} out of range for string '{self.string}'")
        
        local_string = ''.join(self.string[i] for i in sites)
        return PauliString(local_string, self.coeff)
    
    def __eq__(self, other):
        """Check equality based on string and coefficient."""
        if not isinstance(other, PauliString):
            return False
        return self.string == other.string and np.isclose(self.coeff, other.coeff)
    
    def __hash__(self):
        """Hash based on string (for use in dictionaries)."""
        return hash(self.string)
    
    def __len__(self):
        """Return the length of the Pauli string."""
        return len(self.string)
    
    def __mul__(self, scalar):
        """Scale the PauliString by a scalar."""
        return PauliString(self.string, self.coeff * scalar)
    
    def __rmul__(self, scalar):
        """Right multiplication (scalar * PauliString)."""
        return self.__mul__(scalar)
    
    def __neg__(self):
        """Negate the PauliString."""
        return PauliString(self.string, -self.coeff)
    
    def copy(self):
        """Create a copy of the PauliString."""
        return PauliString(self.string, self.coeff)
    
    def conjugate(self):
        """
        Return the Hermitian conjugate of the PauliString.
        
        For Pauli strings, this just conjugates the coefficient since
        all Pauli matrices are Hermitian.
        """
        return PauliString(self.string, np.conj(self.coeff))