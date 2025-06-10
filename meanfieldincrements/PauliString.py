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

    def __init__(self, string:str, coeff = 1.0):
        self.string = string
        self.coeff = coeff
    
    def __repr__(self):
        return f"string={self.string}"

    def matrix(self):
        """
        Convert the Pauli operator string to a matrix representation.
        
        Returns:
            np.ndarray: The matrix representation of the Pauli operator.
        """
        # Initialize with identity
        result = PAULI_I
        
        for char in self.string:
            if char == 'X':
                result = np.kron(result, PAULI_X)
            elif char == 'Y':
                result = np.kron(result, PAULI_Y)
            elif char == 'Z':
                result = np.kron(result, PAULI_Z)
            elif char == 'I':
                result = np.kron(result, PAULI_I)
            else:
                raise ValueError(f"Unknown Pauli character: {char}")
        
        return result
    
    def get_local_string(self, sites:list[int]):
        """
        Get the local Pauli string for the specified sites.
        
        Args:
            sites (list[int]): List of site indices where the Pauli operators act.
        
        Returns:
            PauliString: The local Pauli string for the specified sites.
        """
        local_string = ''.join(self.string[i] for i in sites)
        return PauliString(local_string, self.coeff)