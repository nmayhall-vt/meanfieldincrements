import numpy as np

PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])
PAULI_I = np.array([[1, 0], [0, 1]])


class LocalPauliOperator:
    def __init__(self, string:str):
        self.string = string
        # self.sites = list(sites)
        # self.N = len(sites)
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