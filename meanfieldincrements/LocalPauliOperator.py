import numpy as np
import meanfieldincrements 
from meanfieldincrements import *

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.array([[1, 0], [0, 1]])


class LocalPauliOperator:
    def __init__(self, string:str, sites:list):
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
        result = I
        
        for char in self.string:
            if char == 'X':
                result = np.kron(result, X)
            elif char == 'Y':
                result = np.kron(result, Y)
            elif char == 'Z':
                result = np.kron(result, Z)
            elif char == 'I':
                result = np.kron(result, I)
            else:
                raise ValueError(f"Unknown Pauli character: {char}")
        
        return result