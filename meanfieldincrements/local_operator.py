import numpy as np

class Site:
    def __init__(self, label, dimension):
        """
        Represents a lattice site.

        Args:
            label (int): The site index or label.
            dimension (int): The local Hilbert space dimension at this site.
        """
        self.label = int(label)
        self.dimension = int(dimension)

    def __repr__(self):
        return f"Site(label={self.label}, dimension={self.dimension})"

class LocalOperator:
    def __init__(self, tensor: np.ndarray, sites: list[Site]):
        """
        Initialize a local operator.

        Args:
            tensor (np.ndarray): The operator tensor of shape (d,)*2*N, where d is the local Hilbert space dimension.
            sites (list or tuple): The list of site indices (length N) where the operator acts.
        """
        self.tensor = tensor
        self.sites = list(sites)
        self.N = len(sites)
    
    def __repr__(self):
        return f"LocalOperator(sites={self.sites}, shape={self.tensor.shape})"

    def fold(self):
        """
        Reshape self.tensor from matrix to tensor form based on the operator's site dimensions.
        Matrix form: (total_dim, total_dim) -> Tensor form: (d1, d2, ..., dN, d1, d2, ..., dN)
        Returns:
            self: The LocalOperator instance with reshaped tensor.
        """
        # Get dimensions of each site
        site_dims = [site.dimension for site in self.sites]
        
        # Tensor form has shape: (d1, d2, ..., dN, d1, d2, ..., dN)
        # First N indices are "output" (bra), last N indices are "input" (ket)
        tensor_shape = site_dims + site_dims
        
        self.tensor = self.tensor.reshape(tensor_shape)
        return self

    def unfold(self):
        """
        Reshape self.tensor from tensor to matrix form based on the operator's site dimensions.
        Tensor form: (d1, d2, ..., dN, d1, d2, ..., dN) -> Matrix form: (total_dim, total_dim)
        Returns:
            self: The LocalOperator instance with reshaped tensor.
        """
        # Get dimensions of each site
        site_dims = [site.dimension for site in self.sites]
        
        # Calculate total dimension of the Hilbert space
        total_dim = np.prod(site_dims)
        
        # Reshape to matrix form
        self.tensor = self.tensor.reshape(total_dim, total_dim)
        return self