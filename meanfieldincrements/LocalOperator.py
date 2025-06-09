import numpy as np
from .Site import Site

class LocalOperator:
    def __init__(self, tensor: np.ndarray, sites: list[Site], tensor_format: str = 'matrix'):
        """
        Initialize a local operator.

        Args:
            tensor (np.ndarray): The operator tensor of shape (d,)*2*N, where d is the local Hilbert space dimension.
            sites (list or tuple): The list of site indices (length N) where the operator acts.
        """
        self.tensor = tensor
        self.sites = list(sites)
        self.N = len(sites)
        self._tensor_format = tensor_format 
        if self._tensor_format not in ['matrix', 'tensor']:
            raise ValueError("tensor_format must be either 'matrix' or 'tensor'.")
        if self._tensor_format == 'matrix':
            if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
                raise ValueError("For matrix format, tensor must be a square matrix.")
        elif self._tensor_format == 'tensor':
            if tensor.ndim != 2 * self.N:
                raise ValueError(f"For tensor format, tensor must have shape (d1, d2, ..., dN, d1, d2, ..., dN) where N={self.N}.")
    
    def __repr__(self):
        return f"LocalOperator(sites={self.sites}, shape={self.tensor.shape})"

    def fold(self):
        """
        Reshape self.tensor from matrix to tensor form based on the operator's site dimensions.
        Matrix form: (total_dim, total_dim) -> Tensor form: (d1, d2, ..., dN, d1, d2, ..., dN)
        Returns:
            self: The LocalOperator instance with reshaped tensor.
        """
        if self._tensor_format == 'tensor':
            # Already in tensor form, no need to reshape
            return self

        # Get dimensions of each site
        site_dims = [site.dimension for site in self.sites]
        
        # Tensor form has shape: (d1, d2, ..., dN, d1, d2, ..., dN)
        # First N indices are "output" (bra), last N indices are "input" (ket)
        tensor_shape = site_dims + site_dims
        
        self.tensor = self.tensor.reshape(tensor_shape)
        self._tensor_format = 'tensor'
        return self

    def unfold(self):
        """
        Reshape self.tensor from tensor to matrix form based on the operator's site dimensions.
        Tensor form: (d1, d2, ..., dN, d1, d2, ..., dN) -> Matrix form: (total_dim, total_dim)
        Returns:
            self: The LocalOperator instance with reshaped tensor.
        """
        if self._tensor_format == 'matrix':
            # Already in matrix form, no need to reshape
            return self
        # Get dimensions of each site
        site_dims = [site.dimension for site in self.sites]
        
        # Calculate total dimension of the Hilbert space
        total_dim = np.prod(site_dims)
        
        # Reshape to matrix form
        self.tensor = self.tensor.reshape(total_dim, total_dim)
        self._tensor_format = 'matrix'
        return self

    def trace(self):
        """
        Compute the trace of the local operator tensor.

        Returns:
            float or complex: The trace of the operator.
        """
        return np.trace(self.unfold().tensor)
    
    def partial_trace(self, traced_sites: list) -> 'LocalOperator':
        """
        Compute partial trace over specified sites.
        
        Args:
            traced_sites: List of site indices to trace over
            
        Returns:
            LocalOperator: Reduced operator after partial trace
            
        Example:
            >>> # Trace out site 1 from a two-site operator
            >>> sites = [Site(0, 2), Site(1, 2)]
            >>> op = LocalOperator(tensor, sites)
            >>> reduced_op = op.partial_trace([1])  # Returns single-site operator
        """
        
        for i in traced_sites:
            if i >= self.N or i < 0:
                raise ValueError(f"Invalid site index {i} for partial trace. Operator has {self.N} sites.")

        # Ensure tensor is in folded form
        self.fold()
        
        # Find indices of sites to trace and keep
        traced_indices = []
        kept_sites = []
        
        for i, site in enumerate(self.sites):
            if site.label in traced_sites:
                traced_indices.append(i)
            else:
                kept_sites.append(site)
        
        if len(traced_indices) ==  0:
            return self 
        
        # Contract traced indices using einsum
        N = self.N
        letters = 'abcdefghijklmnopqrstuvwxyz'
        
        if N <= 26:
            # Build einsum string
            input_indices = list(letters[:N]) + list(letters.upper()[:N])

            # Mark traced indices to be contracted
            for idx in traced_indices:
                input_indices[idx + N] = input_indices[idx]  # Same letter for bra/ket
            
            output_indices = []
            for i in range(N):
                if i not in traced_indices:
                    output_indices.append(input_indices[i])
            for i in range(N):
                if i not in traced_indices:
                    output_indices.append(input_indices[i + N])

            einsum_string = ''.join(input_indices) + '->' + ''.join(output_indices)
            reduced_tensor = np.einsum(einsum_string, self.tensor)
            
            return LocalOperator(reduced_tensor, kept_sites, tensor_format='tensor')
        else:
            raise NotImplementedError("Partial trace for >26 sites not implemented")
