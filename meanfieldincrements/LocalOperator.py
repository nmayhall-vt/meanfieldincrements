import numpy as np
import copy as cp
from .Site import Site
from typing import List, Union, Optional

class LocalOperator:
    def __init__(self, tensor: np.ndarray, sites: List[Site], tensor_format: str = 'matrix'):
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


        site_dims = [site.dimension for site in sites]
        total_dim = np.prod(site_dims)
        if np.prod(self.tensor.shape) != total_dim ** 2:
            raise ValueError(f"Tensor shape {tensor.shape} does not match the expected shape for sites with dimensions {site_dims}.")
        
        if tensor_format == 'matrix':
            if tensor.shape != (total_dim, total_dim):
                raise ValueError(f"Matrix shape {tensor.shape} incompatible with sites dimensions {site_dims}")
        elif tensor_format == 'tensor':
            expected_shape = tuple(site_dims + site_dims)
            if tensor.shape != expected_shape:
                raise ValueError(f"Tensor shape {tensor.shape} incompatible with expected {expected_shape}")


    def __repr__(self):
        return f"LocalOperator(sites={self.sites}, shape={self.tensor.shape})"

    def __str__(self):
        out = "OP: "
        for si in self.sites:
            out += str(si.label) + " "
        out += str(self.tensor.shape)    
        return out
    
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
    
    def partial_trace(self, traced_sites: List[int]) -> 'LocalOperator':
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
            if i not in [i.label for i in self.sites]:
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


    def compute_nbody_marginal(self, sites:List[Site]):
        """
        Compute the 2-body marginal operator for the specified pair of sites.
        This method calculates the reduced density matrix for the two given sites
        by tracing out the environment (all other sites) and constructs a 
        LocalOperator representing the 2-body marginal.

        Args:
            si (Site): The first site for which to compute the 2-body marginal.
            sj (Site): The second site for which to compute the 2-body marginal.

        Returns:
            LocalOperator: The 2-body marginal operator for the specified sites.
        """
        # Ensure tensor is in folded form
        self.fold()
        
        env = [s.label for s in self.sites if s not in sites]
        rho_ij = self.partial_trace(env)
        return LocalOperator(rho_ij.tensor, sites, tensor_format='tensor')

    def compute_2body_cumulant(self, si:Site, sj:Site):
        """
        Compute the 2-body cumulant for a pair of sites.
        This function calculates the 2-body cumulant operator for two sites, `si` and `sj`.
        The cumulant is derived from the 2-body marginal density matrix and the 
        1-body reduced density matrices of the individual sites.
        Args:
            si (Site): The first site for which the 2-body cumulant is computed.
            sj (Site): The second site for which the 2-body cumulant is computed.
        Returns:
            LocalOperator: The 2-body cumulant operator represented as a `LocalOperator` 
            object in matrix format.
        """
        rho_ij = self.compute_nbody_marginal([si, sj])
        rho_j = rho_ij.partial_trace([si.label]).unfold()
        rho_i = rho_ij.partial_trace([sj.label]).unfold()
        
        lambda_ij = rho_ij.unfold().tensor - np.kron(rho_i.tensor, rho_j.tensor)
        return LocalOperator(lambda_ij, [si, sj], tensor_format='matrix')

    def compute_3body_cumulant(self, si:Site, sj:Site, sk:Site):
        """
        Compute the 2-body cumulant for a pair of sites.
        This function calculates the 2-body cumulant operator for two sites, `si` and `sj`.
        The cumulant is derived from the 2-body marginal density matrix and the 
        1-body reduced density matrices of the individual sites.
        Args:
            si (Site): The first site for which the 2-body cumulant is computed.
            sj (Site): The second site for which the 2-body cumulant is computed.
        Returns:
            LocalOperator: The 2-body cumulant operator represented as a `LocalOperator` 
            object in matrix format.
        """
        r_ijk = self.compute_nbody_marginal([si, sj, sk]).fold()
        l_ij = r_ijk.compute_2body_cumulant(si, sj).fold()
        l_ik = r_ijk.compute_2body_cumulant(si, sk).fold()
        l_jk = r_ijk.compute_2body_cumulant(sj, sk).fold()

        r_i = r_ijk.compute_nbody_marginal([si]).fold()
        r_j = r_ijk.compute_nbody_marginal([sj]).fold()
        r_k = r_ijk.compute_nbody_marginal([sk]).fold()

        l = r_ijk.tensor.copy()

        l -= np.einsum('ijIJ, kK->ijkIJK', l_ij.tensor, r_k.tensor)
        l -= np.einsum('ikIK, jJ->ijkIJK', l_ik.tensor, r_j.tensor)
        l -= np.einsum('jkJK, iI->ijkIJK', l_jk.tensor, r_i.tensor)
        l -= np.einsum('iI,jJ,kK->ijkIJK', r_i.tensor, r_j.tensor, r_k.tensor)

        return LocalOperator(l, [si, sj, sk], tensor_format='tensor')
    
    def __add__(self, other: 'LocalOperator') -> 'LocalOperator':
        """
        Add two LocalOperator instances together.

        Args:
            other (LocalOperator): The other LocalOperator to add.

        Returns:
            LocalOperator: A new LocalOperator representing the sum.

        Raises:
            ValueError: If the operators act on different sites or have incompatible shapes.
        """
        if self.sites != other.sites:
            raise ValueError("Cannot add operators acting on different sites.")
        # if self.tensor.shape != other.tensor.shape:
        #     raise ValueError("Cannot add operators with different tensor shapes.")
        # if self._tensor_format != other._tensor_format:
        #     raise ValueError("Cannot add operators with different tensor formats.")

        new_tensor = self.tensor + other.tensor
        return LocalOperator(new_tensor, self.sites, tensor_format=self._tensor_format)
    
    def scale(self, scalar: Union[float, complex]) -> 'LocalOperator':
        """
        Scale the operator by a scalar.

        Args:
            scalar (float or complex): The scalar to multiply the operator by.

        Returns:
            LocalOperator: A new LocalOperator scaled by the given scalar.
        """
        scaled_tensor = self.tensor * scalar
        return LocalOperator(scaled_tensor, self.sites, tensor_format=self._tensor_format)