import numpy as np
from typing import List, Union, Dict
from .Site import Site
from .LocalTensor import LocalTensor
from .HilbertSpace import HilbertSpace
from .SiteOperators import SiteOperators
from .Marginal import Marginal


class FactorizedMarginal(Marginal):
    """
    Factorized representation of a marginal density matrix.
    
    Instead of storing the density matrix ρ directly, stores a factor A such that:
    ρ = A*A† / tr(A*A†)
    
    This can be more memory efficient for low-rank density matrices and 
    naturally ensures positive semidefiniteness.
    """
    
    def __init__(self, factor_A: np.ndarray, sites: List[Site], tensor_format: str = 'matrix'):
        """
        Initialize a FactorizedMarginal.
        
        Args:
            factor_A (np.ndarray): The factor A such that ρ = A*A† / tr(A*A†)
                For matrix format: shape should be (total_dim, rank)
                For tensor format: shape should be (d1, d2, ..., dN, rank)
            sites (List[Site]): Sites this marginal acts on
            tensor_format (str): Format of the factor ('matrix' or 'tensor')
        """
        self.sites = sites
        self._factor_A = factor_A
        self._tensor_format = tensor_format
        self._cached_rho = None  # Cache for the computed density matrix
        self._cache_valid = True
        
        # Validate factor dimensions
        self._validate_factor()
    
    def _validate_factor(self):
        """Validate that the factor has correct dimensions."""
        site_dims = [site.dimension for site in self.sites]
        total_dim = np.prod(site_dims)
        
        if self._tensor_format == 'matrix':
            if self._factor_A.ndim != 2:
                raise ValueError("For matrix format, factor must be 2D")
            if self._factor_A.shape[0] != total_dim:
                raise ValueError(f"Factor first dimension {self._factor_A.shape[0]} doesn't match total dimension {total_dim}")
        elif self._tensor_format == 'tensor':
            expected_shape = tuple(site_dims) + (self._factor_A.shape[-1],)
            if self._factor_A.shape != expected_shape:
                raise ValueError(f"Factor shape {self._factor_A.shape} doesn't match expected {expected_shape}")
        else:
            raise ValueError("tensor_format must be 'matrix' or 'tensor'")
    
    @property
    def factor_A(self) -> np.ndarray:
        """Get the factor A."""
        return self._factor_A
    
    @factor_A.setter
    def factor_A(self, value: np.ndarray):
        """Set the factor A and invalidate cache."""
        self._factor_A = value
        self._cache_valid = False
        self._cached_rho = None
        self._validate_factor()
    
    @property
    def rank(self) -> int:
        """Get the rank of the factorization."""
        return self._factor_A.shape[-1]
    
    def _compute_density_matrix(self) -> np.ndarray:
        """Compute ρ = A*A† / tr(A*A†) from the factor."""
        if self._cache_valid and self._cached_rho is not None:
            return self._cached_rho
        
        A = self._factor_A
        
        if self._tensor_format == 'tensor':
            # Reshape to matrix form for computation
            site_dims = [site.dimension for site in self.sites]
            total_dim = np.prod(site_dims)
            A_mat = A.reshape(total_dim, -1)
        else:
            A_mat = A
        
        # Compute ρ = A*A†
        rho = A_mat @ A_mat.conj().T
        
        # Normalize: ρ = A*A† / tr(A*A†)
        trace_val = np.trace(rho)
        if abs(trace_val) < 1e-14:
            raise ValueError("Factor A leads to zero trace - invalid density matrix")
        rho = rho / trace_val
        
        # Cache the result
        self._cached_rho = rho
        self._cache_valid = True
        
        return rho
    
    @property
    def tensor(self) -> np.ndarray:
        """Get the density matrix tensor, computing from factor if needed."""
        rho = self._compute_density_matrix()
        
        if self._tensor_format == 'tensor':
            # Reshape to tensor format
            site_dims = [site.dimension for site in self.sites]
            tensor_shape = site_dims + site_dims
            return rho.reshape(tensor_shape)
        else:
            return rho
    
    @tensor.setter
    def tensor(self, value: np.ndarray):
        """Set the tensor by computing a new factorization."""
        # This is tricky - we need to factorize the given density matrix
        # For simplicity, we'll use eigendecomposition
        if value.ndim == len(self.sites) * 2:  # Tensor format
            site_dims = [site.dimension for site in self.sites]
            total_dim = np.prod(site_dims)
            rho_mat = value.reshape(total_dim, total_dim)
        else:  # Matrix format
            rho_mat = value
        
        # Eigendecomposition: ρ = U @ diag(λ) @ U†
        eigenvals, eigenvecs = np.linalg.eigh(rho_mat)
        
        # Keep only positive eigenvalues (within tolerance)
        pos_mask = eigenvals > 1e-12
        pos_eigenvals = eigenvals[pos_mask]
        pos_eigenvecs = eigenvecs[:, pos_mask]
        
        # Factor: A = U @ diag(√λ)
        sqrt_eigenvals = np.sqrt(pos_eigenvals)
        new_factor = pos_eigenvecs @ np.diag(sqrt_eigenvals)
        
        if self._tensor_format == 'tensor':
            # Reshape to tensor format
            site_dims = [site.dimension for site in self.sites]
            new_factor = new_factor.reshape(tuple(site_dims) + (new_factor.shape[1],))
        
        self._factor_A = new_factor
        self._cache_valid = False
        self._cached_rho = None
    
    def fold(self):
        """Convert to tensor format."""
        if self._tensor_format == 'matrix':
            site_dims = [site.dimension for site in self.sites]
            tensor_shape = tuple(site_dims) + (self.rank,)
            self._factor_A = self._factor_A.reshape(tensor_shape)
            self._tensor_format = 'tensor'
            self._cache_valid = False
        return self
    
    def unfold(self):
        """Convert to matrix format."""
        if self._tensor_format == 'tensor':
            site_dims = [site.dimension for site in self.sites]
            total_dim = np.prod(site_dims)
            self._factor_A = self._factor_A.reshape(total_dim, -1)
            self._tensor_format = 'matrix'
            self._cache_valid = False
        return self
    
    def trace(self) -> Union[float, complex]:
        """Compute trace of the density matrix (should always be 1)."""
        return 1.0  # By construction, tr(ρ) = 1
    
    def partial_trace(self, traced_sites: List[int]) -> 'Marginal':
        """
        Compute partial trace over specified sites.
        
        Args:
            traced_sites (List[Site]): Sites to trace over
            
        Returns:
            Marginal: Reduced marginal after partial trace
        """
        # For partial trace of factorized form, we need to:
        # 1. Compute the full density matrix
        # 2. Perform partial trace
        # 3. Factorize the result
        
        # Convert Site objects to site labels for LocalTensor.partial_trace
        # traced_labels = [site.label for site in traced_sites]
        
        # Get the full density matrix as a LocalTensor
        rho_full = self._compute_density_matrix()
        remaining_sites = []
        for si in self.sites:
            if si.label not in traced_sites:
                remaining_sites.append(si)
        # remaining_sites = [site for site in self.sites if site not in traced_sites]
        
        if not remaining_sites:
            # Tracing all sites - return scalar
            scalar_result = np.trace(rho_full)
            # Return a 0-site FactorizedMarginal (edge case)
            factor = np.array([[np.sqrt(scalar_result)]])
            return FactorizedMarginal(factor, [], 'matrix')
        
        # Create LocalTensor and perform partial trace
        local_op = LocalTensor(rho_full, self.sites, 'matrix')
        reduced_local = local_op.partial_trace(traced_sites)
        
        # Create new FactorizedMarginal from the reduced density matrix
        return FactorizedMarginal.from_density_matrix(reduced_local.tensor, remaining_sites)
    
    @classmethod
    def from_Marginal(cls, rho: 'Marginal') -> 'FactorizedMarginal':
        """
        Create a FactorizedMarginal from an existing Marginal.
        
        Args:
            rho (Marginal): The marginal density matrix
            
        Returns:
            FactorizedMarginal: New factorized marginal
        """
        if not isinstance(rho, Marginal):
            raise TypeError("Input must be a Marginal instance")
        
        return cls.from_density_matrix(rho.tensor, rho.sites, tensor_format=rho._tensor_format)

    @classmethod
    def from_density_matrix(cls, rho: np.ndarray, sites: List[Site], 
                           tensor_format: str = 'matrix') -> 'FactorizedMarginal':
        """
        Create a FactorizedMarginal from a density matrix.
        
        Args:
            rho (np.ndarray): The density matrix
            sites (List[Site]): Sites this marginal acts on
            tensor_format (str): Format of the result
            
        Returns:
            FactorizedMarginal: New factorized marginal
        """
        # Handle tensor format input
        if tensor_format == 'tensor' and rho.ndim == len(sites) * 2:
            site_dims = [site.dimension for site in sites]
            total_dim = np.prod(site_dims)
            rho_mat = rho.reshape(total_dim, total_dim)
        else:
            rho_mat = rho
        
        
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(rho_mat)
        
        # Keep positive eigenvalues
        pos_mask = eigenvals > 1e-12
        pos_eigenvals = eigenvals[pos_mask]
        pos_eigenvecs = eigenvecs[:, pos_mask]
        
        # Factor: A = U @ diag(√λ)
        sqrt_eigenvals = np.sqrt(pos_eigenvals)
        factor = pos_eigenvecs @ np.diag(sqrt_eigenvals)
        
        # Create instance
        result = cls.__new__(cls)
        result.sites = sites
        result._factor_A = factor
        result._tensor_format = 'matrix'
        result._cached_rho = None
        result._cache_valid = False
        
        # Convert to desired format
        if tensor_format == 'tensor':
            result.fold()
        
        return result
    
    @classmethod
    def from_pure_state(cls, psi: np.ndarray, sites: List[Site], 
                       tensor_format: str = 'matrix') -> 'FactorizedMarginal':
        """
        Create a FactorizedMarginal from a pure state |ψ⟩.
        
        Args:
            psi (np.ndarray): The state vector
            sites (List[Site]): Sites this marginal acts on
            tensor_format (str): Format of the result
            
        Returns:
            FactorizedMarginal: New factorized marginal (rank-1)
        """
        # For pure state |ψ⟩, we have ρ = |ψ⟩⟨ψ| = A*A† where A = |ψ⟩
        psi = psi.reshape(-1, 1)  # Column vector
        
        # Normalize
        norm = np.linalg.norm(psi)
        if norm < 1e-14:
            raise ValueError("State vector has zero norm")
        psi = psi / norm
        
        # Create instance
        result = cls.__new__(cls)
        result.sites = sites
        result._factor_A = psi
        result._tensor_format = 'matrix'
        result._cached_rho = None
        result._cache_valid = False
        
        # Convert to desired format
        if tensor_format == 'tensor':
            result.fold()
        
        return result
    
    def expectation_value(self, op: LocalTensor) -> Union[float, complex]:
        """
        Compute expectation value tr(ρ * O) where ρ is this marginal and O is the operator.
        
        Args:
            op (LocalTensor): The operator
            
        Returns:
            float or complex: The expectation value tr(ρ * O)
        """
        rho = self._compute_density_matrix()
        return np.trace(rho @ op.unfold().tensor)
    
    def contract_operators(self, opstr: List[str], oplib: Dict[HilbertSpace, SiteOperators]) -> float:
        """
        Contract the density matrix with a string of operators.
        
        Args:
            opstr (List[str]): List of operator names
            oplib (Dict[HilbertSpace, SiteOperators]): Operator libraries
            
        Returns:
            float: The expectation value
        """
        nsites = self.nsites
        assert len(opstr) == nsites, "Operator string length must match number of sites"

        if nsites == 0:
            return 0.0
        
        # Get the density matrix in the appropriate format
        self.fold() 
        
        if nsites == 1:
            O1 = oplib[self.sites[0].hilbert_space][opstr[0]]
            # return np.einsum('aA,Aa->', self.tensor, O1)
            A = self.factor_A
            return np.einsum('ax,Ax,Aa->', A, A.conj(), O1, optimize=True)
        elif nsites == 2:
            O1 = oplib[self.sites[0].hilbert_space][opstr[0]]
            O2 = oplib[self.sites[1].hilbert_space][opstr[1]]
            A = self.factor_A
            return np.einsum('abx,ABx,Aa,Bb->', A, A.conj(), O1, O2, optimize=True)
        elif nsites == 3:
            O1 = oplib[self.sites[0].hilbert_space][opstr[0]]
            O2 = oplib[self.sites[1].hilbert_space][opstr[1]]
            O3 = oplib[self.sites[2].hilbert_space][opstr[2]]
            A = self.factor_A
            return np.einsum('abcx,ABCx,Aa,Bb,Cc->', A, A.conj(), O1, O2, O3, optimize=True)
        elif nsites == 4:
            O1 = oplib[self.sites[0].hilbert_space][opstr[0]]
            O2 = oplib[self.sites[1].hilbert_space][opstr[1]]
            O3 = oplib[self.sites[2].hilbert_space][opstr[2]]
            O4 = oplib[self.sites[3].hilbert_space][opstr[3]]
            A = self.factor_A
            return np.einsum('abcdx,ABCDx,Aa,Bb,Cc,Dd->', A, A.conj(), O1, O2, O3, O4, optimize=True)
        else:
            raise NotImplementedError("Contracting more than 4 sites is not implemented yet.")
    
    def __repr__(self) -> str:
        site_labels = [site.label for site in self.sites]
        return f"FactorizedMarginal(sites={site_labels}, rank={self.rank}, shape={self._factor_A.shape})"