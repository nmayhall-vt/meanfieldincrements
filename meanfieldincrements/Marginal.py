from typing import List, Union, Dict
import numpy as np
from .Site import Site
from .LocalTensor import LocalTensor
from .HilbertSpace import HilbertSpace
from .SiteOperators import SiteOperators

class Marginal(LocalTensor):
    """
    Marginal density matrix derived from LocalTensor.
    
    This class extends LocalTensor with convenience methods for
    density matrix operations in many-body expansions.
    """
    
    def __init__(self, tensor: np.ndarray, sites: List[Site], tensor_format: str = 'matrix'):
        """
        Initialize a Marginal.
        
        Args:
            tensor (np.ndarray): The density matrix tensor
            sites (List[Site]): Sites this marginal acts on
            tensor_format (str): Format of the tensor ('matrix' or 'tensor')
        """
        super().__init__(tensor, sites, tensor_format)
    
    @property
    def nsites(self) -> int:
        """
        Get number of sites in this
        marginal.
        Returns:
            int: Number of sites
        """
        return len(self.sites)
    

    def partial_trace(self, traced_sites: List[int]) -> 'Marginal':
        """
        Compute partial trace over specified sites.
        
        Args:
            traced_sites (List[Site]): Sites to trace over
            
        Returns:
            Marginal: Reduced marginal after partial trace
        """
        # Convert Site objects to site labels for LocalTensor.partial_trace
        
        # Compute partial trace using parent method
        reduced_tensor = super().partial_trace(traced_sites)
        
        # Determine remaining sites
        remaining_sites = []
        for si in self.sites:
            if si.label not in traced_sites:
                remaining_sites.append(si)
        # remaining_sites = [site for site in self.sites if site not in traced_sites]
        
        return Marginal(reduced_tensor.tensor, remaining_sites, reduced_tensor._tensor_format)
    
    def expectation_value(self, op: LocalTensor) -> Union[float, complex]:
        """
        Compute expectation value tr(ρ * O) where ρ is this marginal and O is the operator.
        
        Args:
            op (LocalTensor): The operator
            
        Returns:
            float or complex: The expectation value tr(ρ * O)
        """
        # Ensure both are in matrix form
        
        # Compute tr(ρ * O)
        return np.trace(self.unfold().tensor @ op.unfold().tensor)
    
    def __repr__(self) -> str:
        site_labels = [site.label for site in self.sites]
        return f"Marginal(sites={site_labels}, shape={self.tensor.shape})"

    def contract_operators(self, opstr:List[str], oplib:Dict[HilbertSpace, SiteOperators]) -> float:
        nsites = self.nsites
        assert len(opstr) == nsites, "Operator string length must match number of sites"

        if nsites == 0:
            return 0.0
        elif nsites == 1:
            O1 = oplib[self.sites[0].hilbert_space][opstr[0]]
            return np.einsum('aA,Aa->', self.tensor, O1)
        elif nsites == 2:
            O1 = oplib[self.sites[0].hilbert_space][opstr[0]]
            O2 = oplib[self.sites[1].hilbert_space][opstr[1]]
            return np.einsum('abAB,Aa,Bb->', self.tensor, O1, O2, optimize=True)
        elif nsites == 3:
            O1 = oplib[self.sites[0].hilbert_space][opstr[0]]
            O2 = oplib[self.sites[1].hilbert_space][opstr[1]]
            O3 = oplib[self.sites[2].hilbert_space][opstr[2]]
            return np.einsum('abcABC,Aa,Bb,Cc->', self.tensor, O1, O2, O3, optimize=True)
        else:
            raise NotImplementedError("Contracting more than 3 sites is not implemented yet.")
    
    @classmethod
    def from_LocalTensor(cls, local_tensor: 'LocalTensor') -> 'Marginal':
        """
        Create a Marginal from a LocalTensor.
        
        Args:
            local_tensor (LocalTensor): The LocalTensor to convert
            
        Returns:
            Marginal: A new Marginal instance
        """
        return cls(local_tensor.tensor.copy(), local_tensor.sites, local_tensor._tensor_format)