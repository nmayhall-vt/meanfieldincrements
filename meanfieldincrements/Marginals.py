from typing import Dict, Tuple, Union, List
from .Marginal import Marginal
from itertools import combinations
from collections import OrderedDict

class Marginals:
    """
    Simple container for marginal density matrices.
    
    Maps tuples of site indices to Marginal or FactorizedMarginal instances.
    Provides basic dictionary-like interface and printing functionality.
    
    Example:
        marginals = Marginals()
        marginals[(0,)] = single_site_marginal
        marginals[(0, 1)] = two_site_marginal
        marginals[(1, 2, 3)] = three_site_marginal
    """
    
    def __init__(self):
        """Initialize empty Marginals container."""
        self.marginals = OrderedDict()  
    
    def __getitem__(self, key: Tuple[int, ...]):
        """Get marginal by site indices."""
        return self.marginals[key]
    
    def __setitem__(self, key: Tuple[int, ...], value):
        """Set marginal for given site indices."""
        if not isinstance(key, tuple):
            raise TypeError("Key must be a tuple of site indices")
        self.marginals[key] = value
    
    def __contains__(self, key: Tuple[int, ...]) -> bool:
        """Check if marginal exists for given site indices."""
        return key in self.marginals
    
    def __len__(self) -> int:
        """Number of marginals stored."""
        return len(self.marginals)
    
    def __iter__(self):
        """Iterate over marginal values."""
        return iter(self.marginals.values())
    
    def keys(self):
        """Get all site index tuples."""
        return self.marginals.keys()
    
    def values(self):
        """Get all marginal instances."""
        return self.marginals.values()
    
    def items(self):
        """Get (site_indices, marginal) pairs."""
        return self.marginals.items()
    
    def __str__(self) -> str:
        """Pretty string representation."""
        if not self.marginals:
            return "Marginals: (empty)"
        
        lines = [f"Marginals ({len(self.marginals)} marginals):"]
        
        # Group by n-body order
        by_order = {}
        for sites, marginal in self.marginals.items():
            order = len(sites)
            if order not in by_order:
                by_order[order] = []
            by_order[order].append((sites, marginal))
        
        # Print by order
        for order in sorted(by_order.keys()):
            lines.append(f"  {order}-body marginals:")
            for sites, marginal in sorted(by_order[order]):
                if hasattr(marginal, 'trace'):
                    trace_val = marginal.trace()
                    if hasattr(trace_val, 'real'):
                        trace_str = f"{trace_val.real:.6f}"
                    else:
                        trace_str = f"{trace_val:.6f}"
                else:
                    trace_str = "N/A"
                
                shape_str = str(marginal.tensor.shape) if hasattr(marginal, 'tensor') else "N/A"
                lines.append(f"    {sites}: shape={shape_str}, trace={trace_str}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Developer string representation."""
        return f"Marginals({len(self.marginals)} marginals)"
    
    def fold(self) -> 'Marginals':
        """
        Fold all marginals to tensor format.
        
        Returns:
            Marginals: Self for method chaining
        """
        for term in self.keys():
            self.marginals[term].fold()
        return self
    
    def unfold(self) -> 'Marginals':
        """
        Unfold all marginals to matrix format.
        
        Returns:
            Marginals: Self for method chaining
        """
        for term in self.keys():
            self.marginals[term].unfold()
        return self


def build_Marginals_from_LocalTensor(lt:'LocalTensor', n_body:int=2):
    """
    Decompose a dense density matrix, provided as a `LocalTensor`, into a Marginals
    """
    sites = lt.sites
    rho = Marginals()

    for si in sites:
        rho[(si.label,)] = Marginal.from_LocalTensor(lt.compute_nbody_marginal([si])) 

    if n_body < 2:
        return rho

    # Test 2: Add a 2-body correction
    for (si, sj) in combinations(sites, 2):
        rho[(si.label, sj.label)] = Marginal.from_LocalTensor(lt.compute_nbody_marginal([si,sj]))
    
    
    if n_body < 3:
        return rho

    for (si, sj, sk) in combinations(sites, 3):
        rho[(si.label, sj.label, sk.label)] = Marginal.from_LocalTensor(lt.compute_nbody_marginal([si,sj,sk]))
    
    
    if n_body < 4:
        return rho

    for (si, sj, sk, sl) in combinations(sites, 4):
        rho.terms[(si.label, sj.label, sk.label, sl.label)] = Marginal.from_LocalTensor(lt.compute_nbody_marginal([si,sj,sk,sl]))
    
    if n_body > 4:
        raise NotImplementedError
    
    return rho