from typing import Dict, Tuple, Union, List
import numpy as np

from meanfieldincrements import LocalTensor, Site
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
                str_curr = f"    {sites}: shape={shape_str}, trace={trace_str}"
                str_curr += " entropy=%12.8f"%marginal.entropy()
                lines.append(str_curr)
        
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

    def export_to_vector(self) -> Tuple[np.ndarray, Dict]:
        """
        Export all A factors from FactorizedMarginal objects to a single 1D vector.
        
        Returns:
            Tuple[np.ndarray, Dict]: 
                - vector: 1D numpy array containing all A factor elements
                - metadata: Dictionary containing information needed for reconstruction
        
        Note:
            Only FactorizedMarginal objects contribute to the vector. Regular Marginal
            objects are ignored but their presence is recorded in metadata.
        """
        from .FactorizedMarginal import FactorizedMarginal
        
        vector_parts = []
        metadata = {
            'marginal_info': {},  # Maps site keys to (is_factorized, shape, start_idx, end_idx)
            'total_length': 0,
            'tensor_formats': {}  # Maps site keys to tensor format
        }
        
        current_idx = 0
        
        # Process marginals in a consistent order (sorted by keys)
        for sites_key in sorted(self.marginals.keys()):
            marginal = self.marginals[sites_key]
            
            if isinstance(marginal, FactorizedMarginal):
                # Extract A factor and flatten
                factor_A = marginal.factor_A
                factor_flat = factor_A.flatten()
                
                # Store metadata
                start_idx = current_idx
                end_idx = current_idx + len(factor_flat)
                
                metadata['marginal_info'][sites_key] = {
                    'is_factorized': True,
                    'shape': factor_A.shape,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                }
                metadata['tensor_formats'][sites_key] = marginal._tensor_format
                
                vector_parts.append(factor_flat)
                current_idx = end_idx
            else:
                # Regular Marginal - record but don't include in vector
                metadata['marginal_info'][sites_key] = {
                    'is_factorized': False,
                    'shape': None,
                    'start_idx': None,
                    'end_idx': None
                }
        
        # Concatenate all parts
        if vector_parts:
            vector = np.concatenate(vector_parts)
        else:
            vector = np.array([])
        
        metadata['total_length'] = len(vector)
        
        return vector, metadata

    def import_from_vector(self, vector: np.ndarray, metadata: Dict) -> None:
        """
        Import A factors from a 1D vector and update FactorizedMarginal objects.
        
        Args:
            vector (np.ndarray): 1D array containing all A factor elements
            metadata (Dict): Metadata dictionary returned by export_to_vector
        
        Raises:
            ValueError: If vector length doesn't match expected length
            KeyError: If metadata is missing required information
        """
        from .FactorizedMarginal import FactorizedMarginal
        
        if len(vector) != metadata['total_length']:
            raise ValueError(f"Vector length {len(vector)} doesn't match expected length {metadata['total_length']}")
        
        # Process marginals in the same order as export
        for sites_key in sorted(self.marginals.keys()):
            marginal_info = metadata['marginal_info'][sites_key]
            
            if marginal_info['is_factorized']:
                marginal = self.marginals[sites_key]
                
                if not isinstance(marginal, FactorizedMarginal):
                    raise ValueError(f"Marginal at {sites_key} was expected to be FactorizedMarginal but is {type(marginal)}")
                
                # Extract the relevant portion of the vector
                start_idx = marginal_info['start_idx']
                end_idx = marginal_info['end_idx']
                factor_flat = vector[start_idx:end_idx]
                
                # Reshape to original factor shape
                # original_shape = marginal_info['shape']
                factor_A_new = factor_flat.reshape(marginal.factor_A.shape)

                # print(marginal._factor_A.shape, factor_A_new.shape) 
                # Update the marginal's factor_A
                marginal.factor_A = factor_A_new
                
                # Restore tensor format if needed
                target_format = metadata['tensor_formats'][sites_key]
                if target_format == 'tensor' and marginal._tensor_format == 'matrix':
                    marginal.fold()
                elif target_format == 'matrix' and marginal._tensor_format == 'tensor':
                    marginal.unfold()

    def initialize_maximally_mixed(self, sites: List['Site'], nbody: int = 2, rank = None) -> 'Marginals':
        """
        Initialize maximally mixed marginals using FactorizedMarginal representation.
        
        Each marginal is created as A*A† where A is chosen to give the maximally mixed state.
        This can be more memory efficient for large systems.
        
        Args:
            sites (List[Site]): All sites in the system
            nbody (int): Maximum n-body order to initialize (default: 2)
            rank (int): Rank of factorization (default: full rank = dimension)
            
        Returns:
            Marginals: Self for method chaining
        """
        from itertools import combinations
        from .FactorizedMarginal import FactorizedMarginal
        
        print(f"Initializing maximally mixed factorized marginals up to {nbody}-body...")
        
        # Clear any existing marginals
        self.marginals.clear()
        
        for n in range(1, nbody + 1):
            print(f"  Creating {n}-body factorized marginals...")
            
            for site_combination in combinations(sites, n):
                total_dim = np.prod([site.dimension for site in site_combination])
                
                # Choose rank (default to full rank for maximally mixed)
                actual_rank = rank if rank is not None else total_dim
                actual_rank = min(actual_rank, total_dim)  # Can't exceed dimension
                
                # For maximally mixed state, we want A such that A*A† = I/dim
                # One choice: A = (1/√dim) * [I, 0] where I is dim×rank identity
                scale_factor = 1.0 / np.sqrt(total_dim)
                
                if actual_rank == total_dim:
                    # Full rank: A = (1/√dim) * I
                    factor_A = scale_factor * np.eye(total_dim, dtype=complex)
                else:
                    # Reduced rank: A = (1/√dim) * [I_rank; 0]
                    factor_A = np.zeros((total_dim, actual_rank), dtype=complex)
                    factor_A[:actual_rank, :actual_rank] = scale_factor * np.eye(actual_rank)
                
                # Create site labels tuple (sorted for consistency)
                site_labels = tuple(sorted([site.label for site in site_combination]))
                
                # Create FactorizedMarginal and store it
                factorized_marginal = FactorizedMarginal(factor_A, list(site_combination), tensor_format='matrix')
                self[site_labels] = factorized_marginal
                
                print(f"    Added factorized marginal {site_labels}: "
                    f"dim={total_dim}, rank={actual_rank}, trace={factorized_marginal.trace():.6f}")
        
        total_marginals = len(self.marginals)
        print(f"  Total factorized marginals created: {total_marginals}")
        
        return self

    def compute_constraint_violations(self):
        violations = []
        self.fold()
        for sites in self.keys():
            if len(sites) == 2:
                si,sj = sites 

                mi  = self.marginals[(si, )]
                mj  = self.marginals[(sj, )]
                mij = self.marginals[(si, sj)]

                trace_i = mij.partial_trace([si])
                violation = np.linalg.norm(trace_i.tensor - mj.tensor)
                violations.append(violation)

                trace_j = mij.partial_trace([sj])
                violation = np.linalg.norm(trace_j.tensor - mi.tensor)
                violations.append(violation)
        return np.real_if_close(violations)
    
    def print_cumulants(self):
        print(" 2-Body Cumulants: ")
        self.unfold()
        for sites in self.keys():
            if len(sites) == 2:
                si,sj = sites
                cumul = self[sites].tensor - np.kron(self[(si,)].tensor, self[(sj,)].tensor)
                print("    %2i,%2i : Norm(lambda) = %12.8f" %(si,sj,np.linalg.norm(cumul)))
                
    def build_full_matrix(self):
        N = 0
        for margs in self.keys():
            if len(margs) == 1:
                N = max(N, margs[0]+1)
        
        letters = 'abcdefghijklmnopqrstuvwxyz'[0:N]
        letters += letters.upper()
        print(letters)
        raise NotImplementedError("Not finished")

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
        rho[(si.label, sj.label, sk.label, sl.label)] = Marginal.from_LocalTensor(lt.compute_nbody_marginal([si,sj,sk,sl]))
    
    if n_body > 4:
        raise NotImplementedError
    
    return rho