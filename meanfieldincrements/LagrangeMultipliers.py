import numpy as np
from collections import OrderedDict
from itertools import combinations
from typing import List, Tuple, Dict, Union


class LagrangeMultipliers:
    """
    Container for Lagrange multipliers used in constrained optimization.
    
    Stores multipliers for various n-body constraints:
    - 1-body: scalar multipliers for trace constraints
    - 2-body: matrix multipliers for partial trace consistency constraints  
    - 3-body: matrix multipliers for higher-order consistency constraints
    """
    
    def __init__(self, sites):
        """
        Initialize LagrangeMultipliers container.
        
        Args:
            sites: List of Site objects
        """
        self.sites = sites
        self.multipliers = OrderedDict()  # tuple -> scalar or matrix
    
    def initialize_to_zero(self, nbody: int = 2):
        """
        Initialize all multipliers to zero.
        
        Args:
            nbody (int): Maximum n-body order to initialize
            
        Returns:
            LagrangeMultipliers: Self for method chaining
        """
        # 1-body: scalar multipliers for trace constraints
        for si in self.sites:
            self.multipliers[(si.label,)] = 0.0 

        if nbody < 2:
            return self

        # 2-body: matrix multipliers for partial trace consistency
        for (si, sj) in combinations(self.sites, 2):
            # Multiplier for tr_i(ρ_ij) - ρ_j constraint
            dim = sj.dimension 
            # self.multipliers[(si.label, sj.label)] = np.zeros((dim, dim))
            self.multipliers[(si.label, sj.label)] = 0 
            
            # Multiplier for tr_j(ρ_ij) - ρ_i constraint
            dim = si.dimension 
            self.multipliers[(sj.label, si.label)] = 0
            # self.multipliers[(sj.label, si.label)] = np.zeros((dim, dim))

        if nbody < 3:
            return self

        # 3-body: matrix multipliers for 3-body partial trace consistency
        for (si, sj, sk) in combinations(self.sites, 3):
            raise NotImplementedError
            # Multiplier for tr_i(ρ_ijk) - ρ_jk constraint
            dim = sj.dimension * sk.dimension
            self.multipliers[(si.label, sj.label, sk.label)] = np.zeros((dim, dim))
            
            # Multiplier for tr_j(ρ_ijk) - ρ_ik constraint  
            dim = si.dimension * sk.dimension
            self.multipliers[(sj.label, si.label, sk.label)] = np.zeros((dim, dim))
            
            # Multiplier for tr_k(ρ_ijk) - ρ_ij constraint
            dim = si.dimension * sj.dimension
            self.multipliers[(sk.label, si.label, sj.label)] = np.zeros((dim, dim))
            
        return self
    
    # Basic dictionary-like interface
    def __getitem__(self, key: Tuple[int, ...]):
        """Get multiplier by site indices."""
        return self.multipliers[key]
    
    def __setitem__(self, key: Tuple[int, ...], value: Union[float, np.ndarray]):
        """Set multiplier for given site indices."""
        self.multipliers[key] = value
    
    def __contains__(self, key: Tuple[int, ...]) -> bool:
        """Check if multiplier exists for given site indices."""
        return key in self.multipliers
    
    def __len__(self) -> int:
        """Number of multipliers stored."""
        return len(self.multipliers)
    
    def __iter__(self):
        """Iterate over multiplier values."""
        return iter(self.multipliers.values())
    
    def keys(self):
        """Get all site index tuples."""
        return self.multipliers.keys()
    
    def values(self):
        """Get all multiplier values."""
        return self.multipliers.values()
    
    def items(self):
        """Get (site_indices, multiplier) pairs."""
        return self.multipliers.items()
    
    def export_to_vector(self) -> Tuple[np.ndarray, Dict]:
        """
        Export all multipliers to a single 1D vector.
        
        Returns:
            Tuple[np.ndarray, Dict]: 
                - vector: 1D numpy array containing all multiplier elements
                - metadata: Dictionary containing information needed for reconstruction
        """
        vector_parts = []
        metadata = {
            'multiplier_info': {},  # Maps site keys to (is_scalar, shape, start_idx, end_idx)
            'total_length': 0
        }
        
        current_idx = 0
        
        # Process multipliers in consistent order (sorted by keys)
        for sites_key in sorted(self.multipliers.keys()):
            multiplier = self.multipliers[sites_key]
            
            if np.isscalar(multiplier) or (isinstance(multiplier, np.ndarray) and multiplier.ndim == 0):
                # Scalar multiplier
                multiplier_flat = np.array([float(multiplier)])
                
                metadata['multiplier_info'][sites_key] = {
                    'is_scalar': True,
                    'shape': (),
                    'start_idx': current_idx,
                    'end_idx': current_idx + 1
                }
                current_idx += 1
            else:
                # Matrix multiplier
                multiplier_flat = multiplier.flatten()
                
                metadata['multiplier_info'][sites_key] = {
                    'is_scalar': False,
                    'shape': multiplier.shape,
                    'start_idx': current_idx,
                    'end_idx': current_idx + len(multiplier_flat)
                }
                current_idx += len(multiplier_flat)
            
            vector_parts.append(multiplier_flat)
        
        # Concatenate all parts
        if vector_parts:
            vector = np.concatenate(vector_parts)
        else:
            vector = np.array([])
        
        metadata['total_length'] = len(vector)
        
        return vector, metadata
    
    def import_from_vector(self, vector: np.ndarray, metadata: Dict) -> None:
        """
        Import multipliers from a 1D vector.
        
        Args:
            vector (np.ndarray): 1D array containing all multiplier elements
            metadata (Dict): Metadata dictionary returned by export_to_vector
        
        Raises:
            ValueError: If vector length doesn't match expected length
        """
        if len(vector) != metadata['total_length']:
            raise ValueError(f"Vector length {len(vector)} doesn't match expected length {metadata['total_length']}")
        
        # Process multipliers in the same order as export
        for sites_key in sorted(self.multipliers.keys()):
            multiplier_info = metadata['multiplier_info'][sites_key]
            
            # Extract the relevant portion of the vector
            start_idx = multiplier_info['start_idx']
            end_idx = multiplier_info['end_idx']
            multiplier_flat = vector[start_idx:end_idx]
            
            if multiplier_info['is_scalar']:
                # Scalar multiplier
                self.multipliers[sites_key] = multiplier_flat[0]
                # self.multipliers[sites_key] = float(multiplier_flat[0])
            else:
                # Matrix multiplier
                original_shape = multiplier_info['shape']
                multiplier_new = multiplier_flat.reshape(original_shape)
                self.multipliers[sites_key] = multiplier_new
    
    def __str__(self) -> str:
        """Pretty string representation."""
        if not self.multipliers:
            return "LagrangeMultipliers: (empty)"
        
        lines = [f"LagrangeMultipliers ({len(self.multipliers)} multipliers):"]
        
        # Group by n-body order
        by_order = {}
        for sites, multiplier in self.multipliers.items():
            order = len(sites)
            if order not in by_order:
                by_order[order] = []
            by_order[order].append((sites, multiplier))
        
        # Print by order
        for order in sorted(by_order.keys()):
            lines.append(f"  {order}-body multipliers:")
            for sites, multiplier in sorted(by_order[order]):
                if np.isscalar(multiplier):
                    lines.append(f"    {sites}: {multiplier:.6f}")
                else:
                    lines.append(f"    {sites}: shape={multiplier.shape}, norm={np.linalg.norm(multiplier):.6f}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Developer string representation."""
        return f"LagrangeMultipliers({len(self.multipliers)} multipliers)"