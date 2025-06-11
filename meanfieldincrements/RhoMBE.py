import numpy as np
import copy as cp
from typing import Union, List, Dict, Set, Tuple, Optional
from itertools import combinations

from .Site import Site
from .LocalOperator import LocalOperator

class RhoMBE:
    """
    Many-Body Expansion (MBE) representation of a density matrix.
    
    The density matrix is represented as:
    ρ = ∏_i ρ_i + Σ_{ij} λ_{ij} ∏_{k≠i,j} ρ_k + Σ_{ijk} λ_{ijk} ∏_{l≠i,j,k} ρ_l + ...
    
    where:
    - ρ_i are 1-body marginals
    - λ_{ij} = ρ_{ij} - ρ_i ⊗ ρ_j (2-body corrections)
    - λ_{ijk} = ρ_{ijk} - ρ_{ij} ⊗ ρ_k - ρ_{ik} ⊗ ρ_j - ρ_{jk} ⊗ ρ_i + 2ρ_i ⊗ ρ_j ⊗ ρ_k
    
    Attributes:
        sites (List[Site]): All sites in the system
        terms (Dict[int, Dict]): Dictionary mapping n-body order to correction terms
    """
    
    def __init__(self, sites: List[Site]):
        self.sites = sites
        self.terms = {}
    
    def __getitem__(self, key):
        return self.terms[key]
    def __setitem__(self, key, value:'LocalOperator'):
        self.terms[key] = value
    def __iter__(self):
        return iter(self.terms.values())

    def initialize_mixed(self):
        """
        Initialize with an unentangled maximally mixed state.
        Sets each site to ρ_i = I/d_i where d_i is the local dimension.
        Only 1-body terms are non-zero initially.
        """
        for site in self.sites:
            mat = np.identity(site.dimension) / site.dimension
            self.terms[(site.label,)] = LocalOperator(mat, [site])
        return self

    
    def compute_2body_cumulant(self, rho_ij: LocalOperator) -> LocalOperator:
        """
        Compute 2-body correction: λ_{ij} = ρ_{ij} - ρ_i ⊗ ρ_j
        
        Args:
            rho_ij (LocalOperator): The 2-body density matrix ρ_{ij}.
            
        Returns:
            LocalOperator: The correction term λ_{ij}.
        """
        if len(rho_ij.sites) != 2:
            raise ValueError("Input LocalOperator must represent a 2-body density matrix.")
        site_i, site_j = [i.label for i in rho_ij.sites]

        # Get 1-body marginals
        rho_i = self.terms[(site_i,)]
        rho_j = self.terms[(site_j,)]
        
        # Compute tensor product ρ_i ⊗ ρ_j
        rho_i.fold()
        rho_j.fold()
        lam_ij = cp.deepcopy(rho_ij).fold()
        lam_ij.tensor -= np.einsum("iI,jJ->ijIJ", rho_i.tensor, rho_j.tensor)
        
        return lam_ij 

    def compute_3body_cumulant(self, rho_ijk: LocalOperator) -> LocalOperator:
        site_i, site_j, site_k = [i.label for i in rho_ijk.sites]
        
        # Get 1-body marginals from MBE expansion
        rho_i = self.terms[(site_i,)].fold()
        rho_j = self.terms[(site_j,)].fold() 
        rho_k = self.terms[(site_k,)].fold()
        
        # Get 2-body corrections from MBE expansion
        lambda_ij = self.terms[(site_i, site_j)].fold()
        lambda_ik = self.terms[(site_i, site_k)].fold()
        lambda_jk = self.terms[(site_j, site_k)].fold()
        
        # Start with the 3-body joint density
        rho_ijk.fold()
        correction_tensor = rho_ijk.tensor.copy()
        
        # Subtract mean-field term: ρ_i ⊗ ρ_j ⊗ ρ_k
        correction_tensor -= np.einsum('iI,jJ,kK->ijkIJK', 
                                    rho_i.tensor, rho_j.tensor, rho_k.tensor)
        
        # Subtract 2-body correction terms:
        # λ_{ij} ⊗ ρ_k
        correction_tensor -= np.einsum('ijIJ,kK->ijkIJK', 
                                    lambda_ij.tensor, rho_k.tensor)
        
        # λ_{ik} ⊗ ρ_j
        correction_tensor -= np.einsum('ikIK,jJ->ijkIJK', 
                                    lambda_ik.tensor, rho_j.tensor)
        
        # λ_{jk} ⊗ ρ_i  
        correction_tensor -= np.einsum('jkJK,iI->ijkIJK', 
                                    lambda_jk.tensor, rho_i.tensor)
        
        return LocalOperator(correction_tensor, rho_ijk.sites, tensor_format='tensor')


    def get_marginal_density_matrix(self, site_label: int) -> LocalOperator:
        """
        Get the 1-body marginal density matrix for a specific site.
        
        Args:
            site_label (int): Label of the site.
            
        Returns:
            LocalOperator: The marginal density matrix ρ_i.
        """
        if (site_label,) in self.terms:
            return self.terms[(site_label,)]
        else:
            # Default to maximally mixed state
            site = next((s for s in self.sites if s.label == site_label), None)
            if site is None:
                raise ValueError(f"Site {site_label} not found")
            
            identity = np.eye(site.dimension) / site.dimension
            return LocalOperator(identity, [site])
    
    
    def __str__(self):
        """String representation showing MBE structure."""
        out = "RhoMBE Density Matrix (MBE Form):\n"
        out += f"  Total sites: {len(self.sites)}\n"

        maxn = 0
        for term in self.terms.keys():
            if len(term) > maxn:
                maxn = len(term)

        for n in range(1,maxn+1):
            out += " %i-body terms: " %n
            out += "\n"
            for idx, lo in self.terms.items():
                if len(idx) != n:
                    continue
                out += "    %s shape=" % str(idx) + str(lo.tensor.shape)
                out += " trace = %.6f" % np.real(lo.trace())
                out += " norm = %.6f" % np.linalg.norm(lo.tensor)
                out += "\n"
        
        return out

    def partial_trace(self, traced_sites: List[int]) -> 'RhoMBE':

        rout = cp.deepcopy(self)        
        for i in traced_sites:
            if i not in [j.label for j in self.sites]:
                raise ValueError(f"Invalid site index {i} for partial trace. Operator has {self.N} sites.")

            rout = rout._trace_out_1site(i, verbose=0)
        return rout        

    def _trace_out_1site(self, site_label: int, verbose=1) -> 'RhoMBE':
        """
        Trace out a single site from the MBE representation.
        
        Args:
            site_label (int): Label of the site to trace out.
            
        Returns:
            RhoMBE: New MBE representation with the specified site traced out.
        """
        new_sites = [site for site in self.sites if site.label != site_label]
        new_rho = RhoMBE(new_sites)
        
        new_rho.fold()

        # Handle n-body corrections involving the traced site
        for term,local_op in self.terms.items():
            if term == (site_label,):
                continue
            if site_label in term:
                # Compute the new correction term
                traced_correction = local_op.partial_trace([site_label]).fold()
                sites_i = tuple([site.label for site in traced_correction.sites]) 
                if sites_i in new_rho.terms.keys():
                    new_rho[sites_i] += traced_correction
                else:
                    new_rho[sites_i] = traced_correction
            else:
                # Copy the correction term as is
                if term not in new_rho:
                    new_rho[term] = local_op.fold()
                else:
                    new_rho[term] += local_op.fold()
        return new_rho
    
    def fold(self) -> 'RhoMBE':
        """
        Fold the MBE representation to ensure all LocalOperators are in standard
        form (i.e., tensor product form).
        Returns:
            RhoMBE: The folded MBE representation.
        """
        for term, local_op in self.terms.items():
            self.terms[term].fold()
        return self
    
    def unfold(self) -> 'RhoMBE':
        """
        Unfold the MBE representation to ensure all LocalOperators are in standard
        form (i.e., tensor product form).
        Returns:
            RhoMBE: The folded MBE representation.
        """
        for term, local_op in self.terms.items():
            self.terms[term].unfold()
        return self