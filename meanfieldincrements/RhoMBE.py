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
        self.terms[1] = {}  # 1-body terms (marginals)
        self.terms[2] = {}  # 2-body terms (marginals)

    def initialize_mixed(self):
        """
        Initialize with an unentangled maximally mixed state.
        Sets each site to ρ_i = I/d_i where d_i is the local dimension.
        Only 1-body terms are non-zero initially.
        """
        for site in self.sites:
            mat = np.identity(site.dimension) / site.dimension
            self.terms[1][(site.label,)] = LocalOperator(mat, [site])
        return self

    def trace(self, sites_to_trace: Optional[List[int]] = None) -> Union[float, complex]:
        """
        Compute the trace of the MBE density matrix over specified sites.
        
        For the MBE form: ρ = ∏_i ρ_i + Σ_{clusters} λ_{cluster} ∏_{k∉cluster} ρ_k
        
        The trace is computed by carefully handling each term in the expansion.
        
        Args:
            sites_to_trace (Optional[List[int]]): Site labels to trace over.
                If None, traces over all sites (returns scalar).
                
        Returns:
            Union[float, complex]: The trace value.
        """
        if sites_to_trace is None:
            return self._compute_full_trace()
        else:
            return self._compute_partial_trace(sites_to_trace)
    
    def _compute_full_trace(self) -> Union[float, complex]:
        """
        Compute the full trace of the MBE density matrix.
        
        For ρ = ∏_i ρ_i + Σ_{clusters} λ_{cluster} ∏_{k∉cluster} ρ_k:
        
        Tr(ρ) = Tr(∏_i ρ_i) + Σ_{clusters} Tr(λ_{cluster}) Tr(∏_{k∉cluster} ρ_k)
               = ∏_i Tr(ρ_i) + Σ_{clusters} Tr(λ_{cluster}) ∏_{k∉cluster} Tr(ρ_k)
        
        Since each ρ_i should be normalized: Tr(ρ_i) = 1
        
        Returns:
            Union[float, complex]: The full trace (should be 1.0 for normalized ρ).
        """
        # Get all 1-body marginal traces
        marginal_traces = {}
        if 1 in self.terms:
            for site_tuple, marginal in self.terms[1].items():
                site_label = site_tuple[0]  # Extract single site from tuple
                marginal_traces[site_label] = marginal.trace()
        else:
            # No 1-body terms - assume identity
            for site in self.sites:
                marginal_traces[site.label] = 1.0
        
        # Product of all marginal traces (mean-field contribution)
        mean_field_trace = np.prod(list(marginal_traces.values()))
        
        total_trace = mean_field_trace
        
        # Add corrections from n-body terms (n ≥ 2)
        for n_body in sorted(self.terms.keys()):
            if n_body == 1:
                continue  # Already handled in mean-field part
                
            for cluster_key, correction_term in self.terms[n_body].items():
                # Trace of the correction term
                correction_trace = correction_term.trace()
                
                # Product of traces for sites not in this cluster
                cluster_sites = list(cluster_key)  # cluster_key is already a tuple
                remaining_sites = set(marginal_traces.keys()) - set(cluster_sites)
                remaining_trace_product = np.prod([marginal_traces[site] for site in remaining_sites])
                
                # Add contribution: Tr(λ_{cluster}) * ∏_{k∉cluster} Tr(ρ_k)
                total_trace += correction_trace * remaining_trace_product
        
        return total_trace 
    
    
    def _get_cluster_sites(self, cluster_key: Tuple[int, ...]) -> List[int]:
        """
        Extract site labels from cluster key.
        
        Args:
            cluster_key (Tuple[int, ...]): Tuple of site labels in the cluster.
            
        Returns:
            List[int]: List of site labels in the cluster.
        """
        return list(cluster_key)
    
    
    def compute_2body_correction(self, site_i: int, site_j: int, joint_density: LocalOperator) -> LocalOperator:
        """
        Compute 2-body correction: λ_{ij} = ρ_{ij} - ρ_i ⊗ ρ_j
        
        Args:
            site_i (int): First site label.
            site_j (int): Second site label.
            joint_density (LocalOperator): The 2-body density matrix ρ_{ij}.
            
        Returns:
            LocalOperator: The correction term λ_{ij}.
        """
        # Get 1-body marginals
        rho_i = self.terms[1][(site_i,)]
        rho_j = self.terms[1][(site_j,)]
        
        # Compute tensor product ρ_i ⊗ ρ_j
        rho_i.unfold()
        rho_j.unfold()
        product_matrix = np.kron(rho_i.tensor, rho_j.tensor)
        
        # Get joint density matrix
        joint_density.unfold()
        
        # Correction: λ_{ij} = ρ_{ij} - ρ_i ⊗ ρ_j
        correction_matrix = joint_density.tensor - product_matrix
        
        return LocalOperator(correction_matrix, joint_density.sites)
    
    def get_marginal_density_matrix(self, site_label: int) -> LocalOperator:
        """
        Get the 1-body marginal density matrix for a specific site.
        
        Args:
            site_label (int): Label of the site.
            
        Returns:
            LocalOperator: The marginal density matrix ρ_i.
        """
        if 1 in self.terms and (site_label,) in self.terms[1]:
            return self.terms[1][(site_label,)]
        else:
            # Default to maximally mixed state
            site = next((s for s in self.sites if s.label == site_label), None)
            if site is None:
                raise ValueError(f"Site {site_label} not found")
            
            identity = np.eye(site.dimension) / site.dimension
            return LocalOperator(identity, [site])
    
    def check_normalization(self) -> float:
        """
        Check if the density matrix is properly normalized.
        
        Returns:
            float: The trace of the density matrix (should be 1.0).
        """
        return abs(self.trace())
    
    def __str__(self):
        """String representation showing MBE structure."""
        out = "RhoMBE Density Matrix (MBE Form):\n"
        out += f"  Total sites: {len(self.sites)}\n"

        for n,terms in self.terms.items(): 
            out += " %i-body terms: " %n
            out += "\n"
            for i in terms.keys():
                # print("    %s shape=" %str(i), terms[i].tensor.shape)
                out += "    %s shape=" %str(i) +  str(terms[i].tensor.shape)
                out += "\n"
        
        total_trace = self.trace()
        out += f" Total trace: {total_trace:.6f}\n"
        
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
        new_sites_idx = [i.label for i in new_sites]

        trace_i = self.terms[1][(site_label,)].trace()
        # print(" trace of rho(i) = ", trace_i) 
        

        for n_body, terms in self.terms.items():
            new_rho.terms[n_body] = {}

        # Copy 1-body terms that do not involve the traced site
        for key, term in self.terms[1].items():
            if site_label not in key:
                new_rho.terms[1][key] = term


        # Handle n-body corrections involving the traced site
        for n_body, terms in self.terms.items():
            if n_body == 1:
                continue
            for sites_i, term in terms.items():
                if site_label in sites_i:
                    # Compute the new correction term
                    traced_correction = term.partial_trace([site_label])

                    traced_correction.scale(1/trace_i)

                    new_sites = traced_correction.sites
                    new_sites_idx = tuple([site.label for site in new_sites]) 
                    if new_sites_idx not in new_rho.terms[n_body-1]:
                        new_rho.terms[n_body-1][new_sites_idx] = traced_correction
                    else:
                        new_rho.terms[n_body-1][new_sites_idx] += traced_correction
        return new_rho