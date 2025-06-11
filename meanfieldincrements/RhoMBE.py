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
        nbody_terms (Dict[int, Dict]): Dictionary mapping n-body order to correction terms
    """
    
    def __init__(self, sites: List[Site]):
        self.sites = sites
        self.nbody_terms = {}
        self.constant = 1   # constant term that multiplies each term in the MBE expansion
        self.nbody_terms[1] = {}  # 1-body terms (marginals)

    def initialize_mixed(self):
        """
        Initialize with an unentangled maximally mixed state.
        Sets each site to ρ_i = I/d_i where d_i is the local dimension.
        Only 1-body terms are non-zero initially.
        """
        for site in self.sites:
            mat = np.identity(site.dimension) / site.dimension
            self.nbody_terms[1][(site.label,)] = LocalOperator(mat, [site])
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
        if 1 in self.nbody_terms:
            for site_tuple, marginal in self.nbody_terms[1].items():
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
        for n_body in sorted(self.nbody_terms.keys()):
            if n_body == 1:
                continue  # Already handled in mean-field part
                
            for cluster_key, correction_term in self.nbody_terms[n_body].items():
                # Trace of the correction term
                correction_trace = correction_term.trace()
                
                # Product of traces for sites not in this cluster
                cluster_sites = list(cluster_key)  # cluster_key is already a tuple
                remaining_sites = set(marginal_traces.keys()) - set(cluster_sites)
                remaining_trace_product = np.prod([marginal_traces[site] for site in remaining_sites])
                
                # Add contribution: Tr(λ_{cluster}) * ∏_{k∉cluster} Tr(ρ_k)
                total_trace += correction_trace * remaining_trace_product
        
        return total_trace * self.constant  # Include the constant term 
    
    # def _compute_partial_trace(self, sites_to_trace: List[int]) -> Union[float, complex]:
    #     """
    #     Compute partial trace over specified sites.
        
    #     For the MBE form, we need to trace out sites from each term:
    #     ρ = ∏_i ρ_i + Σ_{clusters} λ_{cluster} ∏_{k∉cluster} ρ_k
        
    #     When tracing over sites S:
    #     Tr_S(ρ) = ∏_{i∉S} ρ_i ∏_{j∈S} Tr(ρ_j) + Σ_{clusters} Tr_S(λ_{cluster} ∏_{k∉cluster} ρ_k)
        
    #     Args:
    #         sites_to_trace (List[int]): Site labels to trace over.
            
    #     Returns:
    #         Union[float, complex]: The partial trace result.
    #     """
    #     traced_sites_set = set(sites_to_trace)
    #     all_sites_set = {site.label for site in self.sites}
    #     remaining_sites = all_sites_set - traced_sites_set
        
    #     # Get marginal traces for traced sites and operators for remaining sites
    #     traced_marginal_traces = {}
    #     remaining_marginals = {}
        
    #     if 1 in self.nbody_terms:
    #         for site_tuple, marginal in self.nbody_terms[1].items():
    #             site_label = site_tuple[0]  # Extract single site from tuple
    #             if site_label in traced_sites_set:
    #                 traced_marginal_traces[site_label] = marginal.trace()
    #             else:
    #                 remaining_marginals[site_label] = marginal
        
    #     # Handle missing marginals (assume identity/normalized)
    #     for site_label in traced_sites_set:
    #         if site_label not in traced_marginal_traces:
    #             traced_marginal_traces[site_label] = 1.0
        
    #     # Mean-field contribution: ∏_{i∉S} ρ_i ∏_{j∈S} Tr(ρ_j)
    #     traced_product = np.prod(list(traced_marginal_traces.values()))
        
    #     if not remaining_sites:
    #         # All sites traced - return scalar
    #         total_result = traced_product
    #     else:
    #         # Some sites remain - this becomes more complex
    #         # For simplicity, we'll compute the effective trace
    #         total_result = traced_product
        
    #     # Add correction terms
    #     for n_body in sorted(self.nbody_terms.keys()):
    #         if n_body == 1:
    #             continue  # Already handled
                
    #         for cluster_key, correction_term in self.nbody_terms[n_body].items():
    #             cluster_sites_set = set(cluster_key)  # cluster_key is already a tuple
                
    #             # Determine how this cluster overlaps with traced sites
    #             cluster_traced = cluster_sites_set.intersection(traced_sites_set)
    #             cluster_remaining = cluster_sites_set - traced_sites_set
                
    #             if cluster_traced:
    #                 # This correction term involves traced sites
    #                 if cluster_remaining:
    #                     # Partial trace of the correction term
    #                     try:
    #                         partial_correction = correction_term.partial_trace(list(cluster_traced))
    #                         correction_trace = partial_correction.trace()
    #                     except:
    #                         # Fallback: full trace if partial trace fails
    #                         correction_trace = correction_term.trace()
    #                 else:
    #                     # Entire cluster is traced
    #                     correction_trace = correction_term.trace()
                    
    #                 # Product of remaining marginal traces outside this cluster
    #                 other_sites = all_sites_set - cluster_sites_set
    #                 other_traced = other_sites.intersection(traced_sites_set)
    #                 other_remaining = other_sites - traced_sites_set
                    
    #                 other_traced_product = np.prod([traced_marginal_traces.get(site, 1.0) 
    #                                               for site in other_traced])
                    
    #                 total_result += correction_trace * other_traced_product
        
    #     return total_result 
    
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
        rho_i = self.nbody_terms[1][(site_i,)]
        rho_j = self.nbody_terms[1][(site_j,)]
        
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
        if 1 in self.nbody_terms and (site_label,) in self.nbody_terms[1]:
            return self.nbody_terms[1][(site_label,)]
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
        out += " Constant term: " + str(self.constant) + "\n"

        for n,terms in self.nbody_terms.items(): 
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
        
        In order to handle the fact that some local operators may not have unit trace
        (i.e., when this isn't a state), we need to keep track of this trace. 
        We set the self.constant to the trace of the 1body term we are tracing, 
        and then divide each n-body term by this constant. 

        Args:
            site_label (int): Label of the site to trace out.
            
        Returns:
            RhoMBE: New MBE representation with the specified site traced out.
        """
        new_sites = [site for site in self.sites if site.label != site_label]
        new_rho = RhoMBE(new_sites)
        new_sites_idx = [i.label for i in new_sites]

        trace_i = self.nbody_terms[1][(site_label,)].trace()
        # print(" trace of rho(i) = ", trace_i) 
        
        # Get scalar value
        new_rho.constant = self.constant * trace_i 

        for n_body, terms in self.nbody_terms.items():
            new_rho.nbody_terms[n_body] = {}

        # Copy 1-body terms that do not involve the traced site
        for key, term in self.nbody_terms[1].items():
            if site_label not in key:
                new_rho.nbody_terms[1][key] = term


        # Handle n-body corrections involving the traced site
        for n_body, terms in self.nbody_terms.items():
            if n_body == 1:
                continue
            for sites_i, term in terms.items():
                if site_label in sites_i:
                    # Compute the new correction term
                    traced_correction = term.partial_trace([site_label])

                    traced_correction.scale(1/trace_i)

                    new_sites = traced_correction.sites
                    new_sites_idx = tuple([site.label for site in new_sites]) 
                    if new_sites_idx not in new_rho.nbody_terms[n_body-1]:
                        new_rho.nbody_terms[n_body-1][new_sites_idx] = traced_correction
                    else:
                        new_rho.nbody_terms[n_body-1][new_sites_idx] += traced_correction
        return new_rho