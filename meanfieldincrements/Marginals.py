import numpy as np
import copy as cp
from typing import Union, List, Dict, Set, Tuple, Optional
from itertools import combinations
from .Site import Site
from .LocalOperator import LocalOperator

class Marginal(LocalOperator):
    def trace(self):
        return 1
class Cumulant(LocalOperator):
    def trace(self):
        return 0
    

class Marginals:
    def __init__(self, sites:List[Site]):
        self.sites = sites
        self.marginals = {}
        for site in sites:
            dim = site.dimension
            self.marginals[(site.label,)] = Marginal(np.identity(dim), [site,], tensor_format="matrix").fold()
            

    def __getitem__(self, key):
        return self.marginals[key] if key in self.marginals else self.cumulants[key]
    
    def __setitem__(self, key, value):
        self.marginals[key] = value
    
    def __iter__(self):
        return iter(self.marginals.values())
    
    def __str__(self):
        """String representation showing MBE structure."""
        out = "Marginals:\n"
        out += f"  Total sites: {len(self.sites)} {[si.label for si in self.sites]}\n"

        for idx, marg in self.marginals.items(): 
            out += "    %s shape=" %str(idx) +  str(marg.tensor.shape)
            out += " trace=%.6f" % np.real(marg.trace())
            out += "\n"
        
        return out
    
class Cumulants:
    def __init__(self, sites: List[Site]):
        self.sites = sites
        self.cumulants = {}
        for site in sites:
            dim = site.dimension
            self.cumulants[(site.label,)] = Cumulant(np.zeros((dim, dim)), [site,], tensor_format="matrix").fold()

    def __getitem__(self, key):
        return self.cumulants[key]
    def __setitem__(self, key, value):
        self.cumulants[key] = value
    def __iter__(self):
        return iter(self.cumulants.values())

    def __str__(self):
        """String representation showing cumulant structure."""
        out = "Cumulants:\n"
        out += f"  Total sites: {len(self.sites)} {[si.label for si in self.sites]}\n"

        for idx, cumul in self.cumulants.items():
            out += "    %s shape=" % str(idx) + str(cumul.tensor.shape)
            out += " trace = %.6f" % np.real(cumul.trace())
            out += " norm = %.6f" % np.linalg.norm(cumul.tensor)
            out += "\n"

        return out
    
    def partial_trace(self, traced_sites: List[int]) -> 'Cumulants':

        if len(traced_sites) == len(self.sites):
            raise(ValueError("Cannot trace out all sites. Result would be a scalar. Use trace() method instead."))
        lout = cp.deepcopy(self)        
        for i in traced_sites:
            if i not in [j.label for j in self.sites]:
                raise ValueError(f"Invalid site index {i} for partial trace. Operator has {self.N} sites.")

            lout = lout._trace_out_1site(i, verbose=0)
        return lout        

    def _trace_out_1site(self, site_label: int, verbose=1) -> 'Cumulants':
        new_sites = [site for site in self.sites if site.label != site_label]
        new_l = Cumulants(new_sites)

        for sites_i, loc_oper in self.cumulants.items():
            # If the site is not in the current cumulant, just copy it
            if site_label not in sites_i:
                new_l[sites_i] = loc_oper.fold()
            elif (site_label,) == sites_i:
                # If the site is the only one in the cumulant, we can just skip it since the trace is zero
                continue
            else:
                # Compute the new cumulant term
                traced_cumulant = loc_oper.partial_trace([site_label]).fold()
                
                new_sites = tuple([site.label for site in traced_cumulant.sites]) 
                
                if new_sites not in new_l.cumulants:
                    new_l[new_sites] = traced_cumulant
                else:
                    new_l[new_sites] += traced_cumulant

        return new_l