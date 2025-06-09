import numpy as np
from typing import Union, List

from .Site import Site
from .LocalOperator import LocalOperator

class RhoMBE:
    def __init__(self, sites:List[Site]):
        self.sites = sites
        self.nbody_terms = {}
        self.nbody_terms[1] = {} # start by instantiating the 1body dictionary

    def initialize_mixed(self):
        """
        Initialize with an unentangled maximally mixed states
        """
        for si in self.sites:
            mat = np.identity(si.dimension)/si.dimension
            self.nbody_terms[1][si.label] = LocalOperator(mat, [si]) 
        return self

    def __str__(self):
        out = ""
        for n,terms in self.nbody_terms.items():
            out += " %i-body terms: " %n
            # print(" %i-body terms: " %n)
            for term, mat in terms.items():
                out += " " + str(term)
                # print("   ", term)
        return out
    