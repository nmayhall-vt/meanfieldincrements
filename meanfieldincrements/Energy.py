import numpy as np
import copy as cp
from typing import Dict, List, Union

from meanfieldincrements import Marginals
from .Site import Site
from .SiteOperators import SiteOperators
from .GeneralHamiltonian import GeneralHamiltonian
from itertools import combinations

def energy_from_expvals(H:'GeneralHamiltonian', local_evals:Dict):

    E = 0

    for term,coeff in H.items():
        coeff_i = 1
        for si in H.sites:
            coeff_i *= local_evals[(si,)][(term[si.label],)]

        E += coeff_i*coeff
    
    print(" 1-body energy: %12.8f + %12.8fi"%(np.real(E), np.imag(E))) 

    # 2body
    for term,coeff in H.items():
        for (si, sj) in combinations(H.sites, 2):
            eij = local_evals[(si,sj)][term[si.label],term[sj.label]]
            ei  = local_evals[(si,)][term[si.label],]
            ej  = local_evals[(sj,)][term[sj.label],]
            coeff_i = eij - ei*ej
            if np.abs(coeff_i)<1e-13:
                continue
            for si in set(H.sites).difference(set([si,sj])):
                coeff_i *= local_evals[(si,)][(term[si.label],)]
        
            E += coeff_i*coeff

    print(" 2-body energy: %12.8f + %12.8fi"%(np.real(E), np.imag(E))) 
    return E

def build_local_expvals(H: 'GeneralHamiltonian', rho: 'Marginals', oplib: Dict[Site, SiteOperators]) -> Dict[List[Site], Dict[str, float]]:
    # Initialize local_evals dictionary
    local_expvals = {}
    sites = H.sites

    # 1Body terms
    for site in sites:
        local_expvals[(site,)] = {}
    for hi,_ in H.items():
        for (si,) in combinations(sites, 1):
            opstr = (hi[si.label],)
            if opstr in local_expvals[(si,)]:
                continue
            local_expvals[(si,)][opstr] = rho[(si.label,)].contract_operators(opstr, oplib)

    # 2Body terms
    rho.fold()
    for (si, sj) in combinations(sites, 2):
        local_expvals[si,sj] = {}
        for hi,_ in H.items():
            opstr = (hi[si.label], hi[sj.label])
            if opstr in local_expvals[(si,sj)].keys():
                continue
            local_expvals[si,sj][opstr] = rho[si.label, sj.label].contract_operators(opstr, oplib)

    return local_expvals