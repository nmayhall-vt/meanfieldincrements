import numpy as np
import copy as cp
from typing import Dict, List, Union
from .Site import Site
from .SiteOperators import SiteOperators
from .GeneralHamiltonian import GeneralHamiltonian
from itertools import combinations

def energy(H:'GeneralHamiltonian', local_evals:Dict[List[Site], Dict[str, float]]):

    E = 0

    for term,coeff in H.items():
        print(term, coeff)
        coeff_i = 1
        for si in H.sites:
            coeff_i *= local_evals[(si,)][(term[si.label],)]

        E += coeff_i*coeff

    # 2body
    for term,coeff in H.items():
        print(term, coeff)
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

    print(" 1-body term: %12.8f + %12.8fi"%(np.real(E), np.imag(E))) 
    return E

