import numpy as np
import copy as cp
from typing import Dict, List, Union

from meanfieldincrements import Marginals
from meanfieldincrements.LagrangeMultipliers import LagrangeMultipliers
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


def compute_constraints(margs:'Marginals', lagrange_mults:'LagrangeMultipliers', verbose=False):
    """
    lagrange_mults[(i,)] = Lambda^(i) 
    lagrange_mults[(i,j)] = Lambda^(ij) 
    lagrange_mults[(i,j,k)] = Lambda^(ijk) 
    where 
    loss += Lambda^(i) (tr(\rho^{(i)})-1)
    loss += Lambda^(ij) (tr_j(\rho^{(ij)})-\rho^{(i)}) = Lambda^(ij) * tr_j(lambda^(i,j))
    """
    margs.unfold()
    loss = 0.0
    for term,lamb in lagrange_mults.items():
        #
        # Because \Lambda^{(i,j)} \neq \Lambda^{(j,i)} we consider them both. 
        # However, we only store i<j, marginals, so we need to sort the key before we grab the marginal
        t0 = tuple(sorted(term))        # marginal key for current constraint: N body
        t1 = tuple(sorted(term[1:]))    # marginal key for current constraint N-1 body
        trace_idx = term[0]
        if len(term) == 1:
            print(t0, margs[t0])
            loss += lamb * (margs[t0].trace() - 1)
        else: 
            rhoi_j = margs[t0].partial_trace([trace_idx,])
            rho_j = margs[t1]
            loss += np.trace( lamb @ (rhoi_j.tensor - rho_j.tensor) )
    return loss 


