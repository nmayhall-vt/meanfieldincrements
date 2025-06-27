import numpy as np
import copy as cp
from typing import Dict, List, Union

from meanfieldincrements import Marginals
# from meanfieldincrements.LagrangeMultipliers import LagrangeMultipliers
from .Site import Site
from .SiteOperators import SiteOperators
from .GeneralHamiltonian import GeneralHamiltonian
from itertools import combinations

def energy_from_expvals(H:'GeneralHamiltonian', local_evals:Dict, verbose=False):

    E = 0

    E1b = 0
    for term,coeff in H.items():
        coeff_i = 1
        for si in H.sites:
            coeff_i *= local_evals[(si,)][(term[si.label],)]

        E1b += coeff_i*coeff
        assert np.abs(np.imag(E1b)) < 1e-12
    E1b = np.real_if_close(E1b) 
    if verbose: print(" 1-body energy: %12.8f"%E1b) 

    # 2body
    E2b = 0
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
        
            E2b += coeff_i*coeff
    E2b = np.real_if_close(E2b) 
    if verbose: print(" 2-body energy: %12.8f"%E2b) 

    # 3body
    E3b = 0
    for term,coeff in H.items():
        for (si, sj, sk) in combinations(H.sites, 3):
            if (si,sj,sk) not in local_evals:
                continue
            eijk = local_evals[(si,sj,sk)][term[si.label],term[sj.label],term[sk.label]]
            eij = local_evals[(si,sj)][term[si.label],term[sj.label]]
            eik = local_evals[(si,sk)][term[si.label],term[sk.label]]
            ejk = local_evals[(si,sj)][term[sj.label],term[sk.label]]
            ei  = local_evals[(si,)][term[si.label],]
            ej  = local_evals[(sj,)][term[sj.label],]
            ej  = local_evals[(sj,)][term[sk.label],]
            # rho(ijk) = l(ijk) + l(ij)r(k) + l(ik)r(j) + l(jk)r(i) + r(i)r(j)r(k)
            # r(ijk)   = l(ijk) + r(ij)r(k) + r(ik)r(j) + r(jk)r(i) - 2r(i)r(j)r(k)
            # l(ijk)   = r(ijk) - r(ij)r(k) - r(ik)r(j) - r(jk)r(i) + 2r(i)r(j)r(k)
            coeff_i = eijk - eij*ek - eik*ej - ejk*ei + 2*ei*ej*ek
            if np.abs(coeff_i)<1e-13:
                continue
            for si in set(H.sites).difference(set([si,sj])):
                coeff_i *= local_evals[(si,)][(term[si.label],)]
        
            E3b += coeff_i*coeff
    E3b = np.real_if_close(E3b) 
    if verbose: print(" 3-body energy: %12.8f"%E3b) 

    E = E1b + E2b + E3b
    if verbose: print("  Total energy: %12.8f"%E) 
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
            # mij = rho[si.label, sj.label].to_Marginal()
            # mi  = mij.partial_trace([sj.label, ]).fold()
            # mj  = mij.partial_trace([si.label, ]).fold()
            # tmp = np.einsum("pP,qQ->pqPQ", mi.tensor, mj.tensor)
            # eij = mij.contract_operators(opstr, oplib)
            # ei  = mi.contract_operators(opstr, oplib)
            # ej  = mj.contract_operators(opstr, oplib)
            # local_expvals[si, sj][opstr] = eij - ei - ej
    return local_expvals


# def compute_constraints(margs:'Marginals', lagrange_mults:'LagrangeMultipliers', verbose=False):
#     """
#     lagrange_mults[(i,)] = Lambda^(i) 
#     lagrange_mults[(i,j)] = Lambda^(ij) 
#     lagrange_mults[(i,j,k)] = Lambda^(ijk) 
#     where 
#     loss += Lambda^(i) (tr(\rho^{(i)})-1)
#     loss += Lambda^(ij) (tr_j(\rho^{(ij)})-\rho^{(i)}) = Lambda^(ij) * tr_j(lambda^(i,j))
#     """
#     margs.unfold()
#     loss = 0.0
#     for term,lamb in lagrange_mults.items():
#         #
#         # Because \Lambda^{(i,j)} \neq \Lambda^{(j,i)} we consider them both. 
#         # However, we only store i<j, marginals, so we need to sort the key before we grab the marginal
#         t0 = tuple(sorted(term))        # marginal key for current constraint: N body
#         t1 = tuple(sorted(term[1:]))    # marginal key for current constraint N-1 body
#         trace_idx = term[0]
#         if len(term) == 1:
#             if verbose: print(term, (margs[t0].trace() - 1))
#             loss += lamb**2 * (margs[t0].trace() - 1)
#         else: 
#             rhoi_j = margs[t0].partial_trace([trace_idx,]).unfold()
#             rho_j = margs[t1].unfold()
#             loss += lamb**2 * (np.trace(rhoi_j.tensor.conj().T @ rho_j.tensor) - 1)
#             if verbose: print(term, (np.trace(rhoi_j.tensor.conj().T @ rho_j.tensor) - 1))
#             # loss += np.trace( lamb @ (rhoi_j.tensor - rho_j.tensor) )
#     return loss 


# def get_cost_function_full(H:'GeneralHamiltonian', marginals:'Marginals'):
#     oplib = H.build_SiteOperators()
#     mvec, mdata = marginals.export_to_vector()

#     violations = marginals.compute_constraint_violations()
#     print(" Initial Violations norm: ",np.linalg.norm(violations))
#     # print(mdata)
#     n_marg_vals = len(mvec)
#     n_constraints = len(violations) 
#     print(" Number of parameters:  ", n_marg_vals)
#     print(" Number of constraints: ", n_constraints)
#     x0 = np.real_if_close(np.concatenate((mvec, np.zeros(n_constraints))))

#     def cost_function(x):
#         xm = x[0:n_marg_vals]
#         xc = x[n_marg_vals:]
#         marginals.import_from_vector(xm,mdata)
#         expvals = build_local_expvals(H, marginals, oplib)
#         energy = energy_from_expvals(H, expvals)
#         violations = marginals.compute_constraint_violations()
#         energy = np.real_if_close(energy)
#         constraint_val = np.dot(xc, violations)**2 
#         return np.real_if_close(energy + constraint_val) 
    
#     def callback(x):
#         xm = x[0:n_marg_vals]
#         xc = x[n_marg_vals:]
#         marginals.import_from_vector(xm,mdata)
#         expvals = build_local_expvals(H, marginals, oplib)
#         energy = energy_from_expvals(H, expvals)
#         violations = marginals.compute_constraint_violations()
#         energy = np.real_if_close(energy)
#         # constraint_val = np.dot(xc, violations) 
#         constraint_val = np.dot(xc, violations)**2 
#         cost =  np.real_if_close(energy + constraint_val) 
#         print(" Cost = %12.8f Energy = %12.8f violations = %12.8f penalty = %12.8f"%(
#                 cost, energy, np.linalg.norm(violations), constraint_val), flush=True)


#     return cost_function, callback, x0, n_marg_vals, n_constraints

# def get_cost_function_penalty(H:'GeneralHamiltonian', marginals:'Marginals', mu=10):
#     print("\n  mu = ", mu)
#     oplib = H.build_SiteOperators()
#     mvec, mdata = marginals.export_to_vector()

#     violations = marginals.compute_constraint_violations()
#     print(" Initial Violations norm: ",np.linalg.norm(violations))
#     # print(mdata)
#     n_marg_vals = len(mvec)
#     n_constraints = len(violations) 
#     print(" Number of parameters:  ", n_marg_vals)
#     x0 = np.real_if_close(mvec)

#     def cost_function(x):
#         marginals.import_from_vector(x,mdata)
#         expvals = build_local_expvals(H, marginals, oplib)
#         energy = energy_from_expvals(H, expvals)
#         violations = marginals.compute_constraint_violations()
#         energy = np.real_if_close(energy)
#         constraint_val = mu * np.sum(violations) 
#         # print(" Cost = %12.8f Energy = %12.8f violations = %12.8f penalty = %12.8f"%(
#         #         energy + constraint_val, energy, 
#         #             np.linalg.norm(violations), constraint_val), flush=True)

#         return np.real_if_close(energy + constraint_val) 
    
#     def callback(x):
#         marginals.import_from_vector(x,mdata)
#         expvals = build_local_expvals(H, marginals, oplib)
#         energy = energy_from_expvals(H, expvals)
#         violations = marginals.compute_constraint_violations()
#         energy = np.real_if_close(energy)
#         constraint_val = mu * np.sum(violations) 
#         print(" Cost = %12.8f Energy = %12.8f violations = %12.8f penalty = %12.8f"%(
#                 energy + constraint_val, energy, 
#                     np.linalg.norm(violations), constraint_val), flush=True)


#     return cost_function, callback, x0