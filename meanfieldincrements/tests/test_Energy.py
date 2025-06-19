import pytest
import numpy as np
from meanfieldincrements import Site, LocalTensor, PauliHilbertSpace, SpinHilbertSpace, GeneralHamiltonian
from meanfieldincrements.Marginal import Marginal
from meanfieldincrements.FactorizedMarginal import FactorizedMarginal
from meanfieldincrements.Energy import energy
from meanfieldincrements.GeneralHamiltonian import build_heisenberg_hamiltonian, build_ising_hamiltonian
from meanfieldincrements.MBEState import MBEState, build_MBEState_from_LocalTensor
from meanfieldincrements.Marginals import Marginals, build_Marginals_from_LocalTensor
from itertools import combinations

def test_energy():
    N = 4
    sites = [Site(i, SpinHilbertSpace(2)) for i in range(N)]

    #build operator library, which constructs a set of local matrices for acting on different types of hilbert spaces 
    oplib = {}
    for site in sites:
        oplib[site.hilbert_space] = site.hilbert_space.create_operators()
    
    print("oplib")
    print(oplib)

    # rho = MBEState(sites)
    # rho.initialize_mixed()
    # rho.fold()

    H = build_heisenberg_hamiltonian(sites)
    print(H)
    Hmat = H.matrix(oplib)
    # Diagonalize the Hermitian matrix Hmat
    eigvals, eigvecs = np.linalg.eigh(Hmat)
    lowest_idx = np.argmin(eigvals)
    lowest_energy = eigvals[lowest_idx]
    v = eigvecs[:, lowest_idx]
    for i in eigvals:
        print("  %12.8f"%i)
    print("Lowest energy eigenvalue:", lowest_energy)

    # rho = build_MBEState_from_LocalTensor(LocalTensor(np.outer(v,v), sites), n_body=3)
    rho = build_Marginals_from_LocalTensor(LocalTensor(np.outer(v,v), sites), n_body=3)
    rho.fold()
    print(rho)

    # Initialize local_evals dictionary
    local_expvals = {}
    for site in sites:
        local_expvals[(site,)] = {}

    # 1Body terms
    for hi,_ in H.items():
        # for site in sites:
        #     # print(hi, site)
        #     local_expvals[(site,)][hi[site.label]] = rho[(site.label,)].contract_operators([hi[site.label],], oplib)
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
            print(opstr)
            local_expvals[si,sj][opstr] = rho[si.label, sj.label].contract_operators(opstr, oplib)


    print("local_expvals")
    for site,evals in local_expvals.items():
        print(site)
        for opstr,val in evals.items():
            print("  %12.8f %12.8fi" %(np.real(val), np.imag(val)), " ", opstr) 


    print(" Now call energy function")
    e = energy(H, local_expvals)
    
if __name__ == "__main__":
    # Run tests manually
    test_energy()