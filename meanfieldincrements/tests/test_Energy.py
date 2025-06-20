import pytest
import numpy as np
from meanfieldincrements import Site, LocalTensor, PauliHilbertSpace, SpinHilbertSpace, GeneralHamiltonian, SiteOperators
from meanfieldincrements.FactorizedMarginal import FactorizedMarginal
from meanfieldincrements.Energy import compute_constraints, energy_from_expvals, build_local_expvals
from meanfieldincrements.GeneralHamiltonian import build_heisenberg_hamiltonian, build_ising_hamiltonian
from meanfieldincrements.LagrangeMultipliers import LagrangeMultipliers
from meanfieldincrements.Marginals import Marginals, build_Marginals_from_LocalTensor

def test_energy():

    # create lattice containing different types of sites
    sites = []
    sites.append(Site(0, SpinHilbertSpace(2)))
    sites.append(Site(1, SpinHilbertSpace(3)))
    sites.append(Site(2, SpinHilbertSpace(2)))
    sites.append(Site(3, SpinHilbertSpace(1)))
    sites.append(Site(4, SpinHilbertSpace(2)))
    sites.append(Site(5, SpinHilbertSpace(1)))


    #build operator library, which constructs a set of local matrices for acting on different types of hilbert spaces 
    oplib = {}
    for site in sites:
        oplib[site.hilbert_space] = SiteOperators(site.hilbert_space)
    

    H = build_heisenberg_hamiltonian(sites)
    print(H, flush=True)
    Hmat = H.matrix(oplib)
    print("Hamiltonian matrix:", Hmat.shape, flush=True)
    # Diagonalize the Hermitian matrix Hmat
    eigvals, eigvecs = np.linalg.eigh(Hmat)
    lowest_idx = np.argmin(eigvals)
    lowest_energy = eigvals[lowest_idx]
    v = eigvecs[:, lowest_idx]
    for i in eigvals:
        print("  %12.8f"%i)
    print("Lowest energy eigenvalue:", lowest_energy, flush=True)

    rho = build_Marginals_from_LocalTensor(LocalTensor(np.outer(v,v), sites), n_body=2)
    rho.fold()
    print(rho)

    local_expvals = build_local_expvals(H, rho, oplib)

    print("local_expvals")
    for site,evals in local_expvals.items():
        print(site)
        for opstr,val in evals.items():
            if np.abs(val) < 1e-13:
                continue
            print("  %12.8f %12.8fi" %(np.real(val), np.imag(val)), " ", opstr) 


    print(" Now call energy function")
    e = energy_from_expvals(H, local_expvals)

    assert np.isclose(e, lowest_energy ) 


    #
    print("\n Now test factorized marginals")
    for term in rho.keys():
        rho[term] = FactorizedMarginal.from_Marginal(rho[term])
    rho.unfold()
    print(rho)
    local_expvals = build_local_expvals(H, rho, oplib)

    print("local_expvals")
    for site,evals in local_expvals.items():
        print(site)
        for opstr,val in evals.items():
            if np.abs(val) < 1e-13:
                continue
            print("  %12.8f %12.8fi" %(np.real(val), np.imag(val)), " ", opstr) 


    print(" Now call energy function")
    e = energy_from_expvals(H, local_expvals)

    assert np.isclose(e, lowest_energy ) 

    lang_mults = LagrangeMultipliers(sites).initialize_to_zero(nbody=2)

    # rho.unfold()
    mvec, pm_meta_data = rho.export_to_vector() 
    lvec, lm_meta_data = lang_mults.export_to_vector()
    n_lang_mults = len(lvec)
    n_marginal = len(mvec)

    pvec = np.concatenate((mvec, lvec))
    n_params = len(pvec)
    
    print("n_lang_mults: ", n_lang_mults)
    print("n_marginal:   ", n_marginal)
    print("n_params:     ", n_params)
    
    print(lang_mults)

    constraint = compute_constraints(rho, lang_mults)

    print(" Penalty from constraints:")
    print(constraint)

    def loss(v, rho=rho, oplib=oplib):
        # rho.unfold()
        rho.import_from_vector(v[0:n_marginal], pm_meta_data)
        lang_mults.import_from_vector(v[n_marginal:], lm_meta_data)
    
        local_expvals = build_local_expvals(H, rho, oplib)
        e = energy_from_expvals(H, local_expvals)
        constraint = compute_constraints(rho, lang_mults)
        return e + constraint

    l = loss(pvec)
    print(" Loss function: %12.8f %12.8fi" %(np.real(l), np.imag(l)))

    l = loss(np.random.rand(n_params))
    print(" Loss function rand1: %12.8f %12.8fi" %(np.real(l), np.imag(l)))
    
    print("export then import")
    mvec, pm_meta_data = rho.export_to_vector() 
    lvec, lm_meta_data = lang_mults.export_to_vector()
    pvec = np.concatenate((mvec, lvec))
    l2 = loss(pvec)
    print(" Loss function rand2: %12.8f %12.8fi" %(np.real(l2), np.imag(l2)))

    assert np.isclose(l2, l)

if __name__ == "__main__":
    # Run tests manually
    test_energy()