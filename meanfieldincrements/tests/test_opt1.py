from itertools import combinations
import pytest
import numpy as np
import scipy
from meanfieldincrements import Site, LocalTensor, PauliHilbertSpace, SpinHilbertSpace, GeneralHamiltonian, SiteOperators
from meanfieldincrements.FactorizedMarginal import FactorizedMarginal
from meanfieldincrements.Energy import compute_constraints, energy_from_expvals, build_local_expvals, get_cost_function_penalty
from meanfieldincrements.GeneralHamiltonian import build_heisenberg_hamiltonian, build_ising_hamiltonian
from meanfieldincrements.LagrangeMultipliers import LagrangeMultipliers
from meanfieldincrements.Marginals import Marginals, build_Marginals_from_LocalTensor
from scipy.optimize import minimize
from meanfieldincrements import get_cost_function_full

def test_energy():

    np.random.seed(2)

    # create lattice containing different types of sites
    sites = []
    sites.append(Site(0, SpinHilbertSpace(2)))
    sites.append(Site(1, SpinHilbertSpace(3)))
    # sites.append(Site(2, SpinHilbertSpace(2)))
    # sites.append(Site(3, SpinHilbertSpace(1)))
    # sites.append(Site(4, SpinHilbertSpace(2)))
    # sites.append(Site(5, SpinHilbertSpace(2)))



    H = build_heisenberg_hamiltonian(sites)
    for term,coeff in H.items():
        H[term] = np.random.rand() - .5
    print(H, flush=True)

    oplib = H.build_SiteOperators()

    print("\n Compute exact solutions")
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

    print(" First make sure our exact marginals match exact energy")
    rho = build_Marginals_from_LocalTensor(LocalTensor(np.outer(v,v), sites), n_body=2)
    rho.fold()
    print(rho)

    print(" Now call energy function")


    e = energy_from_expvals(H, build_local_expvals(H, rho, oplib))

    assert np.isclose(e, lowest_energy ) 

    #
    print("\n Now test factorized marginals")
    for term in rho.keys():
        rho[term] = FactorizedMarginal.from_Marginal(rho[term])
    rho.unfold()
    print(rho)
    local_expvals = build_local_expvals(H, rho, oplib)


    print(" Now call energy function")
    e = energy_from_expvals(H, local_expvals)
    assert np.isclose(e, lowest_energy ) 


    print(" Now optimize")
    cost, callback, x0 = get_cost_function_penalty(H, rho, mu=100000.)

    e = cost(x0)

    print(" cost_function: ", e)  
    assert np.isclose(e, lowest_energy ) 
    # rhomat = rho.build_full_matrix()

    # rho.initialize_maximally_mixed(sites) 
    cost, callback, x0 = get_cost_function_penalty(H, rho, mu=1e2)
    
    # randomize
    x0 += (np.random.rand(len(x0))-.5)*.001
    rho.print_cumulants()

    result = minimize(cost, x0, method='BFGS', callback=callback, options={'disp': True, 'maxiter': 200})
    # result = minimize(cost, x0, method='bfgs', options={'disp': True, 'maxiter': 200})
    result = minimize(cost, x0, method='cobyla', callback=callback, options={'disp': False, 'maxiter': 2000})

    # Extract the optimized parameters
    optimized_pvec = result.x
    mvec,mdata = rho.export_to_vector()
    rho.import_from_vector(optimized_pvec, mdata)
    print(rho)
    rho.print_cumulants()
    # rhomat = rho.build_full_matrix()

if __name__ == "__main__":
    # Run tests manually
    test_energy()