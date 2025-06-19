"""
Unit tests for the Marginal class.
"""

import pytest
import numpy as np
from meanfieldincrements import Site, LocalTensor, PauliHilbertSpace, SpinHilbertSpace, GeneralHamiltonian
from meanfieldincrements.Marginal import Marginal
from meanfieldincrements.FactorizedMarginal import FactorizedMarginal

def test_marginal_construction():
    """Test basic Marginal construction."""
    sites = [Site(0, PauliHilbertSpace(2)), Site(1, SpinHilbertSpace(4)), Site(2,PauliHilbertSpace(4))]

    np.random.seed(42)  # For reproducibility
    oplib = {}
    for site in sites:
        oplib[site.hilbert_space] = site.hilbert_space.create_operators()
    # ops = [site.hilbert_space.create_operators() for site in sites]

    for site in sites:
        print(site.hilbert_space)
        print(oplib[site.hilbert_space].keys())

    ham = GeneralHamiltonian(sites)
    ham["X","Sz","XZ"] = 1.0
    ham["X","Sx","XY"] = -1.0

    Hmat = ham.matrix(oplib)
    print(np.linalg.norm(Hmat - Hmat.conj().T))
    # print(np.linalg.eigvals(Hmat))


    # Create a random matrix for the marginal 
    dim_tot = np.prod([site.hilbert_space.dimension for site in sites])
    matrix = np.random.random((dim_tot, dim_tot)) + 1j * np.random.random((dim_tot, dim_tot))
    # make positive semidefinite
    matrix = matrix @ matrix.conj().T
    matrix /= np.trace(matrix)  # Normalize to trace 1
    
    marginal = Marginal(matrix.copy(), sites)
    lt = LocalTensor(matrix.copy(), sites)
    assert(marginal.trace() == np.trace(matrix))

    np.isclose(marginal.fold().unfold().tensor, matrix)

    assert(len(marginal.partial_trace([sites[0]]).sites) == 2)
    assert(len(marginal.partial_trace([sites[0], sites[1]]).sites) == 1)
    assert(len(marginal.partial_trace(sites).sites) == 0)

    ev_ref = np.trace(matrix @ Hmat)
    print("Reference expectation value:", ev_ref)

    ev = 0
    for op,coeff in ham.items():
        print(f"Operator: {op}, Coefficient: {coeff}") 
        ev += marginal.contract_operators(op, oplib) * coeff
    print("Computed expectation value:", ev)
    assert np.isclose(ev, ev_ref), f"Expectation value mismatch: {ev} != {ev_ref}"


    print(" Now test factorized marginals")
    # Create Bell state factorization
    fmarginal = FactorizedMarginal.from_density_matrix(matrix, sites)
    assert np.linalg.norm(fmarginal.tensor - matrix) < 1e-10
    print(fmarginal)
    
    assert(len(fmarginal.partial_trace([sites[0]]).sites) == 2)
    assert(len(fmarginal.partial_trace([sites[0], sites[1]]).sites) == 1)
    assert(len(fmarginal.partial_trace(sites).sites) == 0)
    
    ev2 = 0
    for op,coeff in ham.items():
        print(f"Operator: {op}, Coefficient: {coeff}") 
        ev2 += fmarginal.contract_operators(op, oplib) * coeff
    print("Computed expectation value:", ev)
    assert np.isclose(ev2, ev_ref), f"Expectation value mismatch: {ev2} != {ev_ref}"



if __name__ == "__main__":
    # Run tests manually
    test_marginal_construction()