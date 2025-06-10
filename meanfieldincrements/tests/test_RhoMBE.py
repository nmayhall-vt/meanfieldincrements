import pytest
import numpy as np
from meanfieldincrements import Site, LocalOperator, RhoMBE
from itertools import combinations

# Test functions for the corrected MBE trace operations
def test_mbe_trace_functions():
    """Test the MBE trace functions with proper correction terms."""
    np.random.seed(42)  # For reproducibility 
    print("Testing MBE trace functions...")
    
    # Test 1: Pure mean-field state (no corrections)
    sites = [Site(0, 2), Site(1, 4), Site(2, 3)]
    rho = RhoMBE(sites).initialize_mixed()

    # Build target density 
    total_dim = np.prod([site.dimension for site in sites])
    r_exact = np.random.rand(total_dim, total_dim) + 1j * np.random.rand(total_dim, total_dim)
    r_exact = (r_exact + r_exact.conj().T) / 2  # Ensure Hermitian
    r_exact /= np.trace(r_exact)  # Normalize
    
    lo = LocalOperator(r_exact, sites)
    print(lo)

    print(f"   Mean-field trace: {rho.trace():.6f}")
    print(f"   Expected: {1.0:.6f}")
    
    print("\n1. Adding 1-body term:")
    for si in sites:
        env = [s for s in sites if s.label != si.label]
        rho_i = lo.compute_nbody_marginal([si])
        rho.nbody_terms[1][(si.label,)] = rho_i 
    
    # Test 2: Add a 2-body correction
    print("\n2. Adding 2-body correction term:")

    rho.nbody_terms[2] = {}
    for (si, sj) in combinations(sites, 2):
        print(f"   Adding correction for sites {si.label} and {sj.label}")

        lambda_ij = lo.compute_2body_cumulant(si, sj)
        print("     trace of lambda_ij: ", lambda_ij.trace())
        print("     norm  of lambda_ij: ", np.linalg.norm(lambda_ij.tensor))
        rho.nbody_terms[2][si.label, sj.label] = lambda_ij
    
    
    print("\n3. Adding 3-body correction term:")
    
    rho.nbody_terms[3] = {}
    for (si, sj, sk) in combinations(sites, 3):
        print(f"   Adding correction for sites %i %i %i " %(si.label, sj.label, sk.label))

        lambda_ijk = lo.compute_3body_cumulant(si, sj, sk)
        print("     trace of lambda_ijk: ", lambda_ijk.trace())
        print("     norm  of lambda_ijk: ", np.linalg.norm(lambda_ijk.tensor))
        rho.nbody_terms[3][si.label, sj.label, sk.label] = lambda_ijk


    print("\n4. Rebuilding the density matrix:")
    # Check that we rebuild the 3site rdm 
    a = rho.nbody_terms[1][(sites[0].label,)].fold().tensor
    b = rho.nbody_terms[1][(sites[1].label,)].fold().tensor
    c = rho.nbody_terms[1][(sites[2].label,)].fold().tensor
    ab = rho.nbody_terms[2][sites[0].label, sites[1].label].fold().tensor
    ac = rho.nbody_terms[2][sites[0].label, sites[2].label].fold().tensor
    bc = rho.nbody_terms[2][sites[1].label, sites[2].label].fold().tensor
    abc = rho.nbody_terms[3][sites[0].label, sites[1].label, sites[2].label].fold().tensor

    r_rebuilt =  np.einsum('iI,jJ,kK->ijkIJK', a, b, c)
    r_rebuilt += np.einsum('ijIJ,kK->ijkIJK', ab, c)
    r_rebuilt += np.einsum('ikIK,jJ->ijkIJK', ac, b)
    r_rebuilt += np.einsum('jkJK,iI->ijkIJK', bc, a)
    r_rebuilt += abc 

    print(" norm of rho - rho_rebuilt: ", np.linalg.norm(lo.fold().tensor - r_rebuilt))

    np.testing.assert_allclose(lo.fold().tensor, r_rebuilt, atol=1e-10)
if __name__ == "__main__":
    test_mbe_trace_functions()
