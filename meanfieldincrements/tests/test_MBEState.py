import pytest
import numpy as np
import copy as cp
from meanfieldincrements import Site, LocalOperator, MBEState
from itertools import combinations

def _rebuild_2site_rdm(rho):
    sites = rho.sites
    a = rho[(sites[0].label,)].fold().tensor
    b = rho[(sites[1].label,)].fold().tensor
    ab = rho[sites[0].label, sites[1].label].fold().tensor

    r_rebuilt =  np.einsum('iI,jJ->ijIJ', a, b)
    r_rebuilt += ab 
    

    return r_rebuilt

def _rebuild_3site_rdm(rho, nbody=3):
    sites = rho.sites
    a = rho.terms[(sites[0].label,)].fold().tensor
    b = rho.terms[(sites[1].label,)].fold().tensor
    c = rho.terms[(sites[2].label,)].fold().tensor
    ab = rho.terms[sites[0].label, sites[1].label].fold().tensor
    ac = rho.terms[sites[0].label, sites[2].label].fold().tensor
    bc = rho.terms[sites[1].label, sites[2].label].fold().tensor

    r_rebuilt =  np.einsum('iI,jJ,kK->ijkIJK', a, b, c)
    r_rebuilt += np.einsum('ijIJ,kK->ijkIJK', ab, c)
    r_rebuilt += np.einsum('ikIK,jJ->ijkIJK', ac, b)
    r_rebuilt += np.einsum('jkJK,iI->ijkIJK', bc, a)
    if nbody > 2:
        abc = rho.terms[sites[0].label, sites[1].label, sites[2].label].fold().tensor
        r_rebuilt += abc 
    
    
    return r_rebuilt

# Test functions for the corrected MBE trace operations
def test_mbe_trace_functions():
    """Test the MBE trace functions with proper correction terms"""
    np.random.seed(42)  # For reproducibility 
    print("Testing MBE trace functions...")
    
    # Test 1: Pure mean-field state (no corrections)
    sites = [Site(0, 2), Site(1, 4), Site(2, 3)]
    rho = MBEState(sites).initialize_mixed()

    # Build target density 
    total_dim = np.prod([site.dimension for site in sites])
    r_exact = np.random.rand(total_dim, total_dim) + 1j * np.random.rand(total_dim, total_dim)
    r_exact = (r_exact + r_exact.conj().T) / 2  # Ensure Hermitian
    r_exact /= np.trace(r_exact)  # Normalize
    
    lo = LocalOperator(r_exact, sites)
    print(lo)

    print(f"   Expected: {1.0:.6f}")
    
    print("\n1. Adding 1-body term:")
    for si in sites:
        rho.terms[(si.label,)] = lo.compute_nbody_marginal([si]) 

    # Test 2: Add a 2-body correction
    print("\n2. Adding 2-body correction term:")

    for (si, sj) in combinations(sites, 2):
        print(f"   Adding correction for sites {si.label} and {sj.label}")
        rij = lo.compute_nbody_marginal([si,sj])
        rho.terms[(si.label, sj.label)] = rho.compute_2body_cumulant(rij)
    
    
    print("\n3. Adding 3-body correction term:")
    for (si, sj, sk) in combinations(sites, 3):
        print(f"   Adding correction for sites %i %i %i " %(si.label, sj.label, sk.label))
        rijk = lo.compute_nbody_marginal([si,sj,sk])
        rho.terms[(si.label, sj.label, sk.label)] = rho.compute_3body_cumulant(rijk)
    
    


    print("   rho: ")
    print(rho)
    print("\n4. Rebuilding the density matrix:")
    # Check that we rebuild the 3site rdm 
    r_rebuilt = _rebuild_3site_rdm(rho, nbody=3)
    print(" norm of rho - rho_rebuilt: ", np.linalg.norm(lo.fold().tensor - r_rebuilt))
    np.testing.assert_allclose(lo.fold().tensor, r_rebuilt, atol=1e-10)

    print("\n5. Check Partial Trace:")
    print(rho)
    rho_0 = rho.partial_trace([sites[1].label, sites[2].label])
    rho_1 = rho.partial_trace([sites[0].label, sites[2].label])
    rho_2 = rho.partial_trace([sites[0].label, sites[1].label])
    rho_01 = rho.partial_trace([sites[2].label])
    rho_02 = rho.partial_trace([sites[1].label])

    r0_ref = lo.partial_trace([sites[1].label, sites[2].label])
    r1_ref = lo.partial_trace([sites[0].label, sites[2].label])
    r2_ref = lo.partial_trace([sites[0].label, sites[1].label])
    r01_ref = lo.partial_trace([sites[2].label])
    r02_ref = lo.partial_trace([sites[1].label])

    print(" Trace out site 0,1: ")
    tst = rho_2[(2,)].fold().tensor 
    # tst = np.einsum('ijkijK->kK', lo.tensor)
    ref = r2_ref.fold().tensor
    # print(tst)
    # print("ref:")
    # print(ref)
    print("   shapes: ", ref.shape, " ", tst.shape)
    print("   norm of rho - rho_rebuilt: ", np.linalg.norm(ref - tst))
    np.testing.assert_allclose(ref, tst, atol=1e-10)
    
    print(" Trace out site 0,2: ")
    tst = rho_1.terms[(1,)].fold().tensor 
    ref = r1_ref.fold().tensor
    print("   shapes: ", ref.shape, " ", tst.shape)
    print("   norm of rho - rho_rebuilt: ", np.linalg.norm(ref - tst))
    np.testing.assert_allclose(ref, tst, atol=1e-10)
    
    print(" Trace out site 1,2: ")
    tst = rho_0.terms[(0,)].fold().tensor 
    ref = r0_ref.fold().tensor
    print("   shapes: ", ref.shape, " ", tst.shape)
    print("   norm of rho - rho_rebuilt: ", np.linalg.norm(ref - tst))
    np.testing.assert_allclose(ref, tst, atol=1e-10)
    
    print(" Trace out site 0: ")
    rho_12 = rho.partial_trace([sites[0].label])
    r12_ref = lo.partial_trace([sites[0].label]) 
    tst = _rebuild_2site_rdm(rho_12) 
    ref = r12_ref.fold().tensor

    print("   shapes: ", ref.shape, " ", tst.shape)
    print("   norm of rho - rho_rebuilt: ", np.linalg.norm(ref - tst))
    print("   tr(ref): ", np.einsum('ijij->', ref))
    print("   tr(tst): ", np.einsum('ijij->', tst))
    print("   norm(ref): ", np.linalg.norm(ref))
    print("   norm(tst): ", np.linalg.norm(tst))
    # np.testing.assert_allclose(ref, tst, atol=1e-10)
    

if __name__ == "__main__":
    test_mbe_trace_functions()
