import numpy as np
from meanfieldincrements import Site, LocalOperator, RhoMBE, Marginals, Cumulants
from itertools import combinations

def test_marginals():
    """Test the MBE trace functions with proper correction terms."""
    np.random.seed(42)  # For reproducibility 
    
    # Test 1: Pure mean-field state (no corrections)
    sites = [Site(0, 2), Site(1, 4), Site(2, 3), Site(3, 4)]
    marginals = Marginals(sites)
    cumulants = Cumulants(sites)

    # Build target density 
    total_dim = np.prod([site.dimension for site in sites])
    r_exact = np.random.rand(total_dim, total_dim) + 1j * np.random.rand(total_dim, total_dim)
    r_exact = (r_exact + r_exact.conj().T) / 2  # Ensure Hermitian
    r_exact /= np.trace(r_exact)  # Normalize

    r_exact = LocalOperator(r_exact, sites).fold()
    print(" r_exact: ")
    print(r_exact)

    # Fill 1-body Marginals/Cumulants
    for si in sites:
        env = [s.label for s in sites if s.label != si.label]
        marginals[(si.label,)] = r_exact.partial_trace(env).fold()
    
    for (si, sj) in combinations(sites, 2):
        env = [s.label for s in sites if s.label not in (si.label, sj.label)]
        marginals[(si.label, sj.label)] = r_exact.partial_trace(env).fold()
        cumulants[(si.label, sj.label)] = r_exact.compute_2body_cumulant(si, sj).fold()
    
    for (si, sj, sk) in combinations(sites, 3):
        env = [s.label for s in sites if s.label not in (si.label, sj.label, sk.label)]
        marginals[(si.label, sj.label, sk.label)] = r_exact.partial_trace(env).fold()
        cumulants[(si.label, sj.label, sk.label)] = r_exact.compute_3body_cumulant(si, sj, sk).fold()
    
    # for (si, sj, sk, sl) in combinations(sites, 4):
    #     env = [s.label for s in sites if s.label not in (si.label, sj.label, sk.label, sl.label)]
    #     marginals[(si.label, sj.label, sk.label)] = r_exact.partial_trace(env).fold()
    #     cumulants[(si.label, sj.label, sk.label)] = r_exact.compute_4body_cumulant(si, sj, sk).fold()
    
    
    print(marginals)
    print(cumulants)

    print(" Single site traces:")
    for tr_sites in combinations(sites, 1):
        tmp = cumulants.partial_trace([si.label for si in tr_sites])
        print(tmp)
    print(" Two site traces:")
    for tr_sites in combinations(sites, 2):
        tmp = cumulants.partial_trace([si.label for si in tr_sites])
        print(tmp)
    print(" Three site traces:")
    for tr_sites in combinations(sites, 3):
        tmp = cumulants.partial_trace([si.label for si in tr_sites])
        print(tmp)
    return
    l02 = cumulants.partial_trace([sites[1].label,])
    print(l02)
    l12 = cumulants.partial_trace([sites[0].label,])
    print(l12)

    l0 = cumulants.partial_trace([sites[1].label, sites[2].label,])
    print(l0)
    l1 = cumulants.partial_trace([sites[0].label, sites[2].label,])
    print(l1)
    l2 = cumulants.partial_trace([sites[0].label, sites[1].label,])
    print(l2)

if __name__ == "__main__":
    test_marginals()