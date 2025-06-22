from itertools import combinations
import pytest
import numpy as np
import scipy
from meanfieldincrements import Site, LocalTensor, PauliHilbertSpace, SpinHilbertSpace, GeneralHamiltonian, SiteOperators
from meanfieldincrements.FactorizedMarginal import FactorizedMarginal
from meanfieldincrements.Energy import compute_constraints, energy_from_expvals, build_local_expvals
from meanfieldincrements.GeneralHamiltonian import build_heisenberg_hamiltonian, build_ising_hamiltonian
from meanfieldincrements.LagrangeMultipliers import LagrangeMultipliers
from meanfieldincrements.Marginals import Marginals, build_Marginals_from_LocalTensor
from scipy.optimize import minimize

def constrained_optimization_with_monitoring(marginals, hamiltonian, oplib):
    
    # Storage for monitoring
    progress = {
        'iteration': [],
        'energy': [],
        'constraint_violations': [],
        'max_violation': [],
        'total_violation_norm': []
    }
    
    def objective(param_vector):
        marginals.import_from_vector(param_vector, marginal_metadata)
        local_expvals = build_local_expvals(hamiltonian, marginals, oplib)
        energy = energy_from_expvals(hamiltonian, local_expvals)
        return energy
    
    def constraint_functions(param_vector):
        marginals.import_from_vector(param_vector, marginal_metadata)
        violations = []
        
        for (si, sj) in combinations(hamiltonian.sites, 2):
            if (si.label, sj.label) in marginals:
                rho_ij = marginals[(si.label, sj.label)]
                rho_j = marginals[(sj.label,)]
                
                traced = rho_ij.partial_trace([si.label])
                violation = (traced.tensor - rho_j.tensor).flatten()
                # violation = 1 - np.trace(traced.tensor.conj().T @ rho_j.tensor)
                violations.append(violation)
                
                # now swapped
                rho_ij = marginals[(si.label, sj.label)]
                rho_j = marginals[(si.label,)]
                
                traced = rho_ij.partial_trace([sj.label])
                violation = (traced.tensor - rho_j.tensor).flatten()
                # violation = 1 - np.trace(traced.tensor.conj().T @ rho_j.tensor)
                violations.append(violation)

        # return violations        
        return np.concatenate(violations) if violations else np.array([])
    
    # Callback function for monitoring
    def optimization_callback(param_vector):
        iteration = len(progress['iteration'])
        
        # Evaluate energy
        energy = objective(param_vector)
        
        # Evaluate constraint violations
        constraint_vals = constraint_functions(param_vector)
        total_violation_norm = np.linalg.norm(constraint_vals)
        max_violation = np.max(np.abs(constraint_vals)) if len(constraint_vals) > 0 else 0.0
        
        # Store progress
        progress['iteration'].append(iteration)
        progress['energy'].append(energy)
        progress['constraint_violations'].append(constraint_vals.copy())
        progress['max_violation'].append(max_violation)
        progress['total_violation_norm'].append(total_violation_norm)
        
        # Print progress
        # if iteration % 5 == 0 or iteration < 10:  # Print every 5 iterations initially
        print(f"Iteration {iteration:3d}: "
                f"Energy = {energy:12.8f}, "
                f"Max violation = {max_violation:10.6e}, "
                f"Total violation = {total_violation_norm:10.6e}")
        
        # Optional: Early stopping if constraints are well satisfied
        if total_violation_norm < 1e-10:
            print(f"Constraints satisfied to high precision at iteration {iteration}")
            
        return False  # Return True to stop optimization early if needed
    
    # Set up optimization
    constraints = {'type': 'eq', 'fun': constraint_functions}
    x0, marginal_metadata = marginals.export_to_vector()
    
    print("Starting constrained optimization...")
    print("="*70)
    
    result = scipy.optimize.minimize(
        objective, 
        x0, 
        method='SLSQP',
        constraints=constraints,
        callback=optimization_callback,
        options={
            'ftol': 1e-12, 
            'disp': True,
            'maxiter': 1000
        }
    )
    
    print("="*70)
    print(f"Optimization completed. Final energy: {result.fun:.10f}")
    
    # Update marginals with solution
    marginals.import_from_vector(result.x, marginal_metadata)
    
    return result, progress

def test_energy():

    np.random.seed(2)

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
    for term,coeff in H.items():
        H[term] = np.random.rand() - .5

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
    print(len(lvec))

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
        constraint = compute_constraints(rho, lang_mults, verbose=False)
        return e - constraint
        # return -constraint

    global e_curr
    e_curr = 0
    # def penalized_loss(v, rho=rho, oplib=oplib, mu=10.):
    mu=.100
    def penalized_loss(v):
        # rho.unfold()
        rho.import_from_vector(v, pm_meta_data)
        lang_mults.import_from_vector([mu for i in range(n_lang_mults)], lm_meta_data)
    
        local_expvals = build_local_expvals(H, rho, oplib)

        e = energy_from_expvals(H, local_expvals)
        
        global e_curr
        e_curr = e
        
        constraint = compute_constraints(rho, lang_mults, verbose=False)
        # print(" %12.8f %12.8f" %(np.real(e), constraint))
        return np.real(e - mu*constraint)
        # return -constraint
    def callback(v):
        rho.import_from_vector(v, pm_meta_data)
        local_expvals = build_local_expvals(H, rho, oplib)
        e = energy_from_expvals(H, local_expvals)
        current_loss = penalized_loss(v)
        print(" Iteration loss: %12.8f energy: %12.8f |pvec|: %12.8f" %(current_loss, e, np.linalg.norm(v)))

    l = loss(pvec)
    print(" Loss function: %12.8f %12.8fi" %(np.real(l), np.imag(l)))
    
    
    # l = loss(np.random.rand(n_params))
    # print(" Loss function rand1: %12.8f %12.8fi" %(np.real(l), np.imag(l)))
    
    print("export then import")
    mvec, pm_meta_data = rho.export_to_vector() 
    lvec, lm_meta_data = lang_mults.export_to_vector()
    pvec = np.concatenate((mvec, lvec))
    l2 = loss(pvec)
    print(" Loss function rand2: %12.8f %12.8fi" %(np.real(l2), np.imag(l2)))

    assert np.isclose(l2, l)

    if False:
        print("\n ###########\n now optimize")
        # Define a wrapper for the loss function to handle real-valued inputs

        # Use scipy's minimize function to optimize the loss
        # Define a callback function to print the loss value at each iteration

        result = minimize(penalized_loss, mvec, method='BFGS', callback=callback, options={'disp': True, 'maxiter': 20})

        # Extract the optimized parameters
        optimized_pvec = result.x

    # # Re-import the optimized parameters into rho and lang_mults
    # rho.import_from_vector(optimized_pvec[0:n_marginal], pm_meta_data)
    # lang_mults.import_from_vector(optimized_pvec[n_marginal:], lm_meta_data)
    # # Compute the final energy and constraints
    # local_expvals = build_local_expvals(H, rho, oplib)
    # final_energy = energy_from_expvals(H, local_expvals)
    # final_constraint = compute_constraints(rho, lang_mults)

    # print("Optimized energy: %12.8f" % final_energy)
    # print("Final constraint penalty: %12.8f" % -final_constraint)

    def constrained_optimization_scipy(marginals, hamiltonian, oplib):
        
        # Objective function: just the energy
        def objective(param_vector):
            marginals.import_from_vector(param_vector, marginal_metadata)
            local_expvals = build_local_expvals(hamiltonian, marginals, oplib)
            energy = energy_from_expvals(hamiltonian, local_expvals)
            return energy
        
        # Constraint functions: should return 0 when satisfied
        def constraint_functions(param_vector):
            marginals.import_from_vector(param_vector, marginal_metadata)
            violations = []
            
            # For each pair (i,j): tr_i(ρ^{(ij)}) - ρ^{(j)} = 0
            for (si, sj) in combinations(hamiltonian.sites, 2):
                if (si.label, sj.label) in marginals:
                    rho_ij = marginals[(si.label, sj.label)]
                    rho_j = marginals[(sj.label,)]
                    
                    # Partial trace of ij marginal over site i
                    traced = rho_ij.partial_trace([si.label])
                    
                    # Flatten the matrix difference for scipy
                    violation = (traced.tensor - rho_j.tensor).flatten()
                    violations.append(violation)
            
            return np.concatenate(violations) if violations else np.array([])
        
        # Set up constraint dictionary
        constraints = {
            'type': 'eq',
            'fun': constraint_functions
        }
        
        # Initial parameter vector
        x0, marginal_metadata = marginals.export_to_vector()
        
        # Optimize!
        result = scipy.optimize.minimize(
            objective, 
            x0, 
            method='SLSQP',  # or 'trust-constr'
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': True}
        )
        
        # Update marginals with solution
        marginals.import_from_vector(result.x, marginal_metadata)
        return result



    # constrained_optimization_scipy(rho, H, oplib)

    rho.import_from_vector(mvec, pm_meta_data)
    pvec += np.random.rand(len(pvec))*.001
    print(loss(pvec))
    result, progress = constrained_optimization_with_monitoring(rho, H, oplib)



if __name__ == "__main__":
    # Run tests manually
    test_energy()