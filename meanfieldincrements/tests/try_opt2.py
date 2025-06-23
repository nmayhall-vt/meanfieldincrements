import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from meanfieldincrements import LocalTensor, Marginals, Site
from meanfieldincrements import GeneralHamiltonian
from meanfieldincrements.Energy import build_local_expvals, energy_from_expvals
from meanfieldincrements.GeneralHamiltonian import build_heisenberg_hamiltonian
from meanfieldincrements.HilbertSpace import SpinHilbertSpace
from meanfieldincrements import build_Marginals_from_LocalTensor

class OptimizeEnergy:
    def __init__(self, H:'GeneralHamiltonian', M:'Marginals'):
        self.hamiltonian = H
        self.marginals = M
        self.oplib = H.build_SiteOperators()
        self.xcurr, self.mdata = self.marginals.export_to_vector()
        self._iter = 0
 
    def objective(self, x: np.ndarray):
        self.marginals.import_from_vector(x, self.mdata)
        local_expvals = build_local_expvals(self.hamiltonian, self.marginals, self.oplib)
        e = energy_from_expvals(self.hamiltonian, local_expvals)
        return np.real_if_close(e)
    
    def constraint_violations(self, x: np.ndarray):
        # violations = self.marginals.compute_constraint_violations()
        violations = []
        self.marginals.fold()
        for sites in self.marginals.keys():
            if len(sites) == 2:
                si,sj = sites 

                mi  = self.marginals[(si, )]
                mj  = self.marginals[(sj, )]
                mij = self.marginals[(si, sj)]

                trace_i = mij.partial_trace([si])
                violation = np.linalg.norm(trace_i.tensor - mj.tensor)
                violations.append(violation)

                trace_j = mij.partial_trace([sj])
                violation = np.linalg.norm(trace_j.tensor - mi.tensor)
                violations.append(violation)

        return np.real_if_close(violations)


    def setup_single_constraint(self):
        # Set up constraints

        def my_constraint(x):
            self.marginals.import_from_vector(x, self.mdata)
            return np.linalg.norm(self.constraint_violations(x))

        constraints = []
        # Trace constraints (equality)
        trace_constraint = NonlinearConstraint(
            my_constraint, 
            lb=0.0, ub=0.0,  # equality constraint
            jac='3-point'  # finite difference jacobian
        )
        constraints.append(trace_constraint)
        return constraints

    def callback(self, x):
        self._iter += 1
        ecurr = self.objective(x)
        ccurr = np.linalg.norm(self.constraint_violations(x))
        print(" Iteration: %3i Energy = %12.8f Constraint = %12.8f" %(self._iter, ecurr, ccurr))

def run():

    np.random.seed(2)

    # create lattice containing different types of sites
    sites = []
    sites.append(Site(0, SpinHilbertSpace(2)))
    sites.append(Site(1, SpinHilbertSpace(2)))
    sites.append(Site(2, SpinHilbertSpace(2)))
    sites.append(Site(3, SpinHilbertSpace(2)))


    H = build_heisenberg_hamiltonian(sites)
    # for term,coeff in H.items():
    #     if term[0] == "I" and term[3] == "I":
    #         H[term] = 0 
    # for term,coeff in H.items():
    #     H[term] = np.random.rand() - .5
    print(H, flush=True)
    print("\n Compute exact solutions")
    Hmat = H.matrix(H.build_SiteOperators())
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
    M = build_Marginals_from_LocalTensor(LocalTensor(np.outer(v,v), sites), n_body=2)
    print(M)
    M.print_cumulants()

    print("\n Now optimize from scratch")
    M = Marginals().initialize_maximally_mixed(sites)
    print(M)
    M.print_cumulants()
    
    myopt = OptimizeEnergy(H, M)
    x0 = np.real_if_close(myopt.xcurr)
    print(myopt.objective(x0), myopt.constraint_violations(x0))

    # Solve
    result = minimize(
        myopt.objective, 
        x0, 
        method='SLSQP',  # or 'trust-constr'
        constraints=myopt.setup_single_constraint(),
        callback=myopt.callback,
        options={'disp': True}
    )


    M.import_from_vector(result.x, myopt.mdata)
    print(M)
    M.print_cumulants()

run()