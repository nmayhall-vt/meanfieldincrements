import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
from meanfieldincrements import FactorizedMarginal, LocalTensor, Marginals, Site
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
        self.constraints = []
 
    def objective(self, x: np.ndarray):
        self.marginals.import_from_vector(x, self.mdata)
        local_expvals = build_local_expvals(self.hamiltonian, self.marginals, self.oplib)
        e = energy_from_expvals(self.hamiltonian, local_expvals)
        return np.real_if_close(e)
    


    def setup_2b_constraints(self):
        # Set up constraints

        constraints = []
        def my_constraint(x):
            self.marginals.import_from_vector(x, self.mdata)
            return np.linalg.norm(self.marginals.compute_2b_violations())

        trace_constraint = NonlinearConstraint(
            my_constraint, 
            lb=0.0, ub=0.0,  # equality constraint
            jac='3-point'  # finite difference jacobian
        )
        constraints.append(trace_constraint)
        self.constraints.append("2b")
        return constraints

    def setup_3b_constraints(self):
        # Set up constraints

        n_3b = 0
        for subset in self.marginals.keys():
            if len(subset) == 3:
                n_3b += 1
        
        if n_3b < 1:
            raise RuntimeError(" No 3-body marginals found")

        constraints = []
        def my_constraint(x):
            self.marginals.import_from_vector(x, self.mdata)
            return np.linalg.norm(self.marginals.compute_3b_violations())

        trace_constraint = NonlinearConstraint(
            my_constraint, 
            lb=0.0, ub=0.0,  # equality constraint
            jac='3-point'  # finite difference jacobian
        )
        constraints.append(trace_constraint)
        self.constraints.append("3b")
        return constraints


    def callback(self, x):
        self._iter += 1
        ecurr = self.objective(x)
        out = " Iteration: %3i Energy = %12.8f "%(self._iter, ecurr)
        
        if "2b" in self.constraints:
            ccurr2 = np.linalg.norm(self.marginals.compute_2b_violations())
            out += "| Violations: 2B = %12.8f " %(ccurr2)
        if "3b" in self.constraints:
            ccurr3 = np.linalg.norm(self.marginals.compute_3b_violations())
            out += "3B = %12.8f" %(ccurr3)
        print(out)

def run():

    np.random.seed(2)

    # create lattice containing different types of sites
    sites = []
    sites.append(Site(0, SpinHilbertSpace(2)))
    sites.append(Site(1, SpinHilbertSpace(2)))
    sites.append(Site(2, SpinHilbertSpace(2)))
    sites.append(Site(3, SpinHilbertSpace(2)))
    # sites.append(Site(4, SpinHilbertSpace(2)))
    # sites.append(Site(5, SpinHilbertSpace(2)))
    # sites.append(Site(6, SpinHilbertSpace(2)))
    # sites.append(Site(7, SpinHilbertSpace(2)))


    H = build_heisenberg_hamiltonian(sites, periodic=False)
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
    for term in M.keys():
        M[term] = FactorizedMarginal.from_Marginal(M[term])
    M.unfold()
    # M.convert_to_FactorizedMarginals()

    print(M)
    M.print_cumulants()

    if True:
        print("\n Now optimize from scratch")
        M = Marginals().initialize_maximally_mixed(sites, nbody=3)
        print(M)
        M.print_cumulants()
        print(M.compute_2b_violations())
        print(M.compute_3b_violations())


    myopt = OptimizeEnergy(H, M)
    x0 = np.real_if_close(myopt.xcurr)
    # x0 += np.random.rand(len(x0))
    print(myopt.objective(x0), myopt.marginals.compute_2b_violations() )
    print(" Norm of 2b violations %12.8f " %np.linalg.norm(myopt.marginals.compute_2b_violations()))
    print(" Norm of 3b violations %12.8f " %np.linalg.norm(myopt.marginals.compute_3b_violations()))

    # Solve
    constraints = myopt.setup_2b_constraints()
    constraints.extend(myopt.setup_3b_constraints())
    print(constraints)
    result = minimize(
        myopt.objective, 
        x0, 
        method='SLSQP',  # or 'trust-constr'
        constraints=constraints,
        callback=myopt.callback,
        options={'disp': True, 'maxiter': 200}
    )


    M.import_from_vector(result.x, myopt.mdata)
    print(M)
    M.print_cumulants()
    

run()