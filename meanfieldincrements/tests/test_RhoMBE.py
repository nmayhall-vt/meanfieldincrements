import pytest
import numpy as np
from meanfieldincrements import Site, LocalOperator, RhoMBE

class RhoMBE_TEST:

    def test_5qubits(self):
        nsites = 5
        sites = [Site(i, 2) for i in range(nsites)]
        
        rho = RhoMBE(sites).initialize_mixed()
        print(rho)

def test_jobs():
    test_instance = RhoMBE_TEST() 
    
    methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    for method_name in methods:
        try:
            print(f"Running {method_name}...")
            getattr(test_instance, method_name)()
            print(f"✓ {method_name} passed")
        except Exception as e:
            print(f"✗ {method_name} failed: {e}")


if __name__ == "__main__":
    # # Run tests with pytest
    """Simple test runner if pytest is not available."""
    test_jobs()
