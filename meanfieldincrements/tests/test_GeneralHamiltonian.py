"""
Test suite for simplified GeneralHamiltonian class.
"""

import pytest
import numpy as np
from meanfieldincrements import (
    Site, PauliHilbertSpace, SpinHilbertSpace, 
    create_qubit_chain, create_spin_chain
)

# Import the simplified GeneralHamiltonian classes
from meanfieldincrements import (
    GeneralHamiltonian, build_heisenberg_hamiltonian, build_ising_hamiltonian
)


class TestGeneralHamiltonianBasics:
    
    def test_empty_hamiltonian(self):
        """Test empty Hamiltonian construction."""
        sites = create_qubit_chain(2)
        ham = GeneralHamiltonian(sites)
        
        assert len(ham.sites) == 2
        assert len(ham) == 0
        assert len(ham.terms) == 0
    
    def test_hamiltonian_with_terms(self):
        """Test Hamiltonian construction with initial terms."""
        sites = create_qubit_chain(2)
        terms = {
            ('X', 'Y'): 0.5,
            ('Z', 'Z'): -0.3,
            ('I', 'X'): 0.1
        }
        ham = GeneralHamiltonian(sites, terms)
        
        assert len(ham.sites) == 2
        assert len(ham) == 3
        assert ham[('X', 'Y')] == 0.5
        assert ham[('Z', 'Z')] == -0.3
        assert ham[('I', 'X')] == 0.1
    
    def test_setitem_getitem(self):
        """Test dictionary-like interface."""
        sites = create_qubit_chain(2)
        ham = GeneralHamiltonian(sites)
        
        # Add terms using setitem
        ham[('X', 'Y')] = 0.5
        ham[('Z', 'Z')] = -0.3
        
        # Test getitem
        assert ham[('X', 'Y')] == 0.5
        assert ham[('Z', 'Z')] == -0.3
        assert len(ham) == 2
        
        # Test contains
        assert ('X', 'Y') in ham
        assert ('I', 'I') not in ham
        
        # Update existing term
        ham[('X', 'Y')] = 0.7
        assert ham[('X', 'Y')] == 0.7
        assert len(ham) == 2  # Still 2 terms
    
    def test_validation_errors(self):
        """Test validation errors for inconsistent terms."""
        sites = create_qubit_chain(2)
        ham = GeneralHamiltonian(sites)
        
        # Wrong number of operators
        with pytest.raises(ValueError):
            ham[('X', 'Y', 'Z')] = 0.5  # 3 operators for 2 sites
        
        # Wrong key type
        with pytest.raises(TypeError):
            ham['X'] = 0.5  # String instead of tuple
    
    def test_iterator(self):
        """Test iteration over coefficients."""
        sites = create_qubit_chain(2)
        terms = {('X', 'Y'): 0.5, ('Z', 'Z'): -0.3}
        ham = GeneralHamiltonian(sites, terms)
        
        coeffs = list(ham)
        assert set(coeffs) == {0.5, -0.3}


class TestGeneralHamiltonianOperators:
    
    def test_single_site_operators(self):
        """Test single-site operator conversion."""
        sites = [Site(0, PauliHilbertSpace(2))]
        terms = {('X',): 0.5, ('Y',): -0.3}
        ham = GeneralHamiltonian(sites, terms)
        
        pauli_ops = sites[0].create_operators()
        site_ops = {PauliHilbertSpace: pauli_ops}
        
        # Test matrix
        H_matrix = ham.matrix(site_ops)
        expected = 0.5 * pauli_ops['X'] + (-0.3) * pauli_ops['Y']
        np.testing.assert_allclose(H_matrix, expected)
    
    def test_two_site_operators(self):
        """Test two-site operator conversion."""
        sites = create_qubit_chain(2)
        terms = {
            ('X', 'X'): 1.0,
            ('Y', 'Y'): 1.0,
            ('Z', 'Z'): 1.0
        }
        ham = GeneralHamiltonian(sites, terms)
        
        pauli_ops = PauliHilbertSpace(2).create_operators()
        site_ops = {PauliHilbertSpace: pauli_ops}
        
        # Test full matrix
        H_matrix = ham.matrix(site_ops)
        
        # Compare with manual construction
        two_site_ops = pauli_ops.kron(pauli_ops)
        expected = (two_site_ops['XX'] + two_site_ops['YY'] + two_site_ops['ZZ'])
        np.testing.assert_allclose(H_matrix, expected)
    
    def test_different_operator_libraries(self):
        """Test with different operator libraries per site."""
        sites = [
            Site(0, PauliHilbertSpace(2)),  # Pauli site
            Site(1, SpinHilbertSpace(2))    # Spin site
        ]
        terms = {('X', 'Sx'): 1.0}
        ham = GeneralHamiltonian(sites, terms)
        
        site_ops = {
            PauliHilbertSpace: sites[0].create_operators(),
            SpinHilbertSpace: sites[1].create_operators()
        }
        
        # Should work with different libraries
        H_matrix = ham.matrix(site_ops)
        assert H_matrix.shape == (4, 4)
    
    def test_three_site_operators(self):
        """Test three-site operator construction."""
        sites = create_qubit_chain(3)
        terms = {
            ('X', 'Y', 'Z'): 0.5,
            ('I', 'I', 'I'): 1.0
        }
        ham = GeneralHamiltonian(sites, terms)
        
        pauli_ops = PauliHilbertSpace(2).create_operators()
        site_ops = {PauliHilbertSpace: pauli_ops}
        
        H_matrix = ham.matrix(site_ops)
        assert H_matrix.shape == (8, 8)
        
        # Check identity term gives identity matrix
        identity_ham = GeneralHamiltonian(sites, {('I', 'I', 'I'): 1.0})
        I_matrix = identity_ham.matrix(site_ops)
        np.testing.assert_allclose(I_matrix, np.eye(8))
    
    def test_empty_hamiltonian_matrix(self):
        """Test matrix of empty Hamiltonian."""
        sites = create_qubit_chain(2)
        ham = GeneralHamiltonian(sites)  # Empty
        
        pauli_ops = PauliHilbertSpace(2).create_operators()
        site_ops = {PauliHilbertSpace: pauli_ops}
        
        H_matrix = ham.matrix(site_ops)
        expected = np.zeros((4, 4))
        np.testing.assert_allclose(H_matrix, expected)


class TestGeneralHamiltonianOperations:
    
    def test_addition(self):
        """Test Hamiltonian addition."""
        sites = create_qubit_chain(2)
        
        ham1 = GeneralHamiltonian(sites, {('X', 'X'): 1.0, ('Y', 'Y'): 0.5})
        ham2 = GeneralHamiltonian(sites, {('X', 'X'): 0.5, ('Z', 'Z'): -0.3})
        
        ham_sum = ham1 + ham2
        
        assert ham_sum[('X', 'X')] == 1.5  # 1.0 + 0.5
        assert ham_sum[('Y', 'Y')] == 0.5
        assert ham_sum[('Z', 'Z')] == -0.3
        assert len(ham_sum) == 3
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        sites = create_qubit_chain(2)
        ham = GeneralHamiltonian(sites, {('X', 'X'): 1.0, ('Y', 'Y'): 0.5})
        
        ham_scaled = 2.0 * ham
        assert ham_scaled[('X', 'X')] == 2.0
        assert ham_scaled[('Y', 'Y')] == 1.0
        
        ham_scaled2 = ham * (-1.0)
        assert ham_scaled2[('X', 'X')] == -1.0
        assert ham_scaled2[('Y', 'Y')] == -0.5
    
    def test_complex_coefficients(self):
        """Test complex coefficient operations."""
        sites = create_qubit_chain(2)
        ham = GeneralHamiltonian(sites, {('X', 'Y'): 1.0 + 2.0j, ('Z', 'Z'): -0.5j})
        
        ham_scaled = 1j * ham
        assert ham_scaled[('X', 'Y')] == 1j * (1.0 + 2.0j)
        assert ham_scaled[('Z', 'Z')] == 1j * (-0.5j)
    
    def test_subtraction(self):
        """Test Hamiltonian subtraction."""
        sites = create_qubit_chain(2)
        
        ham1 = GeneralHamiltonian(sites, {('X', 'X'): 1.0, ('Y', 'Y'): 0.5})
        ham2 = GeneralHamiltonian(sites, {('X', 'X'): 0.3, ('Z', 'Z'): 0.2})
        
        ham_diff = ham1 - ham2
        assert abs(ham_diff[('X', 'X')] - 0.7) < 1e-10
        assert ham_diff[('Y', 'Y')] == 0.5
        assert ham_diff[('Z', 'Z')] == -0.2
    
    def test_addition_errors(self):
        """Test error handling in addition."""
        sites1 = create_qubit_chain(2)
        sites2 = create_qubit_chain(3)  # Different number of sites
        
        ham1 = GeneralHamiltonian(sites1, {('X', 'X'): 1.0})
        ham2 = GeneralHamiltonian(sites2, {('X', 'X', 'X'): 1.0})
        
        # Different number of sites
        with pytest.raises(ValueError):
            ham1 + ham2
        
        # Wrong type
        with pytest.raises(TypeError):
            ham1 + 5.0


class TestPrebuiltHamiltonians:
    
    def test_heisenberg_hamiltonian(self):
        """Test Heisenberg model construction."""
        sites = create_qubit_chain(3)
        J = 1.0
        ham = build_heisenberg_hamiltonian(sites, J, periodic=False)
        
        # Should have 3 * (n_sites - 1) = 6 terms for linear chain
        expected_terms = 3 * (len(sites) - 1)  # XX, YY, ZZ for each bond
        assert len(ham) == expected_terms
        assert len(ham.sites) == len(sites)
        
        # Check specific terms exist
        assert ham[('X', 'X', 'I')] == J  # Sites 0-1
        assert ham[('Y', 'Y', 'I')] == J
        assert ham[('Z', 'Z', 'I')] == J
        assert ham[('I', 'X', 'X')] == J  # Sites 1-2
        
        # Test with periodic boundary conditions
        ham_pbc = build_heisenberg_hamiltonian(sites, J, periodic=True)
        expected_terms_pbc = 3 * len(sites)  # Include wraparound terms
        assert len(ham_pbc) == expected_terms_pbc
        
        # Check wraparound terms
        assert ham_pbc[('X', 'I', 'X')] == J  # Sites 0-2
    
    def test_ising_hamiltonian(self):
        """Test Ising model construction."""
        sites = create_qubit_chain(3)
        J = 1.0
        h = 0.5
        ham = build_ising_hamiltonian(sites, J, h, periodic=False)
        
        # Should have (n_sites - 1) ZZ terms + n_sites X terms
        expected_terms = (len(sites) - 1) + len(sites)
        assert len(ham) == expected_terms
        
        # Check ZZ coupling terms
        assert ham[('Z', 'Z', 'I')] == -J
        assert ham[('I', 'Z', 'Z')] == -J
        
        # Check transverse field terms
        assert ham[('X', 'I', 'I')] == -h
        assert ham[('I', 'X', 'I')] == -h
        assert ham[('I', 'I', 'X')] == -h
    
    def test_build_with_different_sites(self):
        """Test building Hamiltonians with different site types."""
        # Mixed sites
        sites = [
            Site(0, PauliHilbertSpace(2)),
            Site(1, SpinHilbertSpace(2))
        ]
        
        # Should work even with different site types
        ham = build_heisenberg_hamiltonian(sites, 1.0, periodic=False)
        assert len(ham) == 3  # XX, YY, ZZ terms
        assert len(ham.sites) == 2


class TestGeneralHamiltonianStringRepresentation:
    
    def test_string_representation(self):
        """Test string representation methods."""
        sites = create_qubit_chain(2)
        terms = {
            ('X', 'Y'): 1.0,
            ('Z', 'Z'): -0.5,
            ('I', 'X'): 0.3
        }
        ham = GeneralHamiltonian(sites, terms)
        
        # Test __repr__
        repr_str = repr(ham)
        assert "GeneralHamiltonian" in repr_str
        assert "sites=2" in repr_str
        assert "terms=3" in repr_str
        
        # Test __str__
        str_repr = str(ham)
        assert "GeneralHamiltonian" in str_repr
        assert "X⊗Y" in str_repr
        assert "+1.000000" in str_repr
        assert "-0.500000" in str_repr
    
    def test_empty_string_representation(self):
        """Test string representation of empty Hamiltonian."""
        sites = create_qubit_chain(2)
        ham = GeneralHamiltonian(sites)
        
        str_repr = str(ham)
        assert "(empty)" in str_repr
        assert "2 sites" in str_repr


class TestGeneralHamiltonianIntegration:
    
    def test_matrix_consistency(self):
        """Test that matrix representation is consistent."""
        sites = create_qubit_chain(2)
        
        # Build Hamiltonian in two different ways
        ham1 = GeneralHamiltonian(sites, {('X', 'Y'): 1.0, ('Z', 'Z'): 0.5})
        
        ham2 = GeneralHamiltonian(sites)
        ham2[('X', 'Y')] = 1.0
        ham2[('Z', 'Z')] = 0.5
        
        pauli_ops = PauliHilbertSpace(2).create_operators()
        site_ops = {PauliHilbertSpace: pauli_ops}
        
        # Both should give same matrix
        matrix1 = ham1.matrix(site_ops)
        matrix2 = ham2.matrix(site_ops)
        np.testing.assert_allclose(matrix1, matrix2)
    
    def test_arithmetic_matrix_consistency(self):
        """Test that arithmetic operations preserve matrix representation."""
        sites = create_qubit_chain(2)
        
        ham1 = GeneralHamiltonian(sites, {('X', 'X'): 1.0})
        ham2 = GeneralHamiltonian(sites, {('Y', 'Y'): 0.5})
        
        pauli_ops = PauliHilbertSpace(2).create_operators()
        site_ops = {PauliHilbertSpace: pauli_ops}
        
        # Test addition
        ham_sum = ham1 + ham2
        matrix_sum = ham_sum.matrix(site_ops)
        expected_sum = ham1.matrix(site_ops) + ham2.matrix(site_ops)
        np.testing.assert_allclose(matrix_sum, expected_sum)
        
        # Test scalar multiplication
        ham_scaled = 2.0 * ham1
        matrix_scaled = ham_scaled.matrix(site_ops)
        expected_scaled = 2.0 * ham1.matrix(site_ops)
        np.testing.assert_allclose(matrix_scaled, expected_scaled)
    
    def test_large_system(self):
        """Test with larger systems."""
        # 4-qubit system
        sites = create_qubit_chain(4)
        ham = build_heisenberg_hamiltonian(sites, 1.0, periodic=True)
        
        pauli_ops = PauliHilbertSpace(2).create_operators()
        site_ops = {PauliHilbertSpace: pauli_ops}
        
        # Should handle 2^4 = 16 dimensional Hilbert space
        H_matrix = ham.matrix(site_ops)
        assert H_matrix.shape == (16, 16)
        
        # Check Hermiticity
        assert np.allclose(H_matrix, H_matrix.conj().T)


def run_all_tests():
    """Run all tests without pytest."""
    test_classes = [
        TestGeneralHamiltonianBasics,
        TestGeneralHamiltonianOperators,
        TestGeneralHamiltonianOperations,
        TestPrebuiltHamiltonians,
        TestGeneralHamiltonianStringRepresentation,
        TestGeneralHamiltonianIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}:")
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method in methods:
            total_tests += 1
            try:
                getattr(instance, method)()
                print(f"  ✓ {method}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method}: {e}")
    
    print(f"\nTest Summary: {passed_tests}/{total_tests} tests passed")
    return passed_tests == total_tests


if __name__ == "__main__":
    print("Running simplified GeneralHamiltonian tests...")
    success = run_all_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        print("\nExample usage:")
        
        # Quick demonstration
        print("\n=== Example: Two-qubit Heisenberg model ===")
        sites = create_qubit_chain(2)
        pauli_ops = PauliHilbertSpace(2).create_operators()
        site_ops = {PauliHilbertSpace: pauli_ops}
        
        # Build Hamiltonian manually
        terms = {
            ('X', 'X'): 1.0,
            ('Y', 'Y'): 1.0,
            ('Z', 'Z'): 1.0
        }
        ham = GeneralHamiltonian(sites, terms)
        
        print(f"Hamiltonian:\n{ham}")
        
        # Convert to matrix
        H_matrix = ham.matrix(site_ops)
        print(f"\nMatrix shape: {H_matrix.shape}")
        print(f"Is Hermitian: {np.allclose(H_matrix, H_matrix.conj().T)}")
        
        # Test dictionary interface
        print(f"\nXX coefficient: {ham[('X', 'X')]}")
        ham[('I', 'Z')] = 0.5  # Add field term
        print(f"After adding field: {len(ham)} terms")
        
        print("\n=== Example: Three-qubit Ising model ===")
        ising_sites = create_qubit_chain(3)
        ising_ham = build_ising_hamiltonian(ising_sites, J=1.0, h=0.5, periodic=False)
        print(f"Ising Hamiltonian:\n{ising_ham}")
        
    else:
        print("\n❌ Some tests failed!")