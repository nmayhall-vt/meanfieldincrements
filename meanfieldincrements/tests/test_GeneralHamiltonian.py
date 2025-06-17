"""
Comprehensive test suite for GeneralHamiltonian class.
"""

import pytest
import numpy as np
from meanfieldincrements import (
    Site, LocalTensor, PauliHilbertSpace, SpinHilbertSpace, FermionHilbertSpace,
    create_qubit_chain, create_spin_chain
)

# Import the GeneralHamiltonian classes from meanfieldincrements package
from meanfieldincrements import (
    GeneralHamiltonian, build_heisenberg_hamiltonian, build_ising_hamiltonian, from_pauli_strings
)


class TestGeneralHamiltonianBasics:
    
    def test_empty_hamiltonian(self):
        """Test empty Hamiltonian construction."""
        ham = GeneralHamiltonian()
        assert ham.n_sites == 0
        assert ham.n_terms == 0
        assert len(ham.terms) == 0
    
    def test_hamiltonian_with_terms(self):
        """Test Hamiltonian construction with initial terms."""
        terms = {
            ('X', 'Y'): 0.5,
            ('Z', 'Z'): -0.3,
            ('I', 'X'): 0.1
        }
        ham = GeneralHamiltonian(terms)
        
        assert ham.n_sites == 2
        assert ham.n_terms == 3
        assert ham.get_coefficient(('X', 'Y')) == 0.5
        assert ham.get_coefficient(('Z', 'Z')) == -0.3
        assert ham.get_coefficient(('I', 'X')) == 0.1
    
    def test_add_term(self):
        """Test adding terms to Hamiltonian."""
        ham = GeneralHamiltonian()
        
        # Add first term
        ham.add_term(('X', 'Y'), 0.5)
        assert ham.n_sites == 2
        assert ham.n_terms == 1
        
        # Add second term
        ham.add_term(('Z', 'Z'), -0.3)
        assert ham.n_terms == 2
        
        # Add to existing term (should sum coefficients)
        ham.add_term(('X', 'Y'), 0.2)
        assert ham.get_coefficient(('X', 'Y')) == 0.7
        assert ham.n_terms == 2  # Still 2 unique terms
    
    def test_validation_errors(self):
        """Test validation errors for inconsistent terms."""
        ham = GeneralHamiltonian()
        ham.add_term(('X', 'Y'), 0.5)  # 2-site term
        
        # Try to add 3-site term to 2-site Hamiltonian
        with pytest.raises(ValueError):
            ham.add_term(('X', 'Y', 'Z'), 0.3)
    
    def test_remove_term(self):
        """Test removing terms."""
        terms = {('X', 'Y'): 0.5, ('Z', 'Z'): -0.3}
        ham = GeneralHamiltonian(terms)
        
        ham.remove_term(('X', 'Y'))
        assert ham.n_terms == 1
        assert ham.get_coefficient(('X', 'Y')) == 0.0
        
        # Try to remove non-existent term
        with pytest.raises(KeyError):
            ham.remove_term(('I', 'I'))


class TestGeneralHamiltonianOperators:
    
    def test_single_site_operators(self):
        """Test single-site operator conversion."""
        ham = GeneralHamiltonian({('X',): 0.5, ('Y',): -0.3})
        
        site = Site(0, PauliHilbertSpace(2))
        pauli_ops = site.create_operators()
        
        # Test single term
        x_op = ham.get_local_tensor(('X',), [site], pauli_ops)
        assert isinstance(x_op, LocalTensor)
        assert x_op.tensor.shape == (2, 2)
        np.testing.assert_allclose(x_op.tensor, pauli_ops['X'])
        
        # Test full matrix
        H_matrix = ham.to_matrix([site], pauli_ops)
        expected = 0.5 * pauli_ops['X'] + (-0.3) * pauli_ops['Y']
        np.testing.assert_allclose(H_matrix, expected)
    
    def test_two_site_operators(self):
        """Test two-site operator conversion."""
        ham = GeneralHamiltonian({
            ('X', 'X'): 1.0,
            ('Y', 'Y'): 1.0,
            ('Z', 'Z'): 1.0
        })
        
        sites = create_qubit_chain(2)
        pauli_ops = PauliHilbertSpace(2).create_operators()
        
        # Test individual term
        xx_op = ham.get_local_tensor(('X', 'X'), sites, pauli_ops)
        assert xx_op.tensor.shape == (4, 4)
        
        # Test full matrix
        H_matrix = ham.to_matrix(sites, pauli_ops)
        
        # Compare with manual construction
        two_site_ops = pauli_ops.kron(pauli_ops)
        expected = (two_site_ops['XX'] + two_site_ops['YY'] + two_site_ops['ZZ'])
        np.testing.assert_allclose(H_matrix, expected)
    
    def test_different_operator_libraries(self):
        """Test with different operator libraries per site."""
        ham = GeneralHamiltonian({('X', 'Sx'): 1.0})
        
        sites = [
            Site(0, PauliHilbertSpace(2)),  # Pauli site
            Site(1, SpinHilbertSpace(2))    # Spin site
        ]
        
        libraries = [
            sites[0].create_operators(),  # Pauli operators
            sites[1].create_operators()   # Spin operators
        ]
        
        # Should work with different libraries
        mixed_op = ham.get_local_tensor(('X', 'Sx'), sites, libraries)
        assert mixed_op.tensor.shape == (4, 4)
        
        H_matrix = ham.to_matrix(sites, libraries)
        assert H_matrix.shape == (4, 4)
    
    def test_three_site_operators(self):
        """Test three-site operator construction."""
        ham = GeneralHamiltonian({
            ('X', 'Y', 'Z'): 0.5,
            ('I', 'I', 'I'): 1.0
        })
        
        sites = create_qubit_chain(3)
        pauli_ops = PauliHilbertSpace(2).create_operators()
        
        H_matrix = ham.to_matrix(sites, pauli_ops)
        assert H_matrix.shape == (8, 8)
        
        # Check identity term gives identity matrix
        identity_term = GeneralHamiltonian({('I', 'I', 'I'): 1.0})
        I_matrix = identity_term.to_matrix(sites, pauli_ops)
        np.testing.assert_allclose(I_matrix, np.eye(8))


class TestGeneralHamiltonianOperations:
    
    def test_addition(self):
        """Test Hamiltonian addition."""
        ham1 = GeneralHamiltonian({('X', 'X'): 1.0, ('Y', 'Y'): 0.5})
        ham2 = GeneralHamiltonian({('X', 'X'): 0.5, ('Z', 'Z'): -0.3})
        
        ham_sum = ham1 + ham2
        
        assert ham_sum.get_coefficient(('X', 'X')) == 1.5  # 1.0 + 0.5
        assert ham_sum.get_coefficient(('Y', 'Y')) == 0.5
        assert ham_sum.get_coefficient(('Z', 'Z')) == -0.3
        assert ham_sum.n_terms == 3
    
    def test_scalar_multiplication(self):
        """Test scalar multiplication."""
        ham = GeneralHamiltonian({('X', 'X'): 1.0, ('Y', 'Y'): 0.5})
        
        ham_scaled = 2.0 * ham
        assert ham_scaled.get_coefficient(('X', 'X')) == 2.0
        assert ham_scaled.get_coefficient(('Y', 'Y')) == 1.0
        
        ham_scaled2 = ham * (-1.0)
        assert ham_scaled2.get_coefficient(('X', 'X')) == -1.0
        assert ham_scaled2.get_coefficient(('Y', 'Y')) == -0.5
    
    def test_complex_coefficients(self):
        """Test complex coefficient operations."""
        ham = GeneralHamiltonian({('X', 'Y'): 1.0 + 2.0j, ('Z', 'Z'): -0.5j})
        
        ham_conj = ham.conjugate()
        assert ham_conj.get_coefficient(('X', 'Y')) == 1.0 - 2.0j
        assert ham_conj.get_coefficient(('Z', 'Z')) == 0.5j
    
    def test_subtraction(self):
        """Test Hamiltonian subtraction."""
        ham1 = GeneralHamiltonian({('X', 'X'): 1.0, ('Y', 'Y'): 0.5})
        ham2 = GeneralHamiltonian({('X', 'X'): 0.3, ('Z', 'Z'): 0.2})
        
        ham_diff = ham1 - ham2
        assert ham_diff.get_coefficient(('X', 'X')) == 0.7
        assert ham_diff.get_coefficient(('Y', 'Y')) == 0.5
        assert ham_diff.get_coefficient(('Z', 'Z')) == -0.2


class TestGeneralHamiltonianAnalysis:
    
    def test_expectation_value(self):
        """Test expectation value calculation."""
        # Simple Z operator on single qubit
        ham = GeneralHamiltonian({('Z',): 1.0})
        site = Site(0, PauliHilbertSpace(2))
        pauli_ops = site.create_operators()
        
        # |0⟩ state (eigenstate of Z with eigenvalue +1)
        state_0 = np.array([1, 0])
        exp_val = ham.get_expectation_value(state_0, [site], pauli_ops)
        assert np.isclose(exp_val, 1.0)
        
        # |1⟩ state (eigenstate of Z with eigenvalue -1)
        state_1 = np.array([0, 1])
        exp_val = ham.get_expectation_value(state_1, [site], pauli_ops)
        assert np.isclose(exp_val, -1.0)
        
        # |+⟩ = (|0⟩ + |1⟩)/√2 state
        state_plus = np.array([1, 1]) / np.sqrt(2)
        exp_val = ham.get_expectation_value(state_plus, [site], pauli_ops)
        assert np.isclose(exp_val, 0.0)
    
    def test_is_hermitian(self):
        """Test Hermiticity checking."""
        # Real coefficients should be Hermitian
        ham_real = GeneralHamiltonian({('X', 'X'): 1.0, ('Y', 'Y'): 0.5, ('Z', 'Z'): -0.3})
        sites = create_qubit_chain(2)
        pauli_ops = PauliHilbertSpace(2).create_operators()
        
        assert ham_real.is_hermitian(sites, pauli_ops)
        
        # Complex coefficients - not Hermitian
        ham_complex = GeneralHamiltonian({('X', 'Y'): 1.0j})
        assert not ham_complex.is_hermitian(sites, pauli_ops)
    
    def test_filter_terms(self):
        """Test term filtering."""
        ham = GeneralHamiltonian({
            ('X', 'X'): 1.0,
            ('Y', 'Y'): 0.5,
            ('Z', 'Z'): 1e-15,  # Very small
            ('I', 'X'): 0.3
        })
        
        # Filter out small terms
        ham_filtered = ham.filter_terms(lambda op_str, coeff: abs(coeff) > 1e-10)
        assert ham_filtered.n_terms == 3  # Should remove the 1e-15 term
        assert ham_filtered.get_coefficient(('Z', 'Z')) == 0.0
        
        # Test simplify method
        ham_simple = ham.simplify(rtol=1e-10)
        assert ham_simple.n_terms == 3
    
    def test_get_terms_by_weight(self):
        """Test filtering by number of non-identity operators."""
        ham = GeneralHamiltonian({
            ('I', 'I', 'I'): 1.0,    # 0-body
            ('X', 'I', 'I'): 0.5,    # 1-body
            ('I', 'Y', 'I'): 0.3,    # 1-body
            ('X', 'X', 'I'): 0.2,    # 2-body
            ('X', 'Y', 'Z'): 0.1     # 3-body
        })
        
        # Test 1-body terms
        one_body = ham.get_terms_by_weight(1)
        assert one_body.n_terms == 2
        assert one_body.get_coefficient(('X', 'I', 'I')) == 0.5
        assert one_body.get_coefficient(('I', 'Y', 'I')) == 0.3
        
        # Test 2-body terms
        two_body = ham.get_terms_by_weight(2)
        assert two_body.n_terms == 1
        assert two_body.get_coefficient(('X', 'X', 'I')) == 0.2
        
        # Test 0-body terms
        zero_body = ham.get_terms_by_weight(0)
        assert zero_body.n_terms == 1
        assert zero_body.get_coefficient(('I', 'I', 'I')) == 1.0


class TestPrebuiltHamiltonians:
    
    def test_heisenberg_hamiltonian(self):
        """Test Heisenberg model construction."""
        n_sites = 3
        J = 1.0
        ham = build_heisenberg_hamiltonian(n_sites, J, periodic=False)
        
        # Should have 3 * (n_sites - 1) = 6 terms for linear chain
        expected_terms = 3 * (n_sites - 1)  # XX, YY, ZZ for each bond
        assert ham.n_terms == expected_terms
        assert ham.n_sites == n_sites
        
        # Check specific terms exist
        assert ham.get_coefficient(('X', 'X', 'I')) == J  # Sites 0-1
        assert ham.get_coefficient(('Y', 'Y', 'I')) == J
        assert ham.get_coefficient(('Z', 'Z', 'I')) == J
        assert ham.get_coefficient(('I', 'X', 'X')) == J  # Sites 1-2
        
        # Test with periodic boundary conditions
        ham_pbc = build_heisenberg_hamiltonian(n_sites, J, periodic=True)
        expected_terms_pbc = 3 * n_sites  # Include wraparound terms
        assert ham_pbc.n_terms == expected_terms_pbc
        
        # Check wraparound terms
        assert ham_pbc.get_coefficient(('X', 'I', 'X')) == J  # Sites 0-2
    
    def test_ising_hamiltonian(self):
        """Test Ising model construction."""
        n_sites = 3
        J = 1.0
        h = 0.5
        ham = build_ising_hamiltonian(n_sites, J, h, periodic=False)
        
        # Should have (n_sites - 1) ZZ terms + n_sites X terms
        expected_terms = (n_sites - 1) + n_sites
        assert ham.n_terms == expected_terms
        
        # Check ZZ coupling terms
        assert ham.get_coefficient(('Z', 'Z', 'I')) == -J
        assert ham.get_coefficient(('I', 'Z', 'Z')) == -J
        
        # Check transverse field terms
        assert ham.get_coefficient(('X', 'I', 'I')) == -h
        assert ham.get_coefficient(('I', 'X', 'I')) == -h
        assert ham.get_coefficient(('I', 'I', 'X')) == -h
    
    def test_from_pauli_strings(self):
        """Test construction from Pauli strings."""
        pauli_strings = ['XYZ', 'IXI', 'ZZI']
        coefficients = [0.5, -0.3, 0.1]
        
        ham = from_pauli_strings(pauli_strings, coefficients)
        
        assert ham.n_terms == 3
        assert ham.n_sites == 3
        assert ham.get_coefficient(('X', 'Y', 'Z')) == 0.5
        assert ham.get_coefficient(('I', 'X', 'I')) == -0.3
        assert ham.get_coefficient(('Z', 'Z', 'I')) == 0.1


class TestGeneralHamiltonianIntegration:
    
    def test_integration_with_localoperator(self):
        """Test integration with LocalTensor class."""
        ham = GeneralHamiltonian({('X', 'Y'): 1.0, ('Z', 'Z'): 0.5})
        sites = create_qubit_chain(2)
        pauli_ops = PauliHilbertSpace(2).create_operators()
        
        # Convert to LocalTensors
        local_ops = ham.to_local_tensors(sites, pauli_ops)
        
        assert len(local_ops) == 2
        for local_op in local_ops:
            assert isinstance(local_op, LocalTensor)
            assert local_op.tensor.shape == (4, 4)
        
        # Test that sum of LocalTensors equals full matrix
        total_matrix = sum(op.unfold().tensor for op in local_ops)
        ham_matrix = ham.to_matrix(sites, pauli_ops)
        np.testing.assert_allclose(total_matrix, ham_matrix)
    
    def test_mixed_site_types(self):
        """Test with different types of sites."""
        # Create Hamiltonian for mixed system
        ham = GeneralHamiltonian({
            ('X', 'Sx'): 1.0,     # Pauli-Spin interaction
            ('Z', 'Sz'): 0.5      # Pauli-Spin interaction
        })
        
        sites = [
            Site(0, PauliHilbertSpace(2)),  # Qubit
            Site(1, SpinHilbertSpace(2))    # Spin-1/2
        ]
        
        libraries = [
            sites[0].create_operators(),
            sites[1].create_operators()
        ]
        
        # Should work even with different operator libraries
        H_matrix = ham.to_matrix(sites, libraries)
        assert H_matrix.shape == (4, 4)
        
        # Test expectation value
        random_state = np.random.random(4) + 1j * np.random.random(4)
        random_state /= np.linalg.norm(random_state)
        
        exp_val = ham.get_expectation_value(random_state, sites, libraries)
        print("nick")
        print(type(exp_val))
        assert isinstance(exp_val, (float, complex))
    
    def test_large_system_performance(self):
        """Test performance with larger systems."""
        # 4-qubit system
        n_sites = 4
        ham = build_heisenberg_hamiltonian(n_sites, 1.0, periodic=True)
        sites = create_qubit_chain(n_sites)
        pauli_ops = PauliHilbertSpace(2).create_operators()
        
        # Should handle 2^4 = 16 dimensional Hilbert space
        H_matrix = ham.to_matrix(sites, pauli_ops)
        assert H_matrix.shape == (16, 16)
        
        # Check Hermiticity
        assert ham.is_hermitian(sites, pauli_ops)


class TestGeneralHamiltonianStringRepresentation:
    
    def test_string_representation(self):
        """Test string representation methods."""
        ham = GeneralHamiltonian({
            ('X', 'Y'): 1.0,
            ('Z', 'Z'): -0.5,
            ('I', 'X'): 0.3
        })
        
        # Test __repr__
        repr_str = repr(ham)
        assert "GeneralHamiltonian" in repr_str
        assert "n_sites=2" in repr_str
        assert "n_terms=3" in repr_str
        
        # Test __str__
        str_repr = str(ham)
        assert "GeneralHamiltonian" in str_repr
        assert "X⊗Y" in str_repr or "X⊗Y" in str_repr
        assert "+1.000000" in str_repr
        assert "-0.500000" in str_repr


def run_all_tests():
    """Run all tests without pytest."""
    test_classes = [
        TestGeneralHamiltonianBasics,
        TestGeneralHamiltonianOperators,
        TestGeneralHamiltonianOperations,
        TestGeneralHamiltonianAnalysis,
        TestPrebuiltHamiltonians,
        TestGeneralHamiltonianIntegration,
        TestGeneralHamiltonianStringRepresentation
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
    print("Running GeneralHamiltonian tests...")
    success = run_all_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        print("\nExample usage:")
        
        # Quick demonstration
        print("\n=== Example: Two-qubit Heisenberg model ===")
        sites = create_qubit_chain(2)
        pauli_ops = PauliHilbertSpace(2).create_operators()
        
        # Build Hamiltonian manually
        ham = GeneralHamiltonian({
            ('X', 'X'): 1.0,
            ('Y', 'Y'): 1.0,
            ('Z', 'Z'): 1.0
        })
        
        print(f"Hamiltonian:\n{ham}")
        
        # Convert to matrix
        H_matrix = ham.to_matrix(sites, pauli_ops)
        print(f"\nMatrix shape: {H_matrix.shape}")
        print(f"Is Hermitian: {ham.is_hermitian(sites, pauli_ops)}")
        
        # Eigenvalues
        eigenvals = np.linalg.eigvals(H_matrix)
        print(f"Eigenvalues: {sorted(eigenvals.real)}")
        
        print("\n=== Example: Three-qubit Ising model ===")
        ising_ham = build_ising_hamiltonian(3, J=1.0, h=0.5, periodic=False)
        print(f"Ising Hamiltonian:\n{ising_ham}")
        
    else:
        print("\n❌ Some tests failed!")