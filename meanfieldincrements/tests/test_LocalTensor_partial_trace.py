import pytest
import numpy as np
from meanfieldincrements import Site, LocalTensor

class TestPartialTrace:
    """Comprehensive unit tests for the partial_trace function."""
    
    def test_partial_trace_no_sites_traced(self):
        """Test partial trace when no sites are traced out (identity operation)."""
        sites = [Site(0, 2), Site(1, 2)]
        matrix = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
        op = LocalTensor(matrix.copy(), sites)
        
        # Partial trace with empty list should return the same operator
        result = op.partial_trace([])
        
        assert len(result.sites) == 2
        assert result.sites[0].label == 0
        assert result.sites[1].label == 1
        result.unfold()  # Convert back to matrix form
        np.testing.assert_array_equal(result.tensor, matrix)
    
    def test_partial_trace_two_qubits_trace_second(self):
        """Test partial trace of two-qubit system, tracing out second qubit."""
        site1 = Site(0, 2)
        site2 = Site(1, 2)
        sites = [site1, site2]
        
        # Create a specific matrix for predictable results
        # |00⟩⟨00| + |11⟩⟨11| (diagonal matrix)
        matrix = np.zeros((4, 4))
        matrix[0, 0] = 1  # |00⟩⟨00|
        matrix[3, 3] = 1  # |11⟩⟨11|
        
        op = LocalTensor(matrix, sites)
        op.fold()  # Convert to tensor form
        
        # Trace out site 1 (second qubit)
        result = op.partial_trace([1])

        # Expected result: Tr_2(|00⟩⟨00| + |11⟩⟨11|) = |0⟩⟨0| + |1⟩⟨1| = I
        expected = np.eye(2)
        
        assert len(result.sites) == 1
        assert result.sites[0].label == 0
        result.unfold()  # Convert back to matrix form for comparison
        np.testing.assert_array_almost_equal(result.tensor, expected)
    
    def test_partial_trace_two_qubits_trace_first(self):
        """Test partial trace of two-qubit system, tracing out first qubit."""
        site1 = Site(0, 2)
        site2 = Site(1, 2)
        sites = [site1, site2]
        
        # Create |01⟩⟨01| + |10⟩⟨10|
        matrix = np.zeros((4, 4))
        matrix[1, 1] = 1  # |01⟩⟨01|
        matrix[2, 2] = 1  # |10⟩⟨10|
        
        op = LocalTensor(matrix, sites)
        op.fold()
        
        # Trace out site 0 (first qubit)
        result = op.partial_trace([0])
        
        # Expected result: Tr_1(|01⟩⟨01| + |10⟩⟨10|) = |1⟩⟨1| + |0⟩⟨0| = I
        expected = np.eye(2)
        
        assert len(result.sites) == 1
        assert result.sites[0].label == 1
        result.unfold()
        np.testing.assert_array_almost_equal(result.tensor, expected)
    
    def test_partial_trace_bell_state(self):
        """Test partial trace of Bell state density matrix."""
        sites = [Site(0, 2), Site(1, 2)]
        
        # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        # Density matrix ρ = |Φ+⟩⟨Φ+|
        psi = np.array([1, 0, 0, 1]) / np.sqrt(2)  # |00⟩ + |11⟩
        rho = np.outer(psi, psi.conj())
        
        op = LocalTensor(rho, sites)
        op.fold()
        
        # Trace out second qubit
        result = op.partial_trace([1])
        result.unfold()
        
        # Expected: maximally mixed state on first qubit
        expected = 0.5 * np.eye(2)
        
        assert len(result.sites) == 1
        np.testing.assert_array_almost_equal(result.tensor, expected)
    
    def test_partial_trace_three_qubits_trace_middle(self):
        """Test partial trace of three-qubit system, tracing out middle qubit."""
        sites = [Site(0, 2), Site(1, 3), Site(2, 2)]
        
        # Create |000⟩⟨000| + |111⟩⟨111|
        matrix = np.zeros((12, 12))
        matrix[0, 0] = 1  # |000⟩⟨000|
        matrix[7, 7] = 1  # |111⟩⟨111|
        
        op = LocalTensor(matrix, sites)
        op.fold()
        
        # Trace out site 1 (middle qubit)
        result = op.partial_trace([1])
        
        # Expected result on sites 0 and 2: |00⟩⟨00| + |11⟩⟨11|
        expected = np.zeros((4, 4))
        expected[0, 0] = 1  # |00⟩⟨00|
        expected[3, 3] = 1  # |11⟩⟨11|
        
        assert len(result.sites) == 2
        assert result.sites[0].label == 0
        assert result.sites[1].label == 2
        result.unfold()
        np.testing.assert_array_almost_equal(result.tensor, expected)
    
    def test_partial_trace_three_qubits_trace_multiple(self):
        """Test partial trace tracing out multiple qubits."""
        sites = [Site(0, 2), Site(1, 2), Site(2, 2)]
        
        # Random three-qubit state
        matrix = np.random.random((8, 8)) + 1j * np.random.random((8, 8))
        matrix = matrix + matrix.conj().T  # Make Hermitian
        
        op = LocalTensor(matrix, sites)
        op.fold()
        
        # Trace out sites 0 and 2, leaving only site 1
        result = op.partial_trace([0, 2])
        
        # Verify result is single-qubit operator
        assert len(result.sites) == 1
        assert result.sites[0].label == 1
        assert result.tensor.shape == (2, 2)
        
        # Verify trace is preserved (partial trace preserves total trace)
        original_trace = op.trace()
        result_trace = result.trace()
        np.testing.assert_almost_equal(original_trace, result_trace)
    
    def test_partial_trace_different_dimensions(self):
        """Test partial trace with sites of different dimensions."""
        site1 = Site(0, 2)  # Qubit
        site2 = Site(1, 3)  # Qutrit
        sites = [site1, site2]
        
        # Random 6x6 matrix (2*3 = 6)
        matrix = np.random.random((6, 6)) + 1j * np.random.random((6, 6))
        matrix = matrix + matrix.conj().T  # Make Hermitian
        
        op = LocalTensor(matrix, sites)
        op.fold()
        
        # Trace out the qutrit (site 1)
        result = op.partial_trace([1])
        
        assert len(result.sites) == 1
        assert result.sites[0].label == 0
        assert result.sites[0].dimension == 2
        result.unfold()
        assert result.tensor.shape == (2, 2)
        
        # Verify trace conservation
        original_trace = op.trace()
        result_trace = result.trace()
        np.testing.assert_almost_equal(original_trace, result_trace)
    
    def test_partial_trace_identity_operators(self):
        """Test partial trace with identity operators."""
        # Two-qubit identity
        sites = [Site(0, 2), Site(1, 2)]
        identity = np.eye(4)
        op = LocalTensor(identity, sites)
        op.fold()
        
        # Partial trace of identity should be scaled identity
        result = op.partial_trace([1])
        result.unfold()
        
        # Tr_2(I ⊗ I) = 2 * I_1 (factor of 2 from traced dimension)
        expected = 2 * np.eye(2)
        np.testing.assert_array_almost_equal(result.tensor, expected)
    
    def test_partial_trace_trace_all_sites(self):
        """Test partial trace when all sites are traced out."""
        sites = [Site(0, 2), Site(1, 2)]
        matrix = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
        matrix = matrix + matrix.conj().T
        
        op = LocalTensor(matrix, sites)
        op.fold()
        
        # Tracing out all sites should give scalar (trace of the operator)
        result = op.partial_trace([0, 1])
        
        # Result should be a 1x1 "operator" containing just the trace
        assert len(result.sites) == 0
        expected_trace = op.trace()
        
        # The result tensor should be 0-dimensional (scalar)
        assert result.tensor.ndim == 0
        np.testing.assert_almost_equal(result.tensor, expected_trace)
    
    def test_partial_trace_from_matrix_form(self):
        """Test partial trace starting from matrix form (should auto-convert)."""
        sites = [Site(0, 2), Site(1, 2)]
        matrix = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
        
        op = LocalTensor(matrix, sites)
        # Don't fold - test that partial_trace handles matrix form
        
        result = op.partial_trace([1])
        
        # Should work and produce correct single-qubit result
        assert len(result.sites) == 1
        assert result.sites[0].label == 0
    
    def test_partial_trace_properties(self):
        """Test mathematical properties of partial trace."""
        sites = [Site(0, 2), Site(1, 3), Site(2, 2)]
        matrix = np.random.random((12, 12)) + 1j * np.random.random((12, 12))
        matrix = matrix + matrix.conj().T  # Hermitian
        
        op = LocalTensor(matrix, sites)
        op.fold()
        original_trace = op.trace()
        
        # Property 1: Trace is preserved
        op.fold()  # Ensure tensor is in folded form
        result1 = op.partial_trace([0])
        np.testing.assert_almost_equal(result1.trace(), original_trace)
        
        result2 = op.partial_trace([1])
        np.testing.assert_almost_equal(result2.trace(), original_trace)
        
        # Property 2: Order of partial traces shouldn't matter
        result_01 = op.partial_trace([0]).partial_trace([1])  # Trace 0, then 1
        result_10 = op.partial_trace([1]).partial_trace([0])  # Trace 1, then 0
        
        # Both should give the same single-site operator on site 2
        result_01.unfold()
        result_10.unfold()
        np.testing.assert_array_almost_equal(result_01.tensor, result_10.tensor)
    
    def test_partial_trace_error_cases(self):
        """Test error handling in partial trace."""
        sites = [Site(0, 2), Site(1, 2)]
        matrix = np.random.random((4, 4))
        op = LocalTensor(matrix, sites, tensor_format="matrix")
        op.fold()
        
        # Test tracing non-existent site
        with pytest.raises((ValueError, KeyError, IndexError)):
            op.partial_trace([5])  # Site 5 doesn't exist
    
    def test_2(self):
        """Test error handling in partial trace."""
        sites = [Site(0, 2), Site(1, 2), Site(1, 3)]
        matrix = np.random.random((12, 12))
        op = LocalTensor(matrix, sites, tensor_format="matrix")
        op2 = op.compute_nbody_marginal(sites)
        np.testing.assert_array_almost_equal(op.tensor, op2.tensor)


def partial_trace_comprehensive():
    """Run all partial trace tests."""
    test_instance = TestPartialTrace()
    
    test_methods = [
        'test_partial_trace_no_sites_traced',
        'test_partial_trace_two_qubits_trace_second', 
        'test_partial_trace_two_qubits_trace_first',
        'test_partial_trace_bell_state',
        'test_partial_trace_three_qubits_trace_middle',
        'test_partial_trace_three_qubits_trace_multiple',
        'test_partial_trace_different_dimensions',
        'test_partial_trace_identity_operators',
        'test_partial_trace_trace_all_sites',
        'test_partial_trace_from_matrix_form',
        'test_partial_trace_properties',
        'test_partial_trace_error_cases',
        'test_2'
    ]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            print(f"Running {method_name}...")
            getattr(test_instance, method_name)()
            print(f"✓ {method_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name} FAILED: {e}")
            failed += 1
    
    print(f"\nTest Summary: {passed} passed, {failed} failed")


if __name__ == "__main__":
    # Run comprehensive tests
    print("Running comprehensive partial trace tests...\n")
    success = partial_trace_comprehensive()
    