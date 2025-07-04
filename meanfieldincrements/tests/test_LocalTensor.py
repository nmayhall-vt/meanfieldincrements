import pytest
import numpy as np
import meanfieldincrements
from meanfieldincrements import *

class TestLocalOperator:
    
    def test_single_site_fold_unfold(self):
        """Test fold/unfold for single site operator."""
        site = Site(0, 3)
        
        # Create a 3x3 matrix
        matrix = np.random.random((3, 3))
        op = LocalTensor(matrix.copy(), [site])
        
        # Test fold: (3,3) -> (3,3)
        op.fold()
        assert op.tensor.shape == (3, 3)
        
        # Test unfold: (3,3) -> (3,3)  
        op.unfold()
        assert op.tensor.shape == (3, 3)
        np.testing.assert_array_equal(op.tensor, matrix)
    
    def test_two_site_equal_dimensions(self):
        """Test fold/unfold for two sites with equal dimensions."""
        site1 = Site(0, 2)
        site2 = Site(1, 2)
        sites = [site1, site2]
        
        # Create 4x4 matrix (2*2 = 4)
        matrix = np.random.random((4, 4))
        op = LocalTensor(matrix.copy(), sites)
        
        # Test fold: (4,4) -> (2,2,2,2)
        op.fold()
        assert op.tensor.shape == (2, 2, 2, 2)
        
        # Test unfold: (2,2,2,2) -> (4,4)
        op.unfold()
        assert op.tensor.shape == (4, 4)
        np.testing.assert_array_equal(op.tensor, matrix)
    
    def test_two_site_different_dimensions(self):
        """Test fold/unfold for two sites with different dimensions."""
        site1 = Site(0, 2)
        site2 = Site(1, 3)
        sites = [site1, site2]
        
        # Create 6x6 matrix (2*3 = 6)
        matrix = np.random.random((6, 6))
        op = LocalTensor(matrix.copy(), sites)
        
        # Test fold: (6,6) -> (2,3,2,3)
        op.fold()
        assert op.tensor.shape == (2, 3, 2, 3)
        
        # Test unfold: (2,3,2,3) -> (6,6)
        op.unfold()
        assert op.tensor.shape == (6, 6)
        np.testing.assert_array_equal(op.tensor, matrix)
    
    def test_three_site_operator(self):
        """Test fold/unfold for three site operator."""
        sites = [Site(0, 2), Site(1, 2), Site(2, 3)]
        
        # Create 12x12 matrix (2*2*3 = 12)
        matrix = np.random.random((12, 12))
        op = LocalTensor(matrix.copy(), sites)
        
        # Test fold: (12,12) -> (2,2,3,2,2,3)
        op.fold()
        assert op.tensor.shape == (2, 2, 3, 2, 2, 3)
        
        # Test unfold: (2,2,3,2,2,3) -> (12,12)
        op.unfold()
        assert op.tensor.shape == (12, 12)
        np.testing.assert_array_equal(op.tensor, matrix)
    
    def test_round_trip_matrix_to_tensor(self):
        """Test round-trip: matrix -> fold -> unfold -> matrix."""
        sites = [Site(0, 2), Site(1, 3)]
        matrix = np.random.random((6, 6))
        op = LocalTensor(matrix.copy(), sites)
        
        original_matrix = op.tensor.copy()
        
        # Round trip: matrix -> tensor -> matrix
        op.fold().unfold()
        
        np.testing.assert_array_equal(op.tensor, original_matrix)
        assert op.tensor.shape == (6, 6)
    
    def test_round_trip_tensor_to_matrix(self):
        """Test round-trip: tensor -> unfold -> fold -> tensor."""
        sites = [Site(0, 2), Site(1, 3)]
        # Start with tensor form
        tensor = np.random.random((2, 3, 2, 3))
        op = LocalTensor(tensor.copy(), sites, tensor_format='tensor')
        
        original_tensor = op.tensor.copy()
        
        # Round trip: tensor -> matrix -> tensor
        op.unfold().fold()
        
        np.testing.assert_array_equal(op.tensor, original_tensor)
        assert op.tensor.shape == (2, 3, 2, 3)
    
    def test_method_chaining(self):
        """Test that methods return self for chaining."""
        sites = [Site(0, 2), Site(1, 2)]
        matrix = np.random.random((4, 4))
        op = LocalTensor(matrix, sites)
        
        # Test method chaining
        result = op.fold().unfold().fold()
        assert result is op
        assert op.tensor.shape == (2, 2, 2, 2)
    
    def test_identity_operators(self):
        """Test with identity operators to verify correctness."""
        # Single site identity
        site = Site(0, 2)
        identity_2 = np.eye(2)
        op = LocalTensor(identity_2, [site])
        
        op.fold()
        # Identity matrix should become identity tensor
        expected_tensor = np.zeros((2, 2))
        expected_tensor[0, 0] = 1
        expected_tensor[1, 1] = 1
        np.testing.assert_array_equal(op.tensor, expected_tensor)
        
        # Two site identity
        sites = [Site(0, 2), Site(1, 2)]
        identity_4 = np.eye(4)
        op2 = LocalTensor(identity_4, sites)
        
        op2.fold()
        assert op2.tensor.shape == (2, 2, 2, 2)
        
        # Check that diagonal elements are preserved
        op2.unfold()
        np.testing.assert_array_equal(op2.tensor, identity_4)
    
    def test_pauli_operators(self):
        """Test with Pauli operators for physical relevance."""
        site = Site(0, 2)
        
        # Pauli-X matrix
        pauli_x = np.array([[0, 1], [1, 0]], dtype=float)
        op = LocalTensor(pauli_x.copy(), [site])
        
        original = op.tensor.copy()
        
        # Test that Pauli-X is preserved through fold/unfold
        op.fold().unfold()
        np.testing.assert_array_equal(op.tensor, original)
    
    def test_large_dimensions(self):
        """Test with larger local dimensions."""
        sites = [Site(0, 5), Site(1, 4)]
        total_dim = 5 * 4
        
        matrix = np.random.random((total_dim, total_dim))
        op = LocalTensor(matrix.copy(), sites)
        
        # Test shapes
        op.fold()
        assert op.tensor.shape == (5, 4, 5, 4)
        
        op.unfold()
        assert op.tensor.shape == (total_dim, total_dim)
        np.testing.assert_array_equal(op.tensor, matrix)
    
    def test_value_preservation_detailed(self):
        """Detailed test of value preservation during reshaping."""
        sites = [Site(0, 2), Site(1, 2)]
        
        # Create a specific matrix with known values
        matrix = np.arange(16).reshape(4, 4).astype(float)
        op = LocalTensor(matrix.copy(), sites)
        
        # Fold and check specific tensor elements
        op.fold()
        
        # The mapping should preserve the linear indexing
        # matrix[i,j] should map to tensor[i0,i1,j0,j1] where
        # i = i0*2 + i1 and j = j0*2 + j1
        for i in range(4):
            for j in range(4):
                i0, i1 = divmod(i, 2)
                j0, j1 = divmod(j, 2)
                assert op.tensor[i0, i1, j0, j1] == matrix[i, j]
        
        # Unfold and verify we get back the original
        op.unfold()
        np.testing.assert_array_equal(op.tensor, matrix)
    def test_trace_performance(self):
        """Test trace performance and correctness."""
        import time
        from meanfieldincrements import Site
        
        print("Testing optimized trace function...")
        
        # Test 1: Single site operators
        print("\n1. Single site operators:")
        site = Site(0, 2)
        
        # Pauli matrices
        pauli_matrices = {
            'I': np.array([[1, 0], [0, 1]]),
            'X': np.array([[0, 1], [1, 0]]),
            'Y': np.array([[0, -1j], [1j, 0]]),
            'Z': np.array([[1, 0], [0, -1]])
        }
        
        for name, matrix in pauli_matrices.items():
            op = LocalTensor(matrix, [site])
            matrix_trace = op.trace()
            op.fold()
            tensor_trace = op.trace()
            print(f"   Pauli-{name}: matrix={matrix_trace:.3f}, tensor={tensor_trace:.3f}")
            assert np.isclose(matrix_trace, tensor_trace)

        # Test 2: Multi-site operators
        print("\n2. Multi-site operators:")
        
        for n_sites in [2, 3, 4]:
            sites = [Site(i, i+2) for i in range(n_sites)]
            # dim = 2**n_sites
            dim = np.prod([site.dimension for site in sites])
            
            # Random operator
            random_matrix = np.random.random((dim, dim)) + 1j * np.random.random((dim, dim))
            op = LocalTensor(random_matrix, sites)
            
            # Time matrix trace
            start = time.time()
            matrix_trace = op.trace()
            matrix_time = time.time() - start
            
            # Time tensor trace  
            op.fold()
            start = time.time()
            tensor_trace = op.trace()
            tensor_time = time.time() - start
            
            error = abs(matrix_trace - tensor_trace) / abs(matrix_trace)
            print(f"   {n_sites} sites: error={error:.2e}, "
                f"matrix_time={matrix_time:.6f}s, tensor_time={tensor_time:.6f}s")
        
        # Test 3: Identity operators (should give total dimension)
        print("\n3. Identity operators:")
        for n_sites in [1, 2, 3, 4]:
            sites = [Site(i, i+2) for i in range(n_sites)]
            dim = np.prod([site.dimension for site in sites]) 
            identity = np.eye(dim)
            op = LocalTensor(identity, sites)
            
            trace_val = op.trace()
            expected = dim
            print(f"   {n_sites} sites: trace={trace_val}, expected={expected}")

def run_tests():
    """Simple test runner if pytest is not available."""
    test_instance = TestLocalOperator()
    
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
    run_tests()
    # test_trace_performance()