"""
Test for the vector import/export functionality in Marginals class.
"""
import numpy as np
import pytest
from meanfieldincrements import Site, PauliHilbertSpace, SpinHilbertSpace
from meanfieldincrements.Marginals import Marginals
from meanfieldincrements.FactorizedMarginal import FactorizedMarginal


def test_marginals_vector_roundtrip():
    """Test that export->import gives back the same vector."""
    
    # Create test sites and marginals
    sites = [
        Site(0, PauliHilbertSpace(2)),
        Site(1, SpinHilbertSpace(2)),
        Site(2, PauliHilbertSpace(2))
    ]
    
    np.random.seed(42)
    
    # Create FactorizedMarginal objects with different ranks
    marginals = Marginals()
    marginals[(0,)] = FactorizedMarginal(np.random.randn(2, 3), [sites[0]], 'matrix')
    marginals[(1,)] = FactorizedMarginal(np.random.randn(2, 2), [sites[1]], 'matrix')
    marginals[(0, 1)] = FactorizedMarginal(np.random.randn(4, 2), [sites[0], sites[1]], 'matrix')
    
    # Export to vector
    vector, metadata = marginals.export_to_vector()
    
    # Import back
    marginals.import_from_vector(vector, metadata)
    
    # Export again and check consistency
    vector_roundtrip, _ = marginals.export_to_vector()
    
    assert np.allclose(vector, vector_roundtrip), "Round-trip export/import failed"
    assert len(vector) == metadata['total_length'], "Vector length doesn't match metadata"


def test_marginals_vector_modification():
    """Test that modifying the vector changes the marginals."""
    
    sites = [Site(0, PauliHilbertSpace(2)), Site(1, PauliHilbertSpace(2))]
    np.random.seed(123)
    
    # Create simple test case
    A_original = np.random.randn(2, 1) + 1j * np.random.randn(2, 1)
    marginal = FactorizedMarginal(A_original.copy(), [sites[0]], 'matrix')
    
    marginals = Marginals()
    marginals[(0,)] = marginal
    
    # Export and modify
    vector, metadata = marginals.export_to_vector()
    vector_modified = vector + 0.1
    
    # Import modified vector
    marginals.import_from_vector(vector_modified, metadata)
    
    # Check that the A factor actually changed
    A_new = marginals[(0,)].factor_A
    assert not np.allclose(A_original, A_new), "A factor should have changed after import"
    
    # Check that the modification is consistent with what we expect
    expected_A = A_original + 0.1
    assert np.allclose(A_new, expected_A), "A factor doesn't match expected modification"


def test_marginals_trace_preservation():
    """Test that marginals maintain trace=1 after vector operations."""
    
    sites = [Site(0, PauliHilbertSpace(2))]
    np.random.seed(456)
    
    # Create normalized marginal
    A = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    marginal = FactorizedMarginal(A, [sites[0]], 'matrix')
    
    marginals = Marginals()
    marginals[(0,)] = marginal
    
    # Check initial trace
    initial_trace = marginals[(0,)].trace()
    assert np.isclose(initial_trace, 1.0), f"Initial trace should be 1, got {initial_trace}"
    
    # Export, modify slightly, and import
    vector, metadata = marginals.export_to_vector()
    vector_modified = vector + 0.01 * np.random.randn(len(vector))
    marginals.import_from_vector(vector_modified, metadata)
    
    # Check that trace is still 1 (FactorizedMarginal normalizes automatically)
    final_trace = marginals[(0,)].trace()
    assert np.isclose(final_trace, 1.0), f"Final trace should be 1, got {final_trace}"


def test_marginals_tensor_format_consistency():
    """Test that tensor vs matrix format gives consistent results."""
    
    sites = [Site(0, PauliHilbertSpace(2)), Site(1, PauliHilbertSpace(2))]
    np.random.seed(789)
    
    A = np.random.randn(4, 1)  # Two-site, rank-1
    marginal = FactorizedMarginal(A.copy(), sites, 'matrix')
    
    marginals = Marginals()
    marginals[(0, 1)] = marginal
    
    # Export in matrix format
    vector_matrix, metadata_matrix = marginals.export_to_vector()
    
    # Convert to tensor format and export
    marginals.fold()
    vector_tensor, metadata_tensor = marginals.export_to_vector()
    
    # Vectors should be identical
    assert np.allclose(vector_matrix, vector_tensor), "Matrix and tensor format should give same vector"
    
    # Test round-trip with tensor format
    modified_vector = vector_tensor + 0.05
    marginals.import_from_vector(modified_vector, metadata_tensor)
    vector_check, _ = marginals.export_to_vector()
    
    assert np.allclose(vector_check, modified_vector), "Tensor format round-trip failed"


def test_marginals_empty_case():
    """Test edge case with no FactorizedMarginals."""
    
    marginals = Marginals()
    
    # Export empty marginals
    vector, metadata = marginals.export_to_vector()
    
    assert len(vector) == 0, "Empty marginals should give empty vector"
    assert metadata['total_length'] == 0, "Empty marginals should have zero total length"
    
    # Import should work (no-op)
    marginals.import_from_vector(vector, metadata)  # Should not raise


def test_marginals_vector_length_validation():
    """Test that import validates vector length."""
    
    sites = [Site(0, PauliHilbertSpace(2))]
    A = np.random.randn(2, 1)
    marginal = FactorizedMarginal(A, [sites[0]], 'matrix')
    
    marginals = Marginals()
    marginals[(0,)] = marginal
    
    vector, metadata = marginals.export_to_vector()
    
    # Try to import wrong-length vector
    wrong_vector = np.concatenate([vector, [1.0]])  # Add extra element
    
    with pytest.raises(ValueError, match="Vector length .* doesn't match expected length"):
        marginals.import_from_vector(wrong_vector, metadata)