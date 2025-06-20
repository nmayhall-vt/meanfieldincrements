"""
Test for LagrangeMultipliers class.
"""
import numpy as np
import pytest
from meanfieldincrements import Site, PauliHilbertSpace
from meanfieldincrements.LagrangeMultipliers import LagrangeMultipliers


def test_lagrange_multipliers_initialization():
    """Test basic initialization and structure."""
    
    sites = [Site(0, PauliHilbertSpace(2)), Site(1, PauliHilbertSpace(2)), Site(2, PauliHilbertSpace(4))]
    
    # Test 2-body initialization
    lm = LagrangeMultipliers(sites)
    lm.initialize_to_zero(nbody=2)
    
    # Check 1-body terms (scalars)
    assert (0,) in lm
    assert (1,) in lm  
    assert (2,) in lm
    assert lm[(0,)] == 0.0
    
    # Check 2-body terms (matrices)
    assert (0, 1) in lm
    assert (1, 0) in lm
    assert (0, 2) in lm
    assert (2, 0) in lm
    assert (1, 2) in lm
    assert (2, 1) in lm
    
    # Check matrix shapes
    assert lm[(0, 1)].shape == (2, 2)  # dim of site 1
    assert lm[(1, 0)].shape == (2, 2)  # dim of site 0
    assert lm[(0, 2)].shape == (4, 4)  # dim of site 2
    assert lm[(2, 0)].shape == (2, 2)  # dim of site 0


def test_lagrange_multipliers_vector_roundtrip():
    """Test export->import roundtrip."""
    
    sites = [Site(0, PauliHilbertSpace(2)), Site(1, PauliHilbertSpace(2))]
    
    lm = LagrangeMultipliers(sites)
    lm.initialize_to_zero(nbody=2)
    
    # Modify some values
    lm[(0,)] = 1.5
    lm[(1,)] = -0.3
    lm[(0, 1)] = np.array([[1, 2], [3, 4]])
    lm[(1, 0)] = np.array([[5, 6], [7, 8]])
    
    # Export and import
    vector, metadata = lm.export_to_vector()
    lm.import_from_vector(vector, metadata)
    
    # Check values are preserved
    assert lm[(0,)] == 1.5
    assert lm[(1,)] == -0.3
    assert np.allclose(lm[(0, 1)], [[1, 2], [3, 4]])
    assert np.allclose(lm[(1, 0)], [[5, 6], [7, 8]])


def test_lagrange_multipliers_vector_modification():
    """Test that vector modifications affect the multipliers."""
    
    sites = [Site(0, PauliHilbertSpace(2))]
    
    lm = LagrangeMultipliers(sites)
    lm.initialize_to_zero(nbody=1)
    
    # Export, modify, and import
    vector, metadata = lm.export_to_vector()
    vector_modified = vector + 2.0
    lm.import_from_vector(vector_modified, metadata)
    
    # Check that scalar multiplier changed
    assert lm[(0,)] == 2.0


def test_lagrange_multipliers_basic_interface():
    """Test dictionary-like interface."""
    
    sites = [Site(0, PauliHilbertSpace(2))]
    
    lm = LagrangeMultipliers(sites)
    lm.initialize_to_zero(nbody=1)
    
    # Test len, contains, etc.
    assert len(lm) == 1
    assert (0,) in lm
    assert (1,) not in lm
    
    # Test iteration
    keys = list(lm.keys())
    values = list(lm.values()) 
    items = list(lm.items())
    
    assert keys == [(0,)]
    assert values == [0.0]
    assert items == [((0,), 0.0)]


def test_lagrange_multipliers_empty_vector():
    """Test edge case with no multipliers."""
    
    lm = LagrangeMultipliers([])
    
    vector, metadata = lm.export_to_vector()
    
    assert len(vector) == 0
    assert metadata['total_length'] == 0
    
    # Import should work (no-op)
    lm.import_from_vector(vector, metadata)


def test_lagrange_multipliers_vector_validation():
    """Test vector length validation."""
    
    sites = [Site(0, PauliHilbertSpace(2))]
    
    lm = LagrangeMultipliers(sites)
    lm.initialize_to_zero(nbody=1)
    
    vector, metadata = lm.export_to_vector()
    
    # Try wrong length vector
    wrong_vector = np.concatenate([vector, [1.0]])
    
    with pytest.raises(ValueError, match="Vector length .* doesn't match expected length"):
        lm.import_from_vector(wrong_vector, metadata)