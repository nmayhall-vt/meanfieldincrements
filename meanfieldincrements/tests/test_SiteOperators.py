"""
Unit tests for SiteOperators class.
"""

import pytest
import numpy as np
from meanfieldincrements import (HilbertSpace, SiteOperators, PauliHilbertSpace, 
                                SpinHilbertSpace, Site, LocalOperator)


class TestSiteOperators:
    
    def test_basic_construction(self):
        """Test basic SiteOperators construction."""
        hs = HilbertSpace(3)
        ops = SiteOperators(hs)
        
        assert ops.hilbert_space.dimension == 3
        assert "I" in ops
        assert ops["I"].shape == (3, 3)
    
    def test_custom_operators(self):
        """Test construction with custom operators."""
        hs = HilbertSpace(2)
        custom_ops = {
            "A": np.array([[1, 0], [0, -1]]),
            "B": np.array([[0, 1], [1, 0]])
        }
        
        site_ops = SiteOperators(hs, custom_ops)
        assert set(site_ops.keys()) == {"A", "B"}
        assert np.allclose(site_ops["A"], custom_ops["A"])
    
    def test_operator_validation(self):
        """Test operator dimension validation."""
        hs = HilbertSpace(2)
        site_ops = SiteOperators(hs)
        
        # Valid operator
        site_ops["valid"] = np.array([[1, 0], [0, 1]])
        
        # Invalid dimensions
        with pytest.raises(ValueError):
            site_ops["invalid"] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    def test_dict_interface(self):
        """Test dictionary-like interface."""
        hs = HilbertSpace(2)
        ops = SiteOperators(hs)
        
        # Add operator
        test_op = np.array([[1, 2], [3, 4]])
        ops["test"] = test_op
        
        assert "test" in ops
        assert np.allclose(ops["test"], test_op)
        
        # Check keys/values/items
        assert "test" in ops.keys()
        assert np.any([np.allclose(v, test_op) for v in ops.values()])
        
        found = False
        for name, matrix in ops.items():
            if name == "test" and np.allclose(matrix, test_op):
                found = True
        assert found
    
    def test_kron_same_dimension(self):
        """Test Kronecker product with same dimensions."""
        pauli1 = PauliHilbertSpace(2).create_operators()
        pauli2 = PauliHilbertSpace(2).create_operators()
        
        result = pauli1.kron(pauli2)
        
        # Should have 4 * 4 = 16 operators
        assert len(result.operators) == 16
        assert result.hilbert_space.dimension == 4
        
        # Check specific combinations
        assert "II" in result
        assert "XX" in result
        assert "YZ" in result
        
        # Check matrix dimensions
        for op in result.values():
            assert op.shape == (4, 4)
        
        # Verify specific Kronecker products
        expected_xx = np.kron(pauli1["X"], pauli2["X"])
        assert np.allclose(result["XX"], expected_xx)
    
    def test_kron_different_dimensions(self):
        """Test Kronecker product with different dimensions."""
        pauli = PauliHilbertSpace(2).create_operators()  # 4 operators, dim 2
        spin = SpinHilbertSpace(3).create_operators()    # 6 operators, dim 3
        
        result = pauli.kron(spin)
        
        # Should have 4 * 6 = 24 operators
        assert len(result.operators) == 24
        assert result.hilbert_space.dimension == 6
        
        # Check some specific combinations
        assert "ISx" in result
        assert "XSz" in result
        assert "YS+" in result
        
        # Verify a specific product
        expected = np.kron(pauli["X"], spin["Sz"])
        assert np.allclose(result["XSz"], expected)
    
    def test_commutator(self):
        """Test commutator calculation."""
        pauli = PauliHilbertSpace(2).create_operators()
        
        # [X, Y] = 2iZ for Pauli matrices
        comm = pauli.get_commutator("X", "Y")
        expected = 2j * pauli["Z"]
        assert np.allclose(comm, expected)
        
        # [I, X] = 0
        comm_trivial = pauli.get_commutator("I", "X")
        assert np.allclose(comm_trivial, np.zeros((2, 2)))
    
    def test_anticommutator(self):
        """Test anticommutator calculation."""
        pauli = PauliHilbertSpace(2).create_operators()
        
        # {X, Y} = 0 for Pauli matrices
        anticomm = pauli.get_anticommutator("X", "Y")
        assert np.allclose(anticomm, np.zeros((2, 2)))
        
        # {X, X} = 2I
        anticomm_xx = pauli.get_anticommutator("X", "X")
        expected = 2 * pauli["I"]
        assert np.allclose(anticomm_xx, expected)
    
    def test_integration_with_site_localoperator(self):
        """Test integration with existing Site and LocalOperator classes."""
        # Create operators
        pauli = PauliHilbertSpace(2).create_operators()
        spin = SpinHilbertSpace(2).create_operators()
        
        # Create sites
        site0 = Site(0, 2)
        site1 = Site(1, 2)
        
        # Single-site LocalOperators
        pauli_x = LocalOperator(pauli["X"], [site0])
        spin_y = LocalOperator(spin["Sy"], [site1])
        
        assert pauli_x.tensor.shape == (2, 2)
        assert spin_y.tensor.shape == (2, 2)
        
        # Two-site operator from kron
        combined = pauli.kron(spin)
        xy_op = LocalOperator(combined["XSy"], [site0, site1])
        
        assert xy_op.tensor.shape == (4, 4)
        
        # Test fold/unfold
        xy_op.fold()
        assert xy_op.tensor.shape == (2, 2, 2, 2)
        
        xy_op.unfold()
        assert xy_op.tensor.shape == (4, 4)
    
    def test_chained_kron(self):
        """Test chaining multiple Kronecker products."""
        pauli = PauliHilbertSpace(2).create_operators()
        
        # Chain three single-qubit spaces
        two_qubit = pauli.kron(pauli)
        three_qubit = two_qubit.kron(pauli)
        
        assert three_qubit.hilbert_space.dimension == 8
        assert len(three_qubit.operators) == 64  # 4^3
        
        # Check a specific three-qubit operator
        assert "XXX" in three_qubit
        expected_xxx = np.kron(np.kron(pauli["X"], pauli["X"]), pauli["X"])
        assert np.allclose(three_qubit["XXX"], expected_xxx)
    
    def test_add_operator_method(self):
        """Test add_operator method."""
        pauli = PauliHilbertSpace(2).create_operators()
        
        # Add a custom operator
        custom_op = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        pauli.add_operator("H", custom_op)  # Hadamard gate
        
        assert "H" in pauli
        assert np.allclose(pauli["H"], custom_op)
    
    def test_error_handling(self):
        """Test various error conditions."""
        pauli = PauliHilbertSpace(2).create_operators()
        
        # Non-existent operator
        with pytest.raises(KeyError):
            _ = pauli["NonExistent"]
        
        with pytest.raises(KeyError):
            pauli.get_commutator("X", "NonExistent")


class TestHilbertSpaceIntegration:
    """Test integration between HilbertSpace and SiteOperators."""
    
    def test_create_operators_method(self):
        """Test the create_operators convenience method."""
        pauli_space = PauliHilbertSpace(4)
        pauli_ops = pauli_space.create_operators()
        
        assert isinstance(pauli_ops, SiteOperators)
        assert pauli_ops.hilbert_space is pauli_space
        assert len(pauli_ops.operators) == 16
    
    def test_custom_hilbert_space(self):
        """Test with custom Hilbert space implementation."""
        
        class TestSpace(HilbertSpace):
            def __init__(self):
                super().__init__(3, "Test")
            
            def build_operators(self):
                return {
                    "A": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
                    "B": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                    "C": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
                }
        
        test_space = TestSpace()
        test_ops = SiteOperators(test_space)
        
        assert len(test_ops.operators) == 3
        assert set(test_ops.keys()) == {"A", "B", "C"}
        
        # Test kron with standard space
        pauli = PauliHilbertSpace(2).create_operators()
        combined = test_ops.kron(pauli)
        
        assert combined.hilbert_space.dimension == 6
        assert len(combined.operators) == 12  # 3 * 4
        assert "AX" in combined
        assert "CZ" in combined


if __name__ == "__main__":
    # Simple test runner
    test_classes = [TestSiteOperators, TestHilbertSpaceIntegration]
    
    for test_class in test_classes:
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        print(f"\nRunning {test_class.__name__}:")
        for method in methods:
            try:
                getattr(instance, method)()
                print(f"  ✓ {method}")
            except Exception as e:
                print(f"  ✗ {method}: {e}")
                
    print("\nDone!")