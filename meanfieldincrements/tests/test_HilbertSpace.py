"""
Unit tests for HilbertSpace classes.
"""

import pytest
import numpy as np
from fractions import Fraction
from meanfieldincrements import HilbertSpace, PauliHilbertSpace, SpinHilbertSpace, FermionHilbertSpace


class TestHilbertSpace:
    
    def test_basic_construction(self):
        """Test basic HilbertSpace construction."""
        hs = HilbertSpace(3)
        assert hs.dimension == 3
        assert hs.name == "dim3"
        
        hs_named = HilbertSpace(4, "test")
        assert hs_named.dimension == 4
        assert hs_named.name == "test"
    
    def test_invalid_dimension(self):
        """Test error handling for invalid dimensions."""
        with pytest.raises(ValueError):
            HilbertSpace(0)
        with pytest.raises(ValueError):
            HilbertSpace(-1)
    
    def test_build_operators(self):
        """Test base build_operators method."""
        hs = HilbertSpace(3)
        ops = hs.build_operators()
        assert "I" in ops
        assert ops["I"].shape == (3, 3)
        assert np.allclose(ops["I"], np.eye(3))


class TestPauliHilbertSpace:
    
    def test_valid_dimensions(self):
        """Test PauliHilbertSpace with valid dimensions."""
        # Single qubit
        phs1 = PauliHilbertSpace(2)
        assert phs1.dimension == 2
        assert phs1.n_qubits == 1
        
        # Two qubits
        phs2 = PauliHilbertSpace(4)
        assert phs2.dimension == 4
        assert phs2.n_qubits == 2
        
        # Three qubits
        phs3 = PauliHilbertSpace(8)
        assert phs3.dimension == 8
        assert phs3.n_qubits == 3
    
    def test_invalid_dimensions(self):
        """Test error handling for non-power-of-2 dimensions."""
        with pytest.raises(ValueError):
            PauliHilbertSpace(3)
        with pytest.raises(ValueError):
            PauliHilbertSpace(5)
        with pytest.raises(ValueError):
            PauliHilbertSpace(6)
    
    def test_single_qubit_operators(self):
        """Test single qubit Pauli operators."""
        phs = PauliHilbertSpace(2)
        ops = phs.build_operators()
        
        # Should have 4 operators
        assert len(ops) == 4
        assert set(ops.keys()) == {'I', 'X', 'Y', 'Z'}
        
        # Check matrix values
        assert np.allclose(ops['I'], np.eye(2))
        assert np.allclose(ops['X'], np.array([[0, 1], [1, 0]]))
        assert np.allclose(ops['Y'], np.array([[0, -1j], [1j, 0]]))
        assert np.allclose(ops['Z'], np.array([[1, 0], [0, -1]]))
    
    def test_two_qubit_operators(self):
        """Test two qubit Pauli operators."""
        phs = PauliHilbertSpace(4)
        ops = phs.build_operators()
        
        # Should have 16 operators
        assert len(ops) == 16
        
        # Check a few specific ones
        assert np.allclose(ops['II'], np.eye(4))
        
        # XX should be X ⊗ X
        expected_xx = np.kron([[0, 1], [1, 0]], [[0, 1], [1, 0]])
        assert np.allclose(ops['XX'], expected_xx)
        
        # All operators should be 4x4
        for op in ops.values():
            assert op.shape == (4, 4)
    
    def test_pauli_commutation_relations(self):
        """Test Pauli commutation relations."""
        phs = PauliHilbertSpace(2)
        ops = phs.build_operators()
        
        # [X, Y] = 2iZ
        comm_xy = ops['X'] @ ops['Y'] - ops['Y'] @ ops['X']
        assert np.allclose(comm_xy, 2j * ops['Z'])
        
        # [Y, Z] = 2iX
        comm_yz = ops['Y'] @ ops['Z'] - ops['Z'] @ ops['Y']
        assert np.allclose(comm_yz, 2j * ops['X'])
        
        # [Z, X] = 2iY
        comm_zx = ops['Z'] @ ops['X'] - ops['X'] @ ops['Z']
        assert np.allclose(comm_zx, 2j * ops['Y'])


class TestSpinHilbertSpace:
    
    def test_spin_half(self):
        """Test spin-1/2 system."""
        shs = SpinHilbertSpace(2)
        assert shs.dimension == 2
        assert shs.spin == Fraction(1, 2)
        assert shs.j == 0.5
        
        ops = shs.build_operators()
        expected_ops = {'I', 'Sx', 'Sy', 'Sz', 'S+', 'S-'}
        assert set(ops.keys()) == expected_ops
        
        # Check specific values for spin-1/2
        expected_sz = 0.5 * np.array([[1, 0], [0, -1]])
        assert np.allclose(ops['Sz'], expected_sz)
        
        # Check ladder operators
        expected_sp = np.array([[0, 1], [0, 0]])  # S+|↓⟩ = |↑⟩
        expected_sm = np.array([[0, 0], [1, 0]])  # S-|↑⟩ = |↓⟩
        assert np.allclose(ops['S+'], expected_sp)
        assert np.allclose(ops['S-'], expected_sm)
        
        # Check Sx and Sy derived from ladder operators
        expected_sx = (expected_sp + expected_sm) / 2
        expected_sy = (expected_sp - expected_sm) / (2j)
        assert np.allclose(ops['Sx'], expected_sx)
        assert np.allclose(ops['Sy'], expected_sy)
    
    def test_spin_one(self):
        """Test spin-1 system."""
        shs = SpinHilbertSpace(3)
        assert shs.dimension == 3
        assert shs.spin == Fraction(1, 1)
        assert shs.j == 1.0
        
        ops = shs.build_operators()
        expected_ops = {'I', 'Sx', 'Sy', 'Sz', 'S+', 'S-'}
        assert set(ops.keys()) == expected_ops
        
        # Check Sz diagonal (states ordered as |1,1⟩, |1,0⟩, |1,-1⟩)
        expected_sz = np.diag([1, 0, -1])
        assert np.allclose(ops['Sz'], expected_sz)
        
        # Check ladder operators for spin-1
        # S+|1,0⟩ = √2|1,1⟩, S+|1,-1⟩ = √2|1,0⟩
        expected_sp = np.array([[0, np.sqrt(2), 0],
                               [0, 0, np.sqrt(2)],
                               [0, 0, 0]])
        assert np.allclose(ops['S+'], expected_sp)
        
        # S-|1,1⟩ = √2|1,0⟩, S-|1,0⟩ = √2|1,-1⟩
        expected_sm = np.array([[0, 0, 0],
                               [np.sqrt(2), 0, 0],
                               [0, np.sqrt(2), 0]])
        assert np.allclose(ops['S-'], expected_sm)
    
    def test_spin_three_halves(self):
        """Test spin-3/2 system."""
        shs = SpinHilbertSpace(4)
        assert shs.dimension == 4
        assert shs.spin == Fraction(3, 2)
        assert shs.j == 1.5
        
        ops = shs.build_operators()
        
        # Check Sz diagonal (states: |3/2,3/2⟩, |3/2,1/2⟩, |3/2,-1/2⟩, |3/2,-3/2⟩)
        expected_sz = np.diag([1.5, 0.5, -0.5, -1.5])
        assert np.allclose(ops['Sz'], expected_sz)
        
        # Test a few ladder operator matrix elements
        j = 1.5
        # S+|3/2,1/2⟩ = √[j(j+1) - (1/2)(3/2)]|3/2,3/2⟩ = √[15/4 - 3/4]|3/2,3/2⟩ = √3|3/2,3/2⟩
        assert np.allclose(ops['S+'][0, 1], np.sqrt(3))
        # S+|3/2,-1/2⟩ = √[j(j+1) - (-1/2)(1/2)]|3/2,1/2⟩ = √[15/4 + 1/4]|3/2,1/2⟩ = 2|3/2,1/2⟩
        assert np.allclose(ops['S+'][1, 2], 2.0)
    
    def test_invalid_spin_dimension(self):
        """Test error for invalid spin dimensions."""
        with pytest.raises(ValueError):
            SpinHilbertSpace(0)  # No j<=0 spin
    
    def test_spin_commutation_relations(self):
        """Test spin commutation relations [Si, Sj] = iεijk Sk."""
        # Test for spin-1/2
        shs_half = SpinHilbertSpace(2)
        ops_half = shs_half.build_operators()
        
        # [Sx, Sy] = i*Sz
        comm_xy = ops_half['Sx'] @ ops_half['Sy'] - ops_half['Sy'] @ ops_half['Sx']
        assert np.allclose(comm_xy, 1j * ops_half['Sz'])
        
        # [Sy, Sz] = i*Sx
        comm_yz = ops_half['Sy'] @ ops_half['Sz'] - ops_half['Sz'] @ ops_half['Sy']
        assert np.allclose(comm_yz, 1j * ops_half['Sx'])
        
        # [Sz, Sx] = i*Sy
        comm_zx = ops_half['Sz'] @ ops_half['Sx'] - ops_half['Sx'] @ ops_half['Sz']
        assert np.allclose(comm_zx, 1j * ops_half['Sy'])
        
        # Test for spin-1
        shs_one = SpinHilbertSpace(3)
        ops_one = shs_one.build_operators()
        
        # [Sx, Sy] = i*Sz for spin-1
        comm_xy_s1 = ops_one['Sx'] @ ops_one['Sy'] - ops_one['Sy'] @ ops_one['Sx']
        assert np.allclose(comm_xy_s1, 1j * ops_one['Sz'])
    
    def test_spin_squared_operator(self):
        """Test S^2 = Sx^2 + Sy^2 + Sz^2 = j(j+1)I."""
        # Test for spin-1/2
        shs_half = SpinHilbertSpace(2)
        ops_half = shs_half.build_operators()
        
        s_squared = (ops_half['Sx'] @ ops_half['Sx'] + 
                    ops_half['Sy'] @ ops_half['Sy'] + 
                    ops_half['Sz'] @ ops_half['Sz'])
        j = 0.5
        expected = j * (j + 1) * np.eye(2)  # j(j+1) with j=1/2
        assert np.allclose(s_squared, expected)
        
        # Test for spin-1
        shs_one = SpinHilbertSpace(3)
        ops_one = shs_one.build_operators()
        
        s_squared = (ops_one['Sx'] @ ops_one['Sx'] + 
                    ops_one['Sy'] @ ops_one['Sy'] + 
                    ops_one['Sz'] @ ops_one['Sz'])
        j = 1.0
        expected = j * (j + 1) * np.eye(3)  # j(j+1) with j=1
        assert np.allclose(s_squared, expected)
        
        # Test for spin-3/2
        shs_3half = SpinHilbertSpace(4)
        ops_3half = shs_3half.build_operators()
        
        s_squared = (ops_3half['Sx'] @ ops_3half['Sx'] + 
                    ops_3half['Sy'] @ ops_3half['Sy'] + 
                    ops_3half['Sz'] @ ops_3half['Sz'])
        j = 1.5
        expected = j * (j + 1) * np.eye(4)  # j(j+1) with j=3/2
        assert np.allclose(s_squared, expected)
    
    def test_ladder_operator_properties(self):
        """Test ladder operator properties."""
        # Test for spin-1
        shs = SpinHilbertSpace(3)
        ops = shs.build_operators()
        
        sp = ops['S+']
        sm = ops['S-']
        sz = ops['Sz']
        
        # Test [S+, S-] = 2Sz
        comm_pm = sp @ sm - sm @ sp
        expected_comm = 2 * sz
        assert np.allclose(comm_pm, expected_comm)
        
        # Test [Sz, S+] = S+
        comm_zp = sz @ sp - sp @ sz
        assert np.allclose(comm_zp, sp)
        
        # Test [Sz, S-] = -S-
        comm_zm = sz @ sm - sm @ sz
        assert np.allclose(comm_zm, -sm)
    
    def test_fraction_arithmetic(self):
        """Test that Fraction arithmetic works correctly."""
        # Test various spin values
        test_cases = [
            (2, Fraction(1, 2)),    # spin-1/2
            (3, Fraction(1, 1)),    # spin-1  
            (4, Fraction(3, 2)),    # spin-3/2
            (5, Fraction(2, 1)),    # spin-2
            (6, Fraction(5, 2)),    # spin-5/2
        ]
        
        for dim, expected_spin in test_cases:
            shs = SpinHilbertSpace(dim)
            assert shs.spin == expected_spin
            assert shs.j == float(expected_spin)


class TestFermionHilbertSpace:
    
    def test_construction(self):
        """Test fermion Hilbert space construction."""
        fhs = FermionHilbertSpace()
        assert fhs.dimension == 2
        assert fhs.name == "Fermion"
    
    def test_operators(self):
        """Test fermionic operators."""
        fhs = FermionHilbertSpace()
        ops = fhs.build_operators()
        
        expected_ops = {'I', 'c', 'cdag', 'n'}
        assert set(ops.keys()) == expected_ops
        
        # Check specific matrices
        assert np.allclose(ops['I'], np.eye(2))
        assert np.allclose(ops['c'], np.array([[0, 1], [0, 0]]))
        assert np.allclose(ops['cdag'], np.array([[0, 0], [1, 0]]))
        assert np.allclose(ops['n'], np.array([[0, 0], [0, 1]]))
    
    def test_fermionic_anticommutation(self):
        """Test fermionic anticommutation relations."""
        fhs = FermionHilbertSpace()
        ops = fhs.build_operators()
        
        c = ops['c']
        cdag = ops['cdag']
        
        # {c, c†} = 1
        anticomm = c @ cdag + cdag @ c
        assert np.allclose(anticomm, np.eye(2))
        
        # {c, c} = 0  
        anticomm_cc = c @ c + c @ c
        assert np.allclose(anticomm_cc, np.zeros((2, 2)))


if __name__ == "__main__":
    # Simple test runner
    import sys
    
    test_classes = [TestHilbertSpace, TestPauliHilbertSpace, TestSpinHilbertSpace, TestFermionHilbertSpace]
    
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