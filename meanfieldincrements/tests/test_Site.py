"""
Unit tests for the Site class with HilbertSpace integration.
Clean version without artificial "compatibility" tests.
"""

import pytest
import numpy as np
from meanfieldincrements import (
    Site, HilbertSpace, PauliHilbertSpace, SpinHilbertSpace, FermionHilbertSpace,
    qubit_site, spin_site, fermion_site, multi_qubit_site,
    create_sites, create_qubit_chain, create_spin_chain, create_fermion_chain
)


class TestSiteBasicFunctionality:
    
    def test_site_with_hilbert_space(self):
        """Test Site construction with HilbertSpace objects."""
        # Pauli site
        pauli_space = PauliHilbertSpace(2)
        pauli_site = Site(0, pauli_space)
        
        assert pauli_site.label == 0
        assert pauli_site.hilbert_space is pauli_space
        assert pauli_site.dimension == 2
        assert pauli_site.name == "Pauli1Q"
        
        # Spin site
        spin_space = SpinHilbertSpace(3)
        spin_site_obj = Site(1, spin_space)
        
        assert spin_site_obj.label == 1
        assert spin_site_obj.hilbert_space is spin_space
        assert spin_site_obj.dimension == 3
        assert "Spin1" in spin_site_obj.name
        
        # Fermion site
        fermion_space = FermionHilbertSpace()
        fermion_site_obj = Site(2, fermion_space)
        
        assert fermion_site_obj.label == 2
        assert fermion_site_obj.hilbert_space is fermion_space
        assert fermion_site_obj.dimension == 2
        assert fermion_site_obj.name == "Fermion"
    
    def test_backward_compatibility(self):
        """Test backward compatibility with integer dimensions."""
        # Old style construction
        site = Site(0, 3)
        
        assert site.label == 0
        assert site.dimension == 3
        assert isinstance(site.hilbert_space, HilbertSpace)
        assert site.name == "dim3"
        assert site.is_generic()
    
    def test_invalid_construction(self):
        """Test error handling for invalid construction."""
        with pytest.raises(TypeError):
            Site(0, "invalid")
        
        with pytest.raises(TypeError):
            Site(0, 3.5)  # Float not allowed
    
    def test_site_properties(self):
        """Test site type checking properties."""
        # Qubit site
        qubit = Site(0, PauliHilbertSpace(2))
        assert qubit.is_qubit()
        assert not qubit.is_spin()
        assert not qubit.is_fermion()
        assert not qubit.is_generic()
        assert qubit.get_pauli_qubits() == 1
        assert qubit.get_spin_value() is None
        
        # Multi-qubit site
        two_qubit = Site(1, PauliHilbertSpace(4))
        assert not two_qubit.is_qubit()  # Single qubit is special case
        assert isinstance(two_qubit.hilbert_space, PauliHilbertSpace)
        assert two_qubit.get_pauli_qubits() == 2
        
        # Spin site
        spin = Site(2, SpinHilbertSpace(3))
        assert not spin.is_qubit()
        assert spin.is_spin()
        assert not spin.is_fermion()
        assert not spin.is_generic()
        assert spin.get_spin_value() == 1.0
        assert spin.get_pauli_qubits() is None
        
        # Fermion site
        fermion = Site(3, FermionHilbertSpace())
        assert not fermion.is_qubit()
        assert not fermion.is_spin()
        assert fermion.is_fermion()
        assert not fermion.is_generic()
        assert fermion.get_spin_value() is None
        assert fermion.get_pauli_qubits() is None
        
        # Generic site
        generic = Site(4, 5)
        assert not generic.is_qubit()
        assert not generic.is_spin()
        assert not generic.is_fermion()
        assert generic.is_generic()
        assert generic.get_spin_value() is None
        assert generic.get_pauli_qubits() is None


class TestSiteOperators:
    
    def test_create_operators(self):
        """Test operator creation for different site types."""
        # Qubit site operators
        qubit = Site(0, PauliHilbertSpace(2))
        qubit_ops = qubit.create_operators()
        
        expected_qubit_ops = {'I', 'X', 'Y', 'Z'}
        assert set(qubit_ops.keys()) == expected_qubit_ops
        
        # Spin site operators
        spin = Site(1, SpinHilbertSpace(2))  # Spin-1/2
        spin_ops = spin.create_operators()
        
        expected_spin_ops = {'I', 'Sx', 'Sy', 'Sz', 'S+', 'S-'}
        assert set(spin_ops.keys()) == expected_spin_ops
        
        # Fermion site operators
        fermion = Site(2, FermionHilbertSpace())
        fermion_ops = fermion.create_operators()
        
        expected_fermion_ops = {'I', 'c', 'cdag', 'n'}
        assert set(fermion_ops.keys()) == expected_fermion_ops
        
        # Generic site operators
        generic = Site(3, 4)
        generic_ops = generic.create_operators()
        
        assert 'I' in generic_ops.keys()
        assert generic_ops['I'].shape == (4, 4)


class TestSiteEquality:
    
    def test_site_equality(self):
        """Test site equality and hashing."""
        # Same label, same HilbertSpace type - equal
        site1 = Site(0, PauliHilbertSpace(2))
        site2 = Site(0, PauliHilbertSpace(2))
        assert site1 == site2
        assert hash(site1) == hash(site2)
        
        # Different label - not equal
        site3 = Site(1, PauliHilbertSpace(2))
        assert site1 != site3
        
        # Different HilbertSpace type - not equal
        site4 = Site(0, SpinHilbertSpace(2))
        assert site1 != site4
        
        # Different dimension - not equal
        site5 = Site(0, PauliHilbertSpace(4))
        assert site1 != site5
    
    def test_site_hashing(self):
        """Test that sites can be used in sets and as dict keys."""
        sites = [
            Site(0, PauliHilbertSpace(2)),
            Site(1, SpinHilbertSpace(3)),
            Site(0, PauliHilbertSpace(2)),  # Duplicate
        ]
        
        # Test set behavior
        unique_sites = set(sites)
        assert len(unique_sites) == 2  # Should have 2 unique sites
        
        # Test dict keys
        site_dict = {site: f"site_{site.label}" for site in sites}
        assert len(site_dict) == 2  # Should have 2 unique keys


class TestFactoryFunctions:
    
    def test_qubit_site_factory(self):
        """Test qubit_site factory function."""
        site = qubit_site(5)
        
        assert site.label == 5
        assert site.is_qubit()
        assert site.dimension == 2
        assert isinstance(site.hilbert_space, PauliHilbertSpace)
    
    def test_spin_site_factory(self):
        """Test spin_site factory function."""
        # Spin-1/2
        site_half = spin_site(0, 0.5)
        assert site_half.label == 0
        assert site_half.is_spin()
        assert site_half.dimension == 2
        assert site_half.get_spin_value() == 0.5
        
        # Spin-1
        site_one = spin_site(1, 1.0)
        assert site_one.label == 1
        assert site_one.dimension == 3
        assert site_one.get_spin_value() == 1.0
        
        # Spin-3/2
        site_three_half = spin_site(2, 1.5)
        assert site_three_half.label == 2
        assert site_three_half.dimension == 4
        assert site_three_half.get_spin_value() == 1.5
    
    def test_fermion_site_factory(self):
        """Test fermion_site factory function."""
        site = fermion_site(10)
        
        assert site.label == 10
        assert site.is_fermion()
        assert site.dimension == 2
        assert isinstance(site.hilbert_space, FermionHilbertSpace)
    
    def test_multi_qubit_site_factory(self):
        """Test multi_qubit_site factory function."""
        # 2-qubit site
        site2 = multi_qubit_site(0, 2)
        assert site2.label == 0
        assert site2.dimension == 4
        assert site2.get_pauli_qubits() == 2
        
        # 3-qubit site
        site3 = multi_qubit_site(1, 3)
        assert site3.label == 1
        assert site3.dimension == 8
        assert site3.get_pauli_qubits() == 3


class TestSiteChainFactories:
    
    def test_create_sites_backward_compatibility(self):
        """Test create_sites function for backward compatibility."""
        sites = create_sites(3, 2)
        
        assert len(sites) == 3
        for i, site in enumerate(sites):
            assert site.label == i
            assert site.dimension == 2
            assert site.is_generic()
    
    def test_create_qubit_chain(self):
        """Test create_qubit_chain function."""
        chain = create_qubit_chain(4)
        
        assert len(chain) == 4
        for i, site in enumerate(chain):
            assert site.label == i
            assert site.is_qubit()
            assert site.dimension == 2
    
    def test_create_spin_chain(self):
        """Test create_spin_chain function."""
        # Spin-1/2 chain
        chain_half = create_spin_chain(3, 0.5)
        
        assert len(chain_half) == 3
        for i, site in enumerate(chain_half):
            assert site.label == i
            assert site.is_spin()
            assert site.get_spin_value() == 0.5
            assert site.dimension == 2
        
        # Spin-1 chain
        chain_one = create_spin_chain(2, 1.0)
        
        assert len(chain_one) == 2
        for i, site in enumerate(chain_one):
            assert site.label == i
            assert site.is_spin()
            assert site.get_spin_value() == 1.0
            assert site.dimension == 3
    
    def test_create_fermion_chain(self):
        """Test create_fermion_chain function."""
        chain = create_fermion_chain(5)
        
        assert len(chain) == 5
        for i, site in enumerate(chain):
            assert site.label == i
            assert site.is_fermion()
            assert site.dimension == 2


class TestSiteIntegrationWithExistingCode:
    
    def test_site_with_localoperator(self):
        """Test that Site works with LocalOperator."""
        from meanfieldincrements import LocalTensor
        
        # Create sites with specific HilbertSpaces
        qubit_site_obj = Site(0, PauliHilbertSpace(2))
        spin_site_obj = Site(1, SpinHilbertSpace(2))
        
        # Get operators
        qubit_ops = qubit_site_obj.create_operators()
        spin_ops = spin_site_obj.create_operators()
        
        # Create LocalOperator (single site)
        x_op = LocalTensor(qubit_ops['X'], [qubit_site_obj])
        sx_op = LocalTensor(spin_ops['Sx'], [spin_site_obj])
        
        assert x_op.tensor.shape == (2, 2)
        assert sx_op.tensor.shape == (2, 2)
        
        # Create LocalOperator (two sites) - tensor product
        combined_ops = qubit_ops.kron(spin_ops)
        xsx_op = LocalTensor(combined_ops['XSx'], [qubit_site_obj, spin_site_obj])
        
        assert xsx_op.tensor.shape == (4, 4)
    
    def test_site_with_mbestate(self):
        """Test that Site works with MBEState."""
        from meanfieldincrements import MBEState
        
        # Create mixed site types
        sites = [
            Site(0, PauliHilbertSpace(2)),    # Qubit
            Site(1, SpinHilbertSpace(3)),     # Spin-1
            Site(2, 2),                       # Generic 2-level
        ]
        
        # This should work with the Site class
        rho = MBEState(sites).initialize_mixed()
        
        # Check that it has the right structure
        assert (0,) in rho.terms
        assert (1,) in rho.terms
        assert (2,) in rho.terms
        
        # Check dimensions
        assert rho[(0,)].tensor.shape == (2, 2)  # Qubit
        assert rho[(1,)].tensor.shape == (3, 3)  # Spin-1
        assert rho[(2,)].tensor.shape == (2, 2)  # Generic
    
    def test_mixed_site_types_in_operators(self):
        """Test that different site types can be used together."""
        # Create different types of sites
        qubit = Site(0, PauliHilbertSpace(2))
        spin_half = Site(1, SpinHilbertSpace(2))  # Both dimension 2
        fermion = Site(2, FermionHilbertSpace())  # Also dimension 2
        
        # All should work together since they have the same dimension
        qubit_ops = qubit.create_operators()
        spin_ops = spin_half.create_operators()
        fermion_ops = fermion.create_operators()
        
        # Create hybrid operators
        qubit_spin = qubit_ops.kron(spin_ops)
        qubit_fermion = qubit_ops.kron(fermion_ops)
        spin_fermion = spin_ops.kron(fermion_ops)
        
        # Should be able to create LocalOperators
        from meanfieldincrements import LocalTensor
        
        qs_op = LocalTensor(qubit_spin['XSx'], [qubit, spin_half])
        qf_op = LocalTensor(qubit_fermion['Xc'], [qubit, fermion])
        sf_op = LocalTensor(spin_fermion['Sxc'], [spin_half, fermion])
        
        assert qs_op.tensor.shape == (4, 4)
        assert qf_op.tensor.shape == (4, 4)
        assert sf_op.tensor.shape == (4, 4)


class TestSiteStringRepresentations:
    
    def test_site_repr_and_str(self):
        """Test string representations of sites."""
        # Different site types
        qubit = Site(0, PauliHilbertSpace(2))
        spin = Site(1, SpinHilbertSpace(3))
        fermion = Site(2, FermionHilbertSpace())
        generic = Site(3, 4)
        
        # Test __repr__
        assert "Site(id=0" in repr(qubit)
        assert "Pauli1Q" in repr(qubit)
        assert "Site(id=1" in repr(spin)
        assert "Spin1" in repr(spin)
        
        # Test __str__
        assert "Site 0:" in str(qubit)
        assert "dim=2" in str(qubit)
        assert "Site 1:" in str(spin)
        assert "dim=3" in str(spin)


class TestBackwardCompatibility:
    """Tests focused on maintaining backward compatibility."""
    
    def test_label_attribute_exists(self):
        """Test that site.label exists and works correctly."""
        # Test with old-style creation
        site_old = Site(5, 2)
        assert hasattr(site_old, 'label')
        assert site_old.label == 5
        
        # Test with new-style creation
        site_new = Site(10, PauliHilbertSpace(2))
        assert hasattr(site_new, 'label')
        assert site_new.label == 10
    
    def test_mbestate_compatibility(self):
        """Test specific MBEState compatibility that was failing before."""
        from meanfieldincrements import MBEState
        
        # This exact scenario was failing before
        sites = [Site(0, 2), Site(1, 4), Site(2, 3)]
        
        # This should work now
        rho = MBEState(sites).initialize_mixed()
        
        # Verify each term was created correctly
        for site in sites:
            key = (site.label,)  # This is what MBEState uses
            assert key in rho.terms, f"Missing term for site {site.label}"
            
            term = rho.terms[key]
            expected_shape = (site.dimension, site.dimension)
            assert term.tensor.shape == expected_shape, f"Wrong shape for site {site.label}"
    
    def test_existing_site_usage_patterns(self):
        """Test common usage patterns that should continue to work."""
        # Pattern 1: Create sites with dimensions
        sites = [Site(i, 2) for i in range(3)]
        assert all(site.dimension == 2 for site in sites)
        assert all(site.label == i for i, site in enumerate(sites))
        
        # Pattern 2: Use sites in collections
        site_dict = {site.label: site for site in sites}
        assert len(site_dict) == 3
        
        # Pattern 3: Site properties
        for site in sites:
            assert hasattr(site, 'label')
            assert hasattr(site, 'dimension')
            assert hasattr(site, 'name')


if __name__ == "__main__":
    # Simple test runner
    import sys
    
    test_classes = [
        TestSiteBasicFunctionality,
        TestSiteOperators,
        TestSiteEquality,
        TestFactoryFunctions,
        TestSiteChainFactories,
        TestSiteIntegrationWithExistingCode,
        TestSiteStringRepresentations,
        TestBackwardCompatibility,
    ]
    
    all_passed = True
    
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
                all_passed = False
    
    if all_passed:
        print(f"\n🎉 All tests passed!")
        print("The Site class is working correctly with:")
        print("  - site.label as the primary identifier")
        print("  - Full backward compatibility with existing code")
        print("  - MBEState integration working")
        print("  - All factory functions")
        print("  - Mixed site types can be used together freely")
    else:
        print(f"\n❌ Some tests failed!")
        sys.exit(1)