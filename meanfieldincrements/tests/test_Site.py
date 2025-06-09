import pytest
import numpy as np
from meanfieldincrements import Site, LocalOperator

# Unit tests for the enhanced Site class
def test_site_class():
    """Comprehensive unit tests for the enhanced Site class."""
    
    print("Testing enhanced Site class...")
    
    # Test 1: Basic construction and properties
    print("\n1. Basic construction:")
    site1 = Site(0, 2)
    site2 = Site(1, 3)
    print(f"   site1: {site1}")
    print(f"   site2: {site2}")
    print(f"   site1 is qubit: {site1.is_qubit()}")
    print(f"   site2 is qutrit: {site2.is_qutrit()}")
    
    # Test 2: Equality and hashing
    print("\n2. Equality and hashing:")
    site3 = Site(0, 2)  # Same as site1
    site4 = Site(0, 3)  # Different dimension
    
    print(f"   site1 == site3: {site1 == site3}")
    print(f"   site1 == site4: {site1 == site4}")
    print(f"   hash(site1) == hash(site3): {hash(site1) == hash(site3)}")
    print(f"   hash(site1) == hash(site4): {hash(site1) == hash(site4)}")
    
    # Test 3: Use in sets and dictionaries
    print("\n3. Collections usage:")
    sites_set = {site1, site2, site3, site4}  # Should have 3 unique sites
    print(f"   Unique sites in set: {len(sites_set)}")
    
    sites_dict = {
        site1: "first qubit",
        site2: "qutrit", 
        site4: "second qubit"
    }
    print(f"   Dictionary lookup site1: {sites_dict.get(site1)}")
    print(f"   Dictionary lookup site3: {sites_dict.get(site3)}")  # Should be same as site1
    
    # Test 4: Sorting
    print("\n4. Sorting:")
    sites_list = [Site(2, 2), Site(0, 3), Site(1, 2), Site(0, 2)]
    sorted_sites = sorted(sites_list)
    print("   Original order:", [f"Site({s.label},{s.dimension})" for s in sites_list])
    print("   Sorted order:  ", [f"Site({s.label},{s.dimension})" for s in sorted_sites])
    
    # Test 5: Compatibility checking
    print("\n5. Compatibility:")
    qubit1 = Site(0, 2)
    qubit2 = Site(5, 2)
    qutrit = Site(1, 3)
    
    print(f"   qubit1.compatible_with(qubit2): {qubit1.compatible_with(qubit2)}")
    print(f"   qubit1.compatible_with(qutrit): {qubit1.compatible_with(qutrit)}")
    
    # Test 6: Error handling
    print("\n6. Error handling:")
    try:
        Site(0, 0)  # Invalid dimension
        print("   ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly caught invalid dimension: {e}")
    
    try:
        Site("not_a_number", 2)  # Invalid label
        print("   ERROR: Should have raised TypeError")
    except TypeError as e:
        print(f"   ✓ Correctly caught invalid label: {e}")
    
    # Test 7: String conversion
    print("\n7. String representations:")
    site = Site("3", "4")  # Test string conversion
    print(f"   repr(): {repr(site)}")
    print(f"   str():  {str(site)}")
    
    print("\n✅ All Site class tests completed successfully!")


if __name__ == "__main__":
    # Run tests
    test_site_class()