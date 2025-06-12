Getting Started
===============


You might choose to write an overview tutorial or set of tutorials.

.. code-block:: python
    
    import meanfieldincrements
    from meanfieldincrements import Site, LocalOperator, PauliHilbertSpace, SpinHilbertSpace

Basic Usage
-----------

The MeanFieldIncrements package provides tools for quantum many-body calculations
with support for various operator types and Hilbert spaces.

Creating Sites and Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create sites
    site0 = Site(0, 2)  # Qubit at site 0  
    site1 = Site(1, 3)  # Qutrit at site 1

    # Create operators using HilbertSpace classes
    pauli_ops = PauliHilbertSpace(2).create_operators()  # Single qubit Pauli operators
    spin_ops = SpinHilbertSpace(2).create_operators()    # Spin-1/2 operators

    print(f"Pauli operators: {list(pauli_ops.keys())}")  # ['I', 'X', 'Y', 'Z']
    print(f"Spin operators: {list(spin_ops.keys())}")    # ['I', 'Sx', 'Sy', 'Sz']

Working with Multi-Site Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create two-site operators using tensor products
    two_site_ops = pauli_ops.kron(spin_ops)
    print(f"Two-site operators: {len(two_site_ops.operators)} total")
    # Creates operators like 'ISx', 'XSy', 'ZSz', etc.

    # Convert to LocalOperator for use with existing functionality
    xx_interaction = LocalOperator(two_site_ops['XSx'], [site0, site1])
    
    # Use existing fold/unfold functionality
    xx_interaction.fold()    # Convert to tensor form
    xx_interaction.unfold()  # Convert back to matrix form

Building Hamiltonians
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Build Heisenberg model terms
    sites = [Site(i, 2) for i in range(3)]  # 3 qubits
    spin_ops = SpinHilbertSpace(2).create_operators()
    
    hamiltonian_terms = []
    for i in range(len(sites) - 1):
        # Nearest-neighbor interactions
        two_site = spin_ops.kron(spin_ops)
        
        # Add Sx⊗Sx + Sy⊗Sy + Sz⊗Sz terms
        for pauli in ['Sx', 'Sy', 'Sz']:
            term_name = pauli + pauli  # e.g., 'SxSx'
            if term_name in two_site:
                interaction = LocalOperator(two_site[term_name], [sites[i], sites[i+1]])
                hamiltonian_terms.append(interaction)

Working with Different Operator Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Pauli operators for multiple qubits
    two_qubit_paulis = PauliHilbertSpace(4).create_operators()  # 16 operators
    print(f"Two-qubit Pauli strings: {len(two_qubit_paulis.operators)}")
    
    # Spin-1 operators  
    spin1_ops = SpinHilbertSpace(3).create_operators()
    print(f"Spin-1 operators: {list(spin1_ops.keys())}")  # Includes S+, S- ladder operators
    
    # Fermionic operators
    from meanfieldincrements import FermionHilbertSpace
    fermion_ops = FermionHilbertSpace().create_operators()
    print(f"Fermionic operators: {list(fermion_ops.keys())}")  # ['I', 'c', 'cdag', 'n']

Testing Operator Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Test commutation relations
    pauli_ops = PauliHilbertSpace(2).create_operators()
    
    # [X, Y] = 2iZ for Pauli matrices
    commutator = pauli_ops.get_commutator('X', 'Y')
    expected = 2j * pauli_ops['Z']
    print(f"[X,Y] = 2iZ: {np.allclose(commutator, expected)}")
    
    # Spin operators: [Sx, Sy] = i*Sz
    spin_ops = SpinHilbertSpace(2).create_operators()
    spin_comm = spin_ops.get_commutator('Sx', 'Sy')
    expected_spin = 1j * spin_ops['Sz']
    print(f"[Sx,Sy] = i*Sz: {np.allclose(spin_comm, expected_spin)}")