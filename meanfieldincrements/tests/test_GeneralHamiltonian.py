import numpy as np
from meanfieldincrements import Site
# Assuming the new classes are imported
from meanfieldincrements import (GeneralHamiltonian, PauliHamiltonian, 
                                create_identity_operators, create_spin_operators, 
                                create_bosonic_operators, transverse_field_ising_model_general,
                                heisenberg_model_general)

def test_pauli_hamiltonian():
    """Test the PauliHamiltonian class (qubit systems)."""
    print("=== PauliHamiltonian (Qubit Systems) ===")
    
    # Create qubit sites
    sites = [Site(0, 2), Site(1, 2), Site(2, 2)]
    
    # Create Pauli Hamiltonian
    H = PauliHamiltonian(sites)
    H.add_pauli_string("XII", 1.0)
    H.add_pauli_string("ZZI", -0.5)
    H.add_pauli_string("IYY", 0.3)
    
    print(f"Pauli Hamiltonian: {H}")
    print(f"Number of terms: {len(H)}")
    print(f"Matrix shape: {H.matrix().shape}")
    
    # Test using the tuple interface
    H.add_term(("X", "Y", "Z"), 0.2)
    print(f"After adding XYZ term: {H}")
    
    # Create from dictionary
    pauli_dict = {"XII": 1.0, "ZZI": -0.5, "YYI": 0.3}
    H2 = PauliHamiltonian.from_pauli_strings(sites, pauli_dict)
    print(f"From dictionary: {H2}")
    print()

def test_mixed_dimension_system():
    """Test system with mixed dimensions (qubit + qutrit)."""
    print("=== Mixed Dimension System (Qubit + Qutrit) ===")
    
    # Create sites with different dimensions
    sites = [Site(0, 2), Site(1, 3)]  # qubit + qutrit
    
    # Define operators
    # 2x2 Pauli operators for qubit
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity_2 = np.eye(2, dtype=complex)
    
    # 3x3 operators for qutrit
    identity_3 = np.eye(3, dtype=complex)
    # Qutrit Pauli-X analog (cyclic permutation)
    qutrit_x = np.array([[0, 1, 0],
                         [0, 0, 1], 
                         [1, 0, 0]], dtype=complex)
    # Qutrit Pauli-Z analog
    omega = np.exp(2j * np.pi / 3)
    qutrit_z = np.array([[1, 0, 0],
                         [0, omega, 0],
                         [0, 0, omega**2]], dtype=complex)
    
    # Create operator library
    operators = {
        'I2': identity_2,    # Identity for qubit
        'I3': identity_3,    # Identity for qutrit
        'X2': pauli_x,       # Pauli-X for qubit
        'Z2': pauli_z,       # Pauli-Z for qubit
        'X3': qutrit_x,      # X-analog for qutrit
        'Z3': qutrit_z       # Z-analog for qutrit
    }
    
    # Create Hamiltonian
    H = GeneralHamiltonian(sites, operators)
    H.add_term(("X2", "I3"), 1.0)     # X on qubit, I on qutrit
    H.add_term(("Z2", "Z3"), -0.5)    # Z-Z interaction
    H.add_term(("I2", "X3"), 0.3)     # I on qubit, X on qutrit
    
    print(f"Mixed dimension Hamiltonian: {H}")
    print(f"Matrix shape: {H.matrix().shape}")  # Should be 6x6
    print(f"Is Hermitian: {H.is_hermitian()}")
    print()

def test_spin_system():
    """Test spin-1 system using spin operators."""
    print("=== Spin-1 System ===")
    
    # Create spin-1 sites (dimension 3)
    sites = [Site(0, 3), Site(1, 3)]
    
    # Create spin-1 operators
    spin_ops = create_spin_operators(spin=1)
    print("Available spin operators:", list(spin_ops.keys()))
    
    # Create Hamiltonian
    H = GeneralHamiltonian(sites, spin_ops)
    H.add_term(("S_x", "S_x"), 1.0)      # S_x S_x interaction
    H.add_term(("S_y", "S_y"), 1.0)      # S_y S_y interaction  
    H.add_term(("S_z", "S_z"), 1.0)      # S_z S_z interaction
    H.add_term(("S_z", "I"), 0.1)        # Magnetic field on first spin
    
    print(f"Spin-1 Hamiltonian: {H}")
    print(f"Matrix shape: {H.matrix().shape}")  # Should be 9x9
    
    # Check some properties
    print(f"S_x operator:\n{spin_ops['S_x']}")
    print(f"S_z operator:\n{spin_ops['S_z']}")
    print()

def test_bosonic_system():
    """Test bosonic system with creation/annihilation operators."""
    print("=== Bosonic System ===")
    
    # Create bosonic sites (truncated to max 3 particles)
    sites = [Site(0, 4), Site(1, 4)]  # 4-dimensional: |0⟩, |1⟩, |2⟩, |3⟩
    
    # Create bosonic operators
    bosonic_ops = create_bosonic_operators(max_occupation=3)
    print("Available bosonic operators:", list(bosonic_ops.keys()))
    
    # Create Hamiltonian: H = ω(n₀ + n₁) + g(a₀†a₁ + a₁†a₀)
    H = GeneralHamiltonian(sites, bosonic_ops)
    H.add_term(("n", "I"), 1.0)          # ω n₀
    H.add_term(("I", "n"), 1.0)          # ω n₁
    H.add_term(("a_dag", "a"), 0.1)      # g a₀†a₁
    H.add_term(("a", "a_dag"), 0.1)      # g a₁†a₀
    
    print(f"Bosonic Hamiltonian: {H}")
    print(f"Matrix shape: {H.matrix().shape}")  # Should be 16x16
    
    # Show some operator matrices
    print(f"Creation operator a†:\n{bosonic_ops['a_dag']}")
    print(f"Number operator n:\n{bosonic_ops['n']}")
    print()

def test_hybrid_system():
    """Test a hybrid system with different types of sites."""
    print("=== Hybrid System (Qubit + Spin-1/2 + Bosonic) ===")
    
    # Define sites
    sites = [
        Site(0, 2),  # Qubit
        Site(1, 2),  # Spin-1/2 
        Site(2, 3)   # Bosonic (3 levels)
    ]
    
    # Combine operator libraries
    pauli_ops = PauliHamiltonian.PAULI_OPERATORS
    spin_ops = create_spin_operators(spin=0.5)
    bosonic_ops = create_bosonic_operators(max_occupation=2)
    
    # Create combined operator library with prefixes
    operators = {}
    # Pauli operators for site 0
    operators.update({f"P_{k}": v for k, v in pauli_ops.items()})
    # Spin operators for site 1  
    operators.update({f"S_{k}": v for k, v in spin_ops.items()})
    # Bosonic operators for site 2
    operators.update({f"B_{k}": v for k, v in bosonic_ops.items()})
    
    print("Available operators:", list(operators.keys()))
    
    # Create Hamiltonian
    H = GeneralHamiltonian(sites, operators)
    H.add_term(("P_X", "S_I", "B_I"), 1.0)           # Pauli-X on qubit
    H.add_term(("P_Z", "S_S_z", "B_I"), 0.5)         # Qubit-spin interaction
    H.add_term(("P_I", "S_I", "B_n"), 0.1)           # Bosonic number
    H.add_term(("P_I", "S_S_x", "B_a_dag"), 0.05)    # Spin-boson coupling
    
    print(f"Hybrid Hamiltonian: {H}")
    print(f"Matrix shape: {H.matrix().shape}")  # Should be 12x12
    print()

def test_hamiltonian_operations():
    """Test arithmetic operations on general Hamiltonians."""
    print("=== Hamiltonian Operations ===")
    
    sites = [Site(0, 2), Site(1, 2)]
    
    # Create two Hamiltonians
    H1 = PauliHamiltonian(sites)
    H1.add_pauli_string("XI", 1.0)
    H1.add_pauli_string("ZZ", 0.5)
    
    H2 = PauliHamiltonian(sites)
    H2.add_pauli_string("XI", 0.5)  # Same term
    H2.add_pauli_string("YY", -1.0)  # New term
    
    print(f"H1: {H1}")
    print(f"H2: {H2}")
    
    # Addition
    H_sum = H1 + H2
    print(f"H1 + H2: {H_sum}")
    
    # Scalar multiplication
    H_scaled = 2.0 * H1
    print(f"2.0 * H1: {H_scaled}")
    
    # Test matrix operations
    matrix1 = H1.matrix()
    matrix_sum = H_sum.matrix()
    expected_sum = matrix1 + H2.matrix()
    
    print(f"Matrix addition correct: {np.allclose(matrix_sum, expected_sum)}")
    print()

def test_model_hamiltonians():
    """Test predefined model Hamiltonians."""
    print("=== Predefined Model Hamiltonians ===")
    
    sites = [Site(0, 2), Site(1, 2), Site(2, 2)]
    
    # TFIM using general framework
    tfim = transverse_field_ising_model_general(sites, J=1.0, h=0.5)
    print(f"TFIM: {tfim}")
    
    # Heisenberg model using general framework
    heisenberg = heisenberg_model_general(sites, Jx=1.0, Jy=1.0, Jz=1.0)
    print(f"Heisenberg: {heisenberg}")
    
    # Test that they're Hermitian
    print(f"TFIM is Hermitian: {tfim.is_hermitian()}")
    print(f"Heisenberg is Hermitian: {heisenberg.is_hermitian()}")
    print()

def test_conversion_to_local_operator():
    """Test converting Hamiltonian to LocalOperator."""
    print("=== Conversion to LocalOperator ===")
    
    sites = [Site(0, 2), Site(1, 2)]
    
    # Create simple Hamiltonian
    H = PauliHamiltonian(sites)
    H.add_pauli_string("XI", 1.0)
    H.add_pauli_string("ZZ", -0.5)
    
    # Convert to LocalOperator
    local_op = H.to_local_operator()
    print(f"Hamiltonian: {H}")
    print(f"LocalOperator: {local_op}")
    print(f"LocalOperator trace: {local_op.trace()}")
    
    # Verify matrices are the same
    h_matrix = H.matrix()
    lo_matrix = local_op.tensor
    print(f"Matrices match: {np.allclose(h_matrix, lo_matrix)}")
    print()

def test_advanced_features():
    """Test advanced features like conjugation, simplification, etc."""
    print("=== Advanced Features ===")
    
    sites = [Site(0, 2), Site(1, 2)]
    H = PauliHamiltonian(sites)
    
    # Add complex coefficients
    H.add_pauli_string("XI", 1.0 + 0.5j)
    H.add_pauli_string("ZZ", -0.5)
    H.add_pauli_string("YY", 1e-16)  # Very small coefficient
    
    print(f"Original: {H}")
    print(f"Is Hermitian: {H.is_hermitian()}")
    
    # Conjugate
    H_conj = H.conjugate()
    print(f"Conjugate: {H_conj}")
    
    # Simplify (remove small terms)
    H_simple = H.simplify(tolerance=1e-15)
    print(f"Simplified: {H_simple}")
    
    # Copy
    H_copy = H.copy()
    print(f"Copy: {H_copy}")
    print(f"Copy is independent: {H_copy is not H}")
    print()

def run_all_tests():
    """Run all tests for the general Hamiltonian framework."""
    print("Testing General Hamiltonian Framework")
    print("=" * 60)
    
    test_pauli_hamiltonian()
    test_mixed_dimension_system()
    test_spin_system()
    test_bosonic_system()
    test_hybrid_system()
    test_hamiltonian_operations()
    test_model_hamiltonians()
    test_conversion_to_local_operator()
    test_advanced_features()
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    run_all_tests()