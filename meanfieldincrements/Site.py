"""
Site class that uses HilbertSpace instead of just dimension.
Clean version without artificial "compatibility" restrictions.
"""

from typing import Union, Optional
from .HilbertSpace import HilbertSpace, PauliHilbertSpace, SpinHilbertSpace, FermionHilbertSpace


class Site:
    """
    Represents a site in a many-body quantum system.
    
    Each site has an associated HilbertSpace that defines the type of
    quantum system at that location (e.g., qubit, spin-1, fermion, etc.).
    
    Attributes:
        label (int): Unique identifier for this site
        hilbert_space (HilbertSpace): The Hilbert space for this site
    """
    
    def __init__(self, label: int, hilbert_space_or_dimension: Union[HilbertSpace, int]):
        """
        Initialize a Site.
        
        Args:
            label (int): Unique identifier for this site
            hilbert_space_or_dimension (HilbertSpace or int): Either a HilbertSpace 
                object or an integer dimension (for backward compatibility)
        
        Examples:
            # New way - specify the type of quantum system
            >>> Site(0, PauliHilbertSpace(2))       # Qubit site
            >>> Site(1, SpinHilbertSpace(3))        # Spin-1 site  
            >>> Site(2, FermionHilbertSpace())      # Fermionic site
            
            # Old way - just dimension (creates generic HilbertSpace)
            >>> Site(0, 2)  # Generic 2-level system
        """
        self.label = label
        
        if isinstance(hilbert_space_or_dimension, HilbertSpace):
            self.hilbert_space = hilbert_space_or_dimension
        elif isinstance(hilbert_space_or_dimension, int):
            # Backward compatibility - create a generic HilbertSpace
            self.hilbert_space = HilbertSpace(hilbert_space_or_dimension)
        else:
            raise TypeError("Expected HilbertSpace or int, got "
                          f"{type(hilbert_space_or_dimension)}")
    
    @property
    def dimension(self) -> int:
        """Get the dimension of the Hilbert space at this site."""
        return self.hilbert_space.dimension
    
    @property
    def name(self) -> str:
        """Get the name of the Hilbert space at this site."""
        return self.hilbert_space.name
    
    def is_qubit(self) -> bool:
        """Check if this site represents a qubit (2-level Pauli system)."""
        return (isinstance(self.hilbert_space, PauliHilbertSpace) and 
                self.dimension == 2)
    
    def is_spin(self) -> bool:
        """Check if this site represents a spin system."""
        return isinstance(self.hilbert_space, SpinHilbertSpace)
    
    def is_fermion(self) -> bool:
        """Check if this site represents a fermionic system."""
        return isinstance(self.hilbert_space, FermionHilbertSpace)
    
    def is_generic(self) -> bool:
        """Check if this site uses a generic HilbertSpace."""
        return (type(self.hilbert_space) == HilbertSpace and 
                not isinstance(self.hilbert_space, (PauliHilbertSpace, SpinHilbertSpace, FermionHilbertSpace)))
    
    def get_spin_value(self) -> Optional[float]:
        """
        Get the spin value if this is a spin site.
        
        Returns:
            float or None: The spin value (e.g., 0.5, 1.0) or None if not a spin site
        """
        if self.is_spin():
            return self.hilbert_space.j
        return None
    
    def get_pauli_qubits(self) -> Optional[int]:
        """
        Get the number of qubits if this is a Pauli site.
        
        Returns:
            int or None: Number of qubits or None if not a Pauli site
        """
        if isinstance(self.hilbert_space, PauliHilbertSpace):
            return self.hilbert_space.n_qubits
        return None
    
    def __repr__(self) -> str:
        return f"Site(id={self.label}, {self.hilbert_space})"
    
    def __str__(self) -> str:
        return f"Site {self.label}: {self.name} (dim={self.dimension})"
    
    def __eq__(self, other) -> bool:
        """Two sites are equal if they have the same label and HilbertSpace."""
        if not isinstance(other, Site):
            return False
        return (self.label == other.label and 
                self.dimension == other.dimension and
                type(self.hilbert_space) == type(other.hilbert_space))
    
    def __hash__(self) -> int:
        """Hash based on label and dimension for use in sets/dicts."""
        return hash((self.label, self.dimension))


# Convenience factory functions for common site types
def qubit_site(label: int) -> Site:
    """Create a qubit (Pauli) site."""
    return Site(label, PauliHilbertSpace(2))


def spin_site(label: int, spin_value: float) -> Site:
    """Create a spin site with given spin value."""
    dimension = int(2 * spin_value + 1)
    return Site(label, SpinHilbertSpace(dimension))


def fermion_site(label: int) -> Site:
    """Create a fermionic site."""
    return Site(label, FermionHilbertSpace())


def multi_qubit_site(label: int, n_qubits: int) -> Site:
    """Create a site representing n qubits."""
    dimension = 2 ** n_qubits
    return Site(label, PauliHilbertSpace(dimension))


# Backward compatibility function
def create_sites(n_sites: int, dimension: int) -> list:
    """
    Create a list of sites with the same dimension.
    
    Args:
        n_sites (int): Number of sites to create
        dimension (int): Dimension for each site
        
    Returns:
        list[Site]: List of sites with consecutive labels starting from 0
    """
    return [Site(i, dimension) for i in range(n_sites)]


def create_qubit_chain(n_qubits: int) -> list:
    """
    Create a chain of qubit sites.
    
    Args:
        n_qubits (int): Number of qubits
        
    Returns:
        list[Site]: List of qubit sites with consecutive labels
    """
    return [qubit_site(i) for i in range(n_qubits)]


def create_spin_chain(n_sites: int, spin_value: float) -> list:
    """
    Create a chain of spin sites.
    
    Args:
        n_sites (int): Number of sites
        spin_value (float): Spin value (e.g., 0.5, 1.0, 1.5)
        
    Returns:
        list[Site]: List of spin sites with consecutive labels
    """
    return [spin_site(i, spin_value) for i in range(n_sites)]


def create_fermion_chain(n_sites: int) -> list:
    """
    Create a chain of fermionic sites.
    
    Args:
        n_sites (int): Number of sites
        
    Returns:
        list[Site]: List of fermionic sites with consecutive labels
    """
    return [fermion_site(i) for i in range(n_sites)]


if __name__ == "__main__":
    # Example usage and testing
    print("=== Site Class Examples ===")
    
    # 1. Create different types of sites
    print("\n1. Creating different site types:")
    
    # New way - specific HilbertSpaces
    qubit = Site(0, PauliHilbertSpace(2))
    spin1 = Site(1, SpinHilbertSpace(3))
    fermion = Site(2, FermionHilbertSpace())
    
    print(f"Qubit site: {qubit}")
    print(f"Spin-1 site: {spin1}")
    print(f"Fermion site: {fermion}")
    
    # Old way - backward compatibility
    generic = Site(3, 4)  # Generic 4-level system
    print(f"Generic site: {generic}")
    
    # 2. Test site properties
    print("\n2. Site properties:")
    print(f"Qubit is_qubit: {qubit.is_qubit()}")
    print(f"Spin1 is_spin: {spin1.is_spin()}")
    print(f"Spin1 spin value: {spin1.get_spin_value()}")
    print(f"Fermion is_fermion: {fermion.is_fermion()}")
    print(f"Generic is_generic: {generic.is_generic()}")
    
    # # 3. Create operators for each site
    # print("\n3. Site operators:")
    # qubit_ops = SiteOperators(qubit.hilbert_space)
    # spin_ops = spin1.create_operators()
    # fermion_ops = fermion.create_operators()
    
    # print(f"Qubit operators: {list(qubit_ops.keys())}")
    # print(f"Spin-1 operators: {list(spin_ops.keys())}")
    # print(f"Fermion operators: {list(fermion_ops.keys())}")
    
    # # 4. Convenience factory functions
    # print("\n4. Convenience factories:")
    # qubit_chain = create_qubit_chain(3)
    # spin_chain = create_spin_chain(3, 0.5)
    # fermion_chain = create_fermion_chain(2)
    
    # print(f"Qubit chain: {[f'Site({s.label})' for s in qubit_chain]}")
    # print(f"Spin-1/2 chain: {[f'Site({s.label}, j={s.get_spin_value()})' for s in spin_chain]}")
    # print(f"Fermion chain: {[f'Site({s.label})' for s in fermion_chain]}")
    
    # print("\n✅ All examples completed successfully!")