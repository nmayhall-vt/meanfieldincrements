from typing import Union

class Site:
    """
    A class representing a lattice site in a physical system.

    Attributes:
        label (int): The site index or label, uniquely identifying the site.
        dimension (int): The local Hilbert space dimension at this site, 
            representing the number of possible states.

    Methods:
        __init__(label: int, dimension: int):
            Initializes a Site object with a given label and dimension.
        __repr__():
            Returns a string representation of the Site object.
        __eq__(other):
            Check equality based on label and dimension.
        __hash__():
            Hash function for use in sets and dictionaries.
        __lt__(other):
            Less-than comparison for sorting (based on label).
    
    Examples:
        >>> site1 = Site(0, 2)  # Qubit at site 0
        >>> site2 = Site(1, 3)  # Qutrit at site 1
        >>> sites = [site2, site1]
        >>> sorted(sites)  # Will sort by label: [site1, site2]
        >>> {site1, site2}  # Can be used in sets due to __hash__
    """
    
    def __init__(self, label: int, dimension: int):
        """
        Initialize a lattice site.

        Args:
            label (int): The site index or label, uniquely identifying the site.
            dimension (int): The local Hilbert space dimension at this site.
                Must be a positive integer representing the number of possible 
                quantum states at this site.

        Raises:
            ValueError: If dimension is not a positive integer.
            TypeError: If label or dimension cannot be converted to int.
            
        Examples:
            >>> site = Site(0, 2)      # Qubit (2-level system)
            >>> site = Site(1, 3)      # Qutrit (3-level system) 
            >>> site = Site("2", "4")  # String inputs auto-converted
        """
        try:
            self.label = int(label)
            self.dimension = int(dimension)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Label and dimension must be convertible to int: {e}")
        
        if self.dimension <= 0:
            raise ValueError("Dimension must be a positive integer.")

    def __repr__(self) -> str:
        """
        Return a string representation of the Site object.
        
        Returns:
            str: A string in the format "Site(label=X, dimension=Y)"
        """
        return f"Site(label={self.label}, dimension={self.dimension})"
    
    def __str__(self) -> str:
        """
        Return a user-friendly string representation.
        
        Returns:
            str: A readable string representation.
        """
        return f"Site {self.label} (dim={self.dimension})"
    
    def __eq__(self, other) -> bool:
        """
        Check equality between two Site objects.
        
        Two sites are considered equal if they have the same label and dimension.
        This is important for quantum systems where sites must match exactly.
        
        Args:
            other: Another object to compare with.
            
        Returns:
            bool: True if sites have same label and dimension, False otherwise.
            
        Examples:
            >>> site1 = Site(0, 2)
            >>> site2 = Site(0, 2)
            >>> site3 = Site(0, 3)
            >>> site1 == site2  # True
            >>> site1 == site3  # False (different dimensions)
        """
        if not isinstance(other, Site):
            return NotImplemented
        return self.label == other.label and self.dimension == other.dimension
    
    def __hash__(self) -> int:
        """
        Hash function for Site objects.
        
        Enables Site objects to be used as dictionary keys and in sets.
        Hash is based on both label and dimension for consistency with __eq__.
        
        Returns:
            int: Hash value based on label and dimension.
            
        Examples:
            >>> sites_dict = {Site(0, 2): "qubit", Site(1, 3): "qutrit"}
            >>> unique_sites = {Site(0, 2), Site(0, 2), Site(1, 2)}  # Set of 2 sites
        """
        return hash((self.label, self.dimension))
    
    def __lt__(self, other) -> bool:
        """
        Less-than comparison for sorting Site objects.
        
        Sites are ordered primarily by label, then by dimension if labels are equal.
        This enables natural sorting of site collections.
        
        Args:
            other (Site): Another Site object to compare with.
            
        Returns:
            bool: True if this site should come before other in sorted order.
            
        Raises:
            TypeError: If other is not a Site object.
            
        Examples:
            >>> sites = [Site(2, 2), Site(0, 3), Site(1, 2)]
            >>> sorted(sites)  # [Site(0,3), Site(1,2), Site(2,2)]
            >>> min(sites)     # Site(0,3)
        """
        if not isinstance(other, Site):
            return NotImplemented
        return (self.label, self.dimension) < (other.label, other.dimension)
    
    def __le__(self, other) -> bool:
        """Less-than-or-equal comparison."""
        if not isinstance(other, Site):
            return NotImplemented
        return (self.label, self.dimension) <= (other.label, other.dimension)
    
    def __gt__(self, other) -> bool:
        """Greater-than comparison."""
        if not isinstance(other, Site):
            return NotImplemented
        return (self.label, self.dimension) > (other.label, other.dimension)
    
    def __ge__(self, other) -> bool:
        """Greater-than-or-equal comparison."""
        if not isinstance(other, Site):
            return NotImplemented
        return (self.label, self.dimension) >= (other.label, other.dimension)
    
    def is_qubit(self) -> bool:
        """
        Check if this site represents a qubit (2-level system).
        
        Returns:
            bool: True if dimension is 2, False otherwise.
            
        Examples:
            >>> Site(0, 2).is_qubit()  # True
            >>> Site(1, 3).is_qubit()  # False
        """
        return self.dimension == 2
    
    def is_qutrit(self) -> bool:
        """
        Check if this site represents a qutrit (3-level system).
        
        Returns:
            bool: True if dimension is 3, False otherwise.
        """
        return self.dimension == 3
    
    def compatible_with(self, other: 'Site') -> bool:
        """
        Check if this site is compatible with another site for operations.
        
        Sites are compatible if they have the same dimension, regardless of label.
        This is useful for checking if operations can be performed between sites.
        
        Args:
            other (Site): Another Site object to check compatibility with.
            
        Returns:
            bool: True if sites have the same dimension.
            
        Examples:
            >>> site1 = Site(0, 2)
            >>> site2 = Site(5, 2)  # Different label, same dimension
            >>> site3 = Site(0, 3)  # Same label, different dimension
            >>> site1.compatible_with(site2)  # True
            >>> site1.compatible_with(site3)  # False
        """
        if not isinstance(other, Site):
            raise TypeError("Can only check compatibility with another Site")
        return self.dimension == other.dimension
