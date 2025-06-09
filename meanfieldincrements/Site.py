
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
    """
    def __init__(self, label:int, dimension:int):
        """
        Represents a lattice site.

        Args:
            label (int): The site index or label.
            dimension (int): The local Hilbert space dimension at this site.
        """
        self.label = int(label)
        self.dimension = int(dimension)

    def __repr__(self):
        return f"Site(label={self.label}, dimension={self.dimension})"