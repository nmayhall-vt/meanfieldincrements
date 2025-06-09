
class Site:
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