"""Bounding Box."""

from typing import Tuple

import numpy as np

from geomeffibem.vertex import Vertex


class BoundingBox:
    """A crude BoundingBox.

    As you add points to it, the min/max x, y, z are updated.
    """

    def __init__(self):
        """Constructor for BoundingBox."""
        self.minX = None
        self.minY = None
        self.minZ = None
        self.maxX = None
        self.maxY = None
        self.maxZ = None

    def get_figsize(self, width=12) -> Tuple[float, float]:
        """Figure out a figure size (Broken)."""
        (x, y, z) = self.dimensions().to_numpy()
        return (width, width * x / y)

    def corners(self) -> np.ndarray:
        """Returns an  of all 8 corner Points."""
        # TODO: make it return Vertex?
        if self.isEmpty():
            return None
        return np.array(
            [
                [self.minX, self.minY, self.minZ],
                [self.maxX, self.minY, self.minZ],
                [self.minX, self.maxY, self.minZ],
                [self.maxX, self.maxY, self.minZ],
                [self.minX, self.minY, self.maxZ],
                [self.maxX, self.minY, self.maxZ],
                [self.minX, self.maxY, self.maxZ],
                [self.maxX, self.maxY, self.maxZ],
            ]
        )

    def isEmpty(self) -> bool:
        """Checks if the bounding box is initialized."""
        return self.minX is None

    def addPoint(self, vertex) -> None:
        """Adds a single point and updates the min/max x, y, z."""
        if self.isEmpty():
            self.minX = vertex.x
            self.minY = vertex.y
            self.minZ = vertex.z
            self.maxX = vertex.x
            self.maxY = vertex.y
            self.maxZ = vertex.z
        else:
            self.minX = min(self.minX, vertex.x)
            self.minY = min(self.minY, vertex.y)
            self.minZ = min(self.minZ, vertex.z)

            self.maxX = max(self.minX, vertex.x)
            self.maxY = max(self.maxY, vertex.y)
            self.maxZ = max(self.maxZ, vertex.z)

    def addPoints(self, vertices) -> None:
        """Adds multiple points and updates the min/max x, y, z."""
        [self.addPoint(v) for v in vertices]

    def dimensions(self) -> Vertex:
        """Returns the dimensions of the bounding box."""
        return Vertex(self.maxX - self.minX, self.maxY - self.minY, self.maxZ - self.minZ)
