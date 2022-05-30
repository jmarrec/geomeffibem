"""Plane."""

import numpy as np

from geomeffibem.vertex import Vertex, distance


class Plane:
    """A 3D Plane."""

    def __init__(self, a, b, c, d):
        """Plane constructor."""
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def outwardNormal(self) -> np.ndarray:
        """The outwardNormal of the plane."""
        return np.array([self.a, self.b, self.c])

    def is_orthogonal(self) -> bool:
        """Checks if the plane is orthogonal."""
        return sum(abs(self.outwardNormal())) == 1.0

    def pointOnPlane(self, point: Vertex, tol=0.001) -> bool:
        """Checks whether the Vertex is on the Plane."""
        # project point to plane
        projected = self.project(point)

        return distance(point, projected) <= tol

    def project(self, point: Vertex) -> Vertex:
        """Project a point onto a Plane."""
        # http://www.9math.com/book/projection-point-plane
        u = point.x
        v = point.y
        w = point.z

        num = self.a * u + self.b * v + self.c * w + self.d
        den = self.a * self.a + self.b * self.b + self.c * self.c  # this should always be 1.0
        ratio = num / den

        x = u - self.a * ratio
        y = v - self.b * ratio
        z = w - self.c * ratio

        return Vertex(x, y, z)

    def __repr__(self):
        """Repr."""
        return f"Plane ({self.a}, {self.b}, {self.c}, {self.d})"
