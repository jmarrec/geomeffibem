"""Vertex and Vertex related functions.

A Vertex is a 3-coordinate class, that can be used to represent a Point3d or a Vector3d
Includes utilities to go from/to numpy array and OpenStudio's Point3d
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import openstudio


class Vertex:
    """Point3d and Vector3d."""

    @staticmethod
    def from_numpy(arr):
        """Factory method to construct from a numpy array (or a list) of 3 coords."""
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        if arr.shape != (3,):
            raise ValueError(f"Expected a numpy array with a dimension (3, ) (a Vector3d), got {arr.shape}")
        return Vertex(arr[0], arr[1], arr[2])

    @staticmethod
    def from_Point3d(pt: openstudio.Point3d):
        """Factory method to construct from an openstudio.Point3d."""
        return Vertex(pt.x(), pt.y(), pt.z())

    def __init__(self, x, y, z):
        """Vertex constructor."""
        self.x = x
        self.y = y
        self.z = z
        self.surface = None

    def get_coords_on_plane(self, plane='xy') -> Tuple[float, float]:
        """Returns two coordinates on a given plane."""
        if plane == 'xy':
            return self.x, self.y
        elif plane == 'xz':
            return self.x, self.z
        elif plane == 'yz':
            return self.y, self.z

        raise ValueError("Expected plane to be 'xy', 'xz' or 'yz'.")

    def to_numpy(self) -> np.ndarray:
        """Export to a numpy array of 3 coordinates."""
        return np.array([self.x, self.y, self.z])

    def to_Point3d(self) -> openstudio.Point3d:
        """Export to an openstudio.Point3d."""
        return openstudio.Point3d(self.x, self.y, self.z)

    def __eq__(self, other):
        """Operator equal. Raises if not passed a Vertex."""
        if not isinstance(other, Vertex):
            raise NotImplementedError("Not implemented for any other types than Vertex itself")
        return isAlmostEqual3dPt(self, other)

    def __ne__(self, other):
        """Operator not equal."""
        return not self == other

    def __repr__(self):
        """Repr."""
        # return f"({self.x}, {self.y}, {self.z})"
        return f"({self.x:+.4f}, {self.y:+.4f}, {self.z:+.4f})"

    # def __str__(self):
    #     return f"Vertex {self.__repr__()}"


def isAlmostEqual3dPt(v1: Vertex, v2: Vertex, tol=0.0127) -> bool:
    """Checks if both vertices almost equal within tolerance."""
    # 0.0127 m = 1.27 cm = 1/2 inch
    return not (abs((v1.to_numpy() - v2.to_numpy())) >= tol).any()


def distance(lhs: Vertex, rhs: Vertex) -> float:
    """Distance between two vertices."""
    squared_dist = np.sum((lhs.to_numpy() - rhs.to_numpy()) ** 2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist


def distanceFromPointToLine(start: Vertex, end: Vertex, test: Vertex) -> np.floating[Any]:
    """Distance between a point and a line."""
    s = start.to_numpy()
    e = end.to_numpy()
    p = test.to_numpy()
    return np.linalg.norm(np.cross(e - s, p - s) / np.linalg.norm(e - s))


def isPointOnLineBetweenPoints(start: Vertex, end: Vertex, test: Vertex, tol: float = 0.0127) -> bool:
    """Checks whether a Vertex is on a segment.

    If the distance(start, test) + distance(test, end) == distance(start, end) then it's on a line
    But we first check that the distance from the point to the line is also inferior to this tolerance
    """
    if distanceFromPointToLine(start=start, end=end, test=test) < tol:
        return abs(distance(start, end) - (distance(start, test) + distance(test, end))) < tol
    return False
