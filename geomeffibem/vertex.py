from typing import Tuple

import numpy as np
import openstudio


def isAlmostEqual3dPt(v1, v2, tol=0.0127):
    """
    Checks if both vertices almost equal within tolerance
    """
    # 0.0127 m = 1.27 cm = 1/2 inch
    return not (abs((v1.to_numpy() - v2.to_numpy())) >= tol).any()


def distance(lhs, rhs):
    """
    Distance between two vertices
    """
    squared_dist = np.sum((lhs.to_numpy() - rhs.to_numpy()) ** 2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist


class Vertex:
    """
    Point3d and Vector3d
    """

    @staticmethod
    def from_numpy(arr):
        """
        Factory method to construct from a numpy array (or a list) of 3 coords
        """
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        if arr.shape != (3,):
            raise ValueError(f"Expected a numpy array with a dimension (3, ) (a Vector3d), got {arr.shape}")
        return Vertex(arr[0], arr[1], arr[2])

    @staticmethod
    def from_Point3d(pt: openstudio.Point3d):
        return Vertex(pt.x(), pt.y(), pt.z())

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.surface = None

    def get_coords_on_plane(self, plane='xy') -> Tuple[float, float]:
        if plane == 'xy':
            return self.x, self.y
        elif plane == 'xz':
            return self.x, self.z
        elif plane == 'yz':
            return self.y, self.z

        raise ValueError("Expected plane to be 'xy', 'xz' or 'yz'.")

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def to_Point3d(self) -> openstudio.Point3d:
        return openstudio.Point3d(self.x, self.y, self.z)

    def __eq__(self, other):
        return isAlmostEqual3dPt(self, other)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        # return f"({self.x}, {self.y}, {self.z})"
        return f"({self.x:+.4f}, {self.y:+.4f}, {self.z:+.4f})"

    # def __str__(self):
    #     return f"Vertex {self.__repr__()}"
