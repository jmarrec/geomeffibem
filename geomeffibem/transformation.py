"""Transformations: Rotation, Translations, and combinations of both."""

from __future__ import annotations

from typing import Optional

import numpy as np

from geomeffibem.plane import Plane
from geomeffibem.surface import Surface
from geomeffibem.vertex import Vertex


class Transformation:
    """Transformation class."""

    @staticmethod
    def Rotation(axis: Vertex, radians: float, point: Optional[Vertex] = None) -> Transformation:
        """Constructs a Rotation Transformation (factory method)."""
        temp = axis.normalize()
        normalVector = temp.to_numpy()
        P = np.outer(normalVector, normalVector)

        # Rodrigues' rotation formula / Rotation matrix from Euler axis/angle
        # I*cos(radians) + I*(1-cos(radians))*axis*axis^T + Q*sin(radians)
        # Q = [0, -axis[2], axis[1]; axis[2], 0, -axis[0]; -axis[1], axis[0], 0]
        Q = np.zeros(shape=(3, 3))
        Q[0, 1] = -normalVector[2]
        Q[0, 2] = normalVector[1]
        Q[1, 0] = normalVector[2]
        Q[1, 2] = -normalVector[0]
        Q[2, 0] = -normalVector[1]
        Q[2, 1] = normalVector[0]

        c = np.cos(radians)
        identity_3 = np.identity(3)
        R = identity_3 * c + (1.0 - c) * P + Q * np.sin(radians)
        result = np.identity(4)
        result[:-1, :-1] = R
        if point is not None:
            #  translate point to origin, rotate, and then translate back
            # t_ori = Transformation.Translation(point)
            # t_rot = Transformation(result)
            # t_reverse = Transformation.Translation(-point)
            # return (t_ori, t_rot, t_reverse)
            return Transformation.Translation(point) * Transformation(result) * Transformation.Translation(-point)

        return Transformation(result)

    @staticmethod
    def Translation(translation: Vertex) -> Transformation:
        """Constructs a Translation Transformation (factory method)."""
        result = np.identity(4)
        result[0, 3] = translation.x
        result[1, 3] = translation.y
        result[2, 3] = translation.z
        return Transformation(result)

    # @staticmethod
    # def RotationAroundPoint(point: Vertex, axis: Vertex, radians) -> Transformation

    def __init__(self, matrix: np.ndarray = None):
        """Constructor for Transformation."""
        if matrix is None:
            matrix = np.identity(4)
        elif matrix.shape != (4, 4):
            raise ValueError(f"Expected a matrix of dimension (4, 4), got {matrix.shape}")
        self.matrix = matrix

    def rotationMatrix(self) -> np.ndarray:
        """Returns the rotation portion of the Transformation."""
        return self.matrix[:-1, :-1]

    def translation(self) -> Vertex:
        """Returns the translation portion of the Transformation."""
        return Vertex(self.matrix[0, 3], self.matrix[1, 3], self.matrix[2, 3])

    def inverse(self) -> Transformation:
        """Returns a transformation which is the inverse of this"""
        t = Transformation()
        t.matrix = np.linalg.inv(self.matrix)
        return t

    def __mul__(self, other):  # -> Union[Vertex, Transformation, np.ndarray, Surface, Plane]:
        """Multiplies self by other.

        Accepts various objects: Vertex, Transformation, List of Vertex, Surface, Plane.
        """
        if isinstance(other, Vertex):
            temp = np.matmul(self.matrix, np.append(other.to_numpy(), 1.0))
            return Vertex(temp[0], temp[1], temp[2])
        elif isinstance(other, Transformation):
            return Transformation(np.matmul(self.matrix, other.matrix))
        elif isinstance(other, np.ndarray) or isinstance(other, list):
            return np.array([self * v for v in other])
        elif isinstance(other, Surface):
            name = "Rotated unnamed"
            if other.name is not None:
                name = f"Rotated {name}"

            return Surface(vertices=[self * v for v in other.vertices], name=name)
        elif isinstance(other, Plane):
            # translate a point on the plane, just project (0,0,0)
            point = other.project(Vertex(0.0, 0.0, 0.0))

            # Get a point at outward normal
            outwardNormal = other.outwardNormal()
            refPoint = point + outwardNormal

            # Translate the two points and recompute the normal
            newPoint = self * point
            newRefPoint = self * refPoint
            newNormal = newRefPoint - newPoint

            d = (-newNormal).dot(newPoint)

            p = Plane(newNormal.x, newNormal.y, newNormal.z, d)

            return p
        else:
            raise ValueError(f"Not implemented for type {type(other)}")

    def __repr__(self):
        """Repr."""
        return f"Transformation:\n{self.matrix.__str__()}"
