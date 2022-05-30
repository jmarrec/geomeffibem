from typing import List

import numpy as np

from geomeffibem.surface import Surface
from geomeffibem.vertex import Vertex


class Polyhedron:
    def __init__(self, surfaces: List[Surface]):
        assert isinstance(surfaces, list)
        if not isinstance(surfaces, np.ndarray) and not isinstance(surfaces, list):
            raise ValueError("Expected a list or numpy array of Surfaces")

        for i, surface in enumerate(surfaces):
            if not isinstance(surface, Surface):
                raise ValueError(f"Element {i} is not a Surface object")
        self.surfaces = surfaces

    def get_surface_by_name(self, name):
        for s in self.surfaces:
            if s.name is not None and s.name == name:
                return s

    def count_vertices(self):
        count = 0
        for s in self.surfaces:
            count += len(s.vertices)
        return count

    def makeListOfUniqueVertices(self) -> List[Vertex]:
        uniqueVertices: List[Vertex] = []
        for s in self.surfaces:
            for vertex in s.vertices:
                found = False
                for unique_v in uniqueVertices:
                    if unique_v == vertex:
                        found = True
                        break
                if not found:
                    uniqueVertices.append(vertex)
        return uniqueVertices
