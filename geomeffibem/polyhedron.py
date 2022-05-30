"""A Polyhedron is a collection of Surface objects.

It's meant to represent a volume.
You can check whether it's enclosed or not.
"""
from __future__ import annotations

import copy
from typing import List, Tuple

import numpy as np

from geomeffibem.surface import Surface, Surface3dEge
from geomeffibem.vertex import Vertex


class Polyhedron:
    """A collection of Surfaces, meant to represent a Volume."""

    def __init__(self, surfaces: List[Surface]):
        """Constructor from a list of Surface objects."""
        if not isinstance(surfaces, np.ndarray) and not isinstance(surfaces, list):
            raise ValueError("Expected a list or numpy array of Surfaces")

        for i, surface in enumerate(surfaces):
            if not isinstance(surface, Surface):
                raise ValueError(f"Element {i} is not a Surface object")
        self.surfaces = surfaces

    def get_surface_by_name(self, name):
        """Locate a surface by its name."""
        for s in self.surfaces:
            if s.name is not None and s.name == name:
                return s

    def numVertices(self):
        """Counts the total number of vertices for all surfaces."""
        count = 0
        for s in self.surfaces:
            count += len(s.vertices)
        return count

    def uniqueVertices(self) -> List[Vertex]:
        """Get a list of unique vertices (uses Vertex __eq__ operator which has a tolerance)."""
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

    @staticmethod
    def edgesNotTwoForEnclosedVolumeTest(zonePoly: Polyhedron) -> Tuple[List[Surface3dEge], List[Surface3dEge]]:
        """Counts the number of times an Edge is used.

        Returns the ones that isn't used twice (and the ones used twice for debugging/inspection)
        """
        uniqueSurface3dEdges: List[Surface3dEge] = []

        for surface in zonePoly.surfaces:
            for edge in surface.to_Surface3dEdges():
                found = False
                for uniqEdge in uniqueSurface3dEdges:
                    if uniqEdge == edge:
                        uniqEdge.allSurfaces.append(surface)
                        found = True
                        break
                if not found:
                    uniqueSurface3dEdges.append(edge)

        edgesNotTwoCount = [x for x in uniqueSurface3dEdges if x.count() != 2]
        edgesTwoCount = [x for x in uniqueSurface3dEdges if x.count() == 2]
        return edgesNotTwoCount, edgesTwoCount

    def updateZonePolygonsForMissingColinearPoints(self) -> Polyhedron:
        """Creates a new Polyhedron with extra vertices when a point is found to be on a line segment."""
        updZonePoly = copy.deepcopy(self)

        uniqVertices = self.uniqueVertices()

        for surface in updZonePoly.surfaces:
            insertedVertex = True
            while insertedVertex:
                insertedVertex = False
                for i, edge in enumerate(surface.to_Surface3dEdges()):
                    for testVertex in uniqVertices:
                        if edge.containsPoints(testVertex):
                            if i == len(surface.vertices) - 1:
                                inext = 0
                            else:
                                inext = i + 1
                            surface.vertices.insert(inext, testVertex)
                            insertedVertex = True
                            break
                    # Break out of the loop on vertices/edges too, start again at while loop
                    if insertedVertex:
                        break
        return updZonePoly

    def isEnclosedVolume(self) -> Tuple[bool, List[Surface3dEge]]:
        """Checks if the Polyhedron is enclosed, that is all its edges are used exactly twice."""
        edgeNot2orig, _ = Polyhedron.edgesNotTwoForEnclosedVolumeTest(zonePoly=self)
        if not edgeNot2orig:
            return True, []

        print("Updating Polyhedron with collinear vertices on lines")
        updatedZonePoly = self.updateZonePolygonsForMissingColinearPoints()
        edgeNot2again, _ = Polyhedron.edgesNotTwoForEnclosedVolumeTest(updatedZonePoly)
        if not edgeNot2again:
            return True, []

        return False, edgesInBoth(edgeNot2orig, edgeNot2again)

    def to_os_cpp_code(self):
        """For my own convenience when writting OpenStudio tests."""
        for i, sf in enumerate(self.surfaces):
            if sf.name is not None:
                name = sf.name
            else:
                name = f"Surface {i+1}"
            if name[1] == '-':
                cleaned_name = name[2:].lower().replace('-', '')
            else:
                cleaned_name = name.lower().replace('-', '')
            s = "{"
            if sf.name is not None:
                s += "\n "
            n_vertices = len(sf.vertices)
            imax = n_vertices - 1
            for i, v in enumerate(sf.vertices):
                if i > 0:
                    s += " "
                s += f"{{{v.x:+.1f}, {v.y:+.1f}, {v.z:+.1f}}}"
                if i < imax:
                    s += ",\n"
            s += "}"

            print(f'Surface {cleaned_name}({s}, m);')
            print(f'{cleaned_name}.setName("{name}");')
            print(f'{cleaned_name}.setSpace(s);\n')

    def to_eplus_cpp_code(self):
        """For my own convenience when writting EnergyPlus tests."""
        n_surfaces = len(self.surfaces)
        print(
            f"""
            Array1D_bool enteredCeilingHeight;
            state->dataGlobal->NumOfZones = 1;
            enteredCeilingHeight.dimension(state->dataGlobal->NumOfZones, false);
            state->dataHeatBal->Zone.allocate(state->dataGlobal->NumOfZones);
            state->dataHeatBal->Zone(1).HasFloor = true;
            state->dataHeatBal->Zone(1).HTSurfaceFirst = 1;
            state->dataHeatBal->Zone(1).AllSurfaceFirst = 1;
            state->dataHeatBal->Zone(1).AllSurfaceLast = {n_surfaces};

            state->dataSurface->Surface.allocate({n_surfaces});
            """
        )
        for i, sf in enumerate(self.surfaces):
            n_vertices = len(sf.vertices)
            name = sf.name
            if 'ROOF' in name:
                tilt = 0.0
                c = 'SurfaceClass::Roof'
            elif 'FLOOR' in name:
                tilt = 180.0
                c = 'SurfaceClass::Floor'
            else:
                tilt = 90.0
                c = 'SurfaceClass::Wall'

            print(
                f'''
            state->dataSurface->Surface({i+1}).Name = "{name}";
            state->dataSurface->Surface({i+1}).Sides = {n_vertices};
            state->dataSurface->Surface({i+1}).Vertex.dimension({n_vertices});
            state->dataSurface->Surface({i+1}).Class = {c};
            state->dataSurface->Surface({i+1}).Tilt = {tilt};'''
            )

            for j, v in enumerate(sf.vertices):
                print(f"    state->dataSurface->Surface({i+1}).Vertex(1) = Vector({v.x}, {v.y}, {v.z});")


def edgesInBoth(a: List[Surface3dEge], b: List[Surface3dEge]) -> List[Surface3dEge]:
    """Helper function."""
    in_both = []
    for edge_a in a:
        for edge_b in b:
            if edge_a == edge_b:
                in_both.append(edge_a)
                break
    return in_both
