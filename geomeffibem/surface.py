"""Surface and Surface3dEdge objects.

A Surface is a collection of Vertex.
A Surface3dEdge is a side of a Surface.
"""

from __future__ import annotations

import copy
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import openstudio

from geomeffibem.plane import Plane
from geomeffibem.vertex import (
    Vertex,
    distance,
    getAngle,
    getNewellVector,
    getOutwardNormal,
    isAlmostEqual3dPt,
    isPointOnLineBetweenPoints,
)


class Surface3dEge:
    """An Edge has a start and an end Vertex, and a list of surfaces it was found on."""

    def __init__(self, start: Vertex, end: Vertex, firstSurface: Surface):
        """Constructor."""
        self.start = start
        self.end = end
        self.allSurfaces = [firstSurface]

    def containsPoints(self, testVertex: Vertex) -> bool:
        """Checks whether a Point is on the edge.

        It is not almost equal to the start and end points, and,
        isPointOnLineBetweenPoints(start, end, testVertex) is true.
        """
        return (
            not isAlmostEqual3dPt(self.start, testVertex)
            and not isAlmostEqual3dPt(self.end, testVertex)
            and isPointOnLineBetweenPoints(self.start, self.end, testVertex)
        )

    def length(self) -> float:
        """Compute distance from start to end."""
        return distance(self.start, self.end)

    def count(self) -> int:
        """Number of Surfaces it was found on."""
        return len(self.allSurfaces)

    def __eq__(self, other):
        """Operator equal."""
        if not isinstance(other, Surface3dEge):
            raise NotImplementedError("Not implemented for any other types than Surface3dEge itself")

        return (isAlmostEqual3dPt(self.start, other.start) and isAlmostEqual3dPt(self.end, other.end)) or (
            isAlmostEqual3dPt(self.start, other.end) and isAlmostEqual3dPt(self.end, other.start)
        )

    def __ne__(self, other):
        """Operator not equal."""
        return not self == other

    def __repr__(self):
        """Repr."""
        return f"start={self.start}, end={self.end}, count={self.count()}, firstSurface={self.allSurfaces[0].name}"

    def plot_on_first_surface(self, ax=None):
        """Plots this segment in red atop the outline of the Surface it came from."""
        surface = self.allSurfaces[0]

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 9))
        surface.plot(ax=ax)
        plot_vertices([self.start, self.end], plane=surface.get_plot_axis(), c='r', ax=ax, annotate=False)


class Surface:
    """A 3D Surface."""

    @staticmethod
    def from_numpy_array(arr) -> Surface:
        """Factory method to construct from a numpy array of 3-coordinates arrays."""
        if isinstance(arr, list):
            arr = np.array(arr)
        if arr.shape[0] < 3:
            raise ValueError("Need at least 3 vertices to construct a Surface")
        if arr.shape[1] != 3:
            raise ValueError(f"Expected a numpy array with a dimension (N, 3), got {arr.shape}")
        return Surface([Vertex.from_numpy(x) for x in arr])

    @staticmethod
    def from_Point3dVector(points: Union[openstudio.Point3dVector, List[openstudio.Point3d]]) -> Surface:
        """Factory method to construct from an openstudio Point3dVector or a list of Point3d."""
        return Surface(vertices=[Vertex.from_Point3d(x) for x in points])

    @staticmethod
    def from_Surface(openstudio_surface: openstudio.model.Surface) -> Surface:
        """Factory method to construct from an openstudio.model.Surface."""
        if not isinstance(openstudio_surface, openstudio.openstudiomodelgeometry.Surface):
            raise ValueError("Expected an openstudio.model.Surface")
        return Surface(
            vertices=[Vertex.from_Point3d(x) for x in openstudio_surface.vertices()],
            name=openstudio_surface.nameString(),
        )

    @staticmethod
    def Rectangle(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, min_z=0.0, max_z=0.0) -> Surface:
        """Factory method to easily create an rectangle Surface."""
        vertices_arr = np.array(
            [
                [min_x, min_y, max_z],
                [min_x, max_y, min_z],
                [max_x, max_y, min_z],
                [max_x, min_y, max_z],
            ]
        )
        vertices = [Vertex.from_numpy(x) for x in vertices_arr]

        return Surface(vertices=vertices)

    def __init__(self, vertices, name=None):
        """Surface constructor."""
        if not isinstance(vertices, np.ndarray) and not isinstance(vertices, list):
            raise ValueError("Expected a list or numpy array of Vertex")

        for i, vertex in enumerate(vertices):
            if not isinstance(vertex, Vertex):
                raise ValueError(f"Element {i} is not a Vertex object")

        self.name = name
        self.os_plane = None

        self.vertices = copy.deepcopy(vertices)
        for vertex in self.vertices:
            vertex.surface = self

    def get_plane(self) -> Plane:
        """Returns the Plane of the Surface."""
        if self.os_plane is not None:
            return self.os_plane
        plane = openstudio.Plane(self.to_Point3dVector())
        self.os_plane = Plane(plane.a(), plane.b(), plane.c(), plane.d())
        return self.os_plane

    def get_plot_axis(self) -> str:
        """Returns a string representation of the plane it is on.

        TODO: raises if not exactly on 'xy', 'xz' or 'yz'
        """
        plane = self.get_plane()
        tol = 0.001
        if abs(abs(plane.a) - 1) < tol:
            return 'yz'
        if abs(abs(plane.b) - 1) < tol:
            return 'xz'
        if abs(abs(plane.c) - 1) < tol:
            return 'xy'

        # TODO
        raise NotImplementedError("Surface is not on a standard plane!")

    def plane(self) -> Plane:
        """Compute the plane from outwardNormal and the first point, not using OpenStudio."""
        normalVector = self.outwardNormal()
        if not np.isclose(normalVector.length(), 1.0):
            raise ValueError("Normal Unit Vector doesn't appear to be a unit vector")
        self.vertices[0]

        # d = -thisNormal.x() * point.x() - thisNormal.y() * point.y() - thisNormal.z() * point.z();
        d = (-normalVector).dot(self.vertices[0])

        p = Plane(normalVector.x, normalVector.y, normalVector.z, d)
        for i, v in enumerate(self.vertices):
            if not p.pointOnPlane(v):
                print(f"Vertex {i} is not on the plane")
        return p

    def area(self) -> float:
        """Compute area of the surface."""
        newellVector = getNewellVector(self.vertices)
        return newellVector.length() / 2.0

    def outwardNormal(self) -> Vertex:
        """Returns the outward normal (normal unit vector)."""
        return getOutwardNormal(self.vertices)

    def tilt(self) -> float:
        """Returns the tilt of the surface, in radians, that is the angle between the outwardNormal and the Z axis."""
        z = Vertex(0.0, 0.0, 1.0)
        return getAngle(self.outwardNormal(), z)

    def azimuth(self) -> float:
        """Returns the azimuth of the surface, in radians.

        That is the angle between the outwardNormal and the North axis (Y-axis).
        """
        normal = self.outwardNormal()
        north = Vertex(0.0, 1.0, 0.0)
        angle = getAngle(normal, north)
        if normal.x < 0:
            return -angle + 2.0 * np.pi
        return angle

    def os_area(self) -> Vertex:
        """Returns area of the surface via openstudio."""
        return openstudio.getArea(self.to_Point3dVector()).get()

    def perimeter(self) -> float:
        """Returns the perimeter of the surface."""
        return sum([edge.length() for edge in self.to_Surface3dEdges()])

    def rough_centroid(self) -> Vertex:
        """Returns the centroid calculated in a rough way: the mean of the coordinates."""
        return Vertex.from_numpy(np.array([x.to_numpy() for x in self.vertices]).mean(axis=0))

    def os_centroid(self) -> Vertex:
        """Returns the centroid via openstudio."""
        centroid_ = openstudio.getCentroid(self.to_Point3dVector())
        if not centroid_.is_initialized():
            raise ValueError("OpenStudio failed to calculate centroid")
        return Vertex.from_Point3d(centroid_.get())

    def to_Point3dVector(self) -> List[openstudio.Point3d]:
        """Converts vertices to a list openstudio.Point3d."""
        return [v.to_Point3d() for v in self.vertices]

    def to_OSSurface(self, model: openstudio.model.Model) -> openstudio.model.Surface:
        """Creates an openstudio.model.Surface in the model passed as argument."""
        return openstudio.model.Surface(self.to_Point3dVector(), model)

    def to_numpy(self) -> np.ndarray:
        """Get a numpy array representing the vertices."""
        return np.array([v.to_numpy() for v in self.vertices])

    def to_Surface3dEdges(self) -> List[Surface3dEge]:
        """Converts vertex pairs to Surface3dEge."""
        edges = []
        for i, curVertex in enumerate(self.vertices):
            if i == len(self.vertices) - 1:
                nextVertex = self.vertices[0]
            else:
                nextVertex = self.vertices[i + 1]
            edges.append(Surface3dEge(start=curVertex, end=nextVertex, firstSurface=self))
        return edges

    def split_into_n_segments(self, n_segments, axis=None, plot=False) -> List[Surface]:
        """Splits a surface in N equal segments.

        If axis is not passed, it defaults to the first one of the plane
        eg: for a plane 'xy' it splits on 'x'
        """
        plot_axis = self.get_plot_axis()
        if axis is None:
            axis = plot_axis[0]
        if axis not in plot_axis:
            raise ValueError(f"This surface's plane is '{plot_axis}', so can't split on {axis=}")

        if n_segments < 2:
            raise ValueError("At least 2 segments needed")

        axis_to_index = {'x': 0, 'y': 1, 'z': 2}
        idx = axis_to_index[axis]
        v_np = self.to_numpy()
        minimum = v_np[:, idx].min()
        maximum = v_np[:, idx].max()
        segment_length = (maximum - minimum) / n_segments
        is_max = v_np[:, idx] == maximum
        is_min = ~is_max

        cur_min = minimum
        cur_max = cur_min + segment_length

        new_surfaces = []

        for i in range(n_segments):
            # print(cur_min, cur_max)
            v_np_i = v_np.copy()
            v_np_i[is_min, idx] = cur_min
            v_np_i[is_max, idx] = cur_max
            new_surface = Surface.from_numpy_array(v_np_i)
            if self.name:
                new_surface.name = f'{self.name}-{i+1}'
            new_surfaces.append(new_surface)

            cur_min = cur_max
            cur_max += segment_length

        if plot:
            fig, ax = plt.subplots(figsize=(16, 9))
            for new_surface in new_surfaces:
                new_surface.plot(ax=ax)

        return new_surfaces

    def rotate(self, degrees: float, axis=None) -> Surface:
        """Rotates a surface by an amount of degrees.

        Args:
        -----
        * degrees (float): the angle to rotate it by, in degrees. Positive means clockwise
        * axis (Vertex): if none, uses the Z axis

        Returns:
        ---------
        * a new Surface object with rotated vertices
        """
        if axis is None:
            axis = Vertex(0.0, 0.0, 1.0)

        # Lazy load to avoid circular import
        from geomeffibem.transformation import Transformation

        return Transformation.Rotation(axis=axis, radians=-openstudio.degToRad(degrees)) * self

    def translate(self, translation: Vertex) -> Surface:
        """Translates a surface along a translation vector."""
        from geomeffibem.transformation import Transformation

        return Transformation.Translation(translation=translation) * self

    def plot(self, name: Union[bool, str] = True, **kwargs):
        """Calls plot_vertices, cf help(plot_vertices)."""
        if isinstance(name, str):
            name = name
        elif name:
            name = self.name
        return plot_vertices(surface_like=self, name=name, **kwargs)

    def __repr__(self):
        """Repr."""
        s = ""
        if self.name is not None:
            s += f"Surface '{self.name}' = "
        s += "["
        if self.name is not None:
            s += "\n "
        imax = len(self.vertices) - 1
        for i, v in enumerate(self.vertices):
            if i > 0:
                s += " "
            s += f"{v}"
            if i < imax:
                s += ",\n"
        s += "]"
        return s


def get_surface_from_surface_like(surface_like: Union[Surface, List[Vertex], openstudio.model.Surface]) -> Surface:
    """Helper to get a Surface (class) from a surface like object."""
    if isinstance(surface_like, openstudio.openstudiomodelgeometry.Surface):
        surface = Surface.from_Surface(surface_like)
    elif isinstance(surface_like, Surface):
        surface = surface_like
    else:
        if isinstance(surface_like, list):
            surface_like = np.array(surface_like)

        if isinstance(surface_like[0], openstudio.Point3d):
            surface = Surface.from_Point3dVector(surface_like)
        elif isinstance(surface_like[0], np.ndarray):
            surface = Surface.from_numpy_array(surface_like)
        elif isinstance(surface_like[0], Vertex):
            surface = Surface(surface_like)

    return surface


def plot_vertices(
    surface_like: Union[Surface, List[Vertex], openstudio.openstudiomodelgeometry.Surface],
    ax=None,
    center_axes=False,
    with_rough_centroid=False,
    with_os_centroid=False,
    annotate=True,
    linewidth=None,
    name=None,
    plane=None,
    annotate_kwargs=dict(color='r', xytext=(5, 5), textcoords='offset points'),
    # Passed to ax.plot/plt.plot
    **kwargs,
):
    """Plot any surface-like object in 2D.

    Accepts a Surface, a list or numpy array of Vertex, or an openstudio.openstudiomodelgeometry.Surface object

    TODO: Assumes the surface is planar and falls exactly on 'xy', 'xz' or 'yz' plane currently
    """
    surface = get_surface_from_surface_like(surface_like=surface_like)

    points = surface.to_numpy()
    if plane is None:
        plane = surface.get_plot_axis()

    if plane == 'xy':
        xs = points[:, 0]
        ys = points[:, 1]
    elif plane == 'xz':
        xs = points[:, 0]
        ys = points[:, 2]
    elif plane == 'yz':
        xs = points[:, 1]
        ys = points[:, 2]
    else:
        raise ValueError("plane must be in ['xy', 'xz', 'yz']")
    if ax is None:
        # print("Making a figure")
        max_width = xs.max() - xs.min()
        max_height = ys.max() - ys.min()
        h = 6
        w = h * max_width / max_height
        # print(w, h)
        fig, ax = plt.subplots(figsize=(w, h))

    ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), marker='x', markeredgecolor='r', linewidth=linewidth, **kwargs)
    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])
    if annotate:
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.annotate(f"{i+1} ({x}, {y})", xy=(x, y), **annotate_kwargs)

    if with_rough_centroid:
        centroid_x, centroid_y = surface.rough_centroid().get_coords_on_plane(plane=plane)
        ax.annotate(f"rough ({centroid_x}, {centroid_y})", xy=(centroid_x, centroid_y))
        ax.plot(centroid_x, centroid_y, 'rx')
    if with_os_centroid or name is not None and name is not False:
        centroid_x, centroid_y = surface.os_centroid().get_coords_on_plane(plane=plane)
        if with_os_centroid:
            ax.annotate(f"os ({centroid_x}, {centroid_y})", xy=(centroid_x, centroid_y))
            ax.plot(centroid_x, centroid_y, 'gx')
            if name:
                ax.annotate(
                    name,
                    xy=(centroid_x, centroid_y),
                    xytext=(0, 50),
                    textcoords='offset pixels',
                    color='b',
                    arrowprops=dict(edgecolor='b', lw=1, ls='-', arrowstyle='->'),
                )
        else:
            ax.annotate(name, xy=(centroid_x, centroid_y), ha='center', va='center')

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')  # hide top axis

    if center_axes:
        ax.spines['bottom'].set_position('zero')  # x-axis where y=0
        ax.spines['left'].set_position('zero')
        ax.xaxis.set_label_position("top")

    return ax
