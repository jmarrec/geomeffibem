#!/usr/bin/env python
"""Tests for `geomeffibem` Surface class."""

import numpy as np
import openstudio
import pytest

from geomeffibem.surface import Surface, Surface3dEge, get_surface_from_surface_like, plot_vertices
from geomeffibem.vertex import Vertex


def test_surface_rectangle():
    """Tests Surface.Rectangle factory."""
    surface = Surface.Rectangle(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, min_z=0.0, max_z=0.0)
    assert np.array_equal(
        surface.to_numpy(),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [10.0, 10.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        ),
    )
    assert surface.get_plot_axis() == 'xy'

    surface = Surface.Rectangle(min_x=0.0, max_x=10.0, min_y=5.0, max_y=5.0, min_z=10.0, max_z=20.0)
    assert np.array_equal(
        surface.to_numpy(),
        np.array(
            [
                [0.0, 5.0, 20.0],
                [0.0, 5.0, 10.0],
                [10.0, 5.0, 10.0],
                [10.0, 5.0, 20.0],
            ]
        ),
    )

    assert surface.get_plot_axis() == 'xz'


def test_surface_roundtrip_openstudio():
    """Tests to/from openstudio.model.Surface."""
    surface = Surface.Rectangle(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, min_z=0.0, max_z=0.0)
    assert surface.os_area() == 100.0
    m = openstudio.model.Model()
    sf = openstudio.model.Surface(surface.to_Point3dVector(), m)
    assert isinstance(sf, openstudio.openstudiomodelgeometry.Surface)
    mysf = Surface.from_Surface(sf)
    assert np.array_equal(surface.to_numpy(), mysf.to_numpy())

    mysf2 = Surface.from_Point3dVector(mysf.to_Point3dVector())
    assert np.array_equal(surface.to_numpy(), mysf2.to_numpy())
    with pytest.raises(ValueError):
        Surface.from_Surface(mysf.to_numpy())


def test_surface_roundtrip_numpy():
    """Tests to/from numpy."""
    surface = Surface.Rectangle(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, min_z=0.0, max_z=0.0)
    surface2 = Surface.from_numpy_array(surface.to_numpy())
    assert np.array_equal(surface.to_numpy(), surface2.to_numpy())

    # with a list
    Surface.from_numpy_array(surface.to_numpy().tolist())

    with pytest.raises(ValueError):
        Surface.from_numpy_array(np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]]))

    with pytest.raises(ValueError):
        Surface.from_numpy_array(np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0], [10.0, 10.0, 0.0, 0.0]]))

    with pytest.raises(ValueError):
        Surface(vertices=surface)

    with pytest.raises(ValueError):
        Surface(vertices=surface.to_Point3dVector())


def test_Surface3dEdge():
    """Test Surface3dEge."""
    surface = Surface.Rectangle(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, min_z=0.0, max_z=0.0)
    surface.name = 'Floor'
    edge = Surface3dEge(start=Vertex(0.0, 0.0, 0.0), end=Vertex(0.0, 10.0, 0.01), firstSurface=surface)
    assert edge == surface.to_Surface3dEdges()[0]

    assert edge.containsPoints(Vertex(0.0, 1.0, 0.0))
    assert edge.containsPoints(Vertex(0.0, 1.0, 0.01))
    assert not edge.containsPoints(Vertex(0.0, 0.0, 0.0))
    assert not edge.containsPoints(Vertex(0.0, 1.0, 1.0))
    assert edge.count() == 1

    edge = Surface3dEge(start=Vertex(0.0, 0.0, 0.0), end=Vertex(0.0, 10.0, 1.0), firstSurface=surface)
    assert edge != surface.to_Surface3dEdges()[0]

    edge.plot_on_first_surface()


def test_Surface_get_plot_axis():
    """Test Surface.get_plot_axis."""
    floor_surface = Surface.Rectangle(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, min_z=0.0, max_z=0.0)
    assert floor_surface.get_plot_axis() == 'xy'

    yz_wall = Surface.from_numpy_array(np.array([[0.0, 10.0, 0.3], [0.0, 10.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.3]]))
    assert yz_wall.get_plot_axis() == 'yz'

    xz_wall = Surface.from_numpy_array(np.array([[0.0, 0.0, 0.3], [0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [30.0, 0.0, 0.3]]))
    assert xz_wall.get_plot_axis() == 'xz'

    # This should be able to figure out the plane by itself
    floor_surface.plot()
    yz_wall.plot()
    xz_wall.plot()


def test_Surface_centroid():
    """Test the centroid methods."""
    surface = Surface.Rectangle(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, min_z=0.0, max_z=0.0)
    assert surface.os_centroid() == Vertex(+5.0000, +5.0000, +0.0000)
    assert surface.rough_centroid() == Vertex(+5.0000, +5.0000, +0.0000)


def test_Surface_split():
    """Test splitting a surface."""
    surface = Surface.Rectangle(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, min_z=0.0, max_z=0.0)
    surface.name = "Surface"
    surface.os_centroid() == Vertex(+5.0000, +5.0000, +0.0000)

    s1, s2 = surface.split_into_n_segments(n_segments=2, axis=None, plot=False)
    assert s1.name == "Surface-1"
    assert np.array_equal(
        s1.to_numpy(), np.array([[0.0, 0.0, 0.0], [0.0, 10.0, 0.0], [5.0, 10.0, 0.0], [5.0, 0.0, 0.0]])
    )

    assert s2.name == "Surface-2"
    assert np.array_equal(
        s2.to_numpy(), np.array([[5.0, 0.0, 0.0], [5.0, 10.0, 0.0], [10.0, 10.0, 0.0], [10.0, 0.0, 0.0]])
    )

    with pytest.raises(ValueError):
        surface.split_into_n_segments(n_segments=2, axis='z', plot=False)

    s1, s2 = surface.split_into_n_segments(n_segments=2, axis='y', plot=False)
    assert s1.name == "Surface-1"
    assert np.array_equal(
        s1.to_numpy(), np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [10.0, 5.0, 0.0], [10.0, 0.0, 0.0]])
    )

    assert s2.name == "Surface-2"
    assert np.array_equal(
        s2.to_numpy(), np.array([[0.0, 5.0, 0.0], [0.0, 10.0, 0.0], [10.0, 10.0, 0.0], [10.0, 5.0, 0.0]])
    )

    with pytest.raises(ValueError):
        surface.split_into_n_segments(n_segments=1, axis='x', plot=False)


def test_split_plot():
    """Test split and plot."""
    surface = Surface.Rectangle(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, min_z=0.0, max_z=0.0)
    surface.name = "Surface"
    surface.split_into_n_segments(n_segments=2, axis='y', plot=True)


def test_Surface_rotate():
    """Test rotate."""
    surface = Surface.Rectangle(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, min_z=0.0, max_z=0.0)

    surface_rot = surface.rotate(degrees=-25.0)
    assert np.isclose(surface.os_area(), surface_rot.os_area())

    surface_rot = surface.rotate(degrees=-90.0)

    assert np.allclose(
        surface_rot.to_numpy(),
        np.array(
            [
                [+0.0, +0.0, +0.0],
                [-10.0, +0.0, +0.0],
                [-10.0, +10.0, +0.0],
                [+0.0, +10.0, +0.0],
            ]
        ),
    )


def test_surface_plot():
    """Test plotting."""
    surface = Surface.Rectangle(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, min_z=0.0, max_z=0.0)
    surface.name = "Floor"
    surface.plot(with_rough_centroid=True, with_os_centroid=True, center_axes=True)

    surface.plot(with_rough_centroid=True, with_os_centroid=False, center_axes=False)
    # no name
    surface.name = None
    surface.plot(with_rough_centroid=True, with_os_centroid=True, center_axes=False)
    surface.plot(with_rough_centroid=True, with_os_centroid=False, center_axes=False)
    surface.plot(name="Custom", with_os_centroid=False, center_axes=False)
    surface.plot(name=False, with_os_centroid=False, center_axes=False)

    with pytest.raises(ValueError):
        plot_vertices(surface, plane='wrong')


def test_get_surface_from_surface_like():
    """Test get_surface_from_surface_like."""
    mylist = [
        [+0.0, +0.0, +0.0],
        [-10.0, +0.0, +0.0],
        [-10.0, +10.0, +0.0],
        [+0.0, +10.0, +0.0],
    ]
    assert isinstance(get_surface_from_surface_like(mylist), Surface)
    assert isinstance(get_surface_from_surface_like(np.array(mylist)), Surface)
    surface = Surface.Rectangle(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, min_z=0.0, max_z=0.0)
    surface.name = "Floor"
    assert isinstance(get_surface_from_surface_like(surface), Surface)
    assert isinstance(get_surface_from_surface_like(surface.to_Point3dVector()), Surface)

    m = openstudio.model.Model()
    os_sf = surface.to_OSSurface(m)
    assert isinstance(get_surface_from_surface_like(os_sf), Surface)

    assert isinstance(get_surface_from_surface_like(surface.vertices), Surface)
