#!/usr/bin/env python
"""Tests for `geomeffibem` Surface class."""

import numpy as np
import openstudio
import pytest

from geomeffibem.vertex import Vertex, distanceFromPointToLine, isPointOnLineBetweenPoints


def test_vertex_from_numpy():
    """Vertex from numpy."""
    arr = np.array([0.1, 0.2, 0.3])
    v = Vertex.from_numpy(arr)
    assert v.x == arr[0]
    assert v.y == arr[1]
    assert v.z == arr[2]
    with pytest.raises(ValueError):
        Vertex.from_numpy(np.array([1.0, 2.0, 3.0, 4.0]))


def test_vertex_from_list():
    """Vertex from a list."""
    mylist = [0.1, 0.2, 0.3]
    v = Vertex.from_numpy(mylist)
    assert v.x == mylist[0]
    assert v.y == mylist[1]
    assert v.z == mylist[2]


def test_vertex_to_numpy():
    """To numpy."""
    v = Vertex(0.1, 0.2, 0.3)
    arr = v.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    assert v.x == arr[0]
    assert v.y == arr[1]
    assert v.z == arr[2]


def test_vertex_from_Point3d():
    """From openstudio Point3d."""
    pt = openstudio.Point3d(0.1, 0.2, 0.3)
    v = Vertex.from_Point3d(pt)
    assert v.x == pt.x()
    assert v.y == pt.y()
    assert v.z == pt.z()


def test_vertex_to_Point3d():
    """To openstudio Point3d."""
    v = Vertex(0.1, 0.2, 0.3)
    pt = v.to_Point3d()
    assert v.x == pt.x()
    assert v.y == pt.y()
    assert v.z == pt.z()


def test_vertex_equality():
    """Tests equality with a tolerance."""
    # Baked in tolerance is 0.0127 to match E+
    assert Vertex(0.0, 0.0, 0.0) == Vertex(0.0126, 0.0, 0.0)
    assert not (Vertex(0.0, 0.0, 0.0) != Vertex(0.0126, 0.0, 0.0))

    assert Vertex(0.0, 0.0, 0.0) != Vertex(0.0128, 0.0, 0.0)
    assert not (Vertex(0.0, 0.0, 0.0) == Vertex(0.0128, 0.0, 0.0))


def test_coords_on_plane():
    """Tests the Vertex.get_coords_on_plane."""
    v = Vertex(1.0, 2.0, 3.0)
    assert v.get_coords_on_plane('xy') == (1.0, 2.0)
    assert v.get_coords_on_plane('yz') == (2.0, 3.0)
    assert v.get_coords_on_plane('xz') == (1.0, 3.0)
    with pytest.raises(ValueError):
        v.get_coords_on_plane('dkjngfdjkn')


def test_distanceFromPointToLine():
    """Tests free function distanceFromPointToLine."""
    start = Vertex(0.0, 0.0, 0.0)
    end = Vertex(10.0, 0.0, 0.0)
    testVertex = Vertex(5.0, 0.0, 0.0)
    assert distanceFromPointToLine(start=start, end=end, test=testVertex) == 0.0
    testVertex.y = 5.0
    assert distanceFromPointToLine(start=start, end=end, test=testVertex) == 5.0


def test_isPointOnLineBetweenPoints():
    """Test free function isPointOnLineBetweenPoints."""
    start = Vertex(0.0, 0.0, 0.0)
    end = Vertex(10.0, 0.0, 0.0)
    testVertex = Vertex(5.0, 0.0, 0.0)
    assert isPointOnLineBetweenPoints(start=start, end=end, test=testVertex)
    testVertex.x = 11.0
    assert not isPointOnLineBetweenPoints(start=start, end=end, test=testVertex)
    testVertex.x = 5.0
    testVertex.y = 0.0126
    assert isPointOnLineBetweenPoints(start=start, end=end, test=testVertex)

    testVertex.y = 1.0
    assert not isPointOnLineBetweenPoints(start=start, end=end, test=testVertex)
