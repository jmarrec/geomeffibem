#!/usr/bin/env python
"""Tests for `geomeffibem` Surface class."""

import numpy as np
import openstudio

from geomeffibem.vertex import Vertex


def test_vertex_from_numpy():
    arr = np.array([0.1, 0.2, 0.3])
    v = Vertex.from_numpy(arr)
    assert v.x == arr[0]
    assert v.y == arr[1]
    assert v.z == arr[2]


def test_vertex_from_list():
    mylist = [0.1, 0.2, 0.3]
    v = Vertex.from_numpy(mylist)
    assert v.x == mylist[0]
    assert v.y == mylist[1]
    assert v.z == mylist[2]


def test_vertex_to_numpy():
    v = Vertex(0.1, 0.2, 0.3)
    arr = v.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    assert v.x == arr[0]
    assert v.y == arr[1]
    assert v.z == arr[2]


def test_vertex_from_Point3d():
    pt = openstudio.Point3d(0.1, 0.2, 0.3)
    v = Vertex.from_Point3d(pt)
    assert v.x == pt.x()
    assert v.y == pt.y()
    assert v.z == pt.z()


def test_vertex_to_Point3d():
    v = Vertex(0.1, 0.2, 0.3)
    pt = v.to_Point3d()
    assert v.x == pt.x()
    assert v.y == pt.y()
    assert v.z == pt.z()


def test_vertex_equality():
    # Baked in tolerance is 0.0127 to match E+
    assert Vertex(0.0, 0.0, 0.0) == Vertex(0.0126, 0.0, 0.0)
    assert not (Vertex(0.0, 0.0, 0.0) != Vertex(0.0126, 0.0, 0.0))

    assert Vertex(0.0, 0.0, 0.0) != Vertex(0.0128, 0.0, 0.0)
    assert not (Vertex(0.0, 0.0, 0.0) == Vertex(0.0128, 0.0, 0.0))
