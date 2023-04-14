#!/usr/bin/env python
"""Tests for `geomeffibem` Surface class."""

import numpy as np
import pytest

from geomeffibem.plane import Plane
from geomeffibem.surface import Surface
from geomeffibem.vertex import Vertex


def test_plane():
    """Tests we get the right plane from a Surface."""
    floor_surface = Surface.Floor(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, z=3.0)
    p = floor_surface.get_plane()
    assert p == floor_surface.get_plane()
    assert p.is_orthogonal()
    assert np.array_equal(p.outwardNormal(), np.array([-0.0, -0.0, -1.0]))
    assert p.a == 0.0
    assert p.b == 0.0
    assert p.c == -1.0
    assert p.d == 3.0


def test_project():
    """Test projection of a Vertex on a plane and pointOnPlane."""
    p = Plane(0.0, 0.0, -1.0, 3.0)
    v = Vertex(10.0, 10.0, 5.0)
    v2 = p.project(v)
    assert v2 == Vertex(10.0, 10.0, 3.0)
    assert not p.pointOnPlane(v)
    assert p.pointOnPlane(v2)
    with pytest.raises(ValueError):
        p.project(p)
