#!/usr/bin/env python
"""Tests for `geomeffibem` Surface class."""

import numpy as np
import openstudio

from geomeffibem.surface import Surface


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
    m = openstudio.model.Model()
    sf = openstudio.model.Surface(surface.to_Point3dVector(), m)
    assert isinstance(sf, openstudio.openstudiomodelgeometry.Surface)
    mysf = Surface.from_Surface(sf)
    assert np.array_equal(surface.to_numpy(), mysf.to_numpy())

    mysf2 = Surface.from_Point3dVector(mysf.to_Point3dVector())
    assert np.array_equal(surface.to_numpy(), mysf2.to_numpy())


def test_surface_roundtrip_numpy():
    """Tests to/from numpy."""
    surface = Surface.Rectangle(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, min_z=0.0, max_z=0.0)
    surface2 = Surface.from_numpy_array(surface.to_numpy())
    assert np.array_equal(surface.to_numpy(), surface2.to_numpy())
