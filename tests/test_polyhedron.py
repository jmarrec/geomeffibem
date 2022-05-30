#!/usr/bin/env python
"""Tests for `geomeffibem` Polyhedron class."""

import openstudio
import pytest

from geomeffibem.polyhedron import Polyhedron, edgesInBoth
from geomeffibem.surface import Surface, Surface3dEge


@pytest.fixture
def zonePoly():
    """A fixture to create a Zone Polyhedron."""
    floor_surface = Surface.Rectangle(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, min_z=0.0, max_z=0.0)
    m = openstudio.model.Model()
    space = openstudio.model.Space.fromFloorPrint(floor_surface.to_Point3dVector(), 3.0, m).get()
    for sf in space.surfaces():
        if sf.surfaceType() == 'Floor':
            sf.setName("FLOOR")
        elif sf.surfaceType() == 'RoofCeiling':
            sf.setName("ROOF")
        else:
            group = sf.planarSurfaceGroup().get()
            site_transformation = group.siteTransformation()
            site_vertices = site_transformation * sf.vertices()
            site_outward_normal = openstudio.getOutwardNormal(site_vertices).get()
            north = openstudio.Vector3d(0.0, 1.0, 0.0)
            if site_outward_normal.x() < 0.0:
                azimuth = 360.0 - openstudio.radToDeg(openstudio.getAngle(site_outward_normal, north))
            else:
                azimuth = openstudio.radToDeg(openstudio.getAngle(site_outward_normal, north))

            if azimuth >= 315.0 or azimuth < 45.0:
                facade = "4-North"
            elif azimuth >= 45.0 and azimuth < 135.0:
                facade = "3-East"
            elif azimuth >= 135.0 and azimuth < 225.0:
                facade = "1-South"
            elif azimuth >= 225.0 and azimuth < 315.0:
                facade = "2-West"

            sf.setName(f"{facade}".upper())  # - Abs azimuth {azimuth:.2f}".upper())

    return Polyhedron([Surface.from_Surface(s) for s in m.getSurfaces()])


@pytest.fixture
def zonePolySplitWall(zonePoly):
    """A fixture to create a Zone Polyhedron with a split wall."""
    wall = zonePoly.get_surface_by_name('1-SOUTH')
    new_walls = wall.split_into_n_segments(n_segments=2, axis='x')

    return Polyhedron(surfaces=[x for x in zonePoly.surfaces if x.name != '1-SOUTH'] + new_walls)


def test_polyhedron(zonePoly):
    """Test a Polyhedron that is already enclosed in the first pass."""
    # This should work
    zonePoly.get_surface_by_name("1-SOUTH")
    # Six four-sided surfaces in a box
    assert zonePoly.numVertices() == 4 * 6

    isEnclosed, edgesNot2 = zonePoly.isEnclosedVolume()
    assert isEnclosed
    assert not edgesNot2


def test_polyhedron_split_wall(zonePolySplitWall):
    """Test a Polyhedron with a split wall, so it has to call updateZonePolygonsForMissingColinearPoints."""
    # This should work
    zonePolySplitWall.get_surface_by_name("1-SOUTH")
    # Six in a box, except one wall is split, so 7 four-sided surfaces
    assert zonePolySplitWall.numVertices() == 4 * 7

    isEnclosed, edgesNot2 = zonePolySplitWall.isEnclosedVolume()
    assert isEnclosed
    assert not edgesNot2


def test_polyhedron_not_enclosed(zonePoly):
    """Test a non enclosed volume."""
    # This should work
    zonePoly.surfaces.pop(0)
    # Six in a box, minus one, so 5 four-sided surfaces
    assert zonePoly.numVertices() == 4 * 5

    isEnclosed, edgesNot2 = zonePoly.isEnclosedVolume()
    assert isEnclosed is False
    assert edgesNot2


def test_cpp_generation(zonePoly):
    """Just trying to get coverage to ignore that stuff."""
    zonePoly.to_eplus_cpp_code()
    zonePoly.surfaces[0].name = None
    zonePoly.to_os_cpp_code()


def test_polyhedron_constructor():
    """Test the ctor error handling."""
    with pytest.raises(ValueError):
        Polyhedron(surfaces=5)

    with pytest.raises(ValueError):
        Polyhedron(surfaces=[5])


def test_edgesInBoth():
    """Test the helper edgesInBoth."""
    surface = Surface.Rectangle(min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, min_z=0.0, max_z=0.0)
    surface.name = 'Floor'
    edges = surface.to_Surface3dEdges()
    assert len(edges) == 4
    assert len(edgesInBoth(edges, edges)) == 4
    assert len(edgesInBoth(edges, edges[:2])) == 2
    assert len(edgesInBoth(edges, [])) == 0
    assert len(edgesInBoth([], edges)) == 0
